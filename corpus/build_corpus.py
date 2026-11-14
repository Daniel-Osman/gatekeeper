"""Entry point for the corpus build: the only module under ``corpus/`` that writes files.

Default run: fetch the curated docs at the pinned commit (or reuse the cache),
chunk and label them, write ``chunks.jsonl``, embed the chunks, write
``embeddings.npy`` and ``embeddings.json``.

``--check`` reads the committed files, writes nothing, and is the gate: it
prints the counts and exits 1 on drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from corpus import chunk, fetch, labels
from corpus.chunk import Section
from corpus.labels import CorpusError

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
SOURCES_PATH = ROOT / "sources.json"
RULES_PATH = ROOT / "label_rules.json"
FORUM_PATH = ROOT / "forum.jsonl"
CHUNKS_PATH = ROOT / "chunks.jsonl"
EMBEDDINGS_NPY = ROOT / "embeddings.npy"
EMBEDDINGS_JSON = ROOT / "embeddings.json"

RECORD_KEYS = ("chunk_id", "router", "status", "source", "synthetic", "path", "heading", "text")
SOURCE_VALUES = ("official", "forum")

MIN_CHUNKS = 90
MIN_APP_CHUNKS = 30
FORUM_MIN = 15
FORUM_MAX = 20

EMBED_MODEL = "gemini-embedding-001"
EMBED_TASK = "RETRIEVAL_DOCUMENT"
EMBED_DIM = 768
EMBED_BATCH = 12
EMBED_TOKENS_PER_MINUTE = 25_000
EMBED_TOKENS_CEILING = 28_000
EMBED_MAX_ATTEMPTS = 5
EMBED_RETRY_FALLBACK_S = 20.0
EMBED_CHARS_PER_TOKEN = 4

log = logging.getLogger("corpus")


@dataclass(frozen=True)
class Summary:
    chunks: int
    app: int
    pages: int
    both: int
    legacy: int
    forum: int

    def line(self, commit: str, dim: str) -> str:
        return (
            f"corpus: {self.chunks} chunks (app={self.app} pages={self.pages} both={self.both})"
            f" | legacy={self.legacy} | forum={self.forum} synthetic"
            f" | model={EMBED_MODEL} dim={dim} | commit={commit[:12]}"
        )


def load_forum(path: Path = FORUM_PATH) -> tuple[Section, ...]:
    """Read the synthetic forum answers as sections; ``path`` keeps them out of docs/."""
    if not path.exists():
        raise CorpusError(f"missing forum set: {path}")
    sections: list[Section] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise CorpusError(f"{path.name} line {number}: invalid JSON ({exc})") from exc
        missing = {"path", "heading", "text"} - set(entry)
        if missing:
            raise CorpusError(f"{path.name} line {number}: missing keys {sorted(missing)}")
        if not entry["path"].startswith("forum/"):
            raise CorpusError(f"{path.name} line {number}: path must start with 'forum/', got {entry['path']!r}")
        sections.append(Section(entry["path"], entry["heading"], 0, entry["text"], 0))
    if not sections:
        raise CorpusError(f"{path.name}: no entries")
    return tuple(sections)


def assemble_records(
    manifest: dict,
    rules: tuple[labels.Rule, ...],
    *,
    cache_dir: Path = CACHE_DIR,
    forum_path: Path = FORUM_PATH,
) -> list[dict]:
    """Chunk every fetched file plus the forum set, sort, and label into records."""
    sections: list[Section] = []
    for entry in manifest["files"]:
        raw = fetch.read_cached(cache_dir, manifest, entry["path"])
        sections.extend(chunk.split_markdown(entry["path"], raw))
    sections.extend(load_forum(forum_path))
    if not sections:
        raise CorpusError(f"sources.json lists no paths under {cache_dir}")

    sections.sort(key=lambda section: section.sort_key)
    records: list[dict] = []
    for index, section in enumerate(sections, start=1):
        label, _rule = labels.apply(section.path, section.heading, rules)
        is_forum = section.path.startswith("forum/")
        records.append(
            {
                "chunk_id": f"c{index}",
                "router": label.router,
                "status": label.status,
                "source": "forum" if is_forum else "official",
                "synthetic": is_forum,
                "path": section.path,
                "heading": section.heading,
                "text": section.text,
            }
        )
    return records


def render_chunks(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def chunks_sha256(chunks_text: str) -> str:
    return hashlib.sha256(chunks_text.encode("utf-8")).hexdigest()


def summarise(records: list[dict]) -> Summary:
    return Summary(
        chunks=len(records),
        app=sum(1 for record in records if record["router"] == "app"),
        pages=sum(1 for record in records if record["router"] == "pages"),
        both=sum(1 for record in records if record["router"] == "both"),
        legacy=sum(1 for record in records if record["status"] == "legacy"),
        forum=sum(1 for record in records if record["source"] == "forum"),
    )


def check_rules(rules: tuple[labels.Rule, ...], manifest: dict, forum_sections: tuple[Section, ...]) -> list[str]:
    """Every legacy rule names a pinned doc that exists; every forum path is matched by one."""
    errors: list[str] = []
    sources = {entry["path"] for entry in manifest["files"]}
    for rule in rules:
        if rule.status != "legacy":
            continue
        if rule.superseded_by not in sources:
            errors.append(
                f"rule {rule.match!r}: superseded_by {rule.superseded_by!r} is not in sources.json"
            )
    for section in forum_sections:
        matched = [rule for rule in rules if rule.matches(section.path, section.heading)]
        if not matched:
            errors.append(f"forum chunk {section.path!r}: no rule matches, it would default to current")
        elif not any(rule.status == "legacy" for rule in matched):
            errors.append(f"forum chunk {section.path!r}: matched rule is not legacy")
    return errors


def check_records(
    records: list[dict],
    rules: tuple[labels.Rule, ...],
    manifest_paths: set[str],
    *,
    forum_min: int = FORUM_MIN,
    forum_max: int = FORUM_MAX,
) -> list[str]:
    """Structural invariants over the record list; each error names the chunk."""
    errors: list[str] = []
    seen: set[str] = set()

    for expected_index, record in enumerate(records, start=1):
        chunk_id = record.get("chunk_id", f"<row {expected_index}>")
        where = f"chunk {chunk_id} ({record.get('path', '?')})"

        if tuple(record) != RECORD_KEYS:
            errors.append(f"{where}: keys {tuple(record)} != {RECORD_KEYS}")
            continue
        if chunk_id in seen:
            errors.append(f"{where}: duplicate chunk_id")
        seen.add(chunk_id)
        if chunk_id != f"c{expected_index}":
            errors.append(f"{where}: expected id c{expected_index}, ids must run c1..cN in order")
        if record["router"] not in labels.ROUTERS:
            errors.append(f"{where}: router {record['router']!r} outside {labels.ROUTERS}")
        if record["status"] not in labels.STATUSES:
            errors.append(f"{where}: status {record['status']!r} outside {labels.STATUSES}")
        if record["source"] not in SOURCE_VALUES:
            errors.append(f"{where}: source {record['source']!r} outside {SOURCE_VALUES}")
        if not isinstance(record["synthetic"], bool):
            errors.append(f"{where}: synthetic must be a bool")
        if not record["heading"].strip():
            errors.append(f"{where}: empty heading")
        if not record["text"].strip():
            errors.append(f"{where}: empty text")

        is_forum = record["source"] == "forum"
        if is_forum and record["synthetic"] is not True:
            errors.append(f"{where}: forum chunk must carry synthetic: true")
        if not is_forum and record["synthetic"] is not False:
            errors.append(f"{where}: official chunk must carry synthetic: false")
        if is_forum and not record["path"].startswith("forum/"):
            errors.append(f"{where}: forum chunk path must start with 'forum/'")
        if not is_forum and record["path"].startswith("forum/"):
            errors.append(f"{where}: forum path with source {record['source']!r}")
        if is_forum and record["status"] != "legacy":
            errors.append(f"{where}: forum chunk must be legacy (it predates the current API)")
        if not is_forum and record["path"] not in manifest_paths:
            errors.append(f"{where}: path missing from sources.json")

        if record["status"] == "legacy" and not is_forum:
            _label, rule = labels.apply(record["path"], record["heading"], rules)
            if rule is None:
                errors.append(f"{where}: legacy chunk matched no rule with a reason")

    ids = [record.get("chunk_id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk_id values across the corpus")

    forum_count = sum(1 for record in records if record["source"] == "forum")
    if not forum_min <= forum_count <= forum_max:
        errors.append(f"forum chunks: {forum_count} outside {forum_min}..{forum_max}")
    legacy_count = sum(1 for record in records if record["status"] == "legacy")
    if legacy_count < 1:
        errors.append("no legacy chunks: version-confusion queries would have nothing to hit")

    return errors


def check_embeddings(records: list[dict], embeddings_json: dict, vectors: np.ndarray, chunks_text: str) -> list[str]:
    """Rows, order, dimensions and the chunk-text hash must all agree with chunks.jsonl."""
    errors: list[str] = []
    expected_ids = [record["chunk_id"] for record in records]

    if embeddings_json.get("chunk_ids") != expected_ids:
        stored = embeddings_json.get("chunk_ids") or []
        mismatch = next(
            (f"{a} != {b}" for a, b in zip(stored, expected_ids) if a != b),
            f"length {len(stored)} != {len(expected_ids)}",
        )
        errors.append(f"embeddings.json chunk_ids do not match chunk order: {mismatch}")
    if embeddings_json.get("count") != len(records):
        errors.append(f"embeddings.json count {embeddings_json.get('count')} != {len(records)} chunks")
    if embeddings_json.get("chunks_sha256") != chunks_sha256(chunks_text):
        errors.append("embeddings.json chunks_sha256 does not match chunks.jsonl; re-run the build")
    if embeddings_json.get("model") != EMBED_MODEL:
        errors.append(f"embeddings.json model {embeddings_json.get('model')!r} != {EMBED_MODEL!r}")

    dim = embeddings_json.get("dim")
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        errors.append(f"embeddings.npy rows {vectors.shape[0] if vectors.ndim else '?'} != {len(records)} chunks")
    elif dim is not None and vectors.shape[1] != dim:
        errors.append(f"embeddings.npy width {vectors.shape[1]} != dim {dim}")
    return errors


def check_corpus(
    *,
    chunks_path: Path = CHUNKS_PATH,
    rules_path: Path = RULES_PATH,
    sources_path: Path = SOURCES_PATH,
    forum_path: Path = FORUM_PATH,
    embeddings_json: Path = EMBEDDINGS_JSON,
    embeddings_npy: Path = EMBEDDINGS_NPY,
    min_chunks: int = MIN_CHUNKS,
    min_app_chunks: int = MIN_APP_CHUNKS,
    forum_min: int = FORUM_MIN,
    forum_max: int = FORUM_MAX,
    allow_missing_embeddings: bool = False,
) -> tuple[Summary, list[str], list[str]]:
    """Offline gate over committed files. Returns ``(summary, errors, notes)``."""
    errors: list[str] = []
    notes: list[str] = []

    if not chunks_path.exists():
        raise CorpusError(f"missing {chunks_path}; run the build first")
    chunks_text = chunks_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in chunks_text.splitlines() if line.strip()]

    manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest.get("files", [])}
    rules = labels.load_rules(rules_path)
    forum_sections = load_forum(forum_path)

    errors.extend(check_rules(rules, manifest, forum_sections))
    errors.extend(check_records(records, rules, manifest_paths, forum_min=forum_min, forum_max=forum_max))

    summary = summarise(records)
    if summary.chunks < min_chunks:
        errors.append(f"{summary.chunks} chunks is below the minimum {min_chunks}")
    if summary.app < min_app_chunks:
        errors.append(f"app slice is {summary.app} chunks, below the minimum {min_app_chunks}")

    if embeddings_json.exists() and embeddings_npy.exists():
        payload = json.loads(embeddings_json.read_text(encoding="utf-8"))
        vectors = np.load(embeddings_npy)
        if payload.get("source_commit") != manifest.get("source_commit"):
            errors.append(
                f"embeddings.json source_commit {payload.get('source_commit')!r}"
                f" != sources.json {manifest.get('source_commit')!r}"
            )
        errors.extend(check_embeddings(records, payload, vectors, chunks_text))
    elif allow_missing_embeddings:
        notes.append("embeddings absent: chunks-only build, embedding assertions skipped")
    else:
        errors.append("embeddings.json / embeddings.npy missing; run the full build")

    return summary, errors, notes


def resolve_api_key() -> str:
    """GEMINI_API_KEY first (that is the name in this environment), GOOGLE_API_KEY accepted."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise CorpusError("GEMINI_API_KEY is not set; pass --no-embed to build chunks only")
    return key


def _batch_tokens(response, texts: list[str]) -> int:
    """Prompt tokens for one batch: API usage when reported, else a chars/4 estimate."""
    usage = getattr(response, "usage_metadata", None)
    reported = getattr(usage, "total_token_count", None) if usage is not None else None
    if isinstance(reported, int) and reported > 0:
        return reported
    chars = sum(len(text) for text in texts)
    return max(1, -(-chars // EMBED_CHARS_PER_TOKEN))


def _duration_seconds(raw: object) -> float | None:
    """Read a duration as the API states it: ``"20s"``, ``20`` or ``{"seconds": 20}``."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        return _duration_seconds(raw.get("seconds"))
    if isinstance(raw, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", raw)
        return float(match.group(1)) if match else None
    return None


def _retry_delay(exc: Exception) -> float:
    """Seconds the API asked for in ``RetryInfo``; the 20 s fallback when it says nothing."""
    payload = getattr(exc, "details", None)
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details or []:
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith("RetryInfo"):
            continue
        seconds = _duration_seconds(detail.get("retryDelay") or detail.get("retry_delay"))
        if seconds is not None:
            return seconds
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else EMBED_RETRY_FALLBACK_S


def _embed_batch(client, types, errors, *, model, task, dim, texts, number, total):
    """One embed call; a 429 waits the RetryInfo delay and retries, up to EMBED_MAX_ATTEMPTS."""
    for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
        try:
            return client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=dim),
            )
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == EMBED_MAX_ATTEMPTS:
                raise
            delay = _retry_delay(exc)
            log.warning(
                "embed batch %d/%d: 429 on attempt %d, retrying in %.1fs", number, total, attempt, delay
            )
            time.sleep(delay)
    raise AssertionError("embed retry loop exited without a response")


def embed_records(
    records: list[dict],
    *,
    model: str = EMBED_MODEL,
    task: str = EMBED_TASK,
    dim: int = EMBED_DIM,
    batch: int = EMBED_BATCH,
) -> np.ndarray:
    """Embed chunk texts in batches, paced under the free-tier tokens-per-minute quota."""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=resolve_api_key(), http_options={"api_version": "v1beta"})
    total = (len(records) + batch - 1) // batch
    vectors: list[list[float]] = []
    for number, start in enumerate(range(0, len(records), batch), start=1):
        window = records[start : start + batch]
        texts = [record["text"] for record in window]
        began = time.monotonic()
        response = _embed_batch(
            client, types, errors, model=model, task=task, dim=dim, texts=texts, number=number, total=total
        )
        elapsed = time.monotonic() - began
        embeddings = getattr(response, "embeddings", None) or []
        if len(embeddings) != len(window):
            raise CorpusError(
                f"embedding batch at row {start}: got {len(embeddings)} vectors for {len(window)} chunks"
            )
        tokens = _batch_tokens(response, texts)
        vectors.extend(list(embedding.values) for embedding in embeddings)
        if number < total:
            # Target 25k tokens/minute; the ceiling term keeps any fast batch under 28k.
            pause = max(
                tokens * 60.0 / EMBED_TOKENS_PER_MINUTE - elapsed,
                tokens * 60.0 / EMBED_TOKENS_CEILING,
                0.0,
            )
            print(f"embed batch {number}/{total}: {tokens:,} tokens, sleeping {pause:.1f}s", flush=True)
            time.sleep(pause)
        else:
            print(f"embed batch {number}/{total}: {tokens:,} tokens, sleeping 0.0s", flush=True)

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(records), dim):
        raise CorpusError(f"embedding matrix shape {matrix.shape} != ({len(records)}, {dim})")
    return matrix


def write_chunks(records: list[dict], out: Path = CHUNKS_PATH) -> None:
    out.write_text(render_chunks(records), encoding="utf-8")


def write_embeddings(
    records: list[dict],
    vectors: np.ndarray,
    chunks_text: str,
    source_commit: str,
    built_at: str,
) -> None:
    np.save(EMBEDDINGS_NPY, vectors)
    payload = {
        "model": EMBED_MODEL,
        "task_type": EMBED_TASK,
        "dim": int(vectors.shape[1]),
        "count": len(records),
        "chunk_ids": [record["chunk_id"] for record in records],
        "chunks_sha256": chunks_sha256(chunks_text),
        "source_commit": source_commit,
        "built_at": built_at,
    }
    EMBEDDINGS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build(*, no_embed: bool, rebuild: bool, repin: bool) -> str:
    """Default run: fetch → chunk → label → write → embed. Returns the summary line."""
    manifest = fetch.fetch_all(CACHE_DIR, SOURCES_PATH, repin=repin, rebuild=rebuild)
    SOURCES_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rules = labels.load_rules(RULES_PATH)
    records = assemble_records(manifest, rules)
    chunks_text = render_chunks(records)
    CHUNKS_PATH.write_text(chunks_text, encoding="utf-8")
    summary = summarise(records)

    if no_embed:
        if EMBEDDINGS_JSON.exists():
            payload = json.loads(EMBEDDINGS_JSON.read_text(encoding="utf-8"))
            if payload.get("count") != len(records):
                raise CorpusError(
                    f"embeddings.json counts {payload.get('count')} rows, chunks.jsonl now has {len(records)};"
                    " run without --no-embed"
                )
        return summary.line(manifest["source_commit"], "kept") + " | embedded=skipped"

    vectors = embed_records(records)
    write_embeddings(
        records,
        vectors,
        chunks_text,
        manifest["source_commit"],
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return summary.line(manifest["source_commit"], str(vectors.shape[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m corpus.build_corpus", description=__doc__)
    parser.add_argument("--check", action="store_true", help="offline invariant check; writes nothing, exit 1 on drift")
    parser.add_argument("--no-embed", action="store_true", help="re-chunk and re-label only; keep embeddings on disk")
    parser.add_argument("--rebuild", action="store_true", help="refetch every curated path at the pinned commit")
    parser.add_argument("--repin", action="store_true", help="resolve a new commit (writes a new pin)")
    parser.add_argument(
        "--allow-missing-embeddings",
        action="store_true",
        help="with --check: accept a chunks-only build (used before the first embedding run)",
    )
    parser.add_argument("--verbose", action="store_true", help="log each fetch and embedding batch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        if args.check:
            summary, errors, notes = check_corpus(allow_missing_embeddings=args.allow_missing_embeddings)
            commit = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["source_commit"]
            for note in notes:
                print(f"note: {note}")
            print(summary.line(commit, "kept"))
            if errors:
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                return 1
            print("check: ok")
            return 0

        print(build(no_embed=args.no_embed, rebuild=args.rebuild, repin=args.repin))
        return 0
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
