"""Offline proofs for the committed corpus and the corpus build pipeline.

Every test reads the committed artifacts or the fixture tree under
``tests/fixtures/docs``. No test calls a model, opens a socket, or depends on the
working directory: paths come from ``__file__`` and the socket is refused.

Criteria from ``SPEC-corpus.md`` are mapped one test per sentence or branch;
embedding-value determinism needs the model and is not asserted here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from corpus import build_corpus, chunk, labels

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOCS = FIXTURES / "docs"
EXPECTED = FIXTURES / "corpus_expected.jsonl"

CHUNKS_PATH = CORPUS / "chunks.jsonl"
SOURCES_PATH = CORPUS / "sources.json"
RULES_PATH = CORPUS / "label_rules.json"
FORUM_PATH = CORPUS / "forum.jsonl"
EMBEDDINGS_JSON = CORPUS / "embeddings.json"
EMBEDDINGS_NPY = CORPUS / "embeddings.npy"

RECORD_KEYS = ("chunk_id", "router", "status", "source", "synthetic", "path", "heading", "text")
ROUTERS = {"app", "pages", "both"}
STATUSES = {"current", "legacy"}
SOURCE_VALUES = {"official", "forum"}
HTTP_MODULES = {"urllib", "http", "socket", "httpx", "requests", "aiohttp", "urllib3"}
SHA_DRIFT = "chunks_sha256 does not match"

APP_DOC = "docs/01-app/01-getting-started/03-layouts-and-pages.mdx"
LEGACY_DOC = "docs/01-app/03-api-reference/03-file-conventions/middleware.mdx"
PAGES_DOC = "docs/02-pages/01-getting-started/data-fetching.mdx"

FIXTURE_COMMIT = "fixture-commit"
FIXTURE_FORUM = {
    "path": "forum/pages/unstable-revalidate-in-getstaticprops.md",
    "heading": "ISR revalidate not working",
    "text": (
        "In Next 12.1 you set unstable_revalidate in getStaticProps. That key was "
        "removed; the current key is revalidate."
    ),
}

EXPECTED_RECORDS = [
    ("c1", "app", "current", "official", False, "Creating a page"),
    ("c2", "app", "current", "official", False, "Layouts and Pages"),
    ("c3", "app", "current", "official", False, "Long section"),
    ("c4", "app", "current", "official", False, "Long section (part 2)"),
    ("c5", "app", "current", "official", False, "Retrieving data"),
    ("c6", "app", "legacy", "official", False, "Matcher"),
    ("c7", "app", "legacy", "official", False, "Usage"),
    ("c8", "app", "legacy", "official", False, "middleware"),
    ("c9", "pages", "current", "official", False, "Caveats"),
    ("c10", "pages", "current", "official", False, "Data Fetching"),
    ("c11", "pages", "current", "official", False, "Nested note"),
    ("c12", "pages", "current", "official", False, "getServerSideProps"),
    ("c13", "pages", "legacy", "forum", True, "ISR revalidate not working"),
]

EXPECTED_IDS = [
    ("c1", APP_DOC, "Creating a page"),
    ("c2", APP_DOC, "Layouts and Pages"),
    ("c3", APP_DOC, "Long section"),
    ("c4", APP_DOC, "Long section (part 2)"),
    ("c5", APP_DOC, "Retrieving data"),
    ("c6", LEGACY_DOC, "Matcher"),
    ("c7", LEGACY_DOC, "Usage"),
    ("c8", LEGACY_DOC, "middleware"),
    ("c9", PAGES_DOC, "Caveats"),
    ("c10", PAGES_DOC, "Data Fetching"),
    ("c11", PAGES_DOC, "Nested note"),
    ("c12", PAGES_DOC, "getServerSideProps"),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail fast if a corpus test reaches a socket or an embedding call."""

    def refuse(*args, **kwargs):
        raise AssertionError("corpus tests must not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(build_corpus, "embed_records", refuse)
    monkeypatch.setattr(build_corpus, "resolve_api_key", refuse)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> tuple[labels.Rule, ...]:
    return labels.load_rules(RULES_PATH)


@pytest.fixture(scope="module")
def fixture_sections() -> dict[str, tuple[chunk.Section, ...]]:
    return {
        path: chunk.split_markdown(path, (DOCS / path).read_text(encoding="utf-8"))
        for path in (APP_DOC, LEGACY_DOC, PAGES_DOC)
    }


# --- contract literals -------------------------------------------------------


def test_production_constants_match_the_spec_contract():
    assert build_corpus.RECORD_KEYS == RECORD_KEYS
    assert set(labels.ROUTERS) == ROUTERS
    assert set(labels.STATUSES) == STATUSES
    assert set(build_corpus.SOURCE_VALUES) == SOURCE_VALUES
    assert chunk.MAX_SECTION_CHARS == 2000
    assert (build_corpus.FORUM_MIN, build_corpus.FORUM_MAX) == (15, 20)


# --- committed record invariants ---------------------------------------------


def test_the_committed_corpus_holds_246_records(records):
    assert len(records) == 246


def test_every_record_has_exactly_the_eight_keys_in_order(records):
    for record in records:
        assert tuple(record) == RECORD_KEYS, record.get("chunk_id")


def test_every_field_has_the_declared_type(records):
    for record in records:
        where = record.get("chunk_id")
        for key in ("chunk_id", "router", "status", "source", "path", "heading", "text"):
            assert isinstance(record[key], str), (where, key)
        assert type(record["synthetic"]) is bool, where


def test_chunk_ids_are_unique_and_contiguous(records):
    ids = [record["chunk_id"] for record in records]
    assert ids == [f"c{number}" for number in range(1, 247)]
    assert all(re.fullmatch(r"c[1-9]\d*", chunk_id) for chunk_id in ids)
    assert len(set(ids)) == len(ids)


def test_router_values_are_app_pages_or_both(records):
    assert {record["router"] for record in records} <= ROUTERS
    assert all(record["router"] in ROUTERS for record in records)


def test_status_values_are_current_or_legacy(records):
    assert all(record["status"] in STATUSES for record in records)


def test_text_is_non_empty_and_no_body_exceeds_the_split_cap(records):
    for record in records:
        assert record["text"].strip(), record["chunk_id"]
        assert record["heading"].strip(), record["chunk_id"]
        body = record["text"].split("\n", 1)[1].lstrip("\n") if "\n" in record["text"] else ""
        assert len(body) <= 2000, (record["chunk_id"], len(body))


def test_every_official_path_resolves_to_sources_json(records, manifest):
    listed = {entry["path"] for entry in manifest["files"]}
    official = {record["path"] for record in records if record["source"] == "official"}
    assert official == listed
    assert len(listed) == 17
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"])
    assert not [record["chunk_id"] for record in records if record["path"].startswith("forum/") and record["source"] == "official"]


def test_forum_records_are_exactly_the_synthetic_set(records):
    entries = [
        json.loads(line)
        for line in FORUM_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forum = [record for record in records if record["source"] == "forum"]
    assert 15 <= len(forum) <= 20
    assert len(entries) == len(forum) == 17
    by_path = {entry["path"]: entry for entry in entries}
    assert {record["path"] for record in forum} == set(by_path)
    for record in forum:
        entry = by_path[record["path"]]
        assert record["heading"] == entry["heading"], record["chunk_id"]
        assert record["text"] == entry["text"], record["chunk_id"]
        assert record["synthetic"] is True, record["chunk_id"]
        assert record["path"].startswith("forum/"), record["chunk_id"]
    for record in records:
        if record["source"] != "forum":
            assert record["synthetic"] is False, record["chunk_id"]
            assert not record["path"].startswith("forum/"), record["chunk_id"]


def test_summary_counts_match_the_records(records):
    summary = build_corpus.summarise(records)
    assert (summary.chunks, summary.app, summary.pages, summary.both) == (246, 196, 50, 0)
    assert (summary.legacy, summary.forum) == (21, 17)


def test_committed_labels_match_the_rule_table(records, rules):
    """A drift check over the real table: every committed label is reproduced."""
    for record in records:
        label, _rule = labels.apply(record["path"], record["heading"], rules)
        assert (record["router"], record["status"]) == (label.router, label.status), record["chunk_id"]

    legacy = [record for record in records if record["status"] == "legacy"]
    assert len(legacy) >= 1
    assert len(legacy) == 21
    for record in legacy:
        _label, rule = labels.apply(record["path"], record["heading"], rules)
        assert rule is not None, record["chunk_id"]
        assert rule.reason.strip(), record["chunk_id"]


# --- splitter and label oracles ----------------------------------------------


def _serialise(groups: dict[str, tuple[chunk.Section, ...]]) -> bytes:
    payload = [
        {
            "path": section.path,
            "heading": section.heading,
            "level": section.level,
            "text": section.text,
            "ordinal": section.ordinal,
        }
        for path in sorted(groups)
        for section in groups[path]
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def test_splitter_is_deterministic_byte_for_byte(fixture_sections):
    second = {
        path: chunk.split_markdown(path, (DOCS / path).read_text(encoding="utf-8"))
        for path in sorted(fixture_sections)
    }
    assert _serialise(fixture_sections) == _serialise(second)
    assert len(_serialise(fixture_sections)) > 0


def test_fixture_chunk_ids_are_stable(fixture_sections):
    ordered = sorted(
        (section for group in fixture_sections.values() for section in group),
        key=lambda section: section.sort_key,
    )
    ids = [(f"c{number}", section.path, section.heading) for number, section in enumerate(ordered, start=1)]
    assert ids == EXPECTED_IDS


def test_assembly_is_deterministic_and_matches_the_snapshot(tmp_path, rules):
    manifest, cache, forum = _fixture_inputs(tmp_path)
    first = build_corpus.assemble_records(manifest, rules, cache_dir=cache, forum_path=forum)
    second = build_corpus.assemble_records(manifest, rules, cache_dir=cache, forum_path=forum)

    first_text = build_corpus.render_chunks(first)
    assert first_text == build_corpus.render_chunks(second)
    assert first_text == EXPECTED.read_text(encoding="utf-8")

    actual = [
        (record["chunk_id"], record["router"], record["status"], record["source"],
         record["synthetic"], record["heading"])
        for record in first
    ]
    assert actual == EXPECTED_RECORDS


def test_preamble_uses_the_frontmatter_title(fixture_sections):
    preamble = fixture_sections[APP_DOC][0]
    assert preamble.heading == "Layouts and Pages"
    assert preamble.level == 0
    assert preamble.text.startswith("Layouts and Pages\n")
    assert "PREAMBLE-MARK" in preamble.text

    pages_preamble = fixture_sections[PAGES_DOC][0]
    assert pages_preamble.heading == "Data Fetching"
    assert pages_preamble.level == 0
    assert "PAGES-MARK" in pages_preamble.text


def test_a_hash_line_inside_a_fence_is_not_a_heading(fixture_sections):
    headings = [section.heading for group in fixture_sections.values() for section in group]
    assert not [heading for heading in headings if "not a heading" in heading.lower()]
    page = next(section for section in fixture_sections[APP_DOC] if section.heading == "Creating a page")
    assert "# not a heading: this line sits inside a fence" in page.text
    fetch = next(section for section in fixture_sections[PAGES_DOC] if section.heading == "getServerSideProps")
    assert "### not a heading: this line sits inside a fence" in fetch.text


def test_a_short_section_merges_forward_without_losing_text(fixture_sections):
    headings = [section.heading for section in fixture_sections[APP_DOC]]
    assert "Short note" not in headings
    merged = next(section for section in fixture_sections[APP_DOC] if section.heading == "Retrieving data")
    assert merged.text.startswith("## Short note")
    assert "SHORT-MARK" in merged.text
    assert "DATA-MARK" in merged.text


def test_a_long_section_splits_at_paragraph_boundaries(fixture_sections):
    long_sections = [section for section in fixture_sections[APP_DOC] if section.heading.startswith("Long section")]
    assert [section.heading for section in long_sections] == ["Long section", "Long section (part 2)"]
    for section in long_sections:
        body = section.text.split("\n", 1)[1].lstrip("\n")
        assert len(body) <= 2000, len(body)
    combined = "\n".join(section.text for section in long_sections)
    for marker in ("LONG-P1", "LONG-P2", "LONG-P3"):
        assert marker in combined


def test_heading_levels_are_kept(fixture_sections):
    page = next(section for section in fixture_sections[APP_DOC] if section.heading == "Creating a page")
    assert page.text.startswith("## ")
    nested = next(section for section in fixture_sections[PAGES_DOC] if section.heading == "Nested note")
    assert nested.text.startswith("### ")
    assert "NESTED-MARK" in nested.text


def test_first_matching_rule_wins():
    table = (
        labels.Rule(match="docs/*", router="app", status=None, reason="first rule"),
        labels.Rule(match="docs/private/*", router="pages", status=None, reason="second rule"),
    )
    assert labels.apply("docs/private/page.mdx", "Anything", table)[0] == labels.Labels("app", "current")


def test_a_rule_router_overrides_the_path_default():
    table = (labels.Rule(match="docs/01-app/*", router="pages", status=None, reason="override"),)
    assert labels.apply("docs/01-app/page.mdx", "Anything", table)[0] == labels.Labels("pages", "current")


def test_a_heading_keyword_rule_fires_only_on_its_keyword():
    table = (
        labels.Rule(
            match="docs/*",
            router=None,
            status="legacy",
            reason="old API only",
            heading_keywords=("old",),
        ),
    )
    assert labels.apply("docs/01-app/page.mdx", "The old API", table)[0] == labels.Labels("app", "legacy")
    assert labels.apply("docs/01-app/page.mdx", "The new API", table)[0] == labels.Labels("app", "current")


def test_path_defaults_cover_both_routers_and_forum():
    assert labels.apply("docs/01-app/x.mdx", "H", ())[0] == labels.Labels("app", "current")
    assert labels.apply("docs/02-pages/x.mdx", "H", ())[0] == labels.Labels("pages", "current")
    assert labels.apply("forum/app/x.md", "H", ())[0] == labels.Labels("app", "current")
    assert labels.apply("forum/pages/x.md", "H", ())[0] == labels.Labels("pages", "current")
    assert labels.apply("forum/shared/x.md", "H", ())[0] == labels.Labels("both", "current")


def test_load_rules_rejects_a_rule_without_a_reason(tmp_path):
    path = tmp_path / "label_rules.json"
    path.write_text(json.dumps({"rules": [{"match": "docs/*", "router": "app"}]}), encoding="utf-8")
    with pytest.raises(labels.CorpusError, match="missing 'reason'"):
        labels.load_rules(path)


def test_load_rules_rejects_a_legacy_rule_without_a_successor(tmp_path):
    path = tmp_path / "label_rules.json"
    path.write_text(
        json.dumps({"rules": [{"match": "docs/*", "status": "legacy", "reason": "stale"}]}),
        encoding="utf-8",
    )
    with pytest.raises(labels.CorpusError, match="needs 'superseded_by'"):
        labels.load_rules(path)


# --- --check from a subprocess ----------------------------------------------


def test_check_exits_zero_and_prints_the_counts(tmp_path):
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(
        "import socket\n"
        "def _refuse(*args, **kwargs):\n"
        "    raise RuntimeError('network access from corpus --check')\n"
        "socket.socket.connect = _refuse\n"
        "socket.create_connection = _refuse\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}}
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(ROOT)])

    result = subprocess.run(
        [sys.executable, "-m", "corpus.build_corpus", "--check"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "246 chunks" in result.stdout
    assert "app=196 pages=50" in result.stdout
    assert "legacy=21" in result.stdout
    assert "forum=17 synthetic" in result.stdout
    assert "check: ok" in result.stdout


# --- --check failure modes ---------------------------------------------------


def _assert_diagnostics(errors: list[str], *expected: str) -> None:
    actual = [error for error in errors if SHA_DRIFT not in error]
    assert len(actual) == len(expected), actual
    unused = list(actual)
    for substring in expected:
        for index, error in enumerate(unused):
            if substring in error:
                unused.pop(index)
                break
        else:
            raise AssertionError(f"{substring!r} not in {actual}")
    assert not unused, unused


def _committed_records() -> list[dict]:
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _official(records: list[dict]) -> dict:
    return next(record for record in records if record["source"] == "official")


def _forum(records: list[dict]) -> dict:
    return next(record for record in records if record["source"] == "forum")


def _check(tmp_path, records: list[dict] | None = None, **kwargs):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(build_corpus.render_chunks(records or _committed_records()), encoding="utf-8")
    return build_corpus.check_corpus(chunks_path=chunks_path, **kwargs)


def _drop_key(records):
    records[0].pop("synthetic")


def _duplicate_id(records):
    records[1]["chunk_id"] = "c1"


def _gap_id(records):
    records[1]["chunk_id"] = "c999"


def _bad_router(records):
    records[0]["router"] = "edge"


def _bad_status(records):
    records[0]["status"] = "deprecated"


def _bad_source(records):
    _official(records)["source"] = "external"


def _synthetic_not_bool(records):
    _official(records)["synthetic"] = 1


def _empty_heading(records):
    records[0]["heading"] = "   "


def _empty_text(records):
    records[0]["text"] = ""


def _forum_not_synthetic(records):
    _forum(records)["synthetic"] = False


def _official_synthetic(records):
    _official(records)["synthetic"] = True


def _forum_path_moved(records):
    _forum(records)["path"] = "docs/forum-topic.md"


def _official_forum_path(records):
    _official(records)["path"] = "forum/app/borrowed.md"


def _forum_status_current(records):
    _forum(records)["status"] = "current"


def _official_missing_source(records):
    _official(records)["path"] = "docs/01-app/not-in-sources.mdx"


def _legacy_without_rule(records):
    _official(records)["status"] = "legacy"


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        pytest.param(_drop_key, ("keys (",), id="record-keys-mismatch"),
        pytest.param(
            _duplicate_id,
            ("duplicate chunk_id", "duplicate chunk_id", "expected id c2", "chunk_ids do not match chunk order"),
            id="duplicate-id",
        ),
        pytest.param(
            _gap_id,
            ("expected id c2", "chunk_ids do not match chunk order"),
            id="id-gap",
        ),
        pytest.param(_bad_router, ("router 'edge' outside",), id="router-outside"),
        pytest.param(_bad_status, ("status 'deprecated' outside",), id="status-outside"),
        pytest.param(_bad_source, ("source 'external' outside",), id="source-outside"),
        pytest.param(
            _synthetic_not_bool,
            ("synthetic must be a bool", "official chunk must carry synthetic: false"),
            id="synthetic-not-bool",
        ),
        pytest.param(_empty_heading, ("empty heading",), id="empty-heading"),
        pytest.param(_empty_text, ("empty text",), id="empty-text"),
        pytest.param(_forum_not_synthetic, ("forum chunk must carry synthetic: true",), id="forum-not-synthetic"),
        pytest.param(_official_synthetic, ("official chunk must carry synthetic: false",), id="official-synthetic"),
        pytest.param(_forum_path_moved, ("forum chunk path must start with 'forum/'",), id="forum-path-moved"),
        pytest.param(
            _official_forum_path,
            ("forum path with source 'official'", "path missing from sources.json"),
            id="official-forum-path",
        ),
        pytest.param(_forum_status_current, ("forum chunk must be legacy",), id="forum-status-current"),
        pytest.param(_official_missing_source, ("path missing from sources.json",), id="path-missing-from-sources"),
        pytest.param(_legacy_without_rule, ("legacy chunk matched no rule with a reason",), id="legacy-without-rule"),
    ],
)
def test_check_rejects_a_broken_record(tmp_path, mutator, expected):
    records = _committed_records()
    mutator(records)
    _summary, errors, _notes = _check(tmp_path, records)
    _assert_diagnostics(errors, *expected)


def test_check_rejects_a_forum_count_below_the_floor(tmp_path):
    _summary, errors, _notes = _check(tmp_path, forum_min=18)
    _assert_diagnostics(errors, "forum chunks: 17 outside 18..20")


def test_check_rejects_a_forum_count_above_the_ceiling(tmp_path):
    _summary, errors, _notes = _check(tmp_path, forum_max=16)
    _assert_diagnostics(errors, "forum chunks: 17 outside 15..16")


def test_check_rejects_a_corpus_with_no_legacy_chunks(tmp_path):
    records = _committed_records()
    for record in records:
        if record["status"] == "legacy":
            record["status"] = "current"
    _summary, errors, _notes = _check(tmp_path, records)
    _assert_diagnostics(
        errors,
        *(["forum chunk must be legacy"] * 17 + ["no legacy chunks"]),
    )


def test_check_rejects_too_few_chunks(tmp_path):
    _summary, errors, _notes = _check(tmp_path, min_chunks=247)
    _assert_diagnostics(errors, "246 chunks is below the minimum 247")


def test_check_rejects_too_few_app_chunks(tmp_path):
    _summary, errors, _notes = _check(tmp_path, min_app_chunks=197)
    _assert_diagnostics(errors, "app slice is 196 chunks, below the minimum 197")


def test_check_raises_on_a_missing_chunks_file(tmp_path):
    with pytest.raises(build_corpus.CorpusError, match="missing"):
        build_corpus.check_corpus(chunks_path=tmp_path / "absent.jsonl")


def test_check_rejects_a_legacy_rule_with_a_missing_successor(tmp_path):
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    legacy_rule = next(rule for rule in payload["rules"] if rule.get("status") == "legacy")
    legacy_rule["superseded_by"] = "docs/not-in-sources.mdx"
    rules_path = tmp_path / "label_rules.json"
    rules_path.write_text(json.dumps(payload), encoding="utf-8")

    _summary, errors, _notes = _check(tmp_path, rules_path=rules_path)
    _assert_diagnostics(errors, "is not in sources.json")


def test_check_rejects_a_forum_entry_with_no_matching_rule(tmp_path):
    forum_path = tmp_path / "forum.jsonl"
    forum_path.write_text(
        json.dumps(
            {
                "path": "forum/app/unknown-topic.md",
                "heading": "An unmatched topic",
                "text": "No rule names this path, so it would default to current.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _summary, errors, _notes = _check(tmp_path, forum_path=forum_path)
    _assert_diagnostics(errors, "no rule matches")


def test_check_rejects_a_forum_entry_matched_by_a_current_rule(tmp_path):
    rules_path = tmp_path / "label_rules.json"
    rules_path.write_text(
        json.dumps({"rules": [{"match": "forum/*", "router": "both", "status": "current", "reason": "temporary"}]}),
        encoding="utf-8",
    )
    _summary, errors, _notes = _check(tmp_path, rules_path=rules_path)
    assert len(errors) == 21, errors
    assert sum(1 for error in errors if "matched rule is not legacy" in error) == 17
    assert sum(1 for error in errors if "legacy chunk matched no rule with a reason" in error) == 4


# --- --check embedding checks ------------------------------------------------


def _write_embeddings(tmp_path, *, vectors: np.ndarray | None = None, **changes):
    payload = json.loads(EMBEDDINGS_JSON.read_text(encoding="utf-8"))
    payload.update(changes)
    json_path = tmp_path / "embeddings.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    npy_path = tmp_path / "embeddings.npy"
    np.save(npy_path, np.load(EMBEDDINGS_NPY) if vectors is None else vectors)
    return json_path, npy_path


def test_check_rejects_shuffled_embedding_ids(tmp_path):
    payload = json.loads(EMBEDDINGS_JSON.read_text(encoding="utf-8"))
    ids = list(payload["chunk_ids"])
    ids[0], ids[1] = ids[1], ids[0]
    json_path, npy_path = _write_embeddings(tmp_path, chunk_ids=ids)
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "chunk_ids do not match chunk order" in errors[0]


def test_check_rejects_a_wrong_embedding_count(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, count=245)
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "count 245 != 246 chunks" in errors[0]


def test_check_rejects_a_stale_chunk_hash(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, chunks_sha256="0" * 64)
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert SHA_DRIFT in errors[0]


def test_check_rejects_a_different_embedding_model(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, model="text-embedding-004")
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "model 'text-embedding-004'" in errors[0]


def test_check_rejects_a_short_embedding_matrix(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, vectors=np.load(EMBEDDINGS_NPY)[:-1])
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "embeddings.npy rows 245 != 246 chunks" in errors[0]


def test_check_rejects_a_wrong_embedding_width(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, dim=512)
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "embeddings.npy width 768 != dim 512" in errors[0]


def test_check_rejects_a_source_commit_mismatch(tmp_path):
    json_path, npy_path = _write_embeddings(tmp_path, source_commit="0" * 40)
    _summary, errors, _notes = _check(tmp_path, embeddings_json=json_path, embeddings_npy=npy_path)
    assert len(errors) == 1
    assert "embeddings.json source_commit" in errors[0]


def test_check_rejects_missing_embeddings(tmp_path):
    _summary, errors, _notes = _check(
        tmp_path,
        embeddings_json=tmp_path / "absent.json",
        embeddings_npy=tmp_path / "absent.npy",
    )
    _assert_diagnostics(errors, "embeddings.json / embeddings.npy missing")


def test_check_accepts_missing_embeddings_when_told_to(tmp_path):
    _summary, errors, notes = _check(
        tmp_path,
        embeddings_json=tmp_path / "absent.json",
        embeddings_npy=tmp_path / "absent.npy",
        allow_missing_embeddings=True,
    )
    assert errors == []
    assert notes == ["embeddings absent: chunks-only build, embedding assertions skipped"]


# --- network isolation -------------------------------------------------------


def test_only_fetch_imports_a_network_module():
    modules = sorted(CORPUS.rglob("*.py"))
    assert modules, "no modules under corpus/"

    findings: dict[str, set[str]] = {}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        hits = roots & HTTP_MODULES
        if hits:
            findings[path.name] = hits

    assert findings == {"fetch.py": {"httpx"}}


# --- fixture inputs ----------------------------------------------------------


def _fixture_inputs(tmp_path) -> tuple[dict, Path, Path]:
    cache = tmp_path / "cache"
    files = []
    for source in sorted(DOCS.rglob("*.mdx")):
        path = source.relative_to(DOCS).as_posix()
        target = cache / "raw" / FIXTURE_COMMIT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        files.append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"source_commit": FIXTURE_COMMIT, "files": files}
    forum = tmp_path / "forum.jsonl"
    forum.write_text(json.dumps(FIXTURE_FORUM) + "\n", encoding="utf-8")
    return manifest, cache, forum
