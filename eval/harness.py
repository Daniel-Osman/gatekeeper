"""Smoke-grade eval harness: 40 labelled queries, three retrieval policies.

Run it as ``uv run python -m eval.harness``. Each query is answered by every
requested policy, and each answer becomes one record: the outcome, the answer,
how many sentences it used, the verify verdicts, the chunk ids that reached
synthesis, and the cost the receipt reported.

``strict`` and ``fast_fix`` go through ``app.pipeline.run``: the scope gate, the
rerank gate and the verify gate all run, and the router comes from the scope
gate's choice. ``no_gates`` is the baseline this harness builds itself: it calls
``retrieve.run`` then ``synth.run`` directly, with no scope, rerank or verify
stage, and takes the router from the query label (``unclear`` falls back to
``app``). The five ids retrieval reached are recorded as ``kept_ids`` even when
synthesis refuses, so ``hit_at_5`` stays computable for that turn.

Every number in the output is computed from the records. Nothing here is a
recorded result: ``eval/results.json`` is a run's output, not an input.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config, pipeline, retrieve, synth
from app.gates.decisions import StageError
from app.receipt import build_receipt
from app.trace import timed

EVAL_DIR = Path(__file__).parent
DEFAULT_QUERIES = EVAL_DIR / "queries.jsonl"
DEFAULT_OUT = EVAL_DIR / "results.json"

DEFAULT_POLICIES = ("strict", "fast_fix", "no_gates")
NO_GATES = "no_gates"
KINDS = ("official", "forum_only", "unanswerable", "ambiguous")
ANSWERABLE = ("official", "forum_only")
TOP_K = 5
ROUTERS = ("app", "pages")

ANSWERED = ("answered", "answered_flagged")
REFUSED = ("refused", "blocked")
CLARIFY = ("clarify",)

# One turn's call spend, for the rolling-minute bucket. A gated turn runs five
# model calls; a clarify or refusal stops after one or two. These are the
# numbers the free tier's 15-a-minute limit is spent against.
CALLS_GATED = 5
CALLS_LEAN = 2
RETRY_WAITS = (35.0, 70.0, 140.0)


# ---------------------------------------------------------------- queries


def load_queries(path: str | Path) -> list[dict]:
    """Read the query file: one JSON object per line, ids kept unique."""
    path = Path(path)
    queries: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            query = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON: {exc}") from exc
        if query.get("kind") not in KINDS:
            raise ValueError(f"{path}:{number}: unknown kind {query.get('kind')!r}")
        if query["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id {query['id']!r}")
        seen.add(query["id"])
        queries.append(query)
    return queries


# ------------------------------------------------------------------ records


def _wall_ms(stages: list[dict]) -> float:
    """Sum of the stage latencies: the turn's measured time, never a guess."""
    return round(sum(float(stage.get("latency_ms") or 0.0) for stage in stages), 3)


def _kept_from_stages(stages: list[dict]) -> list[str]:
    """The ids the rerank gate passed on, in its own order."""
    for stage in stages:
        if stage.get("stage") == "rerank":
            items = stage.get("items") or []
            return [
                item["chunk_id"]
                for item in items
                if item.get("kept") and item.get("chunk_id")
            ][:TOP_K]
    return []


def _record(query: dict, *, outcome: str | None, answer: str, sentences: list[dict],
            verdicts: list[str], kept_ids: list[str], list_cost_usd: float | None,
            wall_ms: float, policy: str) -> dict:
    gold = list(query.get("gold_ids") or [])
    answerable = query["kind"] in ANSWERABLE
    cited: list[str] = []
    for sentence in sentences:
        for chunk_id in sentence.get("cited_ids") or []:
            if chunk_id not in cited:
                cited.append(chunk_id)
    hit = bool(set(gold) & set(kept_ids)) if answerable else None
    return {
        "id": query["id"],
        "policy": policy,
        "kind": query["kind"],
        "outcome": outcome,
        "answer": answer,
        "sentences": len(sentences),
        "verdicts": list(verdicts),
        "kept_ids": list(kept_ids)[:TOP_K],
        "gold_ids": gold,
        "hit_at_5": hit,
        "cited_hit": bool(set(gold) & set(cited)) if answerable else None,
        "list_cost_usd": list_cost_usd,
        "wall_ms": wall_ms,
    }


async def evaluate_gated(policy_name: str, query: dict) -> dict:
    """One ``strict`` or ``fast_fix`` turn through the real pipeline."""
    envelope = await pipeline.run(query["question"], policy_name)
    trace = envelope.get("trace") or {}
    stages = trace.get("stages") or []
    sentences = envelope.get("sentences") or []
    receipt = envelope.get("receipt") or {}
    return _record(
        query,
        policy=policy_name,
        outcome=trace.get("outcome"),
        answer=envelope.get("answer") or "",
        sentences=sentences,
        verdicts=[s["verdict"] for s in sentences if "verdict" in s],
        kept_ids=_kept_from_stages(stages),
        list_cost_usd=receipt.get("list_cost_usd"),
        wall_ms=_wall_ms(stages),
    )


async def evaluate_no_gates(query: dict) -> dict:
    """Retrieve then synthesize: the baseline with no gate in the way."""
    router = query.get("router")
    if router not in ROUTERS:
        router = "app"

    stages = []
    with timed(stages, "retrieve", config.EMBED_MODEL) as stage:
        candidates = await asyncio.to_thread(
            retrieve.run, query["question"], router, stage
        )
    kept_ids = [item["chunk_id"] for item in candidates][:TOP_K]

    with timed(stages, "synthesize", config.SYNTH_MODEL) as stage:
        written = await asyncio.to_thread(
            synth.run, query["question"], router, kept_ids, stage
        )

    answer = written.get("answer") or ""
    refused = answer.strip() == config.REFUSAL_ANSWER
    receipt = build_receipt(
        stages,
        synthesis_skipped=refused,
        budget_tokens_out=config.SYNTHESIS_OUTPUT_BUDGET,
    )
    return _record(
        query,
        policy=NO_GATES,
        outcome="refused" if refused else "answered",
        answer=answer,
        sentences=written.get("sentences") or [],
        # No verify stage runs on this path, so there are no verdicts to report.
        verdicts=[],
        kept_ids=kept_ids,
        list_cost_usd=receipt.get("list_cost_usd"),
        wall_ms=_wall_ms([stage.as_dict() for stage in stages]),
    )


async def evaluate(policy_name: str, query: dict) -> dict:
    if policy_name == NO_GATES:
        return await evaluate_no_gates(query)
    return await evaluate_gated(policy_name, query)


# ------------------------------------------------------------------- pacing


def calls_for(policy_name: str) -> int:
    """The most calls one turn of this policy can spend.

    Charged before the turn runs, because the outcome is not known yet; the
    unused share is refunded once the record says what the turn did.
    """
    return CALLS_LEAN if policy_name == NO_GATES else CALLS_GATED


def calls_spent(policy_name: str, record: dict) -> int:
    """The calls a finished turn spent: a clarify or refusal stopped early."""
    if policy_name == NO_GATES:
        return CALLS_LEAN
    if record.get("outcome") in CLARIFY + REFUSED:
        return CALLS_LEAN
    return CALLS_GATED


class Pacer:
    """A rolling-minute token bucket: no more than ``rpm`` calls in any 60s.

    ``rpm=0`` disables the wait, so tests and offline runs never sleep. ``calls``
    counts what was booked net of refunds, which is the run's live-call total.
    """

    WINDOW_S = 60.0

    def __init__(self, rpm: int, *, sleep=asyncio.sleep, clock=time.monotonic,
                 stream=sys.stdout):
        self.rpm = max(0, int(rpm))
        self.calls = 0
        self._sleep = sleep
        self._clock = clock
        self._stream = stream
        self._booked: list[float] = []

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    def _expire(self, now: float) -> None:
        if self._booked:
            cutoff = now - self.WINDOW_S
            self._booked = [stamp for stamp in self._booked if stamp > cutoff]

    async def charge(self, calls: int) -> None:
        """Book ``calls`` slots, sleeping until the rolling minute has room."""
        if calls <= 0:
            return
        if not self.enabled:
            self.calls += calls
            return
        booked = 0
        while booked < calls:
            now = self._clock()
            self._expire(now)
            if len(self._booked) < self.rpm:
                self._booked.append(now)
                self.calls += 1
                booked += 1
                continue
            wait_s = max(self.WINDOW_S - (now - self._booked[0]), 0.01)
            print(f"waiting {wait_s:.0f}s for rpm", file=self._stream)
            await self._sleep(wait_s)

    def refund(self, calls: int) -> None:
        """Give back slots the turn did not spend, newest first."""
        if calls <= 0 or not self.enabled:
            return
        for _ in range(min(calls, len(self._booked))):
            self._booked.pop()
            self.calls -= 1


def _is_rate_limited(error: Exception) -> bool:
    """A StageError that carries a 429: the quota, not a bad request."""
    return "429" in str(error)


async def evaluate_paced(
    policy_name: str,
    query: dict,
    *,
    evaluate_one,
    pacer: Pacer,
    sleep=asyncio.sleep,
    stream=sys.stdout,
    retry_waits=RETRY_WAITS,
) -> dict:
    """One turn, paced and retried off a 429; raises when the retries run out."""
    attempt = 0
    while True:
        await pacer.charge(calls_for(policy_name))
        try:
            record = await evaluate_one(policy_name, query)
        except StageError as error:
            if not _is_rate_limited(error) or attempt >= len(retry_waits):
                raise
            wait_s = retry_waits[attempt]
            attempt += 1
            print(
                f"retry {attempt}/{len(retry_waits)} {query['id']} in {wait_s:.0f}s "
                f"after 429",
                file=stream,
            )
            await sleep(wait_s)
            continue
        pacer.refund(calls_for(policy_name) - calls_spent(policy_name, record))
        return record


def error_record(query: dict, policy_name: str, error: Exception) -> dict:
    """A failed turn, kept in the file so the query is not lost with the run."""
    return {
        "id": query["id"],
        "policy": policy_name,
        "kind": query["kind"],
        "error": str(error)[:200],
        "outcome": None,
        "answer": "",
        "sentences": 0,
        "verdicts": [],
        "kept_ids": [],
        "gold_ids": list(query.get("gold_ids") or []),
        "hit_at_5": None,
        "cited_hit": None,
        "list_cost_usd": None,
        "wall_ms": None,
    }


# -------------------------------------------------------------- aggregates


def _rate(hits: int, total: int) -> float | None:
    """A proportion, or ``None`` when the denominator is empty."""
    if total <= 0:
        return None
    return round(hits / total, 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _p95(values: list[float]) -> float | None:
    """Nearest-rank 95th percentile: the smallest value at or above 95%."""
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[position], 3)


def aggregate(records: list[dict]) -> dict:
    """One policy's numbers, each with the denominator it is honest over.

    ``hit_at_5`` and ``cited_hit_answerable`` divide by the answerable records
    (``official`` plus ``forum_only``), ``refusal_correct`` by the
    ``unanswerable`` records and ``clarify_correct`` by the ``ambiguous`` ones.
    A correct refusal is ``refused`` or ``blocked``; a blocked turn is counted
    separately so the outcome counts add up to ``count``.

    An errored record counts in ``count`` and in ``errored``, and is left out of
    every rate's denominator: the turn never produced a judgement to score.
    """
    count = len(records)
    errored = [r for r in records if r.get("error")]
    scored = [r for r in records if not r.get("error")]
    answerable = [r for r in scored if r["kind"] in ANSWERABLE]
    unanswerable = [r for r in scored if r["kind"] == "unanswerable"]
    ambiguous = [r for r in scored if r["kind"] == "ambiguous"]

    answered = sum(1 for r in scored if r["outcome"] in ANSWERED)
    flagged = sum(1 for r in scored if r["outcome"] == "answered_flagged")
    refused = sum(1 for r in scored if r["outcome"] in ("refused",))
    blocked = sum(1 for r in scored if r["outcome"] in ("blocked",))
    clarify = sum(1 for r in scored if r["outcome"] in CLARIFY)

    hit = sum(1 for r in answerable if r["hit_at_5"])
    cited = sum(1 for r in answerable if r["cited_hit"])
    refusal_ok = sum(1 for r in unanswerable if r["outcome"] in REFUSED)
    clarify_ok = sum(1 for r in ambiguous if r["outcome"] in CLARIFY)

    walls = [float(r["wall_ms"]) for r in scored if r.get("wall_ms") is not None]
    costs = [float(r["list_cost_usd"]) for r in scored if r.get("list_cost_usd") is not None]

    return {
        "count": count,
        "errored": len(errored),
        "answered": answered,
        "answered_flagged": flagged,
        "refused": refused,
        "blocked": blocked,
        "clarify": clarify,
        "hit_at_5": _rate(hit, len(answerable)),
        "cited_hit_answerable": _rate(cited, len(answerable)),
        "refusal_correct": _rate(refusal_ok, len(unanswerable)),
        "clarify_correct": _rate(clarify_ok, len(ambiguous)),
        "wall_ms_mean": _mean(walls),
        "wall_ms_p95": _p95(walls),
        "cost_total_usd": round(sum(costs), 10) if costs else None,
        "cost_mean_usd": round(sum(costs) / len(costs), 10) if costs else None,
    }


def aggregate_all(records: list[dict], policies) -> dict[str, dict]:
    return {
        policy: aggregate([r for r in records if r.get("policy") == policy])
        for policy in policies
    }


TABLE_COLUMNS = (
    ("policy", "policy", 8),
    ("count", "n", 3),
    ("errored", "err", 3),
    ("answered", "ans", 4),
    ("answered_flagged", "flag", 5),
    ("refused", "ref", 4),
    ("blocked", "blk", 4),
    ("clarify", "clar", 4),
    ("hit_at_5", "hit@5", 6),
    ("cited_hit_answerable", "cited", 6),
    ("refusal_correct", "refuse", 7),
    ("clarify_correct", "clarify", 8),
    ("wall_ms_mean", "ms_mean", 8),
    ("wall_ms_p95", "ms_p95", 7),
    ("cost_total_usd", "cost_usd", 11),
)


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_table(aggregates: dict[str, dict]) -> str:
    """Render the per-policy summary from the numbers it is handed."""
    header = "  ".join(
        f"{label:<{width}}" for _, label, width in TABLE_COLUMNS
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for policy, numbers in aggregates.items():
        row = {**numbers, "policy": policy}
        lines.append(
            "  ".join(
                f"{_cell(row.get(key)):<{width}}"
                for key, _, width in TABLE_COLUMNS
            )
        )
    return "\n".join(lines)


# ------------------------------------------------------------------- runner


def load_results(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not a results file: {exc}") from exc


def save_results(path: str | Path, payload: dict) -> None:
    """Write the payload atomically: a sibling ``.tmp`` then a replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def run_harness(
    queries: list[dict],
    policies,
    out_path: str | Path,
    *,
    only: list[str] | None = None,
    limit: int | None = None,
    quiet: bool = False,
    rpm: int = 0,
    evaluate_one=evaluate,
    sleep=asyncio.sleep,
    retry_waits=RETRY_WAITS,
    stream=sys.stdout,
) -> dict:
    """Run the selected queries, resuming from the out file, and print a table.

    A ``(policy, id)`` pair already in the out file is not run again, so an
    interrupted eval can be resumed without paying twice. The out file is
    rewritten after every record, so a crash leaves a resumable file rather
    than losing the day's work.

    ``rpm`` caps model calls per rolling minute; ``0`` disables pacing. A turn
    is charged its worst-case spend before it runs and refunded what it did not
    spend, so a clarify does not cost five calls. A ``StageError`` carrying a
    429 is retried after each of ``retry_waits``; when those run out the query
    becomes an error record and the run moves on.
    """
    out_path = Path(out_path)
    previous = load_results(out_path)
    records: list[dict] = list(previous.get("records") or [])
    done = {(record.get("policy"), record.get("id")) for record in records}
    run_at = previous.get("run_at") or _now()
    pacer = Pacer(rpm, sleep=sleep, stream=stream)

    chosen = queries
    if only:
        wanted = [qid.strip() for qid in only if qid.strip()]
        known = {query["id"] for query in queries}
        missing = [qid for qid in wanted if qid not in known]
        if missing:
            raise ValueError(f"--only names unknown ids: {', '.join(missing)}")
        chosen = [query for query in queries if query["id"] in wanted]
    if limit is not None:
        chosen = chosen[: max(0, limit)]

    def payload() -> dict:
        return {
            "run_at": run_at,
            "updated_at": _now(),
            "queries": str(DEFAULT_QUERIES.name),
            "policies": list(policies),
            "count": len(records),
            "live_calls": pacer.calls,
            "records": records,
            "aggregate": aggregate_all(records, policies),
        }

    skipped = 0
    for policy_name in policies:
        for query in chosen:
            if (policy_name, query["id"]) in done:
                skipped += 1
                continue
            if not quiet:
                print(f"{policy_name:<9} {query['id']} {query['kind']}", file=stream)
            try:
                record = await evaluate_paced(
                    policy_name,
                    query,
                    evaluate_one=evaluate_one,
                    pacer=pacer,
                    sleep=sleep,
                    stream=stream,
                    retry_waits=retry_waits,
                )
            except StageError as error:
                record = error_record(query, policy_name, error)
                print(
                    f"{policy_name:<9} {query['id']} error: {record['error']}",
                    file=stream,
                )
            records.append(record)
            done.add((policy_name, query["id"]))
            save_results(out_path, payload())
    if skipped:
        print(f"skipped {skipped}", file=stream)

    saved = payload()
    save_results(out_path, saved)
    print("", file=stream)
    print(render_table(saved["aggregate"]), file=stream)
    return saved


def _parse_policies(raw: str) -> list[str]:
    policies = [name.strip() for name in raw.split(",") if name.strip()]
    if not policies:
        raise ValueError("--policies named nothing")
    return policies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.harness",
        description=(
            "Run the labelled query set through each retrieval policy and write "
            "one record per query."
        ),
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help=f"comma-separated policies (default: {','.join(DEFAULT_POLICIES)})",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="query file, one JSON object per line (default: eval/queries.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="results file, resumed when it exists (default: eval/results.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N selected queries",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated query ids to evaluate",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="no per-query lines; the summary table still prints",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=12,
        help=(
            "model calls per rolling minute; a gated turn costs 5, a clarify or "
            "no_gates turn 2 (default: 12, under the free tier's 15)"
        ),
    )
    args = parser.parse_args(argv)

    queries = load_queries(args.queries)
    policies = _parse_policies(args.policies)
    only = args.only.split(",") if args.only else None

    payload = asyncio.run(
        run_harness(
            queries,
            policies,
            args.out,
            only=only,
            limit=args.limit,
            quiet=args.quiet,
            rpm=args.rpm,
        )
    )
    print(f"\n{payload['count']} records ({payload['live_calls']} calls) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
