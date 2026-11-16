"""Offline tests for the eval harness: aggregate math, resume and the table.

No test here calls a model. The ``no_gates`` test replaces ``retrieve.run`` and
``synth.run`` with fakes that record the stage they were given, which is the
same seam ``tests/conftest.py`` uses for the pipeline tests.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from app import pipeline, retrieve, synth
from eval import harness


def _record(
    *,
    policy="strict",
    id="q01",
    kind="official",
    outcome="answered",
    hit=None,
    cited=None,
    wall=100.0,
    cost=0.001,
):
    answerable = kind in harness.ANSWERABLE
    return {
        "id": id,
        "policy": policy,
        "kind": kind,
        "outcome": outcome,
        "answer": "an answer [c1].",
        "sentences": 1,
        "verdicts": ["verified"],
        "kept_ids": ["c1"],
        "gold_ids": ["c1"] if answerable else [],
        "hit_at_5": hit,
        "cited_hit": cited,
        "list_cost_usd": cost,
        "wall_ms": wall,
    }


def _five_records():
    return [
        _record(id="q01", kind="official", outcome="answered", hit=True, cited=True,
                wall=100.0, cost=0.001),
        _record(id="q02", kind="official", outcome="answered_flagged", hit=False,
                cited=False, wall=300.0, cost=0.003),
        _record(id="q03", kind="forum_only", outcome="refused", hit=True, cited=False,
                wall=200.0, cost=0.002),
        _record(id="q04", kind="unanswerable", outcome="refused", wall=50.0, cost=None),
        _record(id="q05", kind="ambiguous", outcome="clarify", wall=40.0, cost=0.0005),
    ]


def test_aggregate_math():
    numbers = harness.aggregate(_five_records())

    assert numbers["count"] == 5
    assert numbers["answered"] == 2
    assert numbers["answered_flagged"] == 1
    assert numbers["refused"] == 2
    assert numbers["blocked"] == 0
    assert numbers["clarify"] == 1

    # Three answerable records: two hits, one cited.
    assert numbers["hit_at_5"] == pytest.approx(2 / 3, abs=1e-4)
    assert numbers["cited_hit_answerable"] == pytest.approx(1 / 3, abs=1e-4)
    # One unanswerable record, refused. One ambiguous record, clarified.
    assert numbers["refusal_correct"] == 1.0
    assert numbers["clarify_correct"] == 1.0

    assert numbers["wall_ms_mean"] == pytest.approx(138.0)
    assert numbers["wall_ms_p95"] == pytest.approx(300.0)
    # The unmeasured cost is left out of the total, never read as zero.
    assert numbers["cost_total_usd"] == pytest.approx(0.0065)
    assert numbers["cost_mean_usd"] == pytest.approx(0.001625)


def test_aggregate_empty_denominators():
    only_official = [_record(kind="official", hit=True, cited=True)]
    numbers = harness.aggregate(only_official)
    assert numbers["refusal_correct"] is None
    assert numbers["clarify_correct"] is None
    assert numbers["hit_at_5"] == 1.0

    empty = harness.aggregate([])
    assert empty["count"] == 0
    assert empty["hit_at_5"] is None
    assert empty["cited_hit_answerable"] is None
    assert empty["refusal_correct"] is None
    assert empty["clarify_correct"] is None
    assert empty["wall_ms_mean"] is None
    assert empty["wall_ms_p95"] is None
    assert empty["cost_total_usd"] is None
    assert empty["cost_mean_usd"] is None


def test_resume_skips_present_pairs(tmp_path):
    out = tmp_path / "results.json"
    queries = [
        {"id": "q01", "question": "one", "router": "app", "gold_ids": ["c1"],
         "kind": "official", "note": "x"},
        {"id": "q02", "question": "two", "router": "app", "gold_ids": ["c2"],
         "kind": "official", "note": "x"},
    ]
    calls: list[tuple[str, str]] = []

    async def fake_evaluate(policy_name, query):
        calls.append((policy_name, query["id"]))
        return _record(policy=policy_name, id=query["id"], hit=True, cited=True)

    first = io.StringIO()
    asyncio.run(
        harness.run_harness(queries, ["no_gates"], out, evaluate_one=fake_evaluate,
                            stream=first)
    )
    assert calls == [("no_gates", "q01"), ("no_gates", "q02")]

    second = io.StringIO()
    payload = asyncio.run(
        harness.run_harness(queries, ["no_gates"], out, evaluate_one=fake_evaluate,
                            stream=second)
    )

    # Nothing ran twice: the two pairs were already on disk.
    assert calls == [("no_gates", "q01"), ("no_gates", "q02")]
    assert "skipped 2" in second.getvalue()
    assert "skipped" not in first.getvalue()
    assert payload["count"] == 2
    assert [record["id"] for record in payload["records"]] == ["q01", "q02"]

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["count"] == 2
    assert saved["run_at"] == payload["run_at"]


def test_no_gates_runs_no_gate_stage(monkeypatch):
    names: list[str] = []
    kept = ["c1", "c2", "c3"]

    def fake_retrieve(question, router, stage=None):
        names.append(stage.stage)
        return [{"chunk_id": chunk_id} for chunk_id in kept]

    def fake_synth(question, router, kept_ids, stage=None, repair=None):
        names.append(stage.stage)
        return {
            "answer": "Read the segment with params [c1].",
            "sentences": [{"text": "Read the segment with params [c1].",
                           "cited_ids": ["c1"]}],
        }

    def no_pipeline(*args, **kwargs):
        raise AssertionError("no_gates must not call pipeline.run")

    monkeypatch.setattr(retrieve, "run", fake_retrieve)
    monkeypatch.setattr(synth, "run", fake_synth)
    monkeypatch.setattr(pipeline, "run", no_pipeline)

    query = {
        "id": "q01",
        "question": "How do I make a dynamic route segment?",
        "router": "unclear",
        "gold_ids": ["c1"],
        "kind": "official",
        "note": "x",
    }
    record = asyncio.run(harness.evaluate_no_gates(query))

    assert names == ["retrieve", "synthesize"]
    assert record["outcome"] == "answered"
    assert record["kept_ids"] == kept
    assert record["hit_at_5"] is True
    assert record["cited_hit"] is True
    # The verify gate did not run, so there are no verdicts to claim.
    assert record["verdicts"] == []


def test_no_gates_records_refusal():
    def fake_retrieve(question, router, stage=None):
        return [{"chunk_id": "c1"}]

    def fake_synth(question, router, kept_ids, stage=None, repair=None):
        return {"answer": "I have no reliable docs for this.",
                "sentences": [{"text": "I have no reliable docs for this.",
                               "cited_ids": []}]}

    from app import retrieve as retrieve_module
    from app import synth as synth_module

    original_retrieve, original_synth = retrieve_module.run, synth_module.run
    retrieve_module.run, synth_module.run = fake_retrieve, fake_synth
    try:
        record = asyncio.run(
            harness.evaluate_no_gates(
                {"id": "q31", "question": "React Native?", "router": "app",
                 "gold_ids": [], "kind": "unanswerable", "note": "x"}
            )
        )
    finally:
        retrieve_module.run, synth_module.run = original_retrieve, original_synth

    assert record["outcome"] == "refused"
    # The five reached ids are still recorded, so hit@5 stays computable.
    assert record["kept_ids"] == ["c1"]
    assert record["hit_at_5"] is None


def test_table_renders_recorded_numbers():
    aggregates = harness.aggregate_all(_five_records(), ["strict"])
    table = harness.render_table(aggregates)

    lines = table.splitlines()
    assert lines[0].startswith("policy")
    assert "hit@5" in lines[0]
    assert "cost_usd" in lines[0]
    assert len(lines) == 3
    cells = lines[2].split()
    assert cells[0] == "strict"
    assert cells[1] == "5"
    assert "0.6667" in table
    assert "0.0065" in table
    assert "300" in table


def test_error_record_counts_but_stays_out_of_rates():
    records = _five_records()
    records.append({
        "id": "q06",
        "policy": "strict",
        "kind": "official",
        "error": "synthesize: ClientError: 429 RESOURCE_EXHAUSTED",
        "outcome": None,
        "hit_at_5": None,
        "cited_hit": None,
        "list_cost_usd": None,
        "wall_ms": None,
    })

    numbers = harness.aggregate(records)

    assert numbers["count"] == 6
    assert numbers["errored"] == 1
    # Three answerable records remain: two hits of three, not two of four.
    assert numbers["hit_at_5"] == pytest.approx(2 / 3, abs=1e-4)
    assert numbers["cited_hit_answerable"] == pytest.approx(1 / 3, abs=1e-4)
    # The outcome counts still add up to count once errored is included.
    assert (numbers["answered"] + numbers["refused"] + numbers["blocked"]
            + numbers["clarify"] + numbers["errored"]) == numbers["count"]
    # A turn that produced no stage time is not a zero-millisecond turn.
    assert numbers["wall_ms_mean"] == pytest.approx(138.0)


def test_out_file_saved_after_first_record_when_the_second_raises(tmp_path):
    out = tmp_path / "results.json"
    queries = [
        {"id": "q01", "question": "one", "router": "app", "gold_ids": ["c1"],
         "kind": "official", "note": "x"},
        {"id": "q02", "question": "two", "router": "app", "gold_ids": ["c2"],
         "kind": "official", "note": "x"},
    ]

    async def second_raises(policy_name, query):
        if query["id"] == "q02":
            raise RuntimeError("the run died here")
        return _record(policy=policy_name, id=query["id"], hit=True, cited=True)

    with pytest.raises(RuntimeError):
        asyncio.run(
            harness.run_harness(queries, ["strict"], out, evaluate_one=second_raises,
                                rpm=0, stream=io.StringIO())
        )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [record["id"] for record in saved["records"]] == ["q01"]
    assert saved["aggregate"]["strict"]["count"] == 1
    # The write is atomic: no .tmp is left behind for a resume to read.
    assert not (tmp_path / "results.json.tmp").exists()


def test_429_retries_then_records_an_error_without_raising(tmp_path):
    from app.gates.decisions import StageError

    out = tmp_path / "results.json"
    query = {"id": "q01", "question": "one", "router": "app", "gold_ids": ["c1"],
             "kind": "official", "note": "x"}
    attempts: list[str] = []
    waits: list[float] = []

    async def always_429(policy_name, query):
        attempts.append(query["id"])
        raise StageError("synthesize", "ClientError: 429 RESOURCE_EXHAUSTED")

    async def fake_sleep(seconds):
        waits.append(seconds)

    payload = asyncio.run(
        harness.run_harness([query], ["strict"], out, evaluate_one=always_429,
                            rpm=0, sleep=fake_sleep, stream=io.StringIO())
    )

    assert len(attempts) == 4  # the first call, then three retries
    assert waits == [35.0, 70.0, 140.0]
    record = payload["records"][0]
    assert record["id"] == "q01"
    assert record["error"].startswith("synthesize: ClientError: 429")
    assert "outcome" not in record or record["outcome"] is None
    assert payload["aggregate"]["strict"]["count"] == 1
    assert payload["aggregate"]["strict"]["errored"] == 1
    assert payload["aggregate"]["strict"]["hit_at_5"] is None

