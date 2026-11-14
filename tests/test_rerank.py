"""Gate 2: policy keep rule, the relevance floor, the cap, the refusal path."""

from __future__ import annotations

import asyncio

from app import config
from app.gates import rerank
from app.pipeline import run as run_pipeline

SPREAD = [f"s{n:02d}" for n in range(1, 31)]
REASONS = {
    "kept",
    "below_min_relevance",
    "forum_not_allowed",
    "legacy_not_allowed",
    "over_cap",
}


def _items(result: dict) -> dict[str, dict]:
    return {item["chunk_id"]: item for item in result["payload"]["items"]}


def test_strict_and_fast_fix_differ_on_the_forum_and_legacy_chunks(tmp_index):
    question = "do the thirty candidates spread across forum and legacy sources"

    strict = rerank.run(question, "app", SPREAD, rerank.STRICT)
    fast = rerank.run(question, "app", SPREAD, rerank.FAST_FIX)

    assert strict["kept"] == ["s01", "s02"]
    assert fast["kept"] == ["s01", "s02", "s03", "s04"]
    strict_items, fast_items = _items(strict), _items(fast)
    assert strict_items["s03"]["reason"] == "forum_not_allowed"
    assert strict_items["s04"]["reason"] == "legacy_not_allowed"
    assert fast_items["s03"]["kept"] is True
    assert fast_items["s04"]["kept"] is True
    assert strict["payload"]["candidate_count"] == len(SPREAD) == 30
    assert strict["payload"]["kept_count"] == 2
    assert fast["payload"]["kept_count"] == 4
    assert {item["reason"] for item in strict["payload"]["items"]} <= REASONS


def test_relevance_exactly_060_passes_strict(tmp_index):
    result = rerank.run("relevance boundary at sixty", "app", ["s06"], rerank.STRICT)

    assert result["kept"] == ["s06"]
    item = result["payload"]["items"][0]
    assert item["relevance_noul"] == 0.60
    # currency is reported and does not gate: the corpus label decides.
    assert item["currency_noul"] == 0.30


def test_relevance_059_fails_strict(tmp_index):
    result = rerank.run("relevance boundary at sixty", "app", ["s07"], rerank.STRICT)

    assert result["kept"] == []
    item = result["payload"]["items"][0]
    assert item["relevance_noul"] == 0.59
    assert item["kept"] is False
    assert item["reason"] == "below_min_relevance"


def test_relevance_045_kept_by_fast_fix_and_refused_by_strict(tmp_index):
    question = "do the thirty candidates spread across forum and legacy sources"

    strict = rerank.run(question, "app", ["s31"], rerank.STRICT)
    fast = rerank.run(question, "app", ["s31"], rerank.FAST_FIX)

    fast_item = _items(fast)["s31"]
    assert fast_item["relevance_noul"] == 0.45
    assert fast["kept"] == ["s31"]
    assert fast_item["kept"] is True
    assert fast_item["reason"] == "kept"

    strict_item = _items(strict)["s31"]
    assert strict_item["relevance_noul"] == 0.45
    assert strict["kept"] == []
    assert strict_item["kept"] is False
    assert strict_item["reason"] == "below_min_relevance"


def test_six_passing_chunks_keep_five_and_mark_the_sixth_over_cap(tmp_index):
    candidates = ["s08", "s09", "s10", "s11", "s12", "s13"]

    result = rerank.run("over cap six pass the strict floor", "app", candidates, rerank.STRICT)

    assert result["kept"] == candidates[:5]
    assert result["payload"]["kept_count"] == 5
    sixth = _items(result)["s13"]
    assert sixth["kept"] is False
    assert sixth["reason"] == "over_cap"


def test_all_fail_is_refused_without_a_synthesize_stage():
    envelope = asyncio.run(
        run_pipeline("why is my middleware not running in the app router", "strict")
    )

    assert envelope["trace"]["outcome"] == "refused"
    assert envelope["answer"] == config.REFUSAL_ANSWER
    assert envelope["sentences"] == []
    stages = [stage["stage"] for stage in envelope["trace"]["stages"]]
    assert stages == ["scope", "retrieve", "rerank"]
    assert "synthesize" not in stages
    assert envelope["trace"]["stages"][2]["kept_count"] == 0
    assert envelope["receipt"]["synthesis_skipped"] is True
