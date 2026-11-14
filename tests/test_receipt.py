"""Receipt arithmetic: the total is the sum of the stages, and null stays null."""

from __future__ import annotations

from app import config
from app.receipt import build_receipt
from app.trace import Stage


def test_total_cost_equals_the_sum_of_the_stages():
    stages = [
        Stage(stage="scope", model=config.DECISION_MODEL, tokens_in=412, tokens_out=0),
        Stage(stage="retrieve", model=config.EMBED_MODEL, tokens_in=23, tokens_out=0),
        Stage(stage="rerank", model=config.DECISION_MODEL, tokens_in=4816, tokens_out=0),
    ]

    receipt = build_receipt(
        stages,
        synthesis_skipped=True,
        budget_tokens_out=config.SYNTHESIS_OUTPUT_BUDGET,
    )

    per_stage = sum(item["list_cost_usd"] for item in receipt["per_stage"])
    assert abs(receipt["list_cost_usd"] - per_stage) <= 1e-9
    assert len(receipt["per_stage"]) == len(stages)
    assert receipt["tokens_in"] == 5251
    assert receipt["tokens_out"] == 0


def test_all_null_tokens_yield_no_cost():
    stages = [
        Stage(stage="scope", model=config.DECISION_MODEL),
        Stage(stage="rerank", model=config.DECISION_MODEL),
    ]

    receipt = build_receipt(
        stages,
        synthesis_skipped=True,
        budget_tokens_out=config.SYNTHESIS_OUTPUT_BUDGET,
    )

    assert receipt["list_cost_usd"] is None
    assert [item["list_cost_usd"] for item in receipt["per_stage"]] == [None, None]
