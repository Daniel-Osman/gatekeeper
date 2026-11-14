"""Turn observed token counts into a receipt with list prices."""

from __future__ import annotations

from app import config
from app.trace import Stage


def _stage_cost(stage: Stage) -> tuple[float | None, str]:
    """Return ``(list cost, price basis)`` for one stage.

    ``unpriced`` is a known zero: this build publishes no rate for the model,
    so the stage contributes ``0.0`` rather than an invented price.
    ``unmeasured`` keeps its ``None``: no token numbers, so no cost claim.
    """
    price = config.PRICES.get(stage.model)
    if price is None:
        return 0.0, "unpriced"
    if stage.tokens_in is None and stage.tokens_out is None:
        return None, "unmeasured"
    cost = (stage.tokens_in or 0) * price["in"] + (stage.tokens_out or 0) * price["out"]
    return cost, "list_price"


def build_receipt(
    stages: list[Stage], synthesis_skipped: bool, budget_tokens_out: int
) -> dict:
    per_stage = []
    costs: list[float] = []

    for stage in stages:
        stage_cost, basis = _stage_cost(stage)
        if stage_cost is not None:
            costs.append(stage_cost)
        per_stage.append(
            {
                "stage": stage.stage,
                "tokens_in": stage.tokens_in,
                "tokens_out": stage.tokens_out,
                "price_basis": basis,
                "list_cost_usd": None if stage_cost is None else round(stage_cost, 12),
            }
        )

    ins = [stage.tokens_in for stage in stages]
    outs = [stage.tokens_out for stage in stages]

    return {
        "billed_tier": "free",
        "list_cost_usd": round(sum(costs), 12) if costs else None,
        # A total is a claim about every stage that ran: one silent stage makes
        # the total unmeasured rather than understated.
        "tokens_in": sum(ins) if stages and None not in ins else None,
        "tokens_out": sum(outs) if stages and None not in outs else None,
        "synthesis_skipped": synthesis_skipped,
        "synthesis_budget_tokens_out": budget_tokens_out,
        "per_stage": per_stage,
    }
