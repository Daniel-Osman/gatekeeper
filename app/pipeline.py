"""The gate pipeline: scope, retrieve, rerank."""

from __future__ import annotations

import asyncio

from app import config, index, retrieve
from app.gates import decisions, rerank, scope
from app.receipt import build_receipt
from app.trace import Stage, timed


def _trace(outcome: str | None, policy_name: str, stages: list[Stage]) -> dict:
    return {
        "outcome": outcome,
        "policy": policy_name,
        "stages": [stage.as_dict() for stage in stages],
    }


def _envelope(
    answer: str | None,
    outcome: str,
    policy_name: str,
    stages: list[Stage],
) -> dict:
    return {
        "answer": answer,
        "sentences": [],
        "trace": _trace(outcome, policy_name, stages),
        "receipt": build_receipt(
            stages,
            synthesis_skipped=True,
            budget_tokens_out=config.SYNTHESIS_OUTPUT_BUDGET,
        ),
    }


def _debug(policy_name: str, stages: list[Stage]) -> dict:
    """The stages a ``--only`` run reached: no outcome, no answer, no receipt."""
    return {"trace": _trace(None, policy_name, stages)}


async def run(question: str, policy_name: str = "strict", only: str | None = None) -> dict:
    """Run the gates in order and return the envelope.

    ``blocked`` and ``clarify`` stop after scope and still carry a sentence.
    ``refused`` stops after rerank when no candidate survives. ``only`` is a
    debugging stop: it runs up to and including that stage, and returns the
    stages alone when no gate outcome applies. No synthesize stage exists in
    this build, so only ``gate_passed`` returns a ``None`` answer.
    """
    policy = rerank.POLICIES.get(policy_name)
    if policy is None:
        raise ValueError(f"unknown policy: {policy_name!r}")

    index.ensure_loaded()
    stages: list[Stage] = []

    try:
        with timed(stages, "scope", config.DECISION_MODEL) as stage:
            scope_result = await asyncio.to_thread(scope.run, question, stage)

        outcome = scope_result["outcome"]
        if outcome != "proceed":
            return _envelope(
                config.ANSWERS[scope_result["payload"]["reason"]],
                outcome,
                policy_name,
                stages,
            )
        if only == "scope":
            return _debug(policy_name, stages)

        router = scope_result["payload"]["router_choice"]
        with timed(stages, "retrieve", config.EMBED_MODEL) as stage:
            candidates = await asyncio.to_thread(retrieve.run, question, router, stage)

        if only == "retrieve":
            return _debug(policy_name, stages)

        with timed(stages, "rerank", config.DECISION_MODEL) as stage:
            result = await asyncio.to_thread(
                rerank.run, question, router, candidates, policy, stage
            )

        if not result["kept"]:
            return _envelope(config.REFUSAL_ANSWER, "refused", policy_name, stages)
        # No synthesizer exists yet, so gate_passed is the one outcome that
        # returns no answer: it is the only reason ``answer`` is ``None``.
        return _envelope(None, "gate_passed", policy_name, stages)

    except decisions.StageError as exc:
        exc.trace = _trace("error", policy_name, stages)
        raise
