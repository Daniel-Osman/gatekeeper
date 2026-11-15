"""The gate pipeline: scope, retrieve, rerank, synthesize, verify."""

from __future__ import annotations

import asyncio

from app import config, index, retrieve, synth, verify
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
    sentences: list[dict] | None = None,
    synthesis_skipped: bool = True,
) -> dict:
    return {
        "answer": answer,
        "sentences": list(sentences or []),
        "trace": _trace(outcome, policy_name, stages),
        "receipt": build_receipt(
            stages,
            synthesis_skipped=synthesis_skipped,
            budget_tokens_out=config.SYNTHESIS_OUTPUT_BUDGET,
        ),
    }


def _debug(policy_name: str, stages: list[Stage]) -> dict:
    """The stages a ``--only`` run reached: no outcome, no answer, no receipt."""
    return {"trace": _trace(None, policy_name, stages)}


async def _synthesise(
    question: str,
    router: str,
    kept: list[dict],
    stages: list[Stage],
    repair: list[str] | None = None,
) -> dict:
    with timed(stages, "synthesize", config.SYNTH_MODEL) as stage:
        return await asyncio.to_thread(
            synth.run, question, router, kept, stage, repair=repair
        )


async def _check(
    question: str,
    sentences: list[dict],
    stages: list[Stage],
    *,
    kept: list[dict],
    retry_of: int | None = None,
) -> tuple[list[dict], int]:
    """Verify the sentences against the kept chunks; return verdicts and index."""
    position = len(stages)
    with timed(stages, "verify", config.DECISION_MODEL) as stage:
        verdicts = await asyncio.to_thread(
            verify.run, question, sentences, stage, kept=kept
        )
    if retry_of is not None:
        stage.payload = {**(stage.payload or {}), "retry_of": retry_of}
    return verdicts, position


async def run(question: str, policy_name: str = "strict", only: str | None = None) -> dict:
    """Run the gates in order and return the envelope.

    ``blocked`` and ``clarify`` stop after scope and still carry a sentence.
    ``refused`` stops after rerank when no candidate survives. A kept set is
    synthesised, then every sentence is verified against the chunk it cites:
    one unsupported sentence buys exactly one repair pass, and a sentence still
    unsupported after it flags the answer. ``only`` is a debugging stop: it
    runs up to and including that stage, and returns the stages alone when no
    gate outcome applies.
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
        if only == "rerank":
            return _debug(policy_name, stages)

        kept = [{"chunk_id": chunk_id} for chunk_id in result["kept"]]
        written = await _synthesise(question, router, kept, stages)
        verdicts, retry_of = await _check(
            question, written["sentences"], stages, kept=kept
        )

        rejected = [
            sentence["text"]
            for sentence in verdicts
            if sentence["verdict"] == "unsupported"
        ]
        if rejected:
            written = await _synthesise(
                question, router, kept, stages, repair=rejected
            )
            verdicts, _ = await _check(
                question,
                written["sentences"],
                stages,
                kept=kept,
                retry_of=retry_of,
            )

        outcome = (
            "answered_flagged"
            if any(sentence["verdict"] == "unsupported" for sentence in verdicts)
            else "answered"
        )
        return _envelope(
            written["answer"],
            outcome,
            policy_name,
            stages,
            sentences=verdicts,
            synthesis_skipped=False,
        )

    except decisions.StageError as exc:
        exc.trace = _trace("error", policy_name, stages)
        raise
