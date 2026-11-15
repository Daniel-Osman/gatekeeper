"""The pipeline end to end: synthesise, verify, one repair pass, the flag.

Every case here asks one of four wordings of the same route-param question, so
``canonical_slice`` keeps the candidate slice on the one the rerank fixture
answers. Scope, rerank, synthesis, verify and the receipt are the real code.
"""

from __future__ import annotations

import asyncio

from app.pipeline import run

QUESTION = "how do I read a route param in the app router"
ON_THE_CLIENT = "how do I read a route param in the app router on the client"
IN_A_LAYOUT = "how do I read a route param in the app router in a layout"
IN_A_SERVER_COMPONENT = (
    "how do I read a route param in the app router in a server component"
)
MIDDLEWARE = "why is my middleware not running in the app router"


def _ask(question: str) -> dict:
    return asyncio.run(run(question, "strict"))


def _stages(envelope: dict, name: str) -> list[dict]:
    return [stage for stage in envelope["trace"]["stages"] if stage["stage"] == name]


def _positions(envelope: dict, name: str) -> list[int]:
    return [
        position
        for position, stage in enumerate(envelope["trace"]["stages"])
        if stage["stage"] == name
    ]


def test_a_supported_answer_is_answered_after_one_synthesise_and_one_verify(
    canonical_slice,
):
    envelope = _ask(QUESTION)

    assert envelope["trace"]["outcome"] == "answered"
    assert len(envelope["sentences"]) == 3
    assert [sentence["verdict"] for sentence in envelope["sentences"]] == [
        "verified",
        "verified",
        "verified",
    ]
    assert len(_stages(envelope, "synthesize")) == 1
    assert len(_stages(envelope, "verify")) == 1
    assert envelope["trace"]["stages"][-1]["stage"] == "verify"
    assert envelope["answer"] == (
        "The params prop holds the route's dynamic values "
        "The params promise resolves before the page renders "
        "Dynamic segments fill the params object from the file name"
    )


def test_the_synthesis_stage_records_the_fixture_usage(canonical_slice):
    envelope = _ask(QUESTION)

    synthesize = _stages(envelope, "synthesize")[0]

    assert (synthesize["tokens_in"], synthesize["tokens_out"]) == (812, 64)
    assert synthesize["sentence_count"] == 3
    assert synthesize["repair"] is False


def test_one_unsupported_sentence_buys_exactly_one_repair(canonical_slice):
    envelope = _ask(ON_THE_CLIENT)

    assert envelope["trace"]["outcome"] == "answered"
    assert len(_stages(envelope, "synthesize")) == 2
    assert len(_stages(envelope, "verify")) == 2
    assert [stage["repair"] for stage in _stages(envelope, "synthesize")] == [
        False,
        True,
    ]

    first_verify, second_verify = _positions(envelope, "verify")
    assert (
        envelope["trace"]["stages"][second_verify]["retry_of"] == first_verify
    )

    stages = envelope["trace"]["stages"]
    assert [sentence["verdict"] for sentence in stages[first_verify]["sentences"]] == [
        "verified",
        "unsupported",
        "verified",
    ]
    assert [
        sentence["verdict"] for sentence in stages[second_verify]["sentences"]
    ] == ["verified", "verified"]
    assert envelope["sentences"] == stages[second_verify]["sentences"]


def test_a_sentence_still_unsupported_after_repair_flags_the_answer(canonical_slice):
    envelope = _ask(IN_A_SERVER_COMPONENT)

    assert envelope["trace"]["outcome"] == "answered_flagged"
    assert len(_stages(envelope, "synthesize")) == 2
    assert len(_stages(envelope, "verify")) == 2

    last_verify = envelope["trace"]["stages"][_positions(envelope, "verify")[-1]]
    assert envelope["sentences"] == last_verify["sentences"]
    assert [sentence["verdict"] for sentence in envelope["sentences"]] == [
        "verified",
        "unsupported",
        "verified",
    ]
    assert envelope["sentences"][1]["support_noul"] == 0.12


def test_a_refused_question_never_reaches_synthesis(canonical_slice):
    envelope = _ask(MIDDLEWARE)

    assert envelope["trace"]["outcome"] == "refused"
    assert _stages(envelope, "synthesize") == []
    assert envelope["sentences"] == []
    assert envelope["receipt"]["synthesis_skipped"] is True


def test_absent_synthesis_usage_leaves_the_receipt_unmeasured(canonical_slice):
    envelope = _ask(IN_A_LAYOUT)

    assert envelope["trace"]["outcome"] == "answered"
    synthesize = _stages(envelope, "synthesize")[0]
    assert (synthesize["tokens_in"], synthesize["tokens_out"]) == (None, None)

    receipt = envelope["receipt"]
    assert receipt["tokens_in"] is None
    assert receipt["tokens_out"] is None
    synthesized = [
        entry for entry in receipt["per_stage"] if entry["stage"] == "synthesize"
    ][0]
    assert synthesized["price_basis"] == "unmeasured"
    assert synthesized["list_cost_usd"] is None
