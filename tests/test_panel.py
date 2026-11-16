"""Offline tests for the panel's shaping layer. No Streamlit, no model call."""

from __future__ import annotations

import ast
import pathlib

import pytest

from app import config
from app import panel_view as view

PANEL = pathlib.Path(__file__).resolve().parents[1] / "panel.py"


def _stage(name: str, latency_ms: float, **payload) -> dict:
    return {
        "stage": name,
        "latency_ms": latency_ms,
        "tokens_in": 100,
        "tokens_out": 10,
        "model": "test-model",
        **payload,
    }


def _receipt(cost: float | None, tokens_in: int | None = 812, tokens_out=None,
             skipped=True) -> dict:
    return {
        "list_cost_usd": cost,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "synthesis_skipped": skipped,
        "synthesis_budget_tokens_out": config.SYNTHESIS_OUTPUT_BUDGET,
        "per_stage": [],
    }


def _scope(**payload) -> dict:
    fields = {
        "in_scope_noul": 0.94,
        "router_choice": "app",
        "router_confidence": 0.88,
        "injection_noul": 0.02,
        "reason": "proceed",
    }
    fields.update(payload)
    return _stage("scope", 320.0, **fields)


def _rerank(kept: list[str], total: int = 5, **payload) -> dict:
    items = [
        {
            "chunk_id": f"c{index}",
            "relevance_noul": 0.44 if index == 1 else 0.31,
            "currency_noul": 0.9,
            "kept": f"c{index}" in kept,
            "reason": "kept"
            if f"c{index}" in kept
            else ("below_min_relevance" if index < 4 else "over_cap"),
        }
        for index in range(1, total + 1)
    ]
    return _stage(
        "rerank",
        430.0,
        items=items,
        kept_count=len(kept),
        candidate_count=total,
        **payload,
    )


def _verify(verdicts: list[str], retry_of: int | None = None) -> dict:
    stage = _stage(
        "verify",
        900.0,
        sentences=[
            {
                "text": f"sentence {index}",
                "chunk_id": "c231",
                "cited_ids": ["c231"],
                "support_noul": 0.9 if verdict == "verified" else 0.4,
                "verdict": verdict,
            }
            for index, verdict in enumerate(verdicts)
        ],
    )
    if retry_of is not None:
        stage["retry_of"] = retry_of
    return stage


def _answered(**overrides) -> dict:
    envelope = {
        "answer": "add 'use client'. keep the wrapper small.",
        "sentences": [
            {
                "text": "add 'use client'.",
                "chunk_id": "c231",
                "cited_ids": ["c231"],
                "support_noul": 0.91,
                "verdict": "verified",
            },
            {
                "text": "keep the wrapper small.",
                "chunk_id": "c232",
                "cited_ids": ["c232"],
                "support_noul": 0.88,
                "verdict": "verified",
            },
        ],
        "trace": {
            "outcome": "answered",
            "policy": "strict",
            "stages": [
                _scope(),
                _stage("retrieve", 210.0, router="app", candidate_count=30,
                       candidates=[f"c{index}" for index in range(1, 31)]),
                _rerank(["c1", "c2"]),
                _stage("synthesize", 2600.0, sentence_count=2, repair=False,
                       truncated=False),
                _verify(["verified", "verified"]),
            ],
        },
        "receipt": _receipt(0.00252, tokens_in=2516, tokens_out=109, skipped=False),
    }
    envelope.update(overrides)
    return envelope


# ------------------------------------------------------------------ presets


def test_presets_keep_the_labels_and_payloads_the_brief_names():
    assert [label for label, _ in view.PRESETS] == [
        "Official: App Router Params",
        "The Policy Flip: getServerSideProps",
        "Off-topic: Cake recipe",
    ]
    assert view.preset_payload("Official: App Router Params") == (
        "how do I read a route param in the app router"
    )
    assert view.preset_payload("The Policy Flip: getServerSideProps") == (
        "how do I use getServerSideProps inside an App Router server component"
    )
    assert view.preset_payload("Off-topic: Cake recipe") == "how to bake a cake"


# --------------------------------------------------------------- run guard


def test_a_preset_click_runs_under_the_chosen_policy():
    spec = view.next_run(
        preset="Off-topic: Cake recipe", submitted=False, question="old", policy="fast_fix"
    )
    assert spec == {"question": "how to bake a cake", "policy": "fast_fix"}


def test_a_preset_clicked_after_a_policy_switch_runs_the_new_policy():
    """The spec carries the switch's value, not the one stored by the last rerun."""
    session = {"_policy": "strict"}  # what the previous rerun left behind
    policy_at_top = session["_policy"]  # read before the switch widget exists
    policy = "fast_fix"  # the value the switch reports on this rerun
    session["_policy"] = policy

    spec = view.next_run(
        preset="The Policy Flip: getServerSideProps",
        submitted=False,
        question="",
        policy=policy,
    )

    assert policy_at_top == "strict"
    assert spec == {
        "question": view.preset_payload("The Policy Flip: getServerSideProps"),
        "policy": "fast_fix",
    }

    # The shaping layer is right either way; only the panel's call site can
    # hand it the stale read. Pin that call site to the widget's local.
    tree = ast.parse(PANEL.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "next_run"
    ]
    assert len(calls) == 1
    handed = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(handed["policy"]) == "policy"


def test_a_form_submit_runs_the_typed_question():
    spec = view.next_run(
        preset=None, submitted=True, question="  how do I read cookies  ", policy="strict"
    )
    assert spec == {"question": "how do I read cookies", "policy": "strict"}


@pytest.mark.parametrize(
    ("preset", "submitted", "question"),
    [
        (None, False, "how do I read a route param in the app router"),
        (None, False, ""),
        (None, True, "   "),
    ],
)
def test_a_rerun_policy_change_or_text_edit_calls_nothing(preset, submitted, question):
    calls = []

    def fake_runner(question_text, policy_name):
        calls.append((question_text, policy_name))
        return {"trace": {"outcome": "answered", "stages": []}}

    spec = view.next_run(
        preset=preset, submitted=submitted, question=question, policy="fast_fix"
    )
    if spec is not None:
        fake_runner(spec["question"], spec["policy"])
    assert spec is None
    assert calls == []


def test_each_spec_costs_exactly_one_call():
    calls = []

    def fake_runner(question_text, policy_name):
        calls.append((question_text, policy_name))
        return {}

    for spec in (
        view.next_run(preset="Off-topic: Cake recipe", submitted=False,
                      question="", policy="strict"),
        view.next_run(preset=None, submitted=True, question="what is a layout",
                      policy="fast_fix"),
    ):
        if spec is not None:
            fake_runner(spec["question"], spec["policy"])
    assert calls == [
        ("how to bake a cake", "strict"),
        ("what is a layout", "fast_fix"),
    ]


# ------------------------------------------------------------------ answered


def test_answered_splits_sentences_and_keeps_one_badge_each():
    model = view.shape(_answered(), question="q", policy="strict")
    assert model["outcome"] == "answered"
    assert model["policy"] == "strict"
    assert [sentence["verdict"] for sentence in model["sentences"]] == [
        "verified",
        "verified",
    ]
    assert [sentence["cite"] for sentence in model["sentences"]] == ["c231", "c232"]
    assert model["warning"] is None
    assert model["headline"].startswith("answered · 2 of 2 sentences verified")


def test_answered_cost_line_carries_cost_wall_time_and_tokens():
    model = view.shape(_answered(), question="q", policy="strict")
    assert model["cost"] == (
        "$0.00252 at list price · 4.46 s · 2,516 tokens in, 109 out"
    )
    assert model["stages_total"]["count"] == 5
    assert "$0.00252" in model["stages_total"]["right"]


def test_xray_uses_real_fields_and_prints_retry_of_absent():
    model = view.shape(_answered(), question="q", policy="strict")
    names = [row["name"] for row in model["stages"]]
    assert names == ["scope", "retrieve", "rerank", "synthesize", "verify"]

    keys = {
        field["key"]
        for row in model["stages"]
        for field in row["fields"]
        if field["key"] is not None
    }
    assert {
        "in_scope_noul",
        "router_choice",
        "injection_noul",
        "candidate_count",
        "kept_count",
        "relevance",
        "sentence_count",
        "verdicts",
        "retry_of",
    } <= keys

    verify_row = model["stages"][-1]
    retry = [field for field in verify_row["fields"] if field["key"] == "retry_of"][0]
    assert retry["parts"] == [{"text": "absent", "mark": "gone"}]


def test_a_stage_that_did_not_run_has_no_row():
    envelope = _answered()
    envelope["trace"]["outcome"] = "refused"
    envelope["trace"]["stages"] = [
        _scope(),
        _stage("retrieve", 210.0, router="app", candidate_count=5, candidates=[]),
        _rerank([]),
    ]
    envelope["answer"] = config.REFUSAL_ANSWER
    envelope["sentences"] = []
    envelope["receipt"] = _receipt(0.00091)
    model = view.shape(envelope, question="q", policy="strict")
    assert [row["name"] for row in model["stages"]] == ["scope", "retrieve", "rerank"]


# ------------------------------------------------------------------- flagged


def test_flagged_answer_warns_and_reports_the_repair():
    envelope = _answered()
    envelope["trace"]["outcome"] = "answered_flagged"
    envelope["trace"]["stages"][-2]["repair"] = True
    envelope["trace"]["stages"][-2]["sentence_count"] = 2
    envelope["trace"]["stages"][-2]["latency_ms"] = 1200.0
    envelope["trace"]["stages"].append(
        _stage("synthesize", 1300.0, sentence_count=2, repair=True, truncated=False)
    )
    envelope["trace"]["stages"].append(_verify(["verified", "unsupported"], retry_of=4))
    envelope["sentences"][-1]["verdict"] = "unsupported"

    model = view.shape(envelope, question="q", policy="strict")
    assert model["warning"] is not None
    assert "repair pass" in model["warning"]
    assert "answered_flagged" in model["headline"]
    assert [row["name"] for row in model["stages"]].count("verify") == 2
    last = model["stages"][-1]
    retry = [field for field in last["fields"] if field["key"] == "retry_of"][0]
    assert retry["parts"][0]["text"] == "4"


# ------------------------------------------------------------------- refused


def test_refused_counts_the_candidates_the_policy_dropped():
    envelope = _answered()
    envelope["trace"]["outcome"] = "refused"
    envelope["trace"]["stages"] = [
        _scope(),
        _stage("retrieve", 210.0, router="app", candidate_count=5,
               candidates=["c1", "c2"]),
        _rerank([]),
    ]
    envelope["answer"] = config.REFUSAL_ANSWER
    envelope["sentences"] = []
    envelope["receipt"] = _receipt(0.00091)

    model = view.shape(envelope, question="q", policy="strict")
    assert model["answer"] == config.REFUSAL_ANSWER
    assert model["detail"][0] == (
        "5 of 5 candidates failed strict. Best match 0.44 against a floor of 0.60."
    )
    assert model["detail"][1] == "relevance 0.44 · 0.31 · 0.31 · 0.31 · 0.31"
    assert "below_min_relevance" in model["detail"][2]
    assert model["cost"].startswith("$0.00091 at list price")


# ------------------------------------------------------- blocked and clarify


def test_blocked_shows_the_scope_answer_and_its_receipt():
    envelope = {
        "answer": config.ANSWERS["blocked_out_of_scope"],
        "sentences": [],
        "trace": {
            "outcome": "blocked",
            "policy": "strict",
            "stages": [_scope(reason="blocked_out_of_scope", in_scope_noul=0.12)],
        },
        "receipt": _receipt(0.00041, tokens_in=None, tokens_out=None),
    }
    model = view.shape(envelope, question="how to bake a cake", policy="strict")
    assert model["answer"] == config.ANSWERS["blocked_out_of_scope"]
    assert model["detail"] == [
        "Scope stopped the turn before retrieval · reason blocked_out_of_scope."
    ]
    assert [row["name"] for row in model["stages"]] == ["scope"]
    keys = {
        field["key"]
        for field in model["stages"][0]["fields"]
        if field["key"] is not None
    }
    assert "candidate_count" not in keys
    assert model["cost"] == (
        "$0.00041 at list price · 0.32 s · -- tokens in, -- out · "
        "256-token output budget not spent"
    )


def test_clarify_shows_the_router_question_and_its_receipt():
    envelope = {
        "answer": config.ANSWERS["unclear_router"],
        "sentences": [],
        "trace": {
            "outcome": "clarify",
            "policy": "fast_fix",
            "stages": [
                _scope(router_choice="unclear", router_confidence=0.55,
                       reason="unclear_router")
            ],
        },
        "receipt": _receipt(0.00052),
    }
    model = view.shape(envelope, question="how do I read cookies", policy="fast_fix")
    assert model["policy"] == "fast_fix"
    assert model["answer"] == config.ANSWERS["unclear_router"]
    assert "router_confidence 0.55" in model["detail"][0]
    assert [row["name"] for row in model["stages"]] == ["scope"]
    assert "$0.00052" in model["cost"]


# --------------------------------------------------------------------- error


def test_error_names_the_stage_and_claims_no_cost():
    trace = {
        "outcome": "error",
        "policy": "strict",
        "stages": [_scope(), _stage("retrieve", 180.0, router="app",
                                    candidate_count=0, candidates=[])],
    }
    model = view.shape_error("rerank", "RuntimeError: boom", trace)
    assert model["outcome"] == "error"
    assert model["headline"] == "error · the rerank stage failed"
    assert model["cost"] is None
    assert model["detail"][0] == "rerank: RuntimeError: boom"
    assert "question stays in the box" in model["detail"][1]
    assert [row["name"] for row in model["stages"]] == ["scope", "retrieve"]
    assert model["stages_total"]["right"] == "--"


# ----------------------------------------------------------------- scorecard


def test_scorecard_reads_the_harness_aggregate_and_blanks_are_not_zero():
    payload = {
        "run_at": "2026-11-15T20:58:09+00:00",
        "updated_at": "2026-11-15T21:38:04+00:00",
        "count": 120,
        "live_calls": 384,
        "aggregate": {
            "strict": {
                "hit_at_5": 0.6667,
                "answered": 20,
                "wall_ms_mean": 22245.735,
                "cost_mean_usd": 0.0018248875,
            },
            "fast_fix": {
                "hit_at_5": 0.9,
                "answered": 28,
                "wall_ms_mean": 14634.949,
                "cost_mean_usd": 0.00196994,
            },
            "no_gates": {
                "hit_at_5": None,
                "answered": None,
                "wall_ms_mean": None,
                "cost_mean_usd": None,
            },
        },
    }
    card = view.scorecard(payload)
    assert card["missing"] is False
    assert [row["name"] for row in card["rows"]] == ["no_gates", "strict", "fast_fix"]
    assert card["rows"][0]["cells"] == ["--", "--", "--", "--"]
    assert card["rows"][1]["cells"] == ["0.667", "20", "22,246", "1.82"]
    assert card["rows"][2]["cells"] == ["0.900", "28", "14,635", "1.97"]
    assert "2026-11-15T20:58:09+00:00" in card["run_line"]
    assert "not a zero" in card["note"]


def test_missing_results_file_draws_no_table():
    card = view.scorecard(None)
    assert card["missing"] is True
    assert card["rows"] == []
    assert "missing" in card["run_line"]


def test_headers_are_the_harness_keys():
    card = view.scorecard({"aggregate": {}})
    assert card["headers"] == [
        "hit_at_5",
        "answered",
        "wall_ms_mean",
        "cost_mean_usd × 1,000",
    ]


# -------------------------------------------------------------------- labels


def test_corpus_line_names_the_provenance():
    assert view.corpus_line(246, 17) == (
        "Next.js App Router docs · 246 chunks · 17 of them forum posts"
    )
