"""Verify: thresholds, the uncited sentence, the single decision call."""

from __future__ import annotations

import pytest
from google.genai import types

from app import config, index, verify
from app.gates import decisions

KEPT = [{"chunk_id": "c100"}, {"chunk_id": "c109"}]


def _stub(monkeypatch, answers: dict) -> list[dict]:
    """Replace the decision call and record what verify asked."""
    calls: list[dict] = []

    def ask(input_text, questions, stage="decisions", model=config.DECISION_MODEL):
        calls.append(
            {"input_text": input_text, "questions": questions, "stage": stage}
        )
        return {name: answers[name] for name in questions}, {
            "tokens_in": 100,
            "tokens_out": 0,
        }

    monkeypatch.setattr(verify.decisions, "ask", ask)
    return calls


@pytest.mark.parametrize(
    ("support", "verdict"),
    [
        (0.99, "verified"),
        (0.60, "verified"),
        (0.59, "weak"),
        (0.30, "weak"),
        (0.29, "unsupported"),
        (0.01, "unsupported"),
    ],
)
def test_the_thresholds_are_inclusive_at_both_floors(monkeypatch, support, verdict):
    _stub(monkeypatch, {"sup_0": support})

    result = verify.run(
        "question",
        [{"text": "a line", "cited_ids": ["c100"]}],
        kept=KEPT,
    )

    assert result == [
        {
            "text": "a line",
            "chunk_id": "c100",
            "cited_ids": ["c100"],
            "support_noul": support,
            "verdict": verdict,
        }
    ]


def test_an_uncited_sentence_is_checked_against_every_kept_chunk(monkeypatch):
    calls = _stub(monkeypatch, {"sup_0": 0.9, "sup_1": 0.8})
    sentences = [
        {"text": "no citation", "cited_ids": []},
        {"text": "cited line", "cited_ids": ["c100"]},
    ]

    result = verify.run("question", sentences, kept=KEPT)

    assert len(calls) == 1
    assert set(calls[0]["questions"]) == {"sup_0", "sup_1"}
    assert calls[0]["stage"] == "verify"
    shared = calls[0]["input_text"]
    assert f"[c100] {index.by_id['c100']['text']}" in shared
    assert f"[c109] {index.by_id['c109']['text']}" in shared
    assert (
        calls[0]["questions"]["sup_0"]["instructions"]
        == "Is sentence 0 supported by any of the chunk texts tagged [c100], [c109]?"
    )
    assert result[0] == {
        "text": "no citation",
        "chunk_id": None,
        "cited_ids": [],
        "support_noul": 0.9,
        "verdict": "verified",
    }
    assert result[1]["chunk_id"] == "c100"


def test_nothing_checkable_spends_no_call(monkeypatch):
    calls = _stub(monkeypatch, {})

    result = verify.run(
        "question",
        [{"text": "no citation", "cited_ids": []}],
        kept=[],
    )

    assert calls == []
    assert result[0]["verdict"] == "unsupported"
    assert result[0]["chunk_id"] is None
    assert result[0]["cited_ids"] == []


def test_a_citation_outside_the_corpus_is_checked_against_the_kept_chunks(monkeypatch):
    calls = _stub(monkeypatch, {"sup_0": 0.42})

    result = verify.run(
        "question",
        [{"text": "a line", "cited_ids": ["zz999"]}],
        kept=KEPT,
    )

    assert len(calls) == 1
    assert (
        calls[0]["questions"]["sup_0"]["instructions"]
        == "Is sentence 0 supported by any of the chunk texts tagged [c100], [c109]?"
    )
    assert result[0] == {
        "text": "a line",
        "chunk_id": None,
        "cited_ids": ["zz999"],
        "support_noul": 0.42,
        "verdict": "weak",
    }


def test_the_shared_input_carries_the_sentences_and_the_cited_chunk_texts(monkeypatch):
    calls = _stub(monkeypatch, {"sup_0": 0.9})

    verify.run(
        "the question",
        [{"text": "a line", "cited_ids": ["c100"]}],
        kept=KEPT,
    )

    shared = calls[0]["input_text"]
    assert "the question" in shared
    assert "0: a line" in shared
    assert "[c100]" in shared
    assert (
        calls[0]["questions"]["sup_0"]["instructions"]
        == "Is sentence 0 supported by the chunk text tagged [c100]?"
    )


def test_several_cited_ids_are_named_in_the_instruction(monkeypatch):
    calls = _stub(monkeypatch, {"sup_0": 0.9})

    verify.run(
        "q",
        [{"text": "a line", "cited_ids": ["c100", "c109"]}],
        kept=KEPT,
    )

    assert (
        calls[0]["questions"]["sup_0"]["instructions"]
        == "Is sentence 0 supported by the chunk text tagged [c100], [c109]?"
    )


def test_a_valid_citation_outside_the_kept_set_is_still_bound(monkeypatch):
    calls = _stub(monkeypatch, {"sup_0": 0.9, "sup_1": 0.9})
    sentences = [
        {"text": "no citation", "cited_ids": []},
        {"text": "cited line", "cited_ids": ["c1"]},
    ]

    result = verify.run("q", sentences, kept=KEPT)

    assert len(calls) == 1
    shared = calls[0]["input_text"]
    assert f"[c1] {index.by_id['c1']['text']}" in shared
    assert f"[c100] {index.by_id['c100']['text']}" in shared
    assert f"[c109] {index.by_id['c109']['text']}" in shared
    assert result[1]["chunk_id"] == "c1"


def test_verdicts_keep_the_input_order(monkeypatch):
    _stub(
        monkeypatch,
        {"sup_0": 0.0, "sup_1": 0.1, "sup_2": 0.7},
    )
    sentences = [
        {"text": "first", "cited_ids": []},
        {"text": "second", "cited_ids": ["c100"]},
        {"text": "third", "cited_ids": ["c109"]},
    ]

    result = verify.run("q", sentences, kept=KEPT)

    assert [entry["text"] for entry in result] == ["first", "second", "third"]
    assert [entry["verdict"] for entry in result] == [
        "unsupported",
        "unsupported",
        "verified",
    ]


def test_stage_records_usage_and_the_sentence_payload(monkeypatch):
    _stub(monkeypatch, {"sup_0": 0.9})
    from app.trace import Stage

    stage = Stage(stage="verify", model=config.DECISION_MODEL)

    verify.run(
        "q",
        [{"text": "a line", "cited_ids": ["c100"]}],
        stage,
        kept=KEPT,
    )

    assert (stage.tokens_in, stage.tokens_out) == (100, 0)
    sentence = stage.payload["sentences"][0]
    assert sentence["verdict"] == "verified"
    assert sentence["cited_ids"] == ["c100"]
    assert sentence["chunk_id"] == "c100"


def test_a_missing_answer_raises_through_decisions(monkeypatch):
    class _EmptyAnswers:
        def create(self, **kwargs):
            return types.SimpleNamespace(answers={})

        decisions = None

    empty = _EmptyAnswers()
    empty.decisions = empty
    # Drop the offline replay so the real decisions.ask validates the payload.
    monkeypatch.undo()
    monkeypatch.setattr(decisions, "_client", empty)

    with pytest.raises(decisions.StageError) as raised:
        verify.run("q", [{"text": "a line", "cited_ids": ["c100"]}], kept=KEPT)

    assert raised.value.stage == "verify"
