"""Synthesis: the prompt, citation parsing, the refusal line, usage accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import config, synth
from app.gates import decisions
from app.trace import Stage

KEPT = [{"chunk_id": "c100"}, {"chunk_id": "c109"}]
QUESTION = "how do I read a route param in the app router"


def _generate(monkeypatch, text: str, usage: dict | None = None) -> list[str]:
    prompts: list[str] = []

    def fake(prompt: str):
        prompts.append(prompt)
        return text, {"tokens_in": 7, "tokens_out": 3} if usage is None else usage

    monkeypatch.setattr(synth, "_generate", fake)
    return prompts


def test_parses_a_single_trailing_citation(monkeypatch):
    _generate(monkeypatch, "Read the params prop [c100]")

    result = synth.run(QUESTION, "app", KEPT)

    assert result == {
        "answer": "Read the params prop",
        "sentences": [{"text": "Read the params prop", "cited_ids": ["c100"]}],
    }


def test_parses_several_citations_and_a_full_stop_after_them(monkeypatch):
    _generate(monkeypatch, "First line [c100][c109].\nSecond line [c109]")

    sentences = synth.run(QUESTION, "app", KEPT)["sentences"]

    assert sentences == [
        {"text": "First line.", "cited_ids": ["c100", "c109"]},
        {"text": "Second line", "cited_ids": ["c109"]},
    ]


def test_a_line_without_a_citation_keeps_an_empty_list(monkeypatch):
    _generate(monkeypatch, "Cited line [c100]\nUncited line.")

    sentences = synth.run(QUESTION, "app", KEPT)["sentences"]

    assert sentences[1] == {"text": "Uncited line.", "cited_ids": []}


def test_blank_lines_are_dropped_and_the_answer_stops_at_five(monkeypatch):
    text = "\n\n".join(f"Line {n} [c100]" for n in range(1, 8))
    _generate(monkeypatch, text)
    stage = Stage(stage="synthesize", model=config.SYNTH_MODEL)

    result = synth.run(QUESTION, "app", KEPT, stage)

    assert [sentence["text"] for sentence in result["sentences"]] == [
        f"Line {n}" for n in range(1, config.MAX_SENTENCES + 1)
    ]
    assert result["answer"] == " ".join(f"Line {n}" for n in range(1, 6))
    assert stage.payload["truncated"] is True


def test_a_trailing_fragment_is_dropped_and_flagged(monkeypatch):
    _generate(
        monkeypatch,
        "Read the params prop [c100]\nIn async pages or layouts, `params",
    )
    stage = Stage(stage="synthesize", model=config.SYNTH_MODEL)

    result = synth.run(QUESTION, "app", KEPT, stage)

    assert result["sentences"] == [
        {"text": "Read the params prop", "cited_ids": ["c100"]}
    ]
    assert "In async pages" not in result["answer"]
    assert stage.payload["truncated"] is True


def test_a_fragment_only_response_records_truncation_then_raises(monkeypatch):
    _generate(monkeypatch, "In async pages or layouts, `params")
    stage = Stage(stage="synthesize", model=config.SYNTH_MODEL)

    with pytest.raises(decisions.StageError):
        synth.run(QUESTION, "app", KEPT, stage)

    assert stage.payload == {
        "sentence_count": 0,
        "repair": False,
        "truncated": True,
    }


def test_a_refusal_is_one_uncited_sentence(monkeypatch):
    _generate(monkeypatch, config.REFUSAL_ANSWER)

    result = synth.run(QUESTION, "app", KEPT)

    assert result == {
        "answer": config.REFUSAL_ANSWER,
        "sentences": [{"text": config.REFUSAL_ANSWER, "cited_ids": []}],
    }


def test_the_prompt_carries_the_question_router_and_tagged_chunks(monkeypatch):
    prompts = _generate(monkeypatch, "Cited line [c100]")

    synth.run(QUESTION, "pages", [{"chunk_id": "c100", "text": "chunk body"}])

    prompt = prompts[0]
    assert QUESTION in prompt
    assert "ROUTER\npages" in prompt
    assert "[c100] chunk body" in prompt
    assert config.REFUSAL_ANSWER in prompt
    assert str(config.MAX_SENTENCES) in prompt


def test_the_repair_prompt_names_the_rejected_lines(monkeypatch):
    prompts = _generate(monkeypatch, "Cited line [c100]")

    synth.run(QUESTION, "app", KEPT, repair=["The cache never revalidates"])

    assert "UNSUPPORTED LINES\nThe cache never revalidates" in prompts[0]


def test_stage_records_tokens_sentence_count_and_the_repair_flag(monkeypatch):
    _generate(monkeypatch, "One [c100]\nTwo [c109]")
    stage = Stage(stage="synthesize", model=config.SYNTH_MODEL)

    synth.run(QUESTION, "app", KEPT, stage)

    assert (stage.tokens_in, stage.tokens_out) == (7, 3)
    assert stage.payload == {
        "sentence_count": 2,
        "repair": False,
        "truncated": False,
    }


def test_absent_usage_is_none_not_zero(monkeypatch):
    _generate(monkeypatch, "One [c100]", usage={})
    stage = Stage(stage="synthesize", model=config.SYNTH_MODEL)

    synth.run(QUESTION, "app", KEPT, stage)

    assert stage.tokens_in is None
    assert stage.tokens_out is None


def test_the_call_cap_is_eight_hundred(monkeypatch):
    recorded: list[dict] = []

    def generate_content(**kwargs):
        recorded.append(kwargs)
        return SimpleNamespace(text="One [c100]", usage_metadata=None)

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    # Drop the offline replay so the real ``_generate`` builds the call config.
    monkeypatch.undo()
    monkeypatch.setattr(synth, "_client", stub)

    synth.run(QUESTION, "app", KEPT)

    assert len(recorded) == 1
    assert recorded[0]["config"].max_output_tokens == 800
    assert (
        recorded[0]["config"].max_output_tokens
        == config.SYNTH_MAX_OUTPUT_TOKENS
    )


def test_an_api_failure_is_a_synthesize_stage_error(monkeypatch):
    class _Boom:
        def generate_content(self, **kwargs):
            raise RuntimeError("upstream 500")

        models = None

    boom = _Boom()
    boom.models = boom
    # Drop the offline replay so the real ``_generate`` calls the failing client.
    monkeypatch.undo()
    monkeypatch.setattr(synth, "_client", boom)

    with pytest.raises(decisions.StageError) as raised:
        synth.run(QUESTION, "app", KEPT)

    assert raised.value.stage == "synthesize"
    assert "RuntimeError" in raised.value.message


def test_an_empty_response_is_a_stage_error(monkeypatch):
    _generate(monkeypatch, "   ")

    with pytest.raises(decisions.StageError) as raised:
        synth.run(QUESTION, "app", KEPT)

    assert raised.value.stage == "synthesize"
