"""The only module that imports ``google.genai``.

The client is built once with ``http_options={"api_version": "v1beta"}``.
That version is required: ``gemini-decision-1`` is served by the
``createDecision`` method, and the v1beta routing is the documented one. The
key comes from ``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``.

The decision model is not a chat model. ``client.models.generate_content``
returns 404 for it; ``client.decisions.create`` is the call that works.

Question shape
--------------
    {"in_scope": {"type": "noul", "instructions": "..."}}
    {"router": {"type": "choice", "instructions": "...",
                "criteria": {"app": "rubric", "pages": "rubric", "unclear": "rubric"}}}

``noul`` is a number from 0.01 to 0.99. A ``choice`` answer names one of the
criteria keys and carries its own confidence.

Answer shape
------------
    answers = {"in_scope": 0.93,
               "router": {"option": "app", "confidence": 0.91}}
    usage   = {"tokens_in": 410, "tokens_out": None}

A ``choice`` answer arrives as ``choice`` plus ``confidence`` and comes back
as ``option`` plus ``confidence``.

Every requested question must come back valid. A missing answer, a wrong type,
a ``noul`` outside ``[0.01, 0.99]``, or a ``choice`` that names no requested
criterion raises ``StageError`` for the stage that asked. Nothing is defaulted
and nothing is clamped. A usage field that is missing or not an integer reads
as ``None``. No key and no input text are ever logged or put into an error
message.
"""

from __future__ import annotations

import math
import os

from google import genai
from google.genai import types

from app import config

NOUL_MIN = 0.01
NOUL_MAX = 0.99


class StageError(Exception):
    """A pipeline stage failed. Carries the stage name and a safe message."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.trace: dict | None = None


_client: genai.Client | None = None
_key: str | None = None


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise StageError("decisions", "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
    return key


def _client_or_raise() -> genai.Client:
    global _client, _key
    if _client is None:
        _key = _api_key()
        _client = genai.Client(
            api_key=_key, http_options={"api_version": "v1beta"}
        )
    return _client


def _redact(message: str) -> str:
    if _key:
        message = message.replace(_key, "***")
    return message


def _fail(stage: str, exc: Exception) -> StageError:
    return StageError(stage, _redact(f"{type(exc).__name__}: {exc}"))


def _model_path(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def _field(raw, name: str):
    """Read one field from an answer or usage object, or from a plain mapping."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get(name)
    return getattr(raw, name, None)


def _invalid(stage: str, name: str) -> StageError:
    return StageError(stage, f"missing or invalid answer for {name}")


def _number(value) -> float | None:
    """A finite float from a number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _noul(stage: str, name: str, value) -> float:
    number = _number(value)
    if number is None:
        number = _number(_field(value, "noul"))
    if number is None or not NOUL_MIN <= number <= NOUL_MAX:
        raise _invalid(stage, name)
    return number


def _choice(stage: str, name: str, value, criteria: dict) -> dict:
    option = _field(value, "choice")
    if not isinstance(option, str) or option not in criteria:
        raise _invalid(stage, name)
    confidence = _number(_field(value, "confidence"))
    if confidence is None:
        raise _invalid(stage, name)
    return {"option": option, "confidence": confidence}


def _tokens(value) -> int | None:
    """A token count, or ``None`` when the field is missing or not an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def ask(
    input_text: str,
    questions: dict,
    stage: str = "decisions",
    model: str = config.DECISION_MODEL,
) -> tuple[dict, dict]:
    """Run one decision call and return ``(answers, usage)``.

    ``input_text`` is the shared material under judgement. ``questions`` maps
    a name to a ``noul`` or ``choice`` question (see the module docstring).
    ``stage`` names the caller in every error this function raises.
    """
    client = _client_or_raise()
    try:
        response = client.decisions.create(
            input=input_text,
            model=_model_path(model),
            questions=questions,
            api_version="v1beta",
        )
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped into a stage failure
        raise _fail(stage, exc) from None

    raw_answers = _field(response, "answers")
    if not isinstance(raw_answers, dict):
        raw_answers = {}

    answers = {}
    for name, question in questions.items():
        given = raw_answers.get(name)
        if question.get("type") == "noul":
            answers[name] = _noul(stage, name, given)
        elif question.get("type") == "choice":
            answers[name] = _choice(stage, name, given, question["criteria"])
        else:
            raise _invalid(stage, name)

    try:
        usage_meta = _field(response, "usage")
        usage = {
            "tokens_in": _tokens(_field(usage_meta, "total_input_tokens")),
            "tokens_out": _tokens(_field(usage_meta, "total_output_tokens")),
        }
    except Exception as exc:  # noqa: BLE001 - a broken usage payload is a stage failure
        raise StageError(
            stage, f"missing or invalid usage payload: {type(exc).__name__}"
        ) from None
    return answers, usage


def embed_query(text: str, model: str = config.EMBED_MODEL) -> tuple[list[float], dict]:
    """Embed a search query with ``RETRIEVAL_QUERY`` at ``EMBED_DIM`` dimensions."""
    client = _client_or_raise()
    try:
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=config.EMBED_DIM,
            ),
        )
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped into a stage failure
        raise _fail("embed", exc) from None

    embeddings = getattr(response, "embeddings", None) or []
    values = list(embeddings[0].values) if embeddings else []
    if len(values) != config.EMBED_DIM:
        raise StageError(
            "embed", f"expected {config.EMBED_DIM} dimensions, got {len(values)}"
        )
    metadata = _field(response, "metadata")
    usage = {
        "tokens_in": _tokens(_field(metadata, "billable_prompt_token_count")),
        "tokens_out": 0,
    }
    return values, usage
