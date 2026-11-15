"""Synthesis: one chat call writes the answer, cited line by line.

The chat model is ``gemini-3.5-flash``. It reads the question, the router and
the chunks the gates kept, then writes at most ``MAX_SENTENCES`` lines. Every
line ends with the chunk ids it used, as ``[c12]`` or ``[c12][c30]``. A line
with no citation is still a line: it is returned with ``cited_ids: []`` and the
verify gate decides what an uncited sentence is worth. A synthesizer that
cannot answer replies exactly ``config.REFUSAL_ANSWER``; that reply is returned
as one uncited sentence, so it faces the same check as any other line.

A line that does not end its sentence (no trailing citation, no full stop) is
a fragment the cap cut off: it is dropped rather than shipped, and the stage
payload records ``truncated: true``. Lines past ``config.MAX_SENTENCES`` are
dropped for the same reason and set the same flag.

The prompt asks for ``temperature=0`` and ``max_output_tokens`` from
``config.SYNTH_MAX_OUTPUT_TOKENS``. Usage is read from ``usage_metadata``:
``prompt_token_count`` and ``candidates_token_count``. An absent count is
``None``, never ``0``. An API failure is a ``StageError`` named ``synthesize``.

``_generate`` is the one seam the offline tests replace: everything above it
(prompt building, parsing, the stage accounting) is the real code.
"""

from __future__ import annotations

import os
import re

from google import genai
from google.genai import types

from app import config, index
from app.gates import decisions

# One trailing run of ``[id]`` groups, and the ids inside it. A full stop after
# the citations is kept with the sentence text, not with the citations.
_TRAILING_CITES = re.compile(r"(?P<cites>(?:\s*\[[^\[\]\n]+\])+)\s*$")
_CITE = re.compile(r"\[([^\[\]\n]+)\]")
# Terminal punctuation, optionally closed by a quote or bracket.
_STOP = re.compile(r"[.!?][\"')\]]?$")

_INSTRUCTIONS = (
    "Answer only from the chunks above. "
    f"Use at most {config.MAX_SENTENCES} sentences, one per line. "
    "Keep every sentence under 30 words. "
    "End every line with the chunk ids it used, in the form [c12] or "
    "[c12][c30]. "
    "No preamble and no markdown. "
    f'If the chunks do not answer the question, reply exactly with "'
    f'{config.REFUSAL_ANSWER}" and no citations.'
)

_REPAIR_INSTRUCTIONS = (
    "These lines were not supported by the chunks they cite. Drop each one, or "
    "rewrite it so its cited chunks do support it. The citation rules above "
    "still hold."
)

_client: genai.Client | None = None
_key: str | None = None


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise decisions.StageError(
            "synthesize", "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set"
        )
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


def _fail(exc: Exception) -> decisions.StageError:
    return decisions.StageError("synthesize", _redact(f"{type(exc).__name__}: {exc}"))


def _count(value) -> int | None:
    """A token count, or ``None`` when the field is missing or not an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _chunk(item) -> tuple[str, str]:
    """The id and text of one kept chunk, from a dict or a bare id."""
    chunk_id = item.get("chunk_id") if isinstance(item, dict) else item
    if not isinstance(chunk_id, str) or not chunk_id:
        raise decisions.StageError("synthesize", "kept chunk has no chunk_id")
    text = item.get("text") if isinstance(item, dict) else None
    if not isinstance(text, str):
        chunk = index.by_id.get(chunk_id)
        text = chunk["text"] if chunk else ""
    return chunk_id, text


def _prompt(
    question: str, router: str, kept: list, repair: list[str] | None
) -> str:
    blocks = [f"QUESTION\n{question}", f"ROUTER\n{router}", "CHUNKS"]
    for item in kept:
        chunk_id, text = _chunk(item)
        blocks.append(f"[{chunk_id}] {text}")
    blocks.append(f"INSTRUCTIONS\n{_INSTRUCTIONS}")
    if repair:
        blocks.append(
            f"{_REPAIR_INSTRUCTIONS}\n\nUNSUPPORTED LINES\n" + "\n".join(repair)
        )
    return "\n\n".join(blocks)


def _generate(prompt: str) -> tuple[str, dict]:
    """One chat call. Returns ``(text, usage)``; the tests replace this."""
    client = _client_or_raise()
    try:
        response = client.models.generate_content(
            model=config.SYNTH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=config.SYNTH_MAX_OUTPUT_TOKENS,
                # Thinking tokens are billed against ``max_output_tokens`` on
                # this model, so a synthesizer that must finish its sentence
                # runs with thinking off. The temperature stays at 0.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except decisions.StageError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped into a stage failure
        raise _fail(exc) from None

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise decisions.StageError("synthesize", "empty response")

    usage_meta = getattr(response, "usage_metadata", None)
    return text, {
        "tokens_in": _count(getattr(usage_meta, "prompt_token_count", None)),
        "tokens_out": _count(getattr(usage_meta, "candidates_token_count", None)),
    }


def _line(line: str) -> tuple[str, list[str]]:
    """Split one line into its text and the ids cited at its end."""
    body = line.rstrip()
    stop = ""
    if body and body[-1] in ".!?":
        stop = body[-1]
        body = body[:-1].rstrip()
    match = _TRAILING_CITES.search(body)
    if match is None:
        return line.strip(), []
    text = (body[: match.start()].rstrip() + stop).strip()
    cited = [token.strip() for token in _CITE.findall(match.group("cites"))]
    return text, [token for token in cited if token]


def _complete(line: str) -> bool:
    """True when the line ends its sentence: a trailing citation or a full stop."""
    body = line.strip()
    if not body:
        return False
    return _TRAILING_CITES.search(body) is not None or _STOP.search(body) is not None


def _sentences(text: str) -> tuple[list[dict], bool]:
    """The complete lines, and whether a fragment or an extra line was dropped."""
    lines = [line for line in text.splitlines() if line.strip()]
    truncated = False
    if len(lines) > config.MAX_SENTENCES:
        lines = lines[: config.MAX_SENTENCES]
        truncated = True
    if lines and not _complete(lines[-1]):
        lines.pop()
        truncated = True
    sentences = []
    for line in lines:
        sentence, cited = _line(line)
        if not sentence:
            continue
        sentences.append({"text": sentence, "cited_ids": cited})
    return sentences, truncated


def run(
    question: str,
    router: str,
    kept: list,
    stage=None,
    *,
    repair: list[str] | None = None,
) -> dict:
    """Write the answer from the kept chunks.

    ``repair`` is the list of sentence texts the verify gate rejected; when it
    is present the prompt also asks for those lines to be dropped or rewritten.
    """
    text, usage = _generate(_prompt(question, router, kept, repair))
    if stage is not None:
        stage.tokens_in = usage.get("tokens_in")
        stage.tokens_out = usage.get("tokens_out")

    if text.strip() == config.REFUSAL_ANSWER:
        sentences = [{"text": config.REFUSAL_ANSWER, "cited_ids": []}]
        truncated = False
    else:
        sentences, truncated = _sentences(text)

    if stage is not None:
        stage.payload = {
            "sentence_count": len(sentences),
            "repair": repair is not None,
            "truncated": truncated,
        }
    if not sentences:
        raise decisions.StageError("synthesize", "response has no sentences")

    return {
        "answer": " ".join(sentence["text"] for sentence in sentences),
        "sentences": sentences,
    }
