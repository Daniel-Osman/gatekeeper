"""Gate 3: is each synthesised sentence supported by the chunk it cites?

One decision call covers the whole sentence list: one ``noul`` per sentence,
keyed ``sup_<i>``, with ``i`` the sentence's position in the input list. The
instructions name the ids a sentence is checked against, and those chunk texts
sit in the shared input beside the sentence list.

A sentence is checked against the ids it cites. A sentence that cites nothing,
or that cites only ids the corpus does not hold, is checked against every kept
chunk instead: the model's score is the verdict and the recorded ``chunk_id``
is ``null``. Only a sentence with no chunk to check against at all - an
uncited sentence in a turn that kept nothing - is ``unsupported`` without a
call.

Verdicts: ``verified`` at or above ``VERIFY_VERIFIED_MIN``, ``weak`` at or above
``VERIFY_WEAK_MIN``, otherwise ``unsupported``. A missing or invalid answer
raises through ``decisions.ask``: there is no default score.
"""

from __future__ import annotations

from app import config, index
from app.gates import decisions


def _verdict(noul: float) -> str:
    if noul >= config.VERIFY_VERIFIED_MIN:
        return "verified"
    if noul >= config.VERIFY_WEAK_MIN:
        return "weak"
    return "unsupported"


def _chunk(item) -> tuple[str, str]:
    """The id and text of one kept chunk, from a dict or a bare id."""
    chunk_id = item.get("chunk_id") if isinstance(item, dict) else item
    if not isinstance(chunk_id, str) or not chunk_id:
        raise decisions.StageError("verify", "kept chunk has no chunk_id")
    text = item.get("text") if isinstance(item, dict) else None
    if not isinstance(text, str):
        chunk = index.by_id.get(chunk_id)
        text = chunk["text"] if chunk else ""
    return chunk_id, text


def _declared(sentence: dict) -> list[str]:
    """The ids the sentence cites, whether or not the corpus can supply them."""
    declared: list[str] = []
    for chunk_id in sentence.get("cited_ids") or []:
        if isinstance(chunk_id, str) and chunk_id and chunk_id not in declared:
            declared.append(chunk_id)
    return declared


def _cited(sentence: dict) -> list[str]:
    """The declared ids the corpus can supply, in first-seen order."""
    return [chunk_id for chunk_id in _declared(sentence) if chunk_id in index.by_id]


def _input(question: str, rows: list[dict], texts: list[tuple[str, str]]) -> str:
    blocks = [f"QUESTION\n{question}", "SENTENCES"]
    for row in rows:
        blocks.append(f"{row['position']}: {row['text']}")
    blocks.append("CHUNKS")
    for chunk_id, text in texts:
        blocks.append(f"[{chunk_id}] {text}")
    return "\n\n".join(blocks)


def _questions(rows: list[dict]) -> dict:
    questions = {}
    for row in rows:
        tags = ", ".join(f"[{chunk_id}]" for chunk_id in row["checked"])
        supported = (
            f"supported by the chunk text tagged {tags}?"
            if row["cited"]
            else f"supported by any of the chunk texts tagged {tags}?"
        )
        questions[f"sup_{row['position']}"] = {
            "type": "noul",
            "instructions": f"Is sentence {row['position']} {supported}",
            "criteria": {
                "true_description": (
                    "the cited chunk states the claim the sentence makes"
                ),
                "false_description": (
                    "the cited chunk does not state it, or states something else"
                ),
            },
        }
    return questions


def run(
    question: str,
    sentences: list[dict],
    stage=None,
    *,
    kept,
) -> list[dict]:
    """Score every sentence and return one verdict per sentence.

    ``kept`` is the gate's kept list; a sentence that cites nothing the corpus
    holds is checked against every kept chunk and records ``chunk_id: None``.
    """
    kept_ids: list[str] = []
    kept_text: dict[str, str] = {}
    for item in kept:
        chunk_id, text = _chunk(item)
        if chunk_id not in kept_text:
            kept_ids.append(chunk_id)
            kept_text[chunk_id] = text

    results: dict[int, dict] = {}
    rows: list[dict] = []
    checked_all: list[str] = []

    for position, sentence in enumerate(sentences):
        declared = _declared(sentence)
        cited = [chunk_id for chunk_id in declared if chunk_id in index.by_id]
        checked = cited or kept_ids
        if not checked:
            results[position] = {
                "text": sentence["text"],
                "chunk_id": None,
                "cited_ids": declared,
                "support_noul": None,
                "verdict": "unsupported",
            }
            continue
        rows.append(
            {
                "position": position,
                "text": sentence["text"],
                "cited": cited,
                "checked": checked,
                "declared": declared,
            }
        )
        for chunk_id in checked:
            if chunk_id not in checked_all:
                checked_all.append(chunk_id)

    if rows:
        texts = [
            (
                chunk_id,
                kept_text[chunk_id]
                if chunk_id in kept_text
                else index.by_id.get(chunk_id, {}).get("text", ""),
            )
            for chunk_id in checked_all
        ]
        answers, usage = decisions.ask(
            _input(question, rows, texts), _questions(rows), "verify"
        )
        if stage is not None:
            stage.tokens_in = usage["tokens_in"]
            stage.tokens_out = usage["tokens_out"]
        for row in rows:
            support = answers[f"sup_{row['position']}"]
            results[row["position"]] = {
                "text": row["text"],
                "chunk_id": row["cited"][0] if row["cited"] else None,
                "cited_ids": row["declared"],
                "support_noul": support,
                "verdict": _verdict(support),
            }

    ordered = [results[position] for position in range(len(sentences))]
    if stage is not None:
        stage.payload = {"sentences": ordered}
    return ordered
