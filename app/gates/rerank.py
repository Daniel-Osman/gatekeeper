"""Gate 2: relevance and currency per candidate, then the policy keep rule."""

from __future__ import annotations

from dataclasses import dataclass

from app import config, index
from app.gates import decisions


@dataclass(frozen=True)
class Policy:
    name: str
    min_relevance: float
    allow_forum: bool
    allow_legacy: bool
    max_keep: int


STRICT = Policy(
    name="strict",
    min_relevance=0.60,
    allow_forum=False,
    allow_legacy=False,
    max_keep=config.MAX_KEEP,
)
FAST_FIX = Policy(
    name="fast_fix",
    min_relevance=0.40,
    allow_forum=True,
    allow_legacy=True,
    max_keep=config.MAX_KEEP,
)
POLICIES = {STRICT.name: STRICT, FAST_FIX.name: FAST_FIX}


def _input(question: str, candidates: list[str]) -> str:
    blocks = [f"QUESTION\n{question}", "CANDIDATES"]
    for chunk_id in candidates:
        chunk = index.by_id[chunk_id]
        blocks.append(
            f"[{chunk_id}] source={chunk['source']} status={chunk['status']} "
            f"path={chunk['path']}\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _questions(router: str, candidates: list[str]) -> dict:
    questions = {}
    for chunk_id in candidates:
        questions[f"rel_{chunk_id}"] = {
            "type": "noul",
            "instructions": (
                f"Consider only the candidate tagged [{chunk_id}]. "
                "Does it answer the question, whether or not it targets the "
                "version the user is on?"
            ),
            "criteria": {
                "true_description": (
                    "the chunk gives the mechanism, API or steps that answer "
                    "the question, even if the API belongs to the other router"
                ),
                "false_description": (
                    "the chunk is on a nearby topic or only uses the asked "
                    "term in passing"
                ),
            },
        }
        questions[f"cur_{chunk_id}"] = {
            "type": "noul",
            "instructions": (
                f"Consider only the candidate tagged [{chunk_id}]. "
                "Is the API it shows still the recommended one for the "
                f"{router} router?"
            ),
            "criteria": {
                "true_description": (
                    "the API shown is the current recommended one for that router"
                ),
                "false_description": (
                    "the API is legacy or superseded for that router"
                ),
            },
        }
    return questions


def run(question: str, router: str, candidates, policy: Policy, stage=None) -> dict:
    """Score every candidate once and apply the policy keep rule.

    Currency is recorded and never gates. A missing or invalid answer raises
    ``StageError`` naming ``rerank``: a silent model stops the pipeline, it
    never reads as a low score.
    """
    ids = [c["chunk_id"] if isinstance(c, dict) else c for c in candidates]
    if not ids:
        payload = {"items": [], "kept_count": 0, "candidate_count": 0}
        if stage is not None:
            stage.payload = dict(payload)
        return {"payload": payload, "kept": []}

    answers, usage = decisions.ask(
        _input(question, ids), _questions(router, ids), "rerank"
    )

    if stage is not None:
        stage.tokens_in = usage["tokens_in"]
        stage.tokens_out = usage["tokens_out"]

    order = {chunk_id: position for position, chunk_id in enumerate(ids)}
    items = []
    for chunk_id in ids:
        chunk = index.by_id[chunk_id]
        relevance = answers[f"rel_{chunk_id}"]
        currency = answers[f"cur_{chunk_id}"]
        if relevance < policy.min_relevance:
            reason = "below_min_relevance"
        elif chunk["source"] == "forum" and not policy.allow_forum:
            reason = "forum_not_allowed"
        elif chunk["status"] == "legacy" and not policy.allow_legacy:
            reason = "legacy_not_allowed"
        else:
            reason = None
        items.append(
            {
                "chunk_id": chunk_id,
                "relevance_noul": relevance,
                "currency_noul": currency,
                "kept": reason is None,
                "reason": reason,
            }
        )

    eligible = [item for item in items if item["kept"]]
    eligible.sort(key=lambda item: (-item["relevance_noul"], order[item["chunk_id"]]))

    kept = []
    for rank, item in enumerate(eligible):
        if rank < policy.max_keep:
            kept.append(item["chunk_id"])
            item["reason"] = "kept"
        else:
            item["kept"] = False
            item["reason"] = "over_cap"

    payload = {
        "items": items,
        "kept_count": len(kept),
        "candidate_count": len(ids),
    }
    if stage is not None:
        stage.payload = payload
    return {"payload": payload, "kept": kept}
