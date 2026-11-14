"""Hybrid retrieval: BM25 plus dense, fused with reciprocal rank fusion."""

from __future__ import annotations

from app import config, index
from app.gates import decisions


def _positions(ids: list[str]) -> dict[str, int]:
    return {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)}


def _rank_id(chunk_id: str) -> int:
    """Numeric order of a chunk id, so ``c9`` sorts before ``c10``."""
    return int(chunk_id[1:])


def run(question: str, router: str, stage=None) -> list[dict]:
    """Return the top ``CANDIDATES`` chunks for one routed question.

    Both rankers score the whole routed slice, so a rank is always present.
    ``RRF_K`` is the fusion constant only. Ties break on numeric chunk id, so
    the same question and the same query vector always give the same list.
    """
    try:
        vector, usage = decisions.embed_query(question)
    except decisions.StageError as exc:
        raise decisions.StageError("retrieve", exc.message) from exc

    if stage is not None:
        stage.tokens_in = usage["tokens_in"]
        stage.tokens_out = usage["tokens_out"]

    bm25 = _positions(index.bm25_rank(question, router))
    dense = _positions(index.dense_rank(vector, router))

    fused: dict[str, float] = {}
    for chunk_id in set(bm25) | set(dense):
        score = 0.0
        if chunk_id in bm25:
            score += 1.0 / (config.RRF_K + bm25[chunk_id])
        if chunk_id in dense:
            score += 1.0 / (config.RRF_K + dense[chunk_id])
        fused[chunk_id] = score

    ordered = sorted(fused.items(), key=lambda item: (-item[1], _rank_id(item[0])))

    items = [
        {
            "chunk_id": chunk_id,
            "bm25_rank": bm25.get(chunk_id),
            "dense_rank": dense.get(chunk_id),
            "fused_score": score,
        }
        for chunk_id, score in ordered[: config.CANDIDATES]
    ]

    if stage is not None:
        stage.payload = {
            "router": router,
            "candidate_count": len(items),
            "candidates": [item["chunk_id"] for item in items],
        }
    return items
