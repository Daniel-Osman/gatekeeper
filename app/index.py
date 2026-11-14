"""Corpus load, BM25 slices and dense ranks. Loaded once at startup."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import numpy as np

from app import config

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = CORPUS_DIR / "embeddings.npy"

ROUTERS = ("app", "pages")

# Keep paths whole: ``app/page.tsx`` and ``next.config.js`` are one token each.
TOKEN_PATTERN = r"[A-Za-z0-9_./-]+"

chunks: list[dict] = []
by_id: dict[str, dict] = {}
_embeddings: np.ndarray | None = None
_bm25: dict[str, bm25s.BM25] = {}
_slice_positions: dict[str, list[int]] = {}


class CorpusError(RuntimeError):
    """The committed corpus is missing or internally inconsistent."""


def _belongs(chunk: dict, router: str) -> bool:
    return chunk["router"] == router or chunk["router"] == "both"


def ensure_loaded() -> None:
    """Load the corpus, verify it and build one BM25 index per router slice."""
    global chunks, by_id, _embeddings, _bm25, _slice_positions

    if _embeddings is not None:
        return

    if not CHUNKS_PATH.exists():
        raise CorpusError(f"missing corpus file: {CHUNKS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise CorpusError(f"missing corpus file: {EMBEDDINGS_PATH}")

    records = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.load(EMBEDDINGS_PATH)

    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        rows = matrix.shape[0] if matrix.ndim else 0
        raise CorpusError(
            f"{EMBEDDINGS_PATH.name} has {rows} rows but {CHUNKS_PATH.name} "
            f"has {len(records)} records"
        )
    if matrix.shape[1] != config.EMBED_DIM:
        raise CorpusError(
            f"{EMBEDDINGS_PATH.name} has {matrix.shape[1]} columns, "
            f"expected {config.EMBED_DIM}"
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = (matrix / norms).astype(np.float32)

    slices: dict[str, list[int]] = {}
    indexes: dict[str, bm25s.BM25] = {}
    for router in ROUTERS:
        positions = [i for i, record in enumerate(records) if _belongs(record, router)]
        tokens = bm25s.tokenize(
            [records[i]["text"] for i in positions],
            stopwords="en",
            token_pattern=TOKEN_PATTERN,
            show_progress=False,
        )
        index = bm25s.BM25()
        index.index(tokens, show_progress=False)
        slices[router] = positions
        indexes[router] = index

    chunks = records
    by_id = {record["chunk_id"]: record for record in records}
    _slice_positions = slices
    _bm25 = indexes
    _embeddings = normalised


def slice_ids(router: str) -> list[str]:
    """Every chunk id in a router slice, in corpus order."""
    ensure_loaded()
    if router not in _slice_positions:
        raise ValueError(f"unknown router: {router!r}")
    return [chunks[i]["chunk_id"] for i in _slice_positions[router]]


def _check_router(router: str) -> None:
    if router not in ROUTERS:
        raise ValueError(f"unknown router: {router!r}")


def _rank_id(chunk_id: str) -> int:
    """Numeric order of a chunk id, so ``c9`` sorts before ``c10``."""
    return int(chunk_id[1:])


def bm25_rank(query: str, router: str) -> list[str]:
    """Every chunk id in the router slice, best BM25 score first."""
    ensure_loaded()
    _check_router(router)
    positions = _slice_positions[router]
    if not positions:
        return []
    tokens = bm25s.tokenize(
        [query], stopwords="en", token_pattern=TOKEN_PATTERN, show_progress=False
    )
    results, scores = _bm25[router].retrieve(
        tokens, k=len(positions), show_progress=False
    )
    order = np.asarray(results).reshape(-1)[: len(positions)]
    value = np.asarray(scores).reshape(-1)[: len(positions)]
    pairs = [
        (chunks[positions[int(position)]]["chunk_id"], float(value[n]))
        for n, position in enumerate(order)
    ]
    return [
        chunk_id
        for chunk_id, _ in sorted(
            pairs, key=lambda pair: (-pair[1], _rank_id(pair[0]))
        )
    ]


def dense_rank(query_vector, router: str) -> list[str]:
    """Every chunk id in the router slice, best cosine similarity first."""
    ensure_loaded()
    _check_router(router)
    positions = _slice_positions[router]
    if not positions:
        return []

    vector = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    scores = _embeddings[positions] @ vector
    order = sorted(
        range(len(positions)),
        key=lambda i: (-float(scores[i]), _rank_id(chunks[positions[i]]["chunk_id"])),
    )
    return [chunks[positions[i]]["chunk_id"] for i in order]

