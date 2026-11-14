"""Offline test doubles: replayed decisions, a replayed query vector, one corpus.

Every test except ``@pytest.mark.live`` runs with ``decisions.ask`` and
``decisions.embed_query`` replaced. Recordings live in
``tests/fixtures/decisions`` and are picked by kind (scope or rerank) and the
longest ``match`` substring of the input. Every fixture is hand-built:
``recorded: false`` and a ``note`` say so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, index
from app.gates import decisions

FIXTURES = Path(__file__).parent / "fixtures"
DECISIONS = FIXTURES / "decisions"
VECTOR = FIXTURES / "embeddings" / "route_param.json"


def _entries() -> list[dict]:
    if not DECISIONS.exists():
        return []
    entries: list[dict] = []
    for path in sorted(DECISIONS.glob("*.json")):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(loaded if isinstance(loaded, list) else [loaded])
    return entries


def _kind(questions: dict) -> str:
    """Rerank asks ``rel_<id>``/``cur_<id>``; scope asks three named questions."""
    return "rerank" if any(key.startswith(("rel_", "cur_")) for key in questions) else "scope"


class FakeAsk:
    """Replay a hand-built answer, chosen by kind and longest matching substring."""

    def __init__(self, entries: list[dict]):
        self.entries = entries

    def __call__(self, input_text: str, questions: dict, model: str = config.DECISION_MODEL):
        entry = self._select(_kind(questions), input_text)
        if "raise" in entry:
            raise decisions.StageError(entry["raise"]["stage"], entry["raise"]["message"])
        if entry.get("mode") == "zeros_for_candidates":
            answers = {
                key: 0.0 for key in questions if key.startswith(("rel_", "cur_"))
            }
        else:
            answers = dict(entry["answers"])
        missing = set(questions) - set(answers)
        assert not missing, f"fixture {entry['match']!r} does not answer {sorted(missing)}"
        return answers, dict(entry["usage"])

    def _select(self, kind: str, input_text: str) -> dict:
        matches = [
            entry
            for entry in self.entries
            if entry["kind"] == kind and entry["match"] in input_text
        ]
        if not matches:
            pytest.fail(f"no fixture answers this {kind} input: {input_text[:80]!r}")
        return max(matches, key=lambda entry: len(entry["match"]))


def _fake_vector(question: str) -> list[float]:
    recorded = json.loads(VECTOR.read_text(encoding="utf-8"))
    if question == recorded["input"]:
        return list(recorded["vector"])
    seed = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], "big")
    vector = np.random.default_rng(seed).standard_normal(config.EMBED_DIM)
    return (vector / np.linalg.norm(vector)).tolist()


def _embed_replay(text: str, model: str = config.EMBED_MODEL):
    return _fake_vector(text), {"tokens_in": 12, "tokens_out": 0}


@pytest.fixture
def fake_decisions(monkeypatch):
    """Patch ``decisions.ask`` with the fixture replay and return it."""
    fake = FakeAsk(_entries())
    monkeypatch.setattr(decisions, "ask", fake)
    return fake


@pytest.fixture
def fake_embed(monkeypatch):
    """Patch ``decisions.embed_query`` with the recorded/derived vector."""
    monkeypatch.setattr(decisions, "embed_query", _embed_replay)
    return _embed_replay


@pytest.fixture(autouse=True)
def offline(request):
    """Patch both model calls for every non-live test."""
    if request.node.get_closest_marker("live"):
        yield None
        return
    index.ensure_loaded()
    fake = request.getfixturevalue("fake_decisions")
    request.getfixturevalue("fake_embed")
    yield fake


def _chunk(chunk_id: str, source: str, status: str, heading: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "router": "app",
        "status": status,
        "source": source,
        "synthetic": source == "forum",
        "path": f"docs/01-app/{heading}.mdx",
        "heading": heading,
        "text": f"## {heading}\n\nHand-built chunk for the rerank tests.",
    }


def _synthetic_chunks() -> list[dict]:
    """s01-s02 official/current, s03 forum, s04 legacy, s05-s30 official/current, s31 official/current between the two floors."""
    fixed = [
        _chunk("s01", "official", "current", "reading-a-route-param"),
        _chunk("s02", "official", "current", "dynamic-segments"),
        _chunk("s03", "forum", "current", "forum-push-a-route"),
        _chunk("s04", "official", "legacy", "legacy-route-param"),
    ]
    filler = [
        _chunk(f"s{n:02d}", "official", "current", f"filler-{n:02d}")
        for n in range(5, 31)
    ]
    between_floors = _chunk("s31", "official", "current", "filler-between-the-floors")
    return fixed + filler + [between_floors]


@pytest.fixture
def tmp_index(monkeypatch):
    """Install a 30-chunk index with fixed source/status, then restore the corpus."""
    index.ensure_loaded()
    chunks = _synthetic_chunks()
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    monkeypatch.setattr(index, "chunks", chunks)
    monkeypatch.setattr(index, "by_id", by_id)
    return by_id


@pytest.fixture
def client():
    """The real app over TestClient, with the offline fake in place."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def corpus():
    index.ensure_loaded()
    return index
