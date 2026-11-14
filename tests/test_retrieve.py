"""Retrieval: determinism, router slice isolation, reciprocal rank fusion."""

from __future__ import annotations

from app import config, index, retrieve


def test_same_query_twice_gives_identical_candidates():
    question = "how do I read a route param in the app router"
    first = retrieve.run(question, "app")
    second = retrieve.run(question, "app")
    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert [item["bm25_rank"] for item in first] == [item["bm25_rank"] for item in second]
    assert [item["dense_rank"] for item in first] == [item["dense_rank"] for item in second]


def test_app_route_never_returns_a_pages_only_chunk():
    pages_only = {
        chunk["chunk_id"] for chunk in index.chunks if chunk["router"] == "pages"
    }
    candidates = retrieve.run("how do I fetch data in getStaticProps", "app")
    assert candidates, "app slice returned no candidates"
    assert pages_only.isdisjoint(item["chunk_id"] for item in candidates)
    assert set(index.slice_ids("app")) == {
        chunk["chunk_id"] for chunk in index.chunks if chunk["router"] in ("app", "both")
    }


def test_rrf_scores_match_the_rankings():
    candidates = retrieve.run("how do I read a route param in the app router", "app")
    assert len(candidates) == config.CANDIDATES
    scores = [item["fused_score"] for item in candidates]
    assert scores == sorted(scores, reverse=True)
    for item in candidates:
        expected = 0.0
        for rank in (item["bm25_rank"], item["dense_rank"]):
            if rank is not None:
                expected += 1.0 / (config.RRF_K + rank)
        assert abs(item["fused_score"] - round(expected, 12)) <= 1e-12
    assert any(item["bm25_rank"] and item["dense_rank"] for item in candidates)
