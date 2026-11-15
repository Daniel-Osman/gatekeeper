"""The two endpoints: status codes and the envelope keys."""

from __future__ import annotations

ENVELOPE = {"answer", "sentences", "trace", "receipt"}


def test_health_reports_the_corpus_count(client, corpus):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "corpus_chunks": len(corpus.chunks)}


def test_unknown_policy_is_rejected(client):
    response = client.post(
        "/ask",
        json={"question": "how do I read a route param in the app router", "policy": "loose"},
    )

    assert response.status_code == 422


def test_refused_question_returns_200_with_the_envelope(client):
    response = client.post(
        "/ask",
        json={
            "question": "why is my middleware not running in the app router",
            "policy": "strict",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == ENVELOPE
    assert body["trace"]["outcome"] == "refused"
    assert body["trace"]["policy"] == "strict"
    assert body["sentences"] == []


def test_answered_question_returns_200_with_the_envelope(client, canonical_slice):
    response = client.post(
        "/ask",
        json={
            "question": "how do I read a route param in the app router",
            "policy": "strict",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == ENVELOPE
    assert body["trace"]["outcome"] == "answered"
    assert body["answer"]
    assert len(body["sentences"]) == 3
    assert [stage["stage"] for stage in body["trace"]["stages"]] == [
        "scope",
        "retrieve",
        "rerank",
        "synthesize",
        "verify",
    ]


def test_flagged_answer_returns_200_with_the_envelope(client, canonical_slice):
    response = client.post(
        "/ask",
        json={
            "question": (
                "how do I read a route param in the app router in a server component"
            ),
            "policy": "strict",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == ENVELOPE
    assert body["trace"]["outcome"] == "answered_flagged"
    assert body["answer"]
    assert [stage["stage"] for stage in body["trace"]["stages"]] == [
        "scope",
        "retrieve",
        "rerank",
        "synthesize",
        "verify",
        "synthesize",
        "verify",
    ]
    assert [sentence["verdict"] for sentence in body["sentences"]] == [
        "verified",
        "unsupported",
        "verified",
    ]


def test_stage_error_returns_502_naming_the_stage(client):
    response = client.post(
        "/ask",
        json={"question": "how do I force a stage error", "policy": "strict"},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["stage"] == "scope"
    assert "boom" in body["detail"]
    assert [stage["stage"] for stage in body["trace"]["stages"]] == ["scope"]
