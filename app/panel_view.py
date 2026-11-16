"""Shape a pipeline envelope into what the panel renders.

Pure functions only: no Streamlit import, no model call, no file read. The
panel feeds them an envelope from ``app.pipeline.run`` (or a ``StageError``
trace) and a results payload from ``eval/results.json``, and renders the dicts.

Every figure here comes from the envelope. A field the pipeline reports as
``None`` prints ``--``; a stage absent from the trace has no row; no zero is
invented for a missing measurement.
"""

from __future__ import annotations

from app import config
from app.gates import rerank

# The brief names both the labels and the payloads exactly.
PRESETS: tuple[tuple[str, str], ...] = (
    ("Official: App Router Params", "how do I read a route param in the app router"),
    (
        "The Policy Flip: getServerSideProps",
        "how do I use getServerSideProps inside an App Router server component",
    ),
    ("Off-topic: Cake recipe", "how to bake a cake"),
)
PRESET_PAYLOADS = dict(PRESETS)

POLICIES: tuple[str, ...] = (rerank.STRICT.name, rerank.FAST_FIX.name)

STAGES: tuple[str, ...] = ("scope", "retrieve", "rerank", "synthesize", "verify")

SCORECARD_POLICIES: tuple[str, ...] = ("no_gates", "strict", "fast_fix")

SCORECARD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("hit_at_5", "hit_at_5"),
    ("answered", "answered"),
    ("wall_ms_mean", "wall_ms_mean"),
    ("cost_mean_usd", "cost_mean_usd × 1,000"),
)


def preset_payload(label: str) -> str:
    """The question a preset pill sends."""
    return PRESET_PAYLOADS[label]


def next_run(
    *,
    preset: str | None,
    submitted: bool,
    question: str,
    policy: str,
) -> dict | None:
    """Decide whether this rerun may call the model.

    A preset click and a form submit are the only events that spend a call. A
    rerun from the policy switch, a text edit or anything else returns ``None``,
    and the caller re-renders the cached run.
    """
    if preset is not None:
        return {"question": preset_payload(preset), "policy": policy}
    if submitted:
        text = (question or "").strip()
        if text:
            return {"question": text, "policy": policy}
    return None


# ----------------------------------------------------------------- numbers


def _usd(value) -> str:
    if value is None:
        return "--"
    return f"${float(value):.5f}"


def _noul(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _count(value) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def _ms(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):,.0f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value / 1000.0:.2f} s"


def _stage_ms(stage: dict) -> float | None:
    value = stage.get("latency_ms")
    return None if value is None else float(value)


def wall_ms(stages: list[dict]) -> float | None:
    """The turn's measured time: the sum of the stage latencies it reports."""
    values = [_stage_ms(stage) for stage in stages]
    if not values or None in values:
        return None
    return round(sum(values), 3)


def cost_line(envelope: dict) -> str:
    """List cost, wall time and tokens in/out, one quiet line."""
    receipt = envelope.get("receipt") or {}
    stages = (envelope.get("trace") or {}).get("stages") or []
    parts = [
        f"{_usd(receipt.get('list_cost_usd'))} at list price",
        _seconds(wall_ms(stages)),
        f"{_count(receipt.get('tokens_in'))} tokens in, "
        f"{_count(receipt.get('tokens_out'))} out",
    ]
    if receipt.get("synthesis_skipped"):
        parts.append(
            f"{_count(receipt.get('synthesis_budget_tokens_out'))}-token "
            "output budget not spent"
        )
    return " · ".join(parts)


# ------------------------------------------------------------------ stages


def _field(key: str | None, text: str, mark: str | None = None) -> dict:
    return {"key": key, "parts": [{"text": text, "mark": mark}]}


def _values(key: str | None, pairs: list[tuple[str, str | None]]) -> dict:
    return {"key": key, "parts": [{"text": text, "mark": mark} for text, mark in pairs]}


def _reason_counts(items: list[dict]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        reason = item.get("reason") or "kept"
        if reason == "kept":
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return " · ".join(f"{name} {counts[name]}" for name in sorted(counts))


def stage_fields(stage: dict) -> list[dict]:
    """The real trace fields for one stage, in reading order."""
    name = stage.get("stage")
    if name == "scope":
        return [
            _field("in_scope_noul", _noul(stage.get("in_scope_noul"))),
            _field("router_choice", str(stage.get("router_choice") or "--")),
            _field("router_confidence", _noul(stage.get("router_confidence"))),
            _field("injection_noul", _noul(stage.get("injection_noul"))),
            _field("reason", str(stage.get("reason") or "proceed")),
        ]
    if name == "retrieve":
        candidates = [str(chunk_id) for chunk_id in stage.get("candidates") or []]
        first = " ".join(candidates[:5])
        if len(candidates) > 5:
            first = f"{first} …"
        fields = [
            _field("router", str(stage.get("router") or "--")),
            _field("candidate_count", _count(stage.get("candidate_count"))),
        ]
        if first:
            fields.append(_field("candidates", first))
        return fields
    if name == "rerank":
        items = list(stage.get("items") or [])
        fields = [
            _field(
                "kept_count",
                f"{_count(stage.get('kept_count'))} of "
                f"{_count(stage.get('candidate_count'))}",
            )
        ]
        relevance = [
            (
                f"{float(item['relevance_noul']):.2f}",
                "kept" if item.get("kept") else None,
            )
            for item in items
            if item.get("relevance_noul") is not None
        ]
        if relevance:
            fields.append(_values("relevance", relevance))
        reasons = _reason_counts(items)
        if reasons:
            fields.append(_field("reason", reasons))
        return fields
    if name == "synthesize":
        return [
            _field("sentence_count", _count(stage.get("sentence_count"))),
            _field("repair", str(stage.get("repair") is True).lower()),
            _field("truncated", str(stage.get("truncated") is True).lower()),
        ]
    if name == "verify":
        counts: dict[str, int] = {}
        for sentence in stage.get("sentences") or []:
            verdict = sentence.get("verdict") or "unknown"
            counts[verdict] = counts.get(verdict, 0) + 1
        summary = " · ".join(f"{counts[name]} {name}" for name in sorted(counts))
        retry_of = stage.get("retry_of")
        return [
            _field("verdicts", summary or "--"),
            _field(
                "retry_of",
                "absent" if retry_of is None else str(retry_of),
                None if retry_of is not None else "gone",
            ),
        ]
    return []


def xray(stages: list[dict]) -> list[dict]:
    """One row per stage in the trace, in order, with its share of the slowest."""
    times = [
        value for value in (_stage_ms(stage) for stage in stages) if value is not None
    ]
    slowest = max(times) if times else None
    rows = []
    for stage in stages:
        ms = _stage_ms(stage)
        share = 0.0 if not slowest or ms is None else round(ms / slowest * 100.0, 1)
        rows.append(
            {
                "name": str(stage.get("stage") or "--"),
                "fields": stage_fields(stage),
                "ms": ms,
                "ms_text": _ms(ms),
                "share": share,
            }
        )
    return rows


# ---------------------------------------------------------------- outcomes


def _sentences(envelope: dict) -> list[dict]:
    shaped = []
    for sentence in envelope.get("sentences") or []:
        cited = sentence.get("chunk_id")
        if not cited:
            ids = sentence.get("cited_ids") or []
            cited = ids[0] if ids else ""
        shaped.append(
            {
                "text": str(sentence.get("text") or ""),
                "verdict": str(sentence.get("verdict") or "unsupported"),
                "cite": str(cited),
            }
        )
    return shaped


def _verdict_counts(sentences: list[dict]) -> str:
    counts: dict[str, int] = {}
    for sentence in sentences:
        verdict = sentence["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    return " · ".join(
        f"{counts[name]} {name}"
        for name in ("verified", "weak", "unsupported")
        if name in counts
    )


def _rerank_payload(stages: list[dict]) -> dict | None:
    for stage in stages:
        if stage.get("stage") == "rerank":
            return stage
    return None


def refused_detail(payload: dict, policy: str) -> list[str]:
    """How many candidates the policy dropped, and against what floor."""
    items = list(payload.get("items") or [])
    total = payload.get("candidate_count")
    if total is None:
        total = len(items)
    kept = payload.get("kept_count")
    if kept is None:
        kept = sum(1 for item in items if item.get("kept"))
    failed = max(0, int(total) - int(kept))
    policy_rule = rerank.POLICIES.get(policy)
    floor = policy_rule.min_relevance if policy_rule else None
    values = [
        float(item["relevance_noul"])
        for item in items
        if item.get("relevance_noul") is not None
    ]
    best = max(values) if values else None

    lines = [
        f"{failed} of {int(total)} candidates failed {policy}. "
        f"Best match {_noul(best)} against a floor of {_noul(floor)}."
    ]
    if values:
        lines.append("relevance " + " · ".join(f"{value:.2f}" for value in values))
    reasons = _reason_counts(items)
    if reasons:
        lines.append(f"reason {reasons}")
    return lines


def headline(outcome: str, sentences: list[dict], policy: str) -> str:
    if outcome == "answered":
        verified = sum(
            1 for sentence in sentences if sentence["verdict"] == "verified"
        )
        counts = _verdict_counts(sentences)
        base = f"answered · {verified} of {len(sentences)} sentences verified"
        return f"{base} · {counts}" if counts else base
    if outcome == "answered_flagged":
        unsupported = sum(
            1 for sentence in sentences if sentence["verdict"] == "unsupported"
        )
        return (
            f"answered_flagged · {unsupported} of {len(sentences)} sentences "
            "unsupported after the repair"
        )
    if outcome == "refused":
        return f"refused · {policy} kept no chunks"
    if outcome == "blocked":
        return "blocked · scope stopped the turn"
    if outcome == "clarify":
        return "clarify · the router is unclear"
    return str(outcome)


def warning(outcome: str, envelope: dict) -> str | None:
    if outcome != "answered_flagged":
        return None
    repaired = any(
        stage.get("stage") == "verify" and stage.get("retry_of") is not None
        for stage in (envelope.get("trace") or {}).get("stages") or []
    )
    if repaired:
        return (
            "A sentence stayed unsupported after the repair pass, so this answer "
            "is flagged rather than answered."
        )
    return "A sentence is unsupported, so this answer is flagged rather than answered."


def _scope_stage(stages: list[dict]) -> dict:
    for stage in stages:
        if stage.get("stage") == "scope":
            return stage
    return {}


def shape(envelope: dict, *, question: str, policy: str) -> dict:
    """Turn one envelope into the model the panel renders."""
    trace = envelope.get("trace") or {}
    outcome = str(trace.get("outcome") or "answered")
    stages = list(trace.get("stages") or [])
    sentences = _sentences(envelope)
    receipt = envelope.get("receipt") or {}

    detail: list[str] = []
    if outcome == "refused":
        payload = _rerank_payload(stages)
        if payload is None:
            detail = ["No candidate survived the policy."]
        else:
            detail = refused_detail(payload, policy)
    elif outcome == "blocked":
        scope = _scope_stage(stages)
        detail = [
            "Scope stopped the turn before retrieval · reason "
            f"{scope.get('reason') or '--'}."
        ]
    elif outcome == "clarify":
        scope = _scope_stage(stages)
        detail = [
            "Scope could not name a router · router_choice "
            f"{scope.get('router_choice') or '--'} · router_confidence "
            f"{_noul(scope.get('router_confidence'))}."
        ]

    return {
        "outcome": outcome,
        "policy": str(trace.get("policy") or policy),
        "question": question,
        "headline": headline(outcome, sentences, policy),
        "warning": warning(outcome, envelope),
        "answer": str(envelope.get("answer") or ""),
        "sentences": sentences,
        "detail": detail,
        "cost": cost_line(envelope),
        "stages": xray(stages),
        "stages_total": {
            "count": len(stages),
            "right": f"{_seconds(wall_ms(stages))} · "
            f"{_usd(receipt.get('list_cost_usd'))}",
        },
        "error": None,
    }


def shape_error(stage: str, message: str, trace: dict | None) -> dict:
    """A failed turn: the stage named, its partial trace, no cost claim."""
    stages = list((trace or {}).get("stages") or [])
    return {
        "outcome": "error",
        "policy": str((trace or {}).get("policy") or "--"),
        "question": "",
        "headline": f"error · the {stage} stage failed",
        "warning": None,
        "answer": "",
        "sentences": [],
        "detail": [
            f"{stage}: {message}",
            "The question stays in the box. No answer was written and no cost "
            "is claimed.",
        ],
        "cost": None,
        "stages": xray(stages),
        "stages_total": {"count": len(stages), "right": "--"},
        "error": {"stage": stage, "message": message},
    }


# --------------------------------------------------------------- scorecard


def _rate_text(value) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _mean_text(value) -> str:
    return "--" if value is None else f"{float(value):,.0f}"


def _per_k_text(value) -> str:
    return "--" if value is None else f"{float(value) * 1000.0:.2f}"


def _cell(key: str, value) -> str:
    if value is None:
        return "--"
    if key == "hit_at_5":
        return _rate_text(value)
    if key == "answered":
        return _count(value)
    if key == "wall_ms_mean":
        return _mean_text(value)
    if key == "cost_mean_usd":
        return _per_k_text(value)
    return str(value)


def _policy_sub(name: str) -> str:
    if name == "no_gates":
        return "Retrieve and write, with no gate to stop it"
    policy = rerank.POLICIES.get(name)
    if policy is None:
        return ""
    if policy.allow_forum:
        return (
            f"Five stages, floor {policy.min_relevance:.2f}, forum and legacy "
            "chunks allowed"
        )
    return (
        f"Five stages, floor {policy.min_relevance:.2f}, forum and legacy "
        "chunks dropped"
    )


def scorecard(payload: dict | None) -> dict:
    """Three pipelines in the harness's own aggregate columns.

    ``payload`` is ``eval/results.json`` as loaded, or ``None`` when the file is
    missing. A policy with no aggregate block renders ``--`` in every cell: a
    blank is not a zero.
    """
    if payload is None:
        return {
            "missing": True,
            "run_line": "eval/results.json is missing: no scorecard is drawn.",
            "note": (
                "A blank cell means no run, not a zero. Run "
                "`uv run python -m eval.harness` to fill this table."
            ),
            "headers": [label for _, label in SCORECARD_COLUMNS],
            "rows": [],
        }

    aggregate = payload.get("aggregate") or {}
    rows = []
    for name in SCORECARD_POLICIES:
        numbers = aggregate.get(name) or {}
        rows.append(
            {
                "name": name,
                "sub": _policy_sub(name),
                "cells": [_cell(key, numbers.get(key)) for key, _ in SCORECARD_COLUMNS],
            }
        )

    bits = [f"Run {payload.get('run_at') or '--'}"]
    if payload.get("count") is not None:
        bits.append(f"{_count(payload['count'])} records")
    if payload.get("live_calls") is not None:
        bits.append(f"{_count(payload['live_calls'])} model calls")
    if payload.get("updated_at"):
        bits.append(f"last record {payload['updated_at']}")

    return {
        "missing": False,
        "run_line": " · ".join(bits),
        "note": (
            "A blank cell means no run, not a zero. Every figure is the harness "
            "aggregate from eval/results.json; cost per 1,000 queries is "
            "cost_mean_usd × 1,000."
        ),
        "headers": [label for _, label in SCORECARD_COLUMNS],
        "rows": rows,
    }


# ------------------------------------------------------------------ labels


def corpus_line(chunks: int, forum: int) -> str:
    return f"Next.js App Router docs · {chunks:,} chunks · {forum} of them forum posts"


def sentence_floor_line() -> str:
    """The verify floors, printed once so a badge has a definition."""
    return (
        f"verified at or above {config.VERIFY_VERIFIED_MIN:.2f} support · "
        f"weak at or above {config.VERIFY_WEAK_MIN:.2f} · unsupported below that"
    )
