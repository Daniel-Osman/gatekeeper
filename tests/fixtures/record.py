"""Record one live answer into a fixture file. One live call per run.

    uv run python -m tests.fixtures.record vector "<question>" <out.json>
    uv run python -m tests.fixtures.record scope  "<question>" <out.json>
    uv run python -m tests.fixtures.record rerank "<question>" <ids.json> <out.json>

The key is never printed. Raw recordings keep the model's own values; do not
edit them to make a test pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app import config
from app.gates import decisions, rerank, scope
from app.trace import Stage


def _write(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size} bytes)")


def _vector(question: str, out: str) -> None:
    values, usage = decisions.embed_query(question)
    _write(
        out,
        {
            "provenance": "recorded",
            "kind": "embedding",
            "input": question,
            "model": config.EMBED_MODEL,
            "dim": config.EMBED_DIM,
            "usage": usage,
            "vector": values,
        },
    )


def _scope(question: str, out: str) -> None:
    stage = Stage(stage="scope", model=config.DECISION_MODEL)
    result = scope.run(question, stage)
    payload = result["payload"]
    _write(
        out,
        {
            "provenance": "recorded",
            "kind": "scope",
            "input": question,
            "model": config.DECISION_MODEL,
            "usage": {"tokens_in": stage.tokens_in, "tokens_out": stage.tokens_out},
            "answers": {
                "in_scope": payload["in_scope_noul"],
                "router": {
                    "option": payload["router_choice"],
                    "confidence": payload["router_confidence"],
                },
                "is_injection": payload["injection_noul"],
            },
            "outcome": result["outcome"],
        },
    )


def _rerank(question: str, ids_path: str, out: str) -> None:
    candidates = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    stage = Stage(stage="rerank", model=config.DECISION_MODEL)
    result = rerank.run(question, "app", candidates, rerank.STRICT, stage)
    _write(
        out,
        {
            "provenance": "recorded",
            "kind": "rerank",
            "input": question,
            "model": config.DECISION_MODEL,
            "router": "app",
            "candidates": candidates,
            "usage": {"tokens_in": stage.tokens_in, "tokens_out": stage.tokens_out},
            "items": result["payload"]["items"],
        },
    )


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] in ("vector", "scope"):
        {"vector": _vector, "scope": _scope}[argv[0]](argv[1], argv[2])
        return 0
    if len(argv) == 4 and argv[0] == "rerank":
        _rerank(argv[1], argv[2], argv[3])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
