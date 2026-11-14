"""In-process trace for one question. No HTTP.

    uv run python -m scripts.trace "<question>" [--policy strict|fast_fix]
                                     [--only scope|retrieve|rerank] [--json]

Prints the outcome, one line per stage with its latency and input tokens, the
kept chunk ids with their reason, the refused reasons grouped, and the
receipt. ``--json`` prints the raw envelope instead. A ``--only`` run that
stops before a gate outcome prints its stages and no envelope. Exits 1 on
``StageError``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app import index
from app.gates import decisions
from app.pipeline import run


def _noul(value) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _path(chunk_id: str) -> str:
    return index.by_id[chunk_id]["path"]


def _print_stage(stage: dict, policy_name: str) -> None:
    print(
        f"{stage['stage']} latency_ms={stage['latency_ms']} "
        f"tokens_in={stage['tokens_in']}"
    )
    payload = stage
    if stage["stage"] == "scope":
        print(
            f"  in_scope={_noul(payload['in_scope_noul'])} "
            f"router={payload['router_choice']}@{_noul(payload['router_confidence'])} "
            f"injection={_noul(payload['injection_noul'])} "
            f"reason={payload.get('reason') or '-'}"
        )
    elif stage["stage"] == "retrieve":
        print(f"  router={payload['router']} candidates={payload['candidate_count']}")
    elif stage["stage"] == "rerank":
        print(
            f"  policy={policy_name} kept={payload['kept_count']}/"
            f"{payload['candidate_count']}"
        )
        for item in payload.get("items", []):
            if item["reason"] == "kept":
                print(f"  keep {item['chunk_id']} path={_path(item['chunk_id'])} reason=kept relevance={_noul(item['relevance_noul'])}")
        groups: dict[str, list[str]] = {}
        for item in payload.get("items", []):
            if item["reason"] != "kept":
                groups.setdefault(item["reason"], []).append(item["chunk_id"])
                if item["reason"] != "over_cap":
                    print(f"  refuse {item['chunk_id']} path={_path(item['chunk_id'])} reason={item['reason']} relevance={_noul(item['relevance_noul'])}")
        for reason in sorted(groups):
            ids = " ".join(groups[reason])
            print(f"  refuse reason={reason} count={len(groups[reason])} ids={ids}")


def _print_receipt(receipt: dict) -> None:
    print(
        f"receipt list_cost_usd={receipt['list_cost_usd']} "
        f"tokens_in={receipt['tokens_in']} tokens_out={receipt['tokens_out']} "
        f"synthesis_skipped={str(receipt['synthesis_skipped']).lower()} "
        f"synthesis_budget_tokens_out={receipt['synthesis_budget_tokens_out']}"
    )


def _print_human(envelope: dict) -> None:
    trace = envelope["trace"]
    if trace["outcome"] is None:
        for stage in trace["stages"]:
            _print_stage(stage, trace["policy"])
        return
    print(f"outcome: {trace['outcome']}")
    print(f"policy: {trace['policy']}")
    print(f"answer: {envelope['answer'] if envelope['answer'] else '-'}")
    for stage in trace["stages"]:
        _print_stage(stage, trace["policy"])
    _print_receipt(envelope["receipt"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.trace")
    parser.add_argument("question")
    parser.add_argument("--policy", choices=("strict", "fast_fix"), default="strict")
    parser.add_argument("--only", choices=("scope", "retrieve", "rerank"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        envelope = asyncio.run(run(args.question, args.policy, args.only))
    except decisions.StageError as exc:
        print(f"error {exc.stage}: {exc.message}", file=sys.stderr)
        return 1

    if args.as_json and envelope["trace"]["outcome"] is not None:
        print(json.dumps(envelope, indent=2, ensure_ascii=False))
    else:
        _print_human(envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
