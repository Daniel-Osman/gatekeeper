"""Label rules for the corpus: router and status, first matching rule wins.

Pure: a rule table plus a section in, labels out. ``load_rules`` validates the
table so a rule without a reason (or a legacy rule without a named replacement)
cannot reach the build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

ROUTERS = ("app", "pages", "both")
STATUSES = ("current", "legacy")


class CorpusError(RuntimeError):
    """A build invariant failed; the message names the offending path."""


@dataclass(frozen=True)
class Labels:
    router: str
    status: str


@dataclass(frozen=True)
class Rule:
    match: str
    router: str | None
    status: str | None
    reason: str
    heading_keywords: tuple[str, ...] = ()
    superseded_by: str | None = None

    def matches(self, path: str, heading: str) -> bool:
        if not fnmatchcase(path, self.match):
            return False
        if not self.heading_keywords:
            return True
        lowered = heading.lower()
        return any(keyword.lower() in lowered for keyword in self.heading_keywords)


def router_from_path(path: str) -> str:
    """Router implied by a repo-relative docs path (or a forum path)."""
    if path.startswith("docs/01-app/"):
        return "app"
    if path.startswith("docs/02-pages/"):
        return "pages"
    if path.startswith("forum/app/"):
        return "app"
    if path.startswith("forum/pages/"):
        return "pages"
    if path.startswith("forum/"):
        return "both"
    raise CorpusError(f"no router for path {path!r}: expected docs/01-app, docs/02-pages or forum/")


def apply(path: str, heading: str, rules: tuple[Rule, ...]) -> tuple[Labels, Rule | None]:
    """First matching rule wins; otherwise the path default with ``current``."""
    for rule in rules:
        if rule.matches(path, heading):
            return (
                Labels(
                    router=rule.router or router_from_path(path),
                    status=rule.status or "current",
                ),
                rule,
            )
    return Labels(router=router_from_path(path), status="current"), None


def load_rules(path: Path) -> tuple[Rule, ...]:
    """Read and validate ``label_rules.json``; every rule carries a reason."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules: list[Rule] = []
    for index, entry in enumerate(raw.get("rules", [])):
        location = f"{path.name} rule #{index + 1}"
        match = entry.get("match")
        if not match:
            raise CorpusError(f"{location}: missing 'match' glob")
        reason = (entry.get("reason") or "").strip()
        if not reason:
            raise CorpusError(f"{location} ({match}): missing 'reason'")
        router = entry.get("router")
        status = entry.get("status")
        if router is not None and router not in ROUTERS:
            raise CorpusError(f"{location} ({match}): router {router!r} outside {ROUTERS}")
        if status is not None and status not in STATUSES:
            raise CorpusError(f"{location} ({match}): status {status!r} outside {STATUSES}")
        if router is None and status is None:
            raise CorpusError(f"{location} ({match}): rule sets neither router nor status")
        keywords = tuple(entry.get("heading_keywords") or ())
        superseded_by = entry.get("superseded_by")
        if status == "legacy" and not superseded_by:
            raise CorpusError(f"{location} ({match}): legacy rule needs 'superseded_by'")
        rules.append(
            Rule(
                match=match,
                router=router,
                status=status,
                reason=reason,
                heading_keywords=keywords,
                superseded_by=superseded_by,
            )
        )
    if not rules:
        raise CorpusError(f"{path.name}: no rules")
    return tuple(rules)
