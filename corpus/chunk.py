"""Heading-aware markdown splitter for the corpus build.

Pure functions only: no file reads, no global state, no logging. Given the same
input bytes this module returns the same sections in the same order, which is
what keeps ``chunk_id`` values stable between builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_SECTION_CHARS = 200
MAX_SECTION_CHARS = 2000

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_TITLE_LINE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_HEADING = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Section:
    """One heading section of one markdown file."""

    path: str
    heading: str
    level: int
    text: str
    ordinal: int

    @property
    def sort_key(self) -> tuple[str, str, int]:
        return (self.path, self.heading, self.ordinal)


def _strip_frontmatter(raw: str) -> tuple[str, str | None]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return raw, None
    title = _TITLE_LINE.search(match.group(1))
    value = title.group(1).strip().strip("\"'") if title else None
    return raw[match.end() :], value


def _blocks(body: str) -> list[tuple[int, str, str]]:
    """Return ``(level, heading, body)`` triples in source order.

    A leading block before the first heading gets level ``0`` and an empty
    heading; the caller substitutes the frontmatter title.
    """
    blocks: list[tuple[int, str, str]] = []
    level = 0
    heading = ""
    buffered: list[str] = []
    fence: str | None = None

    for line in body.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
        if fence is None:
            heading_match = _HEADING.match(line)
            if heading_match:
                blocks.append((level, heading, "\n".join(buffered).strip("\n")))
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                buffered = []
                continue
        buffered.append(line)

    blocks.append((level, heading, "\n".join(buffered).strip("\n")))
    return [block for block in blocks if block[0] != 0 or block[2].strip()]


def _group_paragraphs(body: str) -> list[str]:
    """Group paragraphs into bodies no longer than ``MAX_SECTION_CHARS``."""
    groups: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            groups.append("\n\n".join(current))
            current = []
            size = 0

    for paragraph in (part for part in body.split("\n\n") if part.strip()):
        if len(paragraph) > MAX_SECTION_CHARS:
            flush()
            groups.extend(_hard_split(paragraph))
            continue
        added = len(paragraph) + (2 if current else 0)
        if size + added > MAX_SECTION_CHARS:
            flush()
            added = len(paragraph)
        current.append(paragraph)
        size += added

    flush()
    return groups


def _hard_split(paragraph: str) -> list[str]:
    """Cut one over-long paragraph at whitespace so a chunk never exceeds the cap."""
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > MAX_SECTION_CHARS:
        cut = remaining.rfind(" ", 0, MAX_SECTION_CHARS)
        if cut <= 0:
            cut = MAX_SECTION_CHARS
        pieces.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _render(level: int, heading: str, body: str) -> str:
    header = f"{'#' * level} {heading}" if level else heading
    return f"{header}\n\n{body}\n" if body.strip() else f"{header}\n"


def _split_long(section: Section) -> list[Section]:
    """Split one section into ``(part N)`` sections at paragraph boundaries."""
    header = f"{'#' * section.level} {section.heading}" if section.level else section.heading
    body = section.text[len(header) :].strip("\n") if section.text.startswith(header) else section.text
    groups = _group_paragraphs(body)
    if len(groups) <= 1:
        return [section]

    parts: list[Section] = []
    for index, group in enumerate(groups, start=1):
        heading = section.heading if index == 1 else f"{section.heading} (part {index})"
        parts.append(Section(section.path, heading, section.level, _render(section.level, heading, group), section.ordinal))
    return parts


def _merge_short(sections: list[Section]) -> list[Section]:
    """Fold sub-``MIN_SECTION_CHARS`` sections into a neighbour; never drop text."""
    if not sections:
        return sections

    merged: list[Section] = []
    pending: str | None = None
    for index, section in enumerate(sections):
        text = section.text if pending is None else pending + "\n\n" + section.text
        pending = None
        if len(text) < MIN_SECTION_CHARS and index < len(sections) - 1:
            pending = text
            continue
        merged.append(Section(section.path, section.heading, section.level, text, 0))

    if pending is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Section(last.path, last.heading, last.level, last.text + "\n\n" + pending, 0)
        else:
            merged = sections
    return merged


def split_markdown(path: str, raw: str) -> tuple[Section, ...]:
    """Split one markdown file into heading sections; deterministic and total."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    body, title = _strip_frontmatter(text)

    sections: list[Section] = []
    for level, heading, section_body in _blocks(body):
        if level == 0:
            heading = title or "Preamble"
            rendered = f"{heading}\n\n{section_body}\n" if section_body.strip() else f"{heading}\n"
        else:
            rendered = _render(level, heading, section_body)
        sections.append(Section(path, heading, level, rendered, 0))

    merged = _merge_short(sections)
    split: list[Section] = []
    for section in merged:
        split.extend(_split_long(section))

    return tuple(
        Section(section.path, section.heading, section.level, section.text, ordinal)
        for ordinal, section in enumerate(split)
    )
