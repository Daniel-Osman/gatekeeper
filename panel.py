"""Gatekeeper panel — the Streamlit entry file.

Run it with ``uv run streamlit run panel.py``. The file is named ``panel.py`` on
purpose: ``app.py`` beside the ``app/`` package would shadow it.

The pipeline runs in this process, not over HTTP. One CSS block carries the
palette and the type from the approved design; there is no JavaScript.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path

import streamlit as st

from app import config, index, panel_view as view, pipeline
from app.gates import decisions

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "eval" / "results.json"

st.set_page_config(
    page_title="Gatekeeper",
    page_icon="▥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------- key
# The environment wins. A Space that stores the key as a secret gets it copied
# into the environment once, silently: the value is never shown or logged.


def _load_key() -> None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    try:
        secret = st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets file, or no such secret: no key either way
        return
    if secret:
        os.environ["GEMINI_API_KEY"] = str(secret)


def _redact(message: str) -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "***")
    return message


_load_key()

# ------------------------------------------------------------------- corpus


@st.cache_resource(show_spinner=False)
def _corpus() -> dict:
    """Load and verify the committed corpus once per process."""
    index.ensure_loaded()
    return {
        "chunks": len(index.chunks),
        "forum": sum(1 for chunk in index.chunks if chunk.get("source") == "forum"),
    }


def _results() -> tuple[dict | None, str | None]:
    """eval/results.json, or a reason string when it cannot be used."""
    if not RESULTS_PATH.exists():
        return None, "missing"
    try:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "unreadable"
    return payload, None


# ---------------------------------------------------------------------- css

CSS = """
<style>
  :root {
    --ground: #1e2022;
    --paper: #181a1b;
    --ink: #dedbd7;
    --muted: #a9a297;
    --line: #2b2e30;
    --rule: #232628;
    --track: #2c3033;
    --accent: #5ea1cb;
    --weak-bg: #241b0f;
    --weak-ink: #d2a15d;
    --weak-line: #4a3720;
    --neg-bg: #3a0906;
    --neg-ink: #e16c6a;
    --neg-line: #5a1c19;
    --radius: 8px;
    --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif;
    --mono: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--ground);
    color: var(--ink);
    font-family: var(--sans);
  }
  [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }
  .block-container {
    max-width: 1200px;
    padding: 1.4rem 48px 76px;
  }
  ::selection { background: #33414f; }

  div[data-testid="stMarkdownContainer"] p { margin: 0; }
  div[data-testid="stMarkdownContainer"] a { color: var(--accent); }

  code {
    font-family: var(--mono);
    font-size: 0.88em;
    letter-spacing: -0.01em;
    color: var(--ink);
    background: none;
    padding: 0;
  }

  /* widgets: dark chrome that matches the sheet */
  [data-testid="stTextInput"] input {
    background: var(--paper);
    color: var(--ink);
    border-radius: var(--radius);
    font-size: 15px;
    padding: 10px 12px;
  }
  [data-testid="stTextInput"] > div > div {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: var(--paper);
  }
  [data-testid="stTextInput"] input::placeholder { color: #7b7d7c; }

  .stButton button,
  [data-testid="stFormSubmitButton"] button {
    font-family: var(--sans);
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: -0.005em;
    color: var(--ink);
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 6px 13px;
    min-height: 0;
    box-shadow: none;
  }
  .stButton button:hover,
  [data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--accent);
    color: var(--ink);
  }
  [data-testid="stFormSubmitButton"] button {
    background: var(--ink);
    color: #16181a;
    border-color: var(--ink);
    border-radius: var(--radius);
    padding: 9px 18px;
    font-size: 13px;
  }

  [data-testid="stSegmentedControl"],
  [role="radiogroup"] {
    background: var(--track);
    border-radius: 6px;
    padding: 3px;
    width: fit-content;
    gap: 2px;
  }
  button[data-variant="segmented_control"] {
    font-size: 12px;
    font-weight: 600;
    color: var(--muted) !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 4px !important;
    padding: 4px 12px !important;
    min-height: 0;
  }
  button[data-variant="segmented_control"][data-selected="true"] {
    background: var(--ink) !important;
    color: #16181a !important;
  }
  [data-testid="stSpinner"] { color: var(--muted); }

  /* ---------------------------------------------------------- masthead */
  .mast {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 18px 0 20px;
  }
  .mark { width: 22px; height: 22px; flex: none; color: var(--ink); }
  .wordmark { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .mast__meta {
    margin: 0 0 0 auto;
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }

  /* ------------------------------------------------------------- intro */
  .intro { padding: 2px 0 26px; }
  .intro h1 {
    margin: 0 0 10px;
    font-family: var(--serif);
    font-weight: 400;
    font-size: 32px;
    line-height: 1.1;
    letter-spacing: -0.018em;
    max-width: 24ch;
  }
  .lede { margin: 0; font-size: 17px; line-height: 1.45; max-width: 62ch; }

  .spacer { height: 10px; }
  .switch__label {
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 6px 0 4px;
  }

  /* --------------------------------------------------------------- sheet */
  .sheet {
    background: var(--paper);
    border-radius: var(--radius);
    margin-top: 18px;
  }
  .run { padding: 0 28px 20px; }
  .run__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 20px 0 16px;
  }
  .switch { display: inline-flex; gap: 2px; padding: 3px; border-radius: 6px; background: var(--track); }
  .seg {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
  }
  .seg--on { background: var(--ink); color: #16181a; }
  .outcome {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .q {
    margin: 0 0 18px;
    font-size: 19px;
    line-height: 1.4;
    letter-spacing: -0.006em;
    max-width: 76ch;
    color: var(--ink);
  }

  .refusal, .s {
    font-family: var(--serif);
    font-size: 26px;
    font-weight: 400;
    line-height: 1.32;
    letter-spacing: -0.012em;
    margin: 0;
  }
  .refusal { margin-bottom: 14px; }
  .sent + .sent { margin-top: 16px; }
  .s__meta { display: flex; align-items: center; gap: 10px; margin: 7px 0 0; }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 7px 4px 6px;
    border-radius: 4px;
    border: 1px solid transparent;
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 600;
    line-height: 1;
    letter-spacing: 0;
    white-space: nowrap;
  }
  .badge__g { width: 7px; height: 7px; flex: none; border-radius: 50%; background: currentColor; }
  .badge__g--hollow { background: none; border: 1.5px solid currentColor; }
  .badge__g--square { border-radius: 1px; }
  .badge--verified { background: var(--ink); border-color: var(--ink); color: #16181a; }
  .badge--weak { background: var(--weak-bg); border-color: var(--weak-line); color: var(--weak-ink); }
  .badge--unsupported { background: var(--neg-bg); border-color: var(--neg-line); color: var(--neg-ink); }
  .cite { font-family: var(--mono); font-size: 12px; color: var(--muted); }

  .detail {
    margin: 0 0 6px;
    font-size: 14px;
    line-height: 1.5;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    max-width: 78ch;
  }
  .warn {
    margin: 16px 0 0;
    padding: 10px 12px;
    border: 1px solid var(--neg-line);
    background: var(--neg-bg);
    color: var(--neg-ink);
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.45;
  }
  .cost {
    margin: 18px 0 0;
    padding-top: 14px;
    border-top: 1px solid var(--rule);
    font-size: 12px;
    line-height: 1.6;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .banner {
    margin: 0 0 6px;
    font-size: 12.5px;
    color: var(--weak-ink);
  }
  .stale { margin: 10px 0 0; font-size: 12px; color: var(--muted); }

  /* --------------------------------------------------------------- empty */
  .empty { padding: 22px 28px 24px; }
  .empty__chain {
    margin: 0 0 8px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
  }
  .empty__chain b { color: var(--ink); font-weight: 400; }
  .empty__line { margin: 0; font-family: var(--serif); font-size: 22px; color: var(--ink); }

  /* ---------------------------------------------------------------- xray */
  .xray { padding: 4px 28px 10px; }
  .xrow {
    display: grid;
    grid-template-columns: 156px minmax(0, 1fr) 92px 58px;
    grid-template-areas: "name fields bar ms";
    align-items: center;
    column-gap: 22px;
  }
  .xhead { padding: 14px 0 10px; font-size: 12px; color: var(--muted); }
  .stage { padding: 15px 0; border-top: 1px solid var(--rule); }
  .stage__name { grid-area: name; font-size: 19px; font-weight: 600; letter-spacing: -0.012em; }
  .stage__fields { grid-area: fields; font-size: 13px; line-height: 1.7; font-variant-numeric: tabular-nums; }
  .stage__fields .sep { padding: 0 6px; color: #4d5257; }
  .stage__fields .kept {
    font-weight: 600;
    text-decoration: underline;
    text-decoration-color: var(--accent);
    text-decoration-thickness: 2px;
    text-underline-offset: 4px;
  }
  .stage__fields .gone { color: var(--muted); }
  .bar { grid-area: bar; height: 5px; background: var(--track); position: relative; }
  .bar i { position: absolute; inset: 0 auto 0 0; background: var(--accent); }
  .ms { grid-area: ms; text-align: right; font-size: 19px; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  .total {
    display: grid;
    grid-template-columns: 156px minmax(0, 1fr) 92px 58px;
    column-gap: 22px;
    align-items: baseline;
    padding: 15px 0 13px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    color: var(--muted);
  }
  .total strong { grid-column: 1 / 3; font-size: 14px; font-weight: 600; color: var(--ink); }
  .total span { grid-column: 3 / 5; text-align: right; font-size: 19px; color: var(--ink); font-variant-numeric: tabular-nums; }
  .xfoot { margin: 12px 0 6px; font-size: 12px; line-height: 1.6; color: var(--muted); max-width: 82ch; }

  .legend {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px 28px;
    margin: 16px 0 0;
    padding: 0;
    list-style: none;
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
  }
  .legend li { display: flex; align-items: flex-start; gap: 8px; }
  .legend .badge { flex: none; margin-top: -1px; }

  /* ----------------------------------------------------------- scorecard */
  .card { padding: 8px 28px 20px; }
  .caption { margin: 6px 0 10px; font-size: 12px; line-height: 1.55; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; }
  th {
    padding: 10px 0 12px;
    font-size: 12px;
    font-weight: 500;
    text-align: right;
    color: var(--muted);
    vertical-align: bottom;
    white-space: nowrap;
    border-bottom: 1px solid var(--rule);
  }
  th:first-child { text-align: left; }
  td {
    padding: 13px 0;
    border-top: 1px solid var(--rule);
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    font-size: 14px;
  }
  td.name { text-align: left; font-weight: 600; font-size: 14px; white-space: normal; }
  td.blank { color: #6c7176; }
  .sub { display: block; margin-top: 1px; font-size: 12px; font-weight: 400; line-height: 1.45; color: var(--muted); }
  .tfoot { margin: 14px 0 4px; font-size: 12px; line-height: 1.6; color: var(--muted); max-width: 84ch; }

  /* ---------------------------------------------------------------- misc */
  h2.gk {
    margin: 44px 0 6px;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.012em;
  }
  .note { margin: 0 0 4px; font-size: 13px; line-height: 1.5; color: var(--muted); max-width: 78ch; }
  footer.gkfoot {
    margin-top: 52px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    display: flex;
    flex-wrap: wrap;
    gap: 8px 32px;
    align-items: baseline;
  }
  .foot__rule { margin: 0; font-size: 14px; }
  .foot__meta { margin: 0 0 0 auto; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }

  @media (max-width: 900px) {
    .block-container { padding: 1rem 24px 64px; }
    .xrow, .total { grid-template-columns: 1fr auto; grid-template-areas: "name ms" "fields fields" "bar bar"; row-gap: 8px; }
    .xhead { display: none; }
    .total strong { grid-column: 1; }
    .total span { grid-column: 2; font-size: 17px; }
    .legend { grid-template-columns: 1fr; gap: 8px; }
  }
  @media (max-width: 620px) {
    .block-container { padding: 0.6rem 18px 56px; }
    .intro h1 { font-size: 26px; }
    .refusal, .s { font-size: 22px; }
    .stage__name { font-size: 17px; }
    .run, .xray, .card, .empty { padding-left: 18px; padding-right: 18px; }
  }

  /* Streamlit's own <p> rules are more specific than a bare class: raise
     these so the sheet's spacing survives. */
  .mast p.mast__meta { margin: 0 0 0 auto; }
  .intro p.lede { margin: 0; }
  .sheet p.q { margin: 0 0 18px; }
  .sheet p.cost { margin: 18px 0 0; }
  .sheet p.warn { margin: 16px 0 0; }
  .sheet p.detail { margin: 0 0 6px; }
  .sheet p.refusal { margin: 0 0 14px; }
  .sheet p.xfoot { margin: 12px 0 6px; }
  .sheet p.caption { margin: 6px 0 10px; }
  .sheet p.tfoot { margin: 14px 0 4px; }
  .sheet p.empty__chain { margin: 0 0 8px; }
  .sheet p.empty__line { margin: 0; }
  .sheet p.note { margin: 0 0 4px; }
  .gkfoot p.foot__rule { margin: 0; }
  .gkfoot p.foot__meta { margin: 0 0 0 auto; }
  h2.gk + div p.note { margin: 0 0 4px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ render


def _esc(text: object) -> str:
    return html.escape(str(text))


def _badge(verdict: str) -> str:
    kind = verdict if verdict in ("verified", "weak", "unsupported") else "unsupported"
    dot = {"verified": "", "weak": " badge__g--hollow", "unsupported": " badge__g--square"}[kind]
    return (
        f'<span class="badge badge--{kind}">'
        f'<span class="badge__g{dot}" aria-hidden="true"></span>{kind}</span>'
    )


def _sentences_html(sentences: list[dict]) -> str:
    blocks = []
    for sentence in sentences:
        cite = (
            f'<span class="cite">{_esc(sentence["cite"])}</span>'
            if sentence.get("cite")
            else ""
        )
        blocks.append(
            '<div class="sent">'
            f'<p class="s">{_esc(sentence["text"])}</p>'
            f'<p class="s__meta">{_badge(sentence["verdict"])}{cite}</p>'
            "</div>"
        )
    return '<div class="sentences">' + "".join(blocks) + "</div>"


def _fields_html(fields: list[dict]) -> str:
    chunks = []
    for field in fields:
        parts = []
        for part in field["parts"]:
            mark = part.get("mark")
            attr = f' class="{mark}"' if mark else ""
            parts.append(f"<span{attr}>{_esc(part['text'])}</span>")
        value = " ".join(parts)
        if field.get("key"):
            chunks.append(f'<code>{_esc(field["key"])}</code> {value}')
        else:
            chunks.append(value)
    return ' <span class="sep">·</span> '.join(chunks)


def _xray_html(model: dict) -> str:
    rows = []
    for row in model["stages"]:
        rows.append(
            '<li class="xrow stage">'
            f'<span class="stage__name">{_esc(row["name"])}</span>'
            f'<span class="stage__fields">{_fields_html(row["fields"])}</span>'
            f'<span class="bar" aria-hidden="true"><i style="width:{row["share"]}%"></i></span>'
            f'<span class="ms">{_esc(row["ms_text"])}</span>'
            "</li>"
        )
    total = model["stages_total"]
    return (
        '<div class="sheet xray">'
        '<div class="xrow xhead" aria-hidden="true">'
        '<span style="grid-area:name">stage</span>'
        '<span style="grid-area:fields">what it wrote</span>'
        '<span style="grid-area:ms">ms</span></div>'
        '<ol class="stages" style="list-style:none;margin:0;padding:0">'
        + "".join(rows)
        + "</ol>"
        f'<div class="total"><strong>{total["count"]} stages</strong>'
        f'<span>{_esc(total["right"])}</span></div>'
        '<p class="xfoot"><code>retry_of</code> appears only when a repair retried '
        "the writer; otherwise it reads <code>absent</code>. Each bar is a stage's "
        "share of the slowest stage. A stage that did not run has no row.</p>"
        "</div>"
    )


def _legend_html() -> str:
    return (
        '<ul class="legend">'
        f"<li>{_badge('verified')} the cited chunk supports the sentence</li>"
        f"<li>{_badge('weak')} the chunk is stale or beside the point</li>"
        f"<li>{_badge('unsupported')} the repair failed, so the answer is flagged</li>"
        "</ul>"
    )


def _run_html(model: dict) -> str:
    segments = "".join(
        f'<span class="seg{" seg--on" if name == model["policy"] else ""}">{_esc(name)}</span>'
        for name in view.POLICIES
    )
    if model["error"] is not None:
        body = "".join(f'<p class="detail">{_esc(line)}</p>' for line in model["detail"])
        return (
            '<div class="sheet run">'
            '<div class="run__head">'
            f'<div class="switch" role="group" aria-label="Policy: {_esc(model["policy"])}">{segments}</div>'
            f'<p class="outcome">{_esc(model["headline"])}</p>'
            "</div>"
            f"{body}"
            "</div>"
        )

    if model["sentences"]:
        body = _sentences_html(model["sentences"])
    else:
        body = f'<p class="refusal">{_esc(model["answer"])}</p>'
        body += "".join(
            f'<p class="detail">{_esc(line)}</p>' for line in model["detail"]
        )

    warning = (
        f'<p class="warn">{_esc(model["warning"])}</p>' if model["warning"] else ""
    )
    question = (
        f'<p class="q">{_esc(model["question"])}</p>' if model["question"] else ""
    )
    cost = f'<p class="cost">{_esc(model["cost"])}</p>' if model["cost"] else ""
    return (
        '<div class="sheet run">'
        '<div class="run__head">'
        f'<div class="switch" role="group" aria-label="Policy: {_esc(model["policy"])}">{segments}</div>'
        f'<p class="outcome"><b>{_esc(model["headline"])}</b></p>'
        "</div>"
        f"{question}{body}{warning}{cost}"
        "</div>"
    )


def _scorecard_html(card: dict) -> str:
    if card["missing"]:
        return (
            '<div class="sheet card">'
            f'<p class="caption">{_esc(card["run_line"])}</p>'
            f'<p class="tfoot">{_esc(card["note"])}</p>'
            "</div>"
        )
    heads = "".join(f"<th><code>{_esc(head)}</code></th>" for head in card["headers"])
    rows = []
    for row in card["rows"]:
        cells = "".join(
            f'<td class="num{" blank" if cell == "--" else ""}">{_esc(cell)}</td>'
            for cell in row["cells"]
        )
        rows.append(
            f'<tr><td class="name">{_esc(row["name"])}'
            f'<span class="sub">{_esc(row["sub"])}</span></td>{cells}</tr>'
        )
    return (
        '<div class="sheet card">'
        f'<p class="caption">{_esc(card["run_line"])}</p>'
        f"<table><thead><tr><th>pipeline</th>{heads}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f'<p class="tfoot">{_esc(card["note"])}</p>'
        "</div>"
    )


# ------------------------------------------------------------------- state

for name, default in (("_policy", "strict"), ("question", ""), ("run", None)):
    st.session_state.setdefault(name, default)

fired = st.session_state.pop("preset_fired", None)
policy_at_top = st.session_state["_policy"]


def _preset_clicked(label: str) -> None:
    """Fill the box before the input widget is created, and mark the click."""
    st.session_state["question"] = view.preset_payload(label)
    st.session_state["preset_fired"] = label


# -------------------------------------------------------------------- page

try:
    corpus = _corpus()
    corpus_line = view.corpus_line(corpus["chunks"], corpus["forum"])
    corpus_problem = None
except index.CorpusError as exc:  # missing or inconsistent committed corpus
    corpus = None
    corpus_line = "corpus unavailable"
    corpus_problem = str(exc)

has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

st.markdown(
    '<header class="mast">'
    '<svg class="mark" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    '<g fill="currentColor"><rect x="1" y="9" width="2" height="6"></rect>'
    '<rect x="6" y="6" width="2" height="12"></rect>'
    '<rect x="11" y="2" width="2" height="20"></rect>'
    '<rect x="16" y="6" width="2" height="12"></rect>'
    '<rect x="21" y="9" width="2" height="6"></rect></g></svg>'
    '<p class="wordmark">Gatekeeper</p>'
    f'<p class="mast__meta">{_esc(corpus_line)}</p>'
    "</header>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="intro">'
    "<h1>The gates decide before the model answers.</h1>"
    '<p class="lede">Gatekeeper answers App Router questions from versioned '
    "Next.js docs, and shows every gate the answer passed.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# presets, directly above the search bar
with st.container(horizontal=True, gap="small"):
    for label, _payload in view.PRESETS:
        st.button(
            label,
            key=f"preset::{label}",
            on_click=_preset_clicked,
            args=(label,),
            width="content",
            disabled=corpus is None,
        )

with st.form("ask", clear_on_submit=False, border=False):
    ask_left, ask_right = st.columns([6, 1], gap="small", vertical_alignment="center")
    with ask_left:
        st.text_input(
            "Question",
            key="question",
            label_visibility="collapsed",
            placeholder="Ask an App Router question…",
        )
    with ask_right:
        submitted = st.form_submit_button(
            "Ask", width="stretch", disabled=corpus is None
        )

st.markdown('<p class="switch__label">policy</p>', unsafe_allow_html=True)
policy_cols = st.columns([2, 5], gap="small")
with policy_cols[0]:
    chosen = st.segmented_control(
        "Policy",
        list(view.POLICIES),
        default=policy_at_top,
        label_visibility="collapsed",
        selection_mode="single",
    )
policy = chosen or policy_at_top
st.session_state["_policy"] = policy

if corpus_problem:
    st.markdown(
        f'<p class="banner">The corpus could not load: {_esc(corpus_problem)}. '
        "No question can run.</p>",
        unsafe_allow_html=True,
    )
elif not has_key:
    st.markdown(
        '<p class="banner">GEMINI_API_KEY is not set, so a run would fail at the '
        "first gate.</p>",
        unsafe_allow_html=True,
    )

# one call site: a preset click or a form submit, never a plain rerun
spec = view.next_run(
    preset=fired,
    submitted=submitted,
    question=st.session_state.get("question", ""),
    policy=policy,
)

if spec is not None:
    with st.spinner("Running scope → retrieve → rerank → synthesize → verify…"):
        try:
            envelope = asyncio.run(pipeline.run(spec["question"], spec["policy"]))
            model = view.shape(
                envelope, question=spec["question"], policy=spec["policy"]
            )
        except decisions.StageError as exc:
            model = view.shape_error(exc.stage, _redact(exc.message), exc.trace)
        except Exception as exc:  # noqa: BLE001 - the page must not die on a run
            model = view.shape_error(
                "pipeline", _redact(f"{type(exc).__name__}: {exc}"), None
            )
    st.session_state["run"] = {
        "question": spec["question"],
        "policy": spec["policy"],
        "model": model,
    }

run = st.session_state.get("run")

if run is None:
    st.markdown(
        '<div class="sheet empty">'
        '<p class="empty__chain">scope · retrieve · rerank · synthesize · verify</p>'
        '<p class="empty__line">Nothing has run yet. Pick a preset or ask a question.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
else:
    model = run["model"]
    st.markdown(_run_html(model), unsafe_allow_html=True)
    if model["outcome"] != "error" and run["policy"] != policy:
        st.markdown(
            f'<p class="stale">Shown run used <code>{_esc(run["policy"])}</code>; '
            f"the switch is on <code>{_esc(policy)}</code>. Ask again to re-run.</p>",
            unsafe_allow_html=True,
        )

    if model["stages"]:
        st.markdown('<h2 class="gk">Gate x-ray</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="note">Stage by stage, with the fields this gate wrote to '
            "the trace. A stage that did not run has no row.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(_legend_html(), unsafe_allow_html=True)
        st.markdown(_xray_html(model), unsafe_allow_html=True)

        st.markdown(
            f'<p class="note" style="margin-top:10px">{_esc(view.sentence_floor_line())}'
            "</p>",
            unsafe_allow_html=True,
        )

st.markdown('<h2 class="gk">Scorecard</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="note">Three pipelines, in the harness\'s own columns. The table is '
    "read from <code>eval/results.json</code>; nothing here is typed by hand.</p>",
    unsafe_allow_html=True,
)

payload, problem = _results()
card = view.scorecard(payload if problem is None else None)
if problem == "unreadable":
    card = {**card, "run_line": "eval/results.json could not be read: no scorecard is drawn."}
st.markdown(_scorecard_html(card), unsafe_allow_html=True)

corpus_tail = (
    f"{corpus['chunks']:,} chunks · {corpus['forum']} forum posts"
    if corpus
    else "corpus unavailable"
)
st.markdown(
    '<footer class="gkfoot">'
    '<p class="foot__rule">No accounts, no roles: a refusal is a gate decision, '
    "not a permission.</p>"
    f'<p class="foot__meta">Five stages · {_esc(corpus_tail)}</p>'
    "</footer>",
    unsafe_allow_html=True,
)
