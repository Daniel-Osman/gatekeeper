# Gatekeeper — Agent Instructions

## Start here (read order)

1. First session: `AGENTS.md` → `NOW.md` → `tasks/todo.md` (§M-01 is the sign-off record).
2. Building: read the one spec your slice needs, from
   `../../research/Optimizing RAG With Jev Reranking/specs/` — they live outside this repo.
3. Never read the specs cover-to-cover; `tasks/todo.md` is the ledger, so consult per slice.
   `tasks/plan.md` holds the durable calls.

## Purpose

Policy-governed RAG over versioned Next.js docs. Decision-model gates decide before any
text is written. The demo is the gate x-ray: one question, two policies, the trace.

## Stack

Python 3.13 + `uv`. FastAPI for the service. Streamlit for the panel. Gemini through the
Decisions API: `gemini-decision-1`, `gemini-embedding-001`, `gemini-3.5-flash-lite`.

## Commands

```bash
uv sync
uv run pytest -q                                    # offline, no model calls
uv run python -m corpus.build_corpus --check         # proven corpus, exit 0
uv run streamlit run panel.py                        # the demo
uv run uvicorn app.main:app --reload                 # the service
uv run python -m eval.harness                        # live: needs GEMINI_API_KEY
```

## Constraints

- The suite is offline only. No model call in `tests/`; live work is the `live` marker or
  the harness.
- The corpus is pinned to `vercel/next.js` `8d1f4ed9c010`. It must not drift;
  `build_corpus --check` is the gate that says so.
- Specs live in `../../research/Optimizing RAG With Jev Reranking/specs/`, not here.
- Hosted on Streamlit Community Cloud, entry file `panel.py`, because Hugging Face Docker
  Spaces need PRO.
- Never commit without Daniel's say-so.

## Current context

See NOW.md — the project is Done and closed as a portfolio piece, so there is no active
slice. See the specs for intent and acceptance criteria, and `tasks/plan.md` for durable
calls.

## Progress tracking

- `Status:` lifecycle: `Draft` → `Ready` → `Building` → `Done`. Never skip `Ready`; never
  `Done` with unchecked boxes.
- `tasks/todo.md` is the single ledger: one task in flight, each with acceptance, verify
  command and files; a box ticks only with the output that closed it quoted in the task,
  `NOW.md` or `HISTORY.md`. No second ledger, in any format.
- Refresh `NOW.md` top block (Status / Next action / Verified date) after meaningful work.
