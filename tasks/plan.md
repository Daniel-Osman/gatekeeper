# Panel — build plan (one slice)

Spec: this file plus the approved design `screens/panel.html` and
`screens/panel-1440.png` (the dark render is the approved look; the HTML's
light tokens are superseded). Task brief: build the Streamlit panel, verify it
live, record the six preset outcomes.

## What the page is

A dark instrument panel. Serif answer first, numbers second.

1. Masthead: mark, `Gatekeeper` wordmark, corpus provenance in small type
   (`Next.js App Router docs · 246 chunks · 17 of them forum posts`).
2. Intro: serif h1 and one line of what it is.
3. Controls: three preset pills directly above the search bar; the search bar;
   the `strict` / `fast_fix` switch, all visible before any run.
4. Run card: the policy that produced the answer, the outcome, the answer split
   into sentences with a `verified` / `weak` / `unsupported` badge each, and one
   quiet cost line (list cost, wall time, tokens in/out).
5. Gate x-ray: one row per stage that ran, real trace field names, per-stage
   latency. A stage that did not run has no row. `retry_of` prints `absent`.
6. Scorecard: three pipelines from `eval/results.json` on disk, harness
   aggregate columns, run id and date, `--` for a blank.
7. Footer rule.

## Decisions that constrain the build

1. Entry file is `panel.py` at the repo root. Never `app.py`: `app/` is the
   package and a sibling `app.py` shadows it.
2. The pipeline runs in-process: `asyncio.run(app.pipeline.run(...))` and
   `app.index.ensure_loaded()` behind `st.cache_resource`. No HTTP.
3. Presets are `st.button` with `width="content"`, not `st.pills`. A pill that
   is already selected cannot fire again, and clearing its key after a run
   raises `StreamlitAPIException`. Buttons fire on every click.
4. Policy is `st.segmented_control`. Changing it calls no model. The run card
   echoes the policy that produced the shown answer and says the switch is
   ahead of it when they differ.
5. `st.session_state["run"]` holds the last question, envelope and policy, so a
   rerun from any widget never re-calls the model. Only Ask or a preset click
   runs the pipeline.
6. Answer, sentence list, x-ray and scorecard are HTML from pure functions in
   `app/panel_view.py`; widgets stay Streamlit. One CSS block carries the
   palette and type. No JS.
7. Numbers are never invented: nothing numeric before a run; `None` prints
   `--`; a stage absent from the trace is absent from the x-ray.
8. `GEMINI_API_KEY` is read from the environment only, never printed. If the
   Space supplies it through `st.secrets`, copy it into the environment once,
   silently.

## Outcome → envelope contract

One renderer per `trace.outcome`. Each reads only fields the pipeline writes
for that outcome.

| outcome | came from | card body | x-ray rows |
|---|---|---|---|
| `answered` | verify, no unsupported | sentences + verdict badges; cost line | every stage in `trace.stages`, repeats kept |
| `answered_flagged` | verify after one repair still has an unsupported sentence | same, plus a warning: "One sentence stayed unsupported after the repair pass." | same; the second verify row carries `retry_of` |
| `refused` | rerank kept 0 | `config.REFUSAL_ANSWER` as the serif line; `M of N candidates failed <policy>. Best match X against a floor of Y.`; relevance values; the failure-reason counts from `items[].reason` | scope, retrieve, rerank only; no synthesize/verify row |
| `blocked` | scope, before retrieval | the scope answer from `config.ANSWERS[reason]`; an outcome chip `blocked`; one line naming the reason (`blocked_out_of_scope` or `blocked_injection`); the receipt line from that turn | scope only |
| `clarify` | scope, router unclear | `config.ANSWERS["unclear_router"]` as the serif line; the observed `router_choice` and `router_confidence`; the receipt line from that turn | scope only |
| error (`StageError`) | any stage | `The <stage> stage failed.` + `exc.message`; "The question stays in the box; no answer and no cost." | the stages in `exc.trace` that did run, each with its real latency; no total |

`refused` counts come from the rerank stage: `M = candidate_count - kept_count`,
`N = candidate_count`, best match is `max(items[].relevance_noul)`, floor is
`rerank.POLICIES[policy].min_relevance`.

Every turn that returns an envelope has a receipt, including a scope-only
`blocked` or `clarify` turn: the scope call measures tokens and costs money, so
the card prints that receipt line. A field the receipt reports as `None` (for
example `tokens_out` on a decision-model stage) prints `--`; no figure is
inferred. Only the error path, which has no envelope, prints no cost.

## Empty, loading and error

- **Empty** (`st.session_state["run"]` is `None`): the controls, then one quiet
  sheet that lists the five stage names as a chain and says "Nothing has run
  yet." No digits anywhere, no x-ray, no cost line, no fabricated zeros.
- **Loading**: one spinner line — "Running scope → retrieve → rerank →
  synthesize → verify…" — while the pipeline call is in flight. No numbers, no
  partial card.
- **Error** (`StageError` caught): the card names the failing stage in its
  headline, prints `exc.message`, keeps the question in the box, renders the
  stages in `exc.trace` that did run with their real latencies, and prints no
  cost or total. A missing corpus raises `CorpusError` before any run; the page
  says which file is missing and draws no card.
- **Missing key**: a quiet banner says `GEMINI_API_KEY is not set` and the Ask
  control stays usable so the message is reproducible; no key value is shown.

## X-ray field mapping (real trace keys)

| stage | fields rendered | latency |
|---|---|---|
| `scope` | `in_scope_noul`, `router_choice`, `router_confidence`, `injection_noul`, `reason` | `latency_ms` |
| `retrieve` | `router`, `candidate_count`, `candidates` (first five ids) | `latency_ms` |
| `rerank` | `kept_count` of `candidate_count`, `items[].relevance_noul` with kept values marked, `items[].reason` counts | `latency_ms` |
| `synthesize` | `sentence_count`, `repair`, `truncated` | `latency_ms` |
| `verify` | verdict counts from `sentences[].verdict`, `retry_of` (`absent` when the key is missing) | `latency_ms` |

Rows come straight from `trace.stages` in order, so the repair's second
`synthesize` and second `verify` are two more rows. A stage not in the list has
no row; nothing is drawn grey or zeroed. The bar is the stage's share of the
slowest stage, and the x-ray foot says so.

## Files

- `panel.py` (new) — entry, widgets, session state, one CSS block
- `app/panel_view.py` (new) — pure render helpers, presets, the run decision
- `tests/test_panel.py` (new) — offline tests over those helpers
- `requirements.txt` (new) — `streamlit`, `google-genai`, `bm25s`, `numpy`,
  each pinned to the exact version `uv sync` resolves for the Space
- `.streamlit/config.toml` (new) — dark base theme so widget chrome matches
- `pyproject.toml` — add `streamlit` to dependencies
- `tasks/todo.md`, `NOW.md` — records, after the gates

## Run sequencing (no accidental model call)

Every widget interaction reruns the script, so the live call is guarded by one
pure decision function, `app/panel_view.next_run(...)`, with exactly one call
site in `panel.py`.

- Preset buttons use `on_click` callbacks that write the payload into the
  question widget's key *before* the input is created, and publish
  `st.session_state["preset_fired"]` with the label. The script pops that key at
  the top, so every click fires once, even a repeated click on the same preset.
- `Ask` submits the form. On submit the question and the chosen policy are
  snapshotted together with the envelope into `st.session_state["run"]`.
- `next_run(...)` returns a `{"question", "policy"}` spec only for a preset
  click or a form submit. It returns `None` when the rerun was caused by the
  policy switch, a text edit, a widget toggle or nothing at all; the cached run
  is then re-rendered.
- The card prints the policy stored with the run, not the switch's current
  value. When they differ it adds one quiet line: `Shown run used strict; the
  switch is on fast_fix. Ask again to re-run.`

## Tests (deterministic, offline)

`tests/test_panel.py`, no network:

1. `next_run` returns a spec for a preset click and for a form submit.
2. `next_run` returns `None` for a plain rerun, for a policy change alone and
   for a text edit; a counting fake runner is called 0 times in those cases and
   once per spec.
3. Answered envelope: one badge per sentence, verdict words present.
4. Flagged envelope: warning line present; the second verify row shows
   `retry_of 0`.
5. Refused envelope: `M of N candidates failed`, best match, floor, reason
   counts; no synthesize or verify row.
6. Blocked envelope: scope answer, the scope reason named, no candidate count,
   and its real receipt line (`--` only where a field is `None`).
7. Clarify envelope: router question, scope row only, its real receipt line.
8. Error: names the stage, keeps the question, shows the partial trace, no cost.
9. X-ray: absent `retry_of` prints `absent`; a repeated stage appears twice.
10. Scorecard: reads a fixture payload, prints `--` for a `None` cell, prints
    `run_at` as the run line, and with no file says so and draws no table.

## Verification

1. `uv run pytest -q` — the 66 existing tests plus the new panel tests.
2. `uv run streamlit run panel.py --server.headless true --server.port 8599`,
   then drive `http://127.0.0.1:8599`: screenshot the empty state, run all three
   presets under both policies (six live runs, budget 40 calls), capture the
   answered, flagged, refused and clarify states actually produced. Save every
   capture as `screens/panel-live-*.png`. These six live runs are evidence, not
   a test suite; the deterministic tests above cover the states six runs cannot
   guarantee to produce.
3. If the Policy Flip does not flip, quote the trace reason and try exactly one
   alternative wording: `getServerSideProps never logs and props are undefined
   inside an app page`. No gate, threshold or prompt change.
4. Record the six outcomes, the pytest output and the live-call count in
   `tasks/todo.md` and `NOW.md`.

## Acceptance (quoted from the brief)

- `Gatekeeper` wordmark, one line of what it is, corpus provenance in small type.
- Three presets with these exact labels and payloads, above the search bar:
  - `Official: App Router Params` → `how do I read a route param in the app router`
  - `The Policy Flip: getServerSideProps` → `how do I use getServerSideProps inside an App Router server component`
  - `Off-topic: Cake recipe` → `how to bake a cake`
- A preset click fills the box and runs under the chosen policy.
- Two-state policy switch, visible before the run, echoed on the answer.
- Sentence verdicts carry shape and word, not colour alone.
- One cost line: list cost, wall time, tokens in/out.
- Four outcome shapes: answered, answered_flagged (with warning), refused (line
  plus failed-chunk count), clarify (router question).
- X-ray: real field names, latency per stage, absent stages absent.
- Scorecard from disk with run line and date; missing file says so and draws no
  chart; blank is `--` with a note.
- Empty/loading/error states as above.
- `uv run pytest -q` green.

## Design deltas to expect

- The design's two side-by-side runs (strict and fast_fix on one question) cost
  two live runs per view; the page shows one run at a time under the chosen
  policy and the user flips the switch. The six presets are exercised instead.
- Design placeholder numbers (floor 0.45, 1,900-token budget) are prototype
  values; the panel prints real ones from `config.py` and the receipt.
- Streamlit's text input is a form field, not a chat composer; the design's
  query sheet becomes the form row.
- The design names no run id, and `eval/results.json` has `run_at` and
  `updated_at` only: the scorecard prints `run_at` as the run line, and `--`
  when it is missing. No id is invented.
