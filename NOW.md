Status: Done (2026-11-17) — complete and live at https://gatekeeper-rag.streamlit.app — one public URL, no signup, hosted on Streamlit Community Cloud because Hugging Face Docker Spaces need PRO. Corpus 246 chunks, proven offline. Five stages live: scope → retrieve (BM25 + dense) → rerank → synthesize on `gemini-3.5-flash-lite` → verify, with refusal, clarify, one-repair and flag paths. Panel built and live: three presets, gate x-ray, `strict`/`fast_fix` switch, scorecard from `eval/results.json`. 40-query scorecard at `eval/scorecard.md`: 120 records, 384 calls, 0 errors, $0.1797 at list price; cited-gold on the 30 answerable queries strict 0.667 / fast_fix 0.867 / no_gates 0.867. 156 tests green offline. First independent review FAILed on four findings; two fixed, the corpus now proven, the record written — signed off by the third independent review.

Next:
- none. The project is closed as a portfolio piece; there is no next work.
- Outside the sign-off, not tasks: the six live preset runs, embedding-value
  determinism, and the per-feature untested criteria.

Findings:
- flash-lite rejects `thinking_budget=0`, so synthesis sends no thinking config.
- free tier allows 15 generate calls a minute per model and one turn spends 4–5, so the harness needed `--rpm 12` and 429 retries at 35s, 70s, 140s.
- `--out` is written atomically after each record, so a crash leaves a resumable file.
- decision-model `tokens_out` is 0 and embedding usage is absent, so a receipt prices only what it measured.
- flat chunk scores persisted until each gate question named its chunk tag; `max_keep=5` lets official chunks fill the window, so the policy switch needs a question whose best answers are stale forum entries.

Verified:
- 2026-11-17 promoted incubator/gatekeeper -> active/gatekeeper; uv run pytest -q -> 156 passed
- 2026-11-17 `uv run pytest -q` → exit 0 — `156 passed in 3.19s`
- 2026-11-17 `uv run pytest -q tests/test_corpus.py` → exit 0 — `62 passed in 2.68s`
- 2026-11-17 `uv run python -B -m corpus.build_corpus --check` → exit 0 — `corpus: 246 chunks (app=196 pages=50 both=0) | legacy=21 | forum=17 synthetic | model=gemini-embedding-001 dim=kept | commit=8d1f4ed9c010` / `check: ok`
- 2026-11-17 panel holdout: a `strict` refusal envelope rendered by the adapter under a `fast_fix` caller → `refused · strict kept no chunks`, detail `1 of 1 candidates failed strict. Best match 0.45 against a floor of 0.60.`
- 2026-11-17 live https://gatekeeper-rag.streamlit.app → answers; under `strict` the Policy Flip preset refuses with `kept 0 of 30`; under `fast_fix` it answers with `2 of 2 sentences verified`
- 2026-11-17 first independent review → FAIL, four findings: policy attribution (High, fixed), missing scorecard caveats (High, fixed), absent milestone record (Medium, closed by `M-01`), unproven corpus (High, closed by the corpus gates above). Second and third reviews ran; the third PASSed with no blockers.
- 2026-11-15 live `uv run python -m eval.harness` → 120 records (384 calls) in `eval/results.json`, 0 errors, $0.1797 at list price; cited-gold on the 30 answerable queries strict 0.667 / fast_fix 0.867 / no_gates 0.867, official-only all three 0.870; median 3169 / 2795 / 1207 ms; scorecard at `eval/scorecard.md`
- 2026-11-15 mutation `FAST_FIX.min_relevance 0.40 → 0.60` → 1 failed, 19 passed; reverted → 20 passed. Last proven red, on an earlier tree; not re-run this pass.
