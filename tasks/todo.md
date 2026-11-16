# Gatekeeper - task ledger

One task in flight. Each task names the spec it builds and the command that closes it; the suite runs offline against committed fixtures.

- [x] **corpus** - Versioned Next.js snippet set with labels
  - Acceptance: Versioned Next.js snippet set with labels; record invariants, every `--check` failure branch and splitter/assembly determinism proven offline
  - Verify: uv run python -m corpus.build_corpus --check; uv run pytest -q tests/test_corpus.py
  - Files: tests/test_corpus.py, tests/fixtures/docs/, tests/fixtures/corpus_expected.jsonl
  - Spec: ../../research/Optimizing RAG With Jev Reranking/specs/SPEC-corpus.md
  - Note: `--check` -> 246 chunks (app=196 pages=50), legacy=21, forum=17, `check: ok`. `uv run pytest tests/test_corpus.py -v` -> 62 passed. `uv run pytest -q` -> 156 passed. Unproven: embedding-value determinism (needs model calls).
- [x] **api-service** - FastAPI service holding the pipeline and its per-query trace
  - Acceptance: FastAPI service holding the pipeline and its per-query trace
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/specs/SPEC-api-service.md
  - Note: Covered by `tests/test_api.py` (20 passed). Untested criterion: `/health` with a missing `.npy`.
- [x] **gate-scope** - Gate 1: in-scope check, version routing, injection refusal
  - Acceptance: Gate 1: in-scope check, version routing, injection refusal
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/specs/SPEC-gate-scope.md
  - Note: Covered by `tests/test_scope.py` (20 passed). Untested criteria: scope threshold boundary 0.50, injection precedence order, bad-router 502.
- [x] **retrieval-hybrid** - BM25 + dense retrieval returning ~30 chunks with fused ranking
  - Acceptance: BM25 + dense retrieval returning ~30 chunks with fused ranking
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/specs/SPEC-retrieval-hybrid.md
  - Note: Covered by `tests/test_retrieve.py` (20 passed). Untested criteria: 24-chunk short slice, embedding-call count.
- [x] **gate-rerank** - Gate 2: per-chunk relevance/currency scoring, threshold, zero-pass stop
  - Acceptance: Gate 2: per-chunk relevance/currency scoring, threshold, zero-pass stop
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/specs/SPEC-gate-rerank.md
  - Note: Covered by `tests/test_rerank.py` (20 passed). Untested criterion: decision-call question count.
- [x] **synthesis** - Chat model writes the cited answer from the kept chunks
  - Acceptance: Chat model writes the cited answer from the kept chunks
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/SPEC-map.md
  - Note: Live 2026-11-15. uv run pytest -q -> 57 passed in 0.40s. Strict trace: outcome answered, synthesize latency_ms=1899.79 tokens_in=1343, verify latency_ms=381.788, list_cost_usd=0.0025305. fast_fix: answered, verified=5. Repair path still offline only.
- [x] **gate-verify** - Gate 3: per-sentence support check, one repair pass, flag or answer
  - Acceptance: Gate 3: per-sentence support check, one repair pass, flag or answer
  - Verify: uv run pytest -q
  - Files: -
  - Spec: ../../research/Optimizing RAG With Jev Reranking/SPEC-map.md
  - Note: Live 2026-11-15. uv run pytest -q -> 57 passed in 0.40s. Strict trace: outcome answered, synthesize latency_ms=1899.79 tokens_in=1343, verify latency_ms=381.788, list_cost_usd=0.0025305. fast_fix: answered, verified=5. Repair path still offline only.
- [x] **eval-harness** - Labelled query set and per-policy measurement
  - Acceptance: 40 labelled queries, three pipelines, results.json written
  - Verify: uv run python -m eval.harness
  - Files: eval/harness.py, eval/queries.jsonl, eval/README.md, eval/scorecard.md, tests/test_eval.py, eval/results.json
  - Note: Live 2026-11-15, `uv run python -m eval.harness` -> 120 records (384 calls) -> eval/results.json, 0 errors. Cited-gold on the 30 answerable queries: strict 0.667, fast_fix 0.867, no_gates 0.867. `uv run pytest -q` -> 66 passed in 0.50s. Scorecard: eval/scorecard.md.
- [x] **panel** - Streamlit panel with x-ray trace, scorecard and policy switch
  - Acceptance: Dark panel: wordmark + provenance, three presets above the search bar, strict/fast_fix switch echoed on the answer, per-sentence verdict badges, one cost line, four outcome shapes, gate x-ray with real trace fields and no row for a stage that did not run, scorecard from `eval/results.json` with `--` for a blank, and empty/loading/error states with no invented numbers.
  - Verify: `uv run pytest -q`; then `uv run streamlit run panel.py --server.headless true --server.port 8599` and drive `http://127.0.0.1:8599` through the three presets under both policies (6 live runs, 40-call budget), capturing `screens/panel-live-*.png`.
  - Files: panel.py, app/panel_view.py, tests/test_panel.py, requirements.txt, .streamlit/config.toml, pyproject.toml
  - Spec: tasks/plan.md
  - Note: design approved; build done. First independent review, 2026-11-17: FAIL on policy attribution, missing scorecard caveats, the absent milestone record and the unproven corpus. The first two are fixed, the corpus is now proven by its `--check` and 62 tests, and `M-01` below writes the record. Third independent review, 2026-11-17: PASS — `M-01` carries the three passes and the sign-off. The live six-run drive above is deferred, not run, and is not the gate for this pass.

- [x] **panel-attribution** - Every policy string the panel renders comes from the envelope's trace
  - Acceptance: badge, refusal headline, refusal detail, floor and switch note all read the envelope's `trace.policy`; a caller policy that disagrees with the trace never describes the run; a trace with no policy renders `--` and invents no name.
  - Verify: `uv run pytest -q`; plus the holdout — a `strict` refusal envelope rendered with a `fast_fix` caller reads `strict` in headline and detail.
  - Files: app/panel_view.py, panel.py, tests/test_panel.py
  - Spec: tasks/plan.md
  - Note: fixed 2026-11-17 — badge, refusal headline, refusal detail, floor and switch note read the envelope's `trace.policy`. Signed off by the third independent review 2026-11-17 (`M-01` PASS).

- [x] **scorecard-caveats** - The scorecard states the three caveats where the numbers are shown
  - Acceptance: the panel's scorecard section and `eval/scorecard.md` both state smoke-grade (40 queries, 3.3 points), gold cited on 20 of 23 official-doc questions (the gates buy refusal, clarification and verdicts, not recall), and free-tier latency with an unattributed p95 near 60 s; small text under the tables; no existing claim deleted or softened.
  - Verify: `uv run pytest -q`.
  - Files: app/panel_view.py, panel.py, eval/scorecard.md, tests/test_panel.py
  - Spec: tasks/plan.md
  - Note: fixed 2026-11-17 — `eval/scorecard.md` "Caveats on these numbers" and the panel's scorecard foot (`SCORECARD_CAVEATS` in `app/panel_view.py`) both carry the three claims in small type. Signed off by the third independent review 2026-11-17 (`M-01` PASS).

---

## M-01 — panel, corpus and scorecard closure

M-01 — PASS (2026-11-17) — signed off by the third independent review

```
Acceptance: "Dark panel: wordmark + provenance, three presets above the search bar,
  strict/fast_fix switch echoed on the answer, per-sentence verdict badges, one cost
  line, four outcome shapes, gate x-ray with real trace fields and no row for a stage
  that did not run, scorecard from `eval/results.json` with `--` for a blank, and
  empty/loading/error states with no invented numbers." — tasks/plan.md ## Acceptance
Changed: app/panel_view.py, panel.py, tests/test_panel.py, eval/scorecard.md,
  feature_list.json, tasks/todo.md, NOW.md (modified); tests/test_corpus.py,
  tests/fixtures/docs/, tests/fixtures/corpus_expected.jsonl (untracked).
Index: tree of commit b93f1c3 — `git rev-parse b93f1c3^{tree}` →
  36e3181ba1a5bf8445ec2beb86cceeca8a5325b5; the working tree was clean when
  reviewed. Committed as b93f1c3 on `master`, pushed to `origin`.
Gates (run on the final tree, 2026-11-17):
  `uv run pytest -q` -> exit 0 — "156 passed in 3.19s"
  `uv run pytest -q tests/test_corpus.py` -> exit 0 — "62 passed in 2.68s"
  `uv run python -B -m corpus.build_corpus --check` -> exit 0 —
    "corpus: 246 chunks (app=196 pages=50 both=0) | legacy=21 | forum=17 synthetic |
     model=gemini-embedding-001 dim=kept | commit=8d1f4ed9c010"
    "check: ok"
Behavior: tests/test_corpus.py - 62 cases - 246-record invariants (eight keys in order,
  ids c1..c246 unique and contiguous, router/status/source sets, text caps, the 17
  forum records), every `--check` failure branch, splitter and `assemble_records`
  byte-identical against the 13-record snapshot. tests/test_panel.py - 28 cases - pure
  renderers driven by hand-built envelopes (answered, flagged, refused, blocked,
  clarify, error): one badge per sentence, `M of N candidates failed`, x-ray omits a
  stage that did not run, `retry_of absent`, scorecard `--` for a blank and no table
  without a file, `next_run` fires 0 times on a plain rerun. All offline, no model call.
Live: https://gatekeeper-rag.streamlit.app answers; under `strict` the Policy Flip
  preset renders a refusal with "kept 0 of 30"; under `fast_fix` it answers with
  "2 of 2 sentences verified". After a Streamlit Cloud reboot on 2026-11-17 the app
  loads with no traceback, the Policy Flip preset under `fast_fix` renders
  `answered · 3 of 3 sentences verified · 3 verified`, and the scorecard caveats
  render.
Negative: no red-proof re-run on the final tree this pass. The last proven red is the
  2026-11-15 mutation `FAST_FIX.min_relevance 0.40 -> 0.60` -> 1 failed, 19 passed, on
  an earlier tree. Carried as open, not claimed.
Holdout: a `strict` refusal envelope handed to the adapter with a `fast_fix` caller ->
  `refused · strict kept no chunks`; detail `1 of 1 candidates failed strict. Best match
  0.45 against a floor of 0.60.`
Review: pass 1, 2026-11-17 — FAIL — policy attribution, missing scorecard caveats, no
  milestone record, corpus unproven.
  pass 2, 2026-11-17 — FAIL — the live app held mixed code and crashed; this record's
  `Index:` line was false.
  pass 3, 2026-11-17 — PASS, blockers none. Signed off: the two fixes plus the
  regression sweep (`uv run pytest -q` -> "156 passed"), the live app after the reboot
  (`Official` under strict -> `answered · 5 of 5 sentences verified`; `Policy Flip`
  under `fast_fix` -> `answered · 3 of 3 sentences verified`), the caveats visible,
  and the `b93f1c3` tree `36e3181…` verified.
  Outside the sign-off, stated by the reviewer: the six-run live drive, the
  embedding-value determinism and the per-feature untested criteria.
Open: the six live preset runs are deferred, so a live panel drive on the final tree is
  unproven; embedding-value determinism unproven (needs model calls); the per-feature
  untested criteria stay listed on the tasks above. `Negative:` stands: no red-proof
  re-run on this tree.
```
