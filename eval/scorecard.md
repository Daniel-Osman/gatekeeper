# Eval scorecard

Run `2026-11-15T20:58:09+00:00`, last record written `2026-11-15T21:38:04+00:00`.

Command: `uv run python -m eval.harness` — all 40 queries, the three policies,
`--rpm 12`, out file `eval/results.json`.

40 queries × 3 policies = 120 records, 384 model calls, 0 errors.

Models from `app/config.py`: decision `gemini-decision-1`, embed
`gemini-embedding-001`, synthesis `gemini-3.5-flash-lite`.

Every figure below is computed from `eval/results.json`; no model call was made
to recompute it.

Columns: **answerable** is `official` + `forum_only` (30 queries). **Cited gold**
is a gold chunk id in the answer's citations. **hit@5** is a gold id in the five
kept ids. **wall** is `wall_ms`, the sum of stage latencies. **cost** is the
receipt's list price, not a bill.

## Table 1 — the three pipelines

| policy | answered | flagged | refused | blocked | clarify | cited gold (30) | hit@5 (30) | median ms | mean ms | p95 ms | cost USD | USD/1,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | 20 | 0 | 11 | 2 | 7 | 0.667 (20/30) | 0.667 (20/30) | 3,169 | 22,246 | 59,867 | 0.0730 | 1.82 |
| fast_fix | 27 | 1 | 3 | 2 | 7 | 0.867 (26/30) | 0.900 (27/30) | 2,795 | 14,635 | 52,728 | 0.0788 | 1.97 |
| no_gates | 31 | 0 | 9 | 0 | 0 | 0.867 (26/30) | 0.900 (27/30) | 1,207 | 1,334 | 1,873 | 0.0280 | 0.70 |

`answered` counts plain `answered` only, so each row sums to 40. fast_fix's
`answered + flagged` is 28. The run total is $0.1797 at list price.

## Table 2 — the seven forum-only queries

`gold` is the one chunk that answers the question. `kept` means it reached the
five synthesis chunks; `cited` means it appears in the citations.

| query | gold | strict | fast_fix | no_gates |
|---|---|---|---|---|
| q24 CSS import from `node_modules` | c230 | refused | answered_flagged, kept, not cited | refused, kept, not cited |
| q25 `next/dynamic` with `ssr: false` | c231 | refused | answered, cited | answered, cited |
| q26 `getServerSideProps` in an app page | c232 | refused | answered, cited | answered, cited |
| q27 raw body in `pages/api` | c239 | refused | answered, cited | answered, cited |
| q28 styled-components first paint | c241 | refused | answered, cited | answered, cited |
| q29 `router.query` empty on first render | c245 | refused | answered, cited | answered, cited |
| q30 `unstable_revalidate` ignored | c246 | refused | answered, cited | answered, cited |

fast_fix on q24 kept the gold chunk but cited nothing: its one sentence failed
verify, and that sentence is the refusal string, so the turn gave no answer.

## Table 3 — outcome per kind

| kind (n) | strict | fast_fix | no_gates |
|---|---|---|---|
| official (23) | 20 answered, 3 clarify | 20 answered, 3 clarify | 22 answered, 1 refused |
| forum_only (7) | 7 refused | 6 answered, 1 answered_flagged | 6 answered, 1 refused |
| unanswerable (6) | 4 refused, 2 blocked | 3 refused, 2 blocked, 1 answered | 6 refused |
| ambiguous (4) | 4 clarify | 4 clarify | 3 answered, 1 refused |

## Reading

Across the 30 answerable questions, fast_fix and no_gates each put a gold chunk
in the top five 27 times (0.900) and each cite one 26 times (0.867); strict cites
20 (0.667) because its policy drops all seven forum chunks.
On the 23 official-doc questions alone all three policies cite a gold chunk 20
times (0.870), so the gates add no cited-gold recall over the ungated baseline.
fast_fix spends 2.3× the median wall time (2,795 ms against 1,207 ms) and 2.8×
the cost per query ($1.97 against $0.70 per 1,000 queries) for that tie.
Strict refusing all seven stale forum answers is the policy switch, not a
defect: fast_fix runs the same code path and cites the gold forum chunk on six
of the seven.

## Caveat on latency

Latency here is free-tier wall time, and cost is receipt list price. `wall_ms`
sums the stage latencies; the harness charges its own rate-limit waits and its
429 backoff outside the recorded turn, so a turn whose stage hung for a minute
records a minute. The distribution says the tail is not steady model work —
median 1.2–3.2 s, p95 59.9 s on strict — but these records keep no per-stage
split, so a slow service response cannot be separated from a quota wait. Read
the near-60 s p95 as free-tier throttling, not as the model's speed; that
attribution is an inference, not a logged counter. Re-run on a paid tier before
quoting any latency from this table.
