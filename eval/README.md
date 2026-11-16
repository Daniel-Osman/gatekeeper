# Eval harness

Smoke-grade measurement of the three retrieval policies over 40 labelled
queries. It answers one question: does a gate earn its cost, or does plain
retrieval score the same?

**Caveat first.** 40 queries is smoke-grade, not a benchmark. One query moves
`hit_at_5` by 3.3 points; the spread between policies is smaller than that on
most runs. Read the numbers as "no obvious regression", never as a ranking.

## Files

| File | What it is |
|---|---|
| `queries.jsonl` | 40 labelled queries, one JSON object per line |
| `harness.py` | runs the policies, aggregates, prints the table |
| `results.json` | output of a run; created and resumed by the harness |

## Labelling

Each line holds:

| Field | Meaning |
|---|---|
| `id` | `q01`…`q40`, stable across runs |
| `question` | the user wording, sent to the pipeline unchanged |
| `router` | `app`, `pages`, or `unclear`; only `no_gates` reads it |
| `gold_ids` | chunk ids that answer the question; `[]` when none do |
| `kind` | `official`, `forum_only`, `unanswerable` or `ambiguous` |
| `note` | ten words or fewer: why that chunk answers it |

Split: 23 `official`, 7 `forum_only`, 6 `unanswerable`, 4 `ambiguous`.

- `official` — an official docs chunk states the answer. The `note` names the
  mechanism, so a reader can check the label without rerunning the eval.
- `forum_only` — one forum thread, and only that thread, answers it: the
  official docs are silent or stale. The `note` names the delegation the thread
  supplies. All seven forum chunks are `legacy`, so `strict` drops them by
  policy and `fast_fix` keeps them. That is the comparison this kind exists for.
- `unanswerable` — nothing in the corpus answers it. A refusal is the correct
  turn, an answer is a miss.
- `ambiguous` — the question needs a router before it can be answered. A
  clarify is the correct turn.

`gold_ids` is `[]` for the last two kinds: there is no chunk to hit.

Adding a query: pick the topic, find the chunk, read the chunk, and only then
write the question around it. If the chunk does not answer the wording, change
the wording, never the `kind`.

## Running

```bash
uv run python -m eval.harness                      # all 40, all three policies
uv run python -m eval.harness --limit 5            # first five queries
uv run python -m eval.harness --only q01,q24       # named ids
uv run python -m eval.harness --policies no_gates  # one policy
uv run python -m eval.harness --quiet              # no per-query lines
```

`--out` moves `results.json`. A pair already in that file is skipped, so an
interrupted run resumes; the harness prints `skipped N`. Delete the file for a
clean run. This calls live models and costs money: run it only with approval.

`--rpm` caps model calls per rolling minute; the default is 12, under the free
tier's 15. A gated turn spends 5 calls, a clarify or `no_gates` turn 2, and the
harness sleeps before a turn when the minute is full, printing
`waiting 12s for rpm`. A 429 is retried after 35s, 70s and 140s; when those run
out the query becomes an error record with its `error` text and the run goes on.
The out file is rewritten atomically after every record, so a crash leaves a
resumable file rather than losing the day: rerun the same command and the
finished pairs are skipped.

The tests are offline and call no model:

```bash
uv run pytest -q tests/test_eval.py
```

## First paced run

```
uv run python -m eval.harness --limit 6 --out eval/results.smoke2.json --rpm 12
```

18 records, 72 calls, 9m07s wall, no error and no clarify. Every gated turn
answered; `hit@5` was 1.0 for `strict` and `fast_fix`, 0.6667 for `no_gates`.

| query | strict | fast_fix | no_gates |
|---|---|---|---|
| q01 | answered, hit | answered, hit | answered, hit |
| q02 | answered, hit | answered, hit | answered, hit |
| q03 | answered, hit | answered, hit | answered, missed |
| q04 | answered, hit | answered, hit | answered, hit |
| q05 | answered, hit | answered, hit | answered, hit |
| q06 | answered, hit | answered, hit | answered, missed |

`hit` is `hit_at_5` true for the gold id; `missed` is false. The six questions
now name the App Router, so the scope gate proceeds. The four clarifies in
`results.smoke.json` (q01, q03, q04, q06) were label flaws, not gate failures:
the gate asked a fair question of a question that named no router. `no_gates`
missing q03 and q06 is the ungated baseline dropping a gold chunk, not a gate.

## Policies

| Policy | Path |
|---|---|
| `strict` | `app.pipeline.run` — scope, rerank (no forum, no legacy), synthesize, verify |
| `fast_fix` | `app.pipeline.run` — same, but forum and legacy chunks stay eligible |
| `no_gates` | `retrieve.run` then `synth.run`, no scope, rerank or verify stage |

`no_gates` takes the router from the query label and falls back to `app` when
the label is `unclear`. It records the five ids retrieval reached as
`kept_ids`, and calls a turn `refused` only when synthesis returns
`config.REFUSAL_ANSWER`. It has no verify stage, so its records carry no
`verdicts`.

## Record fields

One record per query per policy, in `results.json`:

| Field | Meaning |
|---|---|
| `id`, `policy`, `kind` | what ran |
| `error` | first 200 characters of the failure; a good turn has none |
| `outcome` | `answered`, `answered_flagged`, `refused`, `blocked` or `clarify` |
| `answer` | the text the turn returned |
| `sentences` | how many sentences that text holds |
| `verdicts` | per-sentence verify verdicts; empty on `no_gates` |
| `kept_ids` | ids that reached synthesis, first five |
| `gold_ids` | the label, copied from the query |
| `hit_at_5` | any gold id among `kept_ids`; `null` when not answerable |
| `cited_hit` | any gold id cited in the answer; `null` when not answerable |
| `list_cost_usd` | the receipt's list price; `null` when unmeasured |
| `wall_ms` | sum of the stage latencies |

## Table columns

| Column | Meaning | Denominator |
|---|---|---|
| `n` | records for that policy | — |
| `err` | the turn raised and was recorded as an error | — |
| `ans` | `answered` plus `answered_flagged` | `n` |
| `flag` | `answered_flagged`: a sentence stayed unsupported after repair | `n` |
| `ref` | `refused` | `n` |
| `blk` | `blocked`: scope stopped the turn | `n` |
| `clar` | `clarify` | `n` |
| `hit@5` | a gold id reached the synthesis list | `official` + `forum_only` |
| `cited` | a gold id was cited in the answer | `official` + `forum_only` |
| `refuse` | an `unanswerable` query was refused or blocked | `unanswerable` |
| `clarify` | an `ambiguous` query was clarified | `ambiguous` |
| `ms_mean` | mean `wall_ms` | records with a time |
| `ms_p95` | nearest-rank 95th percentile of `wall_ms` | records with a time |
| `cost_usd` | summed list price | — |

An empty denominator prints `-`, not `0`: a rate over nothing is unknown, and
a zero there would read as a failure. A record with an unmeasured cost is left
out of the cost total rather than counted as free. An errored record counts in
`n` and `err` and sits in no rate's denominator: the turn never produced a
judgement to score.

## What it cannot tell you

- It does not judge answer quality. `cited` asks whether a gold id appears in
  the citations, not whether the sentence is right; only the verify gate scores
  the sentence, and it scores against the chunks that were kept.
- 40 queries cannot separate two policies that differ by one query.
- `no_gates` is not a quality ceiling: it answers every in-scope question,
  so its `refuse` score is low by construction. Compare it on `hit@5` and
  `cost_usd`, not on refusals.
