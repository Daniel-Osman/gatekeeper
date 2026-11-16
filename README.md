# Gatekeeper

Status: vertical slice, all five stages live, 40-query scorecard written
(`eval/scorecard.md`). Panel built and live.

Live: https://gatekeeper-rag.streamlit.app — no signup, three presets, the policy switch and the measured scorecard.

Answers Next.js questions from a 246-chunk corpus (`vercel/next.js` at
`8d1f4ed9c010`, MIT; 21 legacy chunks, 17 synthetic forum chunks). Retrieval is
BM25 + dense. Two decision-model gates run before any text is written, and one
verify gate runs after. The eval asks whether those gates earn their cost.

## The five stages

```
question
  scope       route to App or Pages; refuse out of scope; refuse injection
  retrieve    BM25 + dense over the routed slice -> 30 candidates
  rerank      relevance and currency per chunk -> keep at most 5
  synthesize  cited answer from the kept chunks only
  verify      support per sentence, one repair pass, flag what still fails
```

All five are built. Every stage reports its scores in the response trace.

## Measured

40 labelled queries x 3 policies = 120 records, 384 model calls, 0 errors,
$0.1797 total at list price. Full tables: `eval/scorecard.md`.

| policy | answered | cited gold (30) | median | USD/1,000 |
|---|---|---|---|---|
| strict | 20 | 0.667 | 3,169 ms | 1.82 |
| fast_fix | 28 | 0.867 | 2,795 ms | 1.97 |
| no_gates | 31 | 0.867 | 1,207 ms | 0.70 |

fast_fix's 28 counts one `answered_flagged`. `cited gold` asks whether a gold
chunk id appears in the answer's citations, over the 30 answerable queries.

## One question, two policies

"`getServerSideProps` never logs and props are undefined inside an app page.
Why?" (`q26`, forum-only: the official docs do not answer it.)

- `strict` — refused: "I have no reliable docs for this."
- `fast_fix` — answered from forum chunk `c232`, 5 of 5 sentences verified.

Same pipeline, same question, one policy constant. `strict` drops stale forum
chunks on purpose; `fast_fix` keeps them.

## Run

```bash
uv sync
uv run pytest -q                     # 87 tests, offline fixtures
uv run python -m eval.harness        # live: 384 calls, needs GEMINI_API_KEY
uv run uvicorn app.main:app --reload
curl -s localhost:8000/health
```

`eval/harness.py --help` lists `--limit`, `--only`, `--policies`, `--out`,
`--quiet` and `--rpm`.

The hosted app runs `panel.py` on Streamlit Community Cloud and reads
`eval/results.json` from the repo for its scorecard.

## What this does not show

- **The gates do not beat the ungated baseline on recall.** All three policies
  cite a gold chunk on 20 of the 23 official-doc questions (0.870). The gates
  buy refusal, clarification and a per-sentence verdict, not recall.
- **40 queries is smoke-grade.** One query moves a rate by 3.3 points.
- **`cited` is not correctness.** It checks that a gold id is in the citations,
  not that the sentence is true.
- **Latency is free-tier.** The p95 is 59.9 s on `strict`; the attribution to
  throttling is an inference. Re-run on a paid tier before quoting it.
- **The embedding stage is unpriced**, so cost per 1,000 is a floor.

Corpus provenance and the label sources are in
`../../research/Optimizing RAG With Jev Reranking/brief.md`.
