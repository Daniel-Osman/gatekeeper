"""Every model id, price and threshold lives here and nowhere else."""

DECISION_MODEL = "gemini-decision-1"
EMBED_MODEL = "gemini-embedding-001"

EMBED_DIM = 768
CANDIDATES = 30
# Fusion constant only: both rankers score the whole routed slice.
RRF_K = 60
MAX_KEEP = 5

REFUSAL_ANSWER = "I have no reliable docs for this."
SYNTHESIS_OUTPUT_BUDGET = 256

# One sentence per block reason, so a refused turn still answers.
ANSWERS = {
    "blocked_out_of_scope": "That question is outside the Next.js documentation this demo covers.",
    "blocked_injection": "That request tries to change how this assistant works, so it was refused.",
    "unclear_router": "Is this about the App Router or the Pages Router?",
}

# List prices, US dollars per token. The service bills the free tier; the
# receipt reports list price so the cost of a paid tier stays visible.
PRICE_DECISION_IN = 0.10 / 1_000_000
# No published output price for the decision model: input-only billing.
PRICE_DECISION_OUT = 0.0

# This build has no published per-token embedding price and does not invent
# one: the embedding stage is reported as unpriced and costs 0.0.
PRICES = {
    DECISION_MODEL: {"in": PRICE_DECISION_IN, "out": PRICE_DECISION_OUT},
}

IN_SCOPE_MIN = 0.50
INJECTION_MAX = 0.50
