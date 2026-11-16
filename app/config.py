"""Every model id, price and threshold lives here and nowhere else."""

DECISION_MODEL = "gemini-decision-1"
EMBED_MODEL = "gemini-embedding-001"
SYNTH_MODEL = "gemini-3.5-flash-lite"

EMBED_DIM = 768
CANDIDATES = 30
# Fusion constant only: both rankers score the whole routed slice.
RRF_K = 60
MAX_KEEP = 5

REFUSAL_ANSWER = "I have no reliable docs for this."
# The synthesis call's cap. 256 cut a live answer mid-sentence; 800 holds five
# lines of under 30 words each with the citations.
SYNTH_MAX_OUTPUT_TOKENS = 800
# This model has no thinking budget and rejects ``thinking_budget=0``; the knob
# stays so a model that does think can set it to 0.
SYNTH_THINKING_BUDGET = None
MAX_SENTENCES = 5
# What the receipt states as the turn's output budget (the one-line refusal
# turn). The call itself is capped by ``SYNTH_MAX_OUTPUT_TOKENS``.
SYNTHESIS_OUTPUT_BUDGET = 256

# A sentence at or above the verified floor is supported; at or above the weak
# floor it is doubtful but kept; below it the sentence is unsupported.
VERIFY_VERIFIED_MIN = 0.60
VERIFY_WEAK_MIN = 0.30

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

PRICE_SYNTH_IN = 0.30 / 1_000_000
PRICE_SYNTH_OUT = 2.50 / 1_000_000

# This build has no published per-token embedding price and does not invent
# one: the embedding stage is reported as unpriced and costs 0.0.
PRICES = {
    DECISION_MODEL: {"in": PRICE_DECISION_IN, "out": PRICE_DECISION_OUT},
    SYNTH_MODEL: {"in": PRICE_SYNTH_IN, "out": PRICE_SYNTH_OUT},
}

IN_SCOPE_MIN = 0.50
INJECTION_MAX = 0.50
