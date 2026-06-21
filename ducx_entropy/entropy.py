"""Per-token likelihood + entropy primitives.

Two distinct signals (the collaborator wants both side by side):
* ``logprob``  -- log p(token | prefix) of the *actually generated* token. High
  likelihood = the model was confident in what it said.
* ``entropy``  -- H = -sum_v p(v) log p(v) over the *whole next-token distribution*
  at that step. Low entropy = the distribution was peaked, regardless of which
  token was chosen. High likelihood does NOT imply low entropy.

PATH A (vLLM logprobs) precision note
-------------------------------------
vLLM/OpenAI return only the top-K logprobs per step, not the full vocabulary.
``token_entropy_bits`` therefore computes a **top-K renormalised approximation**:
it renormalises the K returned probabilities to sum to 1 and takes their entropy.
This *systematically under-estimates* the true entropy (it ignores the tail
mass), so it is good for comparing trends across conditions but is NOT the exact
number for a paper -- for that, recompute over the full vocabulary with an HF
forward pass (path B). Units are **bits** (log2) to match the existing
``selection_entropy_bits`` in ducx_noise.metrics.

logprob values from vLLM are natural-log; ``perplexity`` uses them directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Chat-template control tokens that are emitted but are not "reasoning". They are
# kept in the per-token granularity but excluded from aggregate statistics.
CONTROL_TOKENS = {"<|im_end|>", "<|im_start|>", "<|endoftext|>"}

_LN2 = math.log(2.0)


def is_control_token(token: str) -> bool:
    return token in CONTROL_TOKENS


def token_logprob(entry: Dict[str, Any]) -> float:
    """The natural-log probability of the generated token at this step."""
    return float(entry["logprob"])


def token_entropy_bits(entry: Dict[str, Any]) -> float:
    """Top-K renormalised entropy (bits) of the step's distribution.

    ``entry`` is one vLLM/OpenAI logprobs element with a ``top_logprobs`` list of
    ``{"token", "logprob"}`` (natural-log). Returns 0.0 when no distribution is
    available (e.g. a single deterministic token).
    """
    tops = entry.get("top_logprobs") or []
    if not tops:
        return 0.0
    # Probabilities of the top-K alternatives (natural-log -> linear).
    probs = [math.exp(float(t["logprob"])) for t in tops]
    z = sum(probs)
    if z <= 0.0:
        return 0.0
    h_nats = 0.0
    for p in probs:
        q = p / z  # renormalise over the returned top-K
        if q > 0.0:
            h_nats -= q * math.log(q)
    return h_nats / _LN2  # nats -> bits


def perplexity(mean_logprob: Optional[float]) -> Optional[float]:
    """exp(-mean natural-log-likelihood). None when undefined."""
    if mean_logprob is None:
        return None
    return math.exp(-mean_logprob)


def aggregate(
    logprobs: Sequence[float],
    entropies_bits: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Mean logprob / perplexity / mean entropy over a token subset.

    Empty input yields all-None (so callers can record "no tokens" cleanly).
    """
    n = len(logprobs)
    if n == 0:
        return {"n_tokens": 0, "mean_logprob": None, "perplexity": None,
                "mean_entropy_bits": None}
    mean_lp = sum(logprobs) / n
    mean_h = sum(entropies_bits) / n if entropies_bits else None
    return {
        "n_tokens": n,
        "mean_logprob": mean_lp,
        "perplexity": perplexity(mean_lp),
        "mean_entropy_bits": mean_h,
    }
