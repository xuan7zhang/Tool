"""DUCX reasoning-entropy module.

Opt-in extraction of per-token reasoning **likelihood** and **entropy** from
rollouts, orthogonal to the tool-*selection* entropy in ``ducx_noise.metrics``.
Nothing here changes the rollout/eval pipeline: it post-processes the run log
(written with ``launch_over_chexbench.py --capture-logprobs K``) the same way the
selection metrics do.

Public API::

    from ducx_entropy.extract import per_turn_seq_rows, per_turn_agg_rows
    from ducx_entropy.entropy import token_entropy_bits, token_logprob

CLI::

    python -m ducx_entropy.cli extract --run-dir DIR --label L --seed 0 --out-dir OUT
    python -m ducx_entropy.analyze --perturn OUT/*_perturn.jsonl --selection-csv M.csv
"""

from .entropy import aggregate, perplexity, token_entropy_bits, token_logprob
from .extract import (
    build_turns,
    per_turn_agg_rows,
    per_turn_seq_rows,
)
from .reasoning import REASONING_MODE, reasoning_turn_index

__all__ = [
    "token_entropy_bits",
    "token_logprob",
    "perplexity",
    "aggregate",
    "build_turns",
    "per_turn_seq_rows",
    "per_turn_agg_rows",
    "reasoning_turn_index",
    "REASONING_MODE",
]
