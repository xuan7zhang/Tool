"""Extract reasoning likelihood/entropy from a run log at three granularities.

Reads the per-question JSONL run log written by ``launch_over_chexbench.py`` when
run with ``--capture-logprobs K`` (each assistant message then carries a
``logprobs`` list of vLLM top-K elements). Produces:

  (A) per-token    : one row per generated token (pos, token, logprob, entropy,
                     is_reasoning) -- the rawest view.
  (B) per-turn seq : one row per assistant turn holding that turn's token list.
  (C) per-turn agg : mean_logprob / perplexity / mean_entropy per turn, plus the
                     reasoning-only versions (reasoning_mean_logprob,
                     reasoning_mean_entropy, n_reasoning_tokens).

Every row carries ``query_id`` (= question_id), ``condition`` (= label) and
``seed`` so Task-2 analysis can line these up with the existing selection metrics.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from .entropy import aggregate, token_entropy_bits, token_logprob
from .reasoning import (
    REASONING_MODE,
    reasoning_token_mask,
    reasoning_turn_index,
)


def load_run_log(path: str) -> List[Dict[str, Any]]:
    """Load the per-question JSONL run log (skips blank/corrupt lines)."""
    entries: List[Dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _ai_messages(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assistant messages from the serialized trace, in order."""
    return [m for m in (entry.get("trace") or []) if m.get("type") == "ai"]


def _tool_names(msg: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for call in msg.get("tool_calls") or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name:
            if "<|channel|>" in name:  # mirror agent.execute_tools cleanup
                name = name.split("<|channel|>", 1)[0].strip()
            names.append(name)
    return names


def build_turns(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structured per-turn view with per-token logprob/entropy/is_reasoning.

    Returns a list of turn dicts; ``turn_idx`` enumerates assistant turns. Turns
    without captured logprobs simply have an empty ``tokens`` list.
    """
    ai_msgs = _ai_messages(entry)
    reasoning_idx = reasoning_turn_index(ai_msgs)

    turns: List[Dict[str, Any]] = []
    for turn_idx, msg in enumerate(ai_msgs):
        is_reasoning_turn = turn_idx == reasoning_idx
        raw = msg.get("logprobs") or []
        mask = reasoning_token_mask(raw, is_reasoning_turn)
        tokens: List[Dict[str, Any]] = []
        for pos, (elem, is_reasoning) in enumerate(zip(raw, mask)):
            tokens.append({
                "pos": pos,
                "token_id": elem.get("token_id"),  # absent in path A (logprobs)
                "token_str": elem.get("token", ""),
                "logprob": token_logprob(elem),
                "entropy_bits": token_entropy_bits(elem),
                "is_reasoning": is_reasoning,
            })
        turns.append({
            "turn_idx": turn_idx,
            "has_tool_calls": bool(msg.get("tool_calls")),
            "tool_names": _tool_names(msg),
            "is_reasoning_turn": is_reasoning_turn,
            "tokens": tokens,
        })
    return turns


def _meta(entry: Dict[str, Any], label: str, seed: int) -> Dict[str, Any]:
    return {
        "query_id": str(entry.get("question_id", "unknown")),
        "condition": label,
        "seed": seed,
    }


def per_token_rows(entry, label: str, seed: int) -> Iterator[Dict[str, Any]]:
    """Granularity A: one row per token."""
    meta = _meta(entry, label, seed)
    for turn in build_turns(entry):
        for tok in turn["tokens"]:
            yield {**meta, "turn_idx": turn["turn_idx"], **tok}


def per_turn_seq_rows(entry, label: str, seed: int) -> Iterator[Dict[str, Any]]:
    """Granularity B: one row per turn, holding the token sequence."""
    meta = _meta(entry, label, seed)
    for turn in build_turns(entry):
        yield {
            **meta,
            "turn_idx": turn["turn_idx"],
            "has_tool_calls": turn["has_tool_calls"],
            "tool_names": turn["tool_names"],
            "is_reasoning_turn": turn["is_reasoning_turn"],
            "n_tokens": len(turn["tokens"]),
            "tokens": turn["tokens"],
        }


def per_turn_agg_rows(entry, label: str, seed: int) -> Iterator[Dict[str, Any]]:
    """Granularity C: per-turn aggregate + reasoning-only aggregate."""
    meta = _meta(entry, label, seed)
    for turn in build_turns(entry):
        toks = turn["tokens"]
        all_lp = [t["logprob"] for t in toks]
        all_h = [t["entropy_bits"] for t in toks]
        r_lp = [t["logprob"] for t in toks if t["is_reasoning"]]
        r_h = [t["entropy_bits"] for t in toks if t["is_reasoning"]]

        all_agg = aggregate(all_lp, all_h)
        r_agg = aggregate(r_lp, r_h)
        yield {
            **meta,
            "turn_idx": turn["turn_idx"],
            "has_tool_calls": turn["has_tool_calls"],
            "is_reasoning_turn": turn["is_reasoning_turn"],
            "reasoning_mode": REASONING_MODE,
            "n_tokens": all_agg["n_tokens"],
            "mean_logprob": all_agg["mean_logprob"],
            "perplexity": all_agg["perplexity"],
            "mean_entropy_bits": all_agg["mean_entropy_bits"],
            "n_reasoning_tokens": r_agg["n_tokens"],
            "reasoning_mean_logprob": r_agg["mean_logprob"],
            "reasoning_perplexity": r_agg["perplexity"],
            "reasoning_mean_entropy_bits": r_agg["mean_entropy_bits"],
        }


def has_any_logprobs(entries: List[Dict[str, Any]]) -> bool:
    """True if at least one assistant message carries captured logprobs."""
    for entry in entries:
        for msg in _ai_messages(entry):
            if msg.get("logprobs"):
                return True
    return False
