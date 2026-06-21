"""Locate the reasoning segment within a rollout.

DECISION (2026-06-17): the agent runs Qwen3-VL-8B-**Instruct** (no <think>
channel; intermediate tool-call turns emit empty content). The collaborator's
"reasoning process" is therefore defined as the model's **final-answer prose** --
the terminal assistant message, which carries the model's verbalised
justification (~1.3k chars on average). Its tokens are the reasoning tokens;
tool-call-turn tokens are not.

This keeps span detection trivial and exact: no char->token offset mapping is
needed because the reasoning segment is an entire generation turn. If a thinking
variant is ever used, swap in a ``<think>...</think>`` span finder here (use the
tokenizer with ``return_offsets_mapping=True`` for char->token alignment) and
flip ``REASONING_MODE`` -- nothing else downstream changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .entropy import is_control_token

REASONING_MODE = "final_answer_prose"  # recorded in outputs for provenance


def _has_tool_calls(turn: Dict[str, Any]) -> bool:
    tc = turn.get("tool_calls")
    return bool(tc)


def reasoning_turn_index(ai_turns: List[Dict[str, Any]]) -> int:
    """Index (into ``ai_turns``) of the reasoning turn, or -1 if none.

    The reasoning turn is the last assistant turn that issues no tool call (the
    final answer). Falls back to the last assistant turn if every turn somehow
    has a tool call (degraded; flagged by callers via n_reasoning_tokens).
    """
    if not ai_turns:
        return -1
    for idx in range(len(ai_turns) - 1, -1, -1):
        if not _has_tool_calls(ai_turns[idx]):
            return idx
    return len(ai_turns) - 1


def reasoning_token_mask(turn_tokens: List[Dict[str, Any]], is_reasoning_turn: bool) -> List[bool]:
    """Per-token reasoning flags for one turn's logprobs token list.

    A token is reasoning iff it belongs to the reasoning turn and is not a
    chat-template control token (e.g. a trailing ``<|im_end|>``). For the
    ``final_answer_prose`` mode the whole reasoning turn minus control tokens is
    the reasoning span -- the documented degraded "whole generation = reasoning"
    behaviour, scoped to the single answer turn.
    """
    mask: List[bool] = []
    for tok in turn_tokens:
        token_str = tok.get("token", "")
        mask.append(is_reasoning_turn and not is_control_token(token_str))
    return mask
