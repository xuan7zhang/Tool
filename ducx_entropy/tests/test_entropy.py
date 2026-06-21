"""Unit tests for ducx_entropy (CPU only, no model/server).

Covers: top-K renormalised entropy on known distributions, perplexity/aggregate
arithmetic, reasoning-turn detection, and end-to-end extraction with correct
position alignment + is_reasoning labelling (control tokens excluded, tool-call
turns not counted as reasoning). Logprob values must survive extraction at the
right position.
"""

import math

from ducx_entropy.entropy import aggregate, perplexity, token_entropy_bits, token_logprob
from ducx_entropy.extract import (
    build_turns,
    has_any_logprobs,
    per_turn_agg_rows,
    per_turn_seq_rows,
)
from ducx_entropy.reasoning import reasoning_turn_index


def _tok(token, logprob, top):
    """top: list of (token, prob) -> stored as natural-log logprob."""
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top],
    }


# --------------------------------------------------------------------------
# entropy primitives
# --------------------------------------------------------------------------
def test_entropy_uniform_topk_is_log2_k():
    # 4 equal alternatives -> renormalised uniform -> 2 bits.
    entry = _tok("a", math.log(0.25), [("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)])
    assert math.isclose(token_entropy_bits(entry), 2.0, abs_tol=1e-9)


def test_entropy_two_equal_is_one_bit():
    entry = _tok("a", math.log(0.5), [("a", 0.5), ("b", 0.5)])
    assert math.isclose(token_entropy_bits(entry), 1.0, abs_tol=1e-9)


def test_entropy_renormalises_partial_topk():
    # Only top-2 returned out of a larger vocab but unequal: 0.8 / 0.2 ->
    # renormalised stays 0.8/0.2 -> H = -(.8 log2 .8 + .2 log2 .2).
    entry = _tok("a", math.log(0.8), [("a", 0.8), ("b", 0.2)])
    expected = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
    assert math.isclose(token_entropy_bits(entry), expected, abs_tol=1e-9)


def test_entropy_empty_top_is_zero():
    assert token_entropy_bits({"token": "x", "logprob": -0.1, "top_logprobs": []}) == 0.0


def test_token_logprob_passthrough():
    assert token_logprob({"token": "x", "logprob": -1.234}) == -1.234


def test_perplexity_and_aggregate():
    agg = aggregate([-1.0, -2.0], [1.0, 3.0])
    assert agg["n_tokens"] == 2
    assert math.isclose(agg["mean_logprob"], -1.5)
    assert math.isclose(agg["mean_entropy_bits"], 2.0)
    assert math.isclose(agg["perplexity"], math.exp(1.5), rel_tol=1e-9)
    assert perplexity(None) is None
    assert aggregate([], [])["mean_logprob"] is None


# --------------------------------------------------------------------------
# reasoning-turn detection + extraction
# --------------------------------------------------------------------------
def _entry_with_logprobs():
    tool_turn = {
        "type": "ai",
        "tool_calls": [{"name": "chest_xray_classifier", "id": "1"}],
        "logprobs": [
            _tok("<tool_call>", math.log(0.99), [("<tool_call>", 0.99), ("x", 0.01)]),
            _tok("{", math.log(0.9), [("{", 0.9), ("y", 0.1)]),
        ],
    }
    answer_turn = {
        "type": "ai",
        "tool_calls": [],
        "logprobs": [
            _tok("Based", math.log(0.5), [("Based", 0.5), ("The", 0.5)]),
            _tok(" on", math.log(0.8), [(" on", 0.8), (" upon", 0.2)]),
            _tok("<|im_end|>", math.log(0.99), [("<|im_end|>", 0.99), ("z", 0.01)]),
        ],
    }
    return {
        "question_id": "q42",
        "trace": [
            {"type": "human", "content": "..."},
            tool_turn,
            {"type": "tool", "name": "chest_xray_classifier", "content": "..."},
            answer_turn,
        ],
    }


def test_reasoning_turn_is_last_non_tool_ai_turn():
    entry = _entry_with_logprobs()
    ai = [m for m in entry["trace"] if m["type"] == "ai"]
    assert reasoning_turn_index(ai) == 1  # the answer turn


def test_build_turns_positions_and_reasoning_mask():
    turns = build_turns(_entry_with_logprobs())
    assert [t["turn_idx"] for t in turns] == [0, 1]
    # tool turn: not reasoning
    assert turns[0]["is_reasoning_turn"] is False
    assert all(tok["is_reasoning"] is False for tok in turns[0]["tokens"])
    # answer turn: reasoning, but the trailing <|im_end|> control token excluded
    ans = turns[1]
    assert ans["is_reasoning_turn"] is True
    flags = [tok["is_reasoning"] for tok in ans["tokens"]]
    assert flags == [True, True, False]
    # positions are 0..n-1 and logprob survives at the right position
    assert [tok["pos"] for tok in ans["tokens"]] == [0, 1, 2]
    assert math.isclose(ans["tokens"][1]["logprob"], math.log(0.8))


def test_per_turn_agg_reasoning_only():
    rows = list(per_turn_agg_rows(_entry_with_logprobs(), label="base", seed=0))
    answer = [r for r in rows if r["is_reasoning_turn"]][0]
    # reasoning tokens = the two prose tokens (im_end excluded)
    assert answer["n_reasoning_tokens"] == 2
    expected_mean_lp = (math.log(0.5) + math.log(0.8)) / 2
    assert math.isclose(answer["reasoning_mean_logprob"], expected_mean_lp, rel_tol=1e-9)
    # entropy of [1 bit, ~0.72 bit] reasoning tokens
    h1 = 1.0
    h2 = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
    assert math.isclose(answer["reasoning_mean_entropy_bits"], (h1 + h2) / 2, rel_tol=1e-9)
    # tool turn contributes zero reasoning tokens
    tool_row = [r for r in rows if not r["is_reasoning_turn"]][0]
    assert tool_row["n_reasoning_tokens"] == 0


def test_per_turn_seq_carries_metadata():
    rows = list(per_turn_seq_rows(_entry_with_logprobs(), label="distractor_5", seed=2))
    assert all(r["query_id"] == "q42" for r in rows)
    assert all(r["condition"] == "distractor_5" and r["seed"] == 2 for r in rows)
    assert rows[0]["n_tokens"] == 2 and rows[1]["n_tokens"] == 3


def test_has_any_logprobs_false_when_capture_off():
    entry = {"question_id": "q", "trace": [{"type": "ai", "tool_calls": [], "content": "hi"}]}
    assert has_any_logprobs([entry]) is False
    turns = build_turns(entry)
    assert turns[0]["tokens"] == []  # no logprobs -> empty, no crash
