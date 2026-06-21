# ducx_entropy — reasoning likelihood & entropy

A metric orthogonal to the tool-**selection** entropy in
[`ducx_noise.metrics`](../ducx_noise/metrics.py). For each generated token it
records two distinct signals:

| signal | meaning | high value means |
| --- | --- | --- |
| **likelihood** `logprob` | `log p(token \| prefix)` of the *actually emitted* token (natural log) | the model was confident in what it said |
| **entropy** `entropy_bits` | `H = −Σ_v p(v) log₂ p(v)` over the *whole next-token distribution* at that step (bits) | the distribution was flat / the model was unsure |

These are **not** redundant: a token can be high-likelihood yet high-entropy
(picked the top option from a flat distribution) or vice-versa. Both are emitted
so you can decide which one answers the question.

**Hypothesis it serves:** reasoning entropy may react to environment noise
*earlier / more sensitively* than selection entropy — the model picks the right
tool but its reasoning is already "struggling" — which would explain why
distractors raise selection confusion without hurting `task_acc`.

## What counts as "reasoning"?

The agent runs **Qwen3-VL-8B-Instruct** (no `<think>` channel; intermediate
tool-call turns emit *empty* `content`). The reasoning segment is therefore
defined as the model's **final-answer prose** — the terminal assistant message,
which carries its verbalised justification (~1.3k chars). Its tokens are the
reasoning tokens; tool-call-turn tokens are not. See
[`reasoning.py`](reasoning.py) — `REASONING_MODE = "final_answer_prose"`.

> If a thinking variant is ever used, swap the span finder in `reasoning.py` for
> a `<think>…</think>` splitter (use the tokenizer with
> `return_offsets_mapping=True` for char→token alignment) and flip
> `REASONING_MODE`. Nothing downstream changes.

## Path A (vLLM logprobs) vs Path B (HF full-vocab) — precision

This module consumes **path A**: rollouts re-run with vLLM `logprobs=True,
top_logprobs=K`. vLLM returns only the **top-K** logprobs per step, so
`token_entropy_bits` is a **top-K renormalised approximation** (renormalise the K
returned probabilities to sum to 1, take their entropy). It **systematically
under-estimates** the true entropy (ignores the tail) — good for *trends across
conditions*, not for an exact paper number. For exact entropy, recompute over the
full vocabulary with an HF teacher-forcing forward pass (path B) — not
implemented here; path A was chosen because the multimodal + hermes-tool prompt
is reconstructed exactly by vLLM, whereas a byte-for-byte HF re-prompt is fragile.

Units: entropy in **bits** (matches `selection_entropy_bits`); `perplexity =
exp(−mean_logprob)` from natural-log logprobs.

## Three granularities

Produced by [`extract.py`](extract.py):

- **(A) per-token** — `pos, token_id, token_str, logprob, entropy_bits, is_reasoning`.
  (`token_id` is `null` on path A; vLLM logprobs carry the token string, not its id.)
- **(B) per-turn sequence** — one record per assistant turn holding that turn's
  ordered token list (the A rows embedded).
- **(C) per-turn aggregate** — `mean_logprob, perplexity, mean_entropy_bits`
  plus the reasoning-only versions `reasoning_mean_logprob,
  reasoning_mean_entropy_bits, n_reasoning_tokens`.

Every record carries `query_id` (= `question_id`), `condition`, `seed`.

## Usage

### 1. Re-run rollouts capturing logprobs (opt-in; default behaviour unchanged)

```bash
python launch_over_chexbench.py ... --noise-config CFG.yaml --capture-logprobs 20
```

`--capture-logprobs K` threads `logprobs=True, top_logprobs=K` into the agent's
`ChatOpenAI` ([main.py](../main.py)) and persists the per-token logprobs in the
run-log trace (`serialize_messages`). Omit it → no logprobs, byte-for-byte the
old behaviour.

### 2. Extract the three granularities (one command — Task-3 acceptance)

```bash
python -m ducx_entropy.cli extract \
    --run-dir logs/entropy_distractor_5_seed0 \
    --label distractor_5 --seed 0 --out-dir logs/entropy_experiment/entropy
# -> {label}_seed{seed}_pertoken.jsonl   (granularity A embedded in B)
# -> {label}_seed{seed}_perturn.jsonl    (granularity C)
```

`--run-dir` auto-picks the latest run log; or pass `--run-log FILE` directly. If
the log has no captured logprobs it warns and emits empty-token rows (no crash).

### 3. Compare reasoning entropy vs selection entropy

```bash
python -m ducx_entropy.analyze \
    --perturn logs/entropy_experiment/entropy/*_perturn.jsonl \
    --selection-csv logs/entropy_experiment/selection_results.csv \
    --out-md  logs/entropy_experiment/reasoning_vs_selection.md \
    --out-csv logs/entropy_experiment/reasoning_vs_selection.csv
```

Token-weighted reasoning entropy/logprob per condition×seed → per-condition
mean ± std across seeds, joined with `selection_entropy_bits` / `misselect` /
`task_acc` from [`ducx_noise.metrics`](../ducx_noise/metrics.py).

### Full pipeline in one shot

[`run_entropy_experiment.sh`](run_entropy_experiment.sh) serves vLLM, re-runs each
`(config, seed)` with `--capture-logprobs`, computes the selection row, extracts
the three granularities, and prints the reasoning-vs-selection table:

```bash
CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
SEEDS="0 1 2" MAX_CASES=20 TOPK=20 bash ducx_entropy/run_entropy_experiment.sh
```

## Verifying the setup

[`smoke_test.sh`](smoke_test.sh) (needs 1 GPU) starts the Instruct server and
confirms, via both the raw OpenAI client and langchain `ChatOpenAI`, that
per-token logprobs come back with the final answer + tool calls — the capture
hook's prerequisite.

## Tests

```bash
python -m pytest ducx_entropy/tests/ -q
```

CPU-only: top-K entropy on known distributions, perplexity/aggregate arithmetic,
reasoning-turn detection, and end-to-end extraction with correct position
alignment + `is_reasoning` labelling (control tokens excluded, tool turns not
reasoning).

## Non-invasiveness

Nothing here changes the rollout/eval/noise pipeline. `--capture-logprobs`
defaults off; `serialize_messages` only adds a `logprobs` key when the field is
present; extraction/analysis are pure post-processing over the run log, exactly
like the selection metrics.
