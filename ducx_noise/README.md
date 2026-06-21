# ducx_noise — Noisy Tool Environments for DUCX agents

Opt-in module that transforms the agent's **tool environment** (the list of
tools passed to the agent) into a *noisy* one, plus an evaluator that measures
how the noise affects tool selection. Everything is config-driven and
reproducible; **with no config supplied, DUCX behaves exactly as before.**

```
ducx_noise/
  config.py        # NoiseConfig dataclasses + YAML/JSON loader (seed lives here)
  provenance.py    # ToolProvenance ground-truth tags (real|distractor|redundant|unreliable)
  wrappers.py      # ProxyTool / DistractorTool / schema builder
  generation.py    # LLM-backed distractor & description generation (cached) + fallbacks
  transforms.py    # the 5 noise transforms, each registered
  registry.py      # transform registry + PIPELINE_ORDER (compositional-env seam)
  apply.py         # apply_noise(base_env, noise_config) -> NoisyEnv  (the entry point)
  metrics.py       # Task 2 selection metrics + base-vs-noisy aggregation
  configs/         # example YAML configs incl. a distractor sweep (0/2/5/10)
  tests/           # unit tests (no GPU/LLM needed)
  run_noise_experiment.sh   # one-command serve -> eval sweep -> metrics
  # --- compositional environment (K operator) ---
  atoms.py         # DenseNet-split atoms: cxr_encoder (image->embedding) + cxr_embedding_classifier
  macro.py         # MacroTool: in-memory chain pipe; intermediate never exposed to the agent
  mine_pairs.py    # typed-IO registry + mine_pairs (IO-compat x non-semantic x low p_b)
  eval_compose.py  # 3-condition eval (p_b/p_c/p_chain) + criteria + macro-selection rate
  run_compose_experiment.sh # one-command mine -> macro -> 3-condition eval -> criteria table
```

## Compositional environment (K operator / macro-tools)

Tests the hypothesis that a pair `(t_a, t_b)` the model is bad at *separately* (because
`t_a`'s output is a **non-semantic intermediate** it can't transcribe) becomes good
*end-to-end* as a macro `t_c = t_b ∘ t_a` that hides the intermediate.

The candidate chain splits the **real** `chest_xray_classifier` DenseNet (parity-exact):
`cxr_encoder` (image → 1024-d embedding) → `cxr_embedding_classifier` (embedding → 18 probs).
The 1024-float embedding cannot be faithfully emitted by the VLM as a tool arg, so `t_b`-alone
and self-chaining collapse while the macro pipes it in memory.

```python
from ducx_noise import apply_noise
# expose only the macro (p_c condition); embedding hidden inside
env = apply_noise([], {"compose": {"enabled": True, "atom_groups": ["cxr_densenet"],
      "macros": [{"name": "cxr_macro", "chain": ["cxr_encoder", "cxr_embedding_classifier"]}],
      "exclusive": True, "device": "cuda"}})
```

**Pair mining:** `python -m ducx_noise.mine_pairs --out pairs.csv` ranks IO-compatible pairs whose
intermediate is non-semantic (the DenseNet chain scores ϕ=1.0; semantic image-path chains are dropped).

**Three-condition eval** (one query batch, one backbone, identical judgment; only tool exposure changes):

| condition | tools exposed | measures |
|---|---|---|
| `p_b` | only `cxr_embedding_classifier`; question carries `t_a`'s **real** embedding | intermediate→disease (model must route the embedding) |
| `p_c` | only the macro `cxr_macro` | end-to-end (embedding hidden) |
| `p_chain` | both atoms | model self-chains |
| `both_macro` | atoms + macro | macro-selection rate |

```bash
# one command (inside a GPU allocation): mine -> macro -> 3-condition eval -> criteria table
MAX_CASES=20 srun --jobid=<ID> --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
    bash ducx_noise/run_compose_experiment.sh
# -> logs/compose_experiment/{mined_pairs.csv, summary.md, runs.json}
```

Reports task_acc per condition (the headline p), Δ_comp = p_c − p_chain, macro-selection rate, and
the three criteria (weak `p_c>p_a·p_b` · mid `p_c>max(p_b,p_chain)` · strong `p_b≈chance & p_c high`).
`p_a` is the **operational** valid-intermediate rate (not task success) — `t_a` alone cannot answer the
MCQ, so a task-success p_a is ill-defined; primary report is p_b/p_c/p_chain. Tool selection is logged
per question (reused from the noise metrics) so we confirm the macro was actually chosen.

## Concept

```python
from ducx_noise import apply_noise
env = apply_noise(base_tools, "ducx_noise/configs/distractor_5.yaml")
agent_tools = env.tools                 # drop-in replacement for the tool list
env.save_manifest("manifest.json")      # ground truth for the evaluator
```

`apply_noise(base_env, noise_config) -> NoisyEnv` is the only entry point. The
returned `NoisyEnv` has `.tools` (the transformed list) and `.manifest` (a
`{name: ToolProvenance}` map). The transformation runs as an ordered pipeline of
registered transforms (`registry.PIPELINE_ORDER`); the registry is the seam for
future **compositional environments** (`registry.compose`, not implemented yet).

## Noise types

| Type | Config block | What it does |
|------|--------------|--------------|
| **1.1 Distractor** | `distractor` | Adds `count` fake tools (LLM-generated names/descriptions, cached; deterministic fallback). They never call a real model — return a plausible-but-useless response. Tagged `source=distractor`. |
| **1.2 Unreliable** | `unreliable` | Wraps real tools so each call fails with prob `p_fail` (`failure_mode: error\|empty\|timeout`). Schema is unchanged; only reliability changes. Tagged `source=unreliable`. |
| **1.3 Redundant** | `redundant` | Adds `copies_per_tool` functional duplicates of real tools (same backend instance — no extra weights), different name. Tagged `source=redundant`. |
| **1.4 Description corruption** | `description_corruption` | Rewrites descriptions to be `vague\|ambiguous\|misleading` (LLM or rule-based). Original kept in `meta.original_description` for Φ comparison. |
| **1.5 Schema noise** | `schema_noise` | Renames params, injects redundant optional params, shuffles order — all callability-preserving (exposed names map back to the real ones). |

### Ground truth

Every exposed tool gets a `ToolProvenance`:
`source ∈ {real, distractor, redundant, unreliable}`, `noises` (all transforms
applied), `is_noise_tool` (selecting it = mis-selection; true for
distractor/redundant), `base_name`, and `meta`. Saved to a JSON manifest the
evaluator reads.

## Config (YAML or JSON, with `seed`)

```yaml
seed: 0
shuffle_tools: true          # seeded final-order shuffle (hides position cues)
label: distractor_5
distractor:
  enabled: true
  count: 5
  llm_generate: false        # true -> use the agent LLM to author distractors (cached)
  cache_path: ducx_noise/cache/distractors.json
  style: plausible           # plausible | misleading
unreliable: {enabled: false}
redundant: {enabled: false}
description_corruption: {enabled: false}
schema_noise: {enabled: false}
```

Each block has an `enabled` switch, strength parameter(s), and an optional
`tools:` whitelist (null = all real tools). See `configs/` for ready-made
examples and the distractor **sweep** (`distractor_2/5/10.yaml` + `base.yaml`).

## Running with the agent (opt-in)

The eval entrypoint gained two flags; omitting `--noise-config` = unchanged DUCX:

```bash
python launch_over_chexbench.py \
  --model qwen3-vl-8b --data-file data/chestagentbench/metadata.jsonl \
  --log-prefix qwen3vl8b-distractor5 --max-cases 20 --llm-parse \
  --noise-config ducx_noise/configs/distractor_5.yaml \
  --noise-metrics        # also writes logs/<prefix>/noise_metrics.csv
```

`initialize_agent(..., noise_config=..., noise_manifest_path=...)` is the
library hook (used by both the eval and the Gradio demo).

## Task 2 — metrics

Post-processes the run log (the per-question `trace` DUCX already writes) + the
noise manifest. No change to the eval loop.

```bash
# one run -> a CSV row
python -m ducx_noise.metrics row --run-log <run>.json --manifest <manifest>.json \
    --label distractor_5 --seed 0 --out results.csv
# aggregate base vs noisy across seeds -> summary table
python -m ducx_noise.metrics aggregate --in results.csv \
    --summary-out summary.md --summary-csv summary.csv
```

Metrics: `tool_selection_accuracy` (hit a real tool), `noise_tool_misselection_rate`
(`p(selected ∈ noise tools)`), `selection_entropy_bits` (action-space entropy),
`avg_tool_calls_per_question`, `task_accuracy`, `unknown_tool_rate`, and per-tool
confusion (`top_confused_noise_tool` + `--confusion-out`).

## Task 3 — one command (base vs noisy sweep)

Inside a GPU allocation (serves vLLM, runs the eval per config×seed, aggregates):

```bash
CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
SEEDS="0 1 2" MAX_CASES=20 bash ducx_noise/run_noise_experiment.sh
# -> logs/noise_experiment/{results.csv, summary.md, summary.csv}
```

It prints a table you can read base-vs-noisy directly, e.g.:

```
| condition    | seeds | sel_acc | misselect | entropy | calls/q | task_acc |
| base         | 3     | 1.0     | 0.0       | 1.10    | 2.3     | 0.33     |
| distractor_5 | 3     | 0.62    | 0.38      | 2.05    | 2.7     | 0.30     |
```

## Answer extraction (`extract.py`)

The shipped scorer (`launch_over_chexbench.py:extract_choice`) reads the final
A–F choice with a line-counting + last-`\b[A-F]\b` heuristic. On ChestAgentBench
the model never emits a bare-letter line, so the case-insensitive fallback fires
100% of the time and latches onto the article **"a"** in prose like
"…presence of **a** cavitary lesion" — recording correct answers as wrong. This
inflated/deflated `task_acc` by ~0.3 (see `extraction_fix_findings.md`).

`extract.py` replaces **only the answer-extraction step** with two separate
stages — deliberately *not* one LLM judge:

1. **LLM extract (temp 0).** A greedy model reads the full generation and returns
   the model's *final* A–F choice (or `NONE`/abstain — it never guesses). It must
   be a model **other than the one under test** to avoid self-grading; default is
   `Qwen2.5-7B-Instruct` (text-only, a different model from the Qwen3-VL-8B agent),
   served on a spare GPU via vLLM. Greedy + fixed prompt + `EXTRACTOR_VERSION` +
   on-disk cache make it reproducible.
2. **Deterministic compare.** ChestAgentBench answers are a **closed set** (single
   letter A–F), so the extracted letter is normalised and exact-matched against the
   ground truth — no judge needed. An open-text **LLM-judge branch** exists in the
   design for free-form answers but is not exercised by this benchmark.

A `RuleExtractor` (regex, no server) is the offline/test fallback and a cross-check
(it agreed with the LLM to within 1 pt on the n=500 runs); the LLM is the default
because it also handles "echo option, reject it, choose another" cases the regex
cannot. Sampling temperature for the *main* experiment is unchanged (0.7) — only
answer extraction is replaced. `sel_acc` / entropy / `calls` are independent of
the answer text and are not recomputed.

```bash
# inside a GPU allocation: serve Qwen2.5-7B, re-extract every run, rebuild table
srun --jobid=<ID> --overlap --ntasks=1 bash ducx_noise/run_extract_recompute.sh
#   -> extraction_cache/, logs/extract_recompute/summary_extract.md
```

Cache: `extraction_cache/{question_id}_{sha1(model_answer)}.json` holds the new
answer, old/new correctness, and extractor version; reruns read it.

## Tests

```bash
python -m pytest ducx_noise/tests/ -q        # no GPU/LLM required
```

`test_extract.py` covers the "XXX is a XXX" article traps (old rule wrong, new
extractor right), closed-set normalisation/compare, abstain, the reject-then-choose
case (rule limitation + LLM fix via a fake client), caching, and run recompute.

Includes a **regression test** that `apply_noise` with no/all-disabled config is
an identity transform (same objects, same order) — the guarantee that existing
experiments are unaffected. Each noise type has its own unit test, plus config
round-trip, seed reproducibility, and metrics tests.

## Engineering notes

- No new heavy dependency: distractor/description LLM generation reuses the
  existing OpenAI-compatible client (the same endpoint driving the agent); YAML
  uses the already-present `pyyaml`. `pytest` is dev-only.
- LLM generation results are cached to JSON (`cache_path`) — never re-called on
  reruns; offline/test runs use deterministic fallbacks.
- Reproducibility: a single top-level `seed` seeds all transforms; unreliable
  rolls are derived from `(seed, tool_name, call_index)`.
- Compositional environments: register new transforms in `registry.py`; the
  multi-environment `compose()` is intentionally a stub for now.
