# Noisy Tool Environments for Tool-Using Chest X-ray Agents — Technical Report

**Module:** `ducx_noise/` (DUCK repo) · **Date:** 2026-06 · **Status:** Tasks 1–3 complete, validated end-to-end on real models (Qwen3-VL-8B + MedRAX tools, L40S).

---

## 1. Motivation

DUCK/DUCX audits *fairness* of tool-using CXR agents. This module adds an orthogonal axis: **robustness of tool selection under a noisy tool environment**. Real deployments expose agents to imperfect tool registries — near-duplicate tools, mislabeled descriptions, flaky backends, and irrelevant "distractor" tools. We provide an opt-in layer that injects controlled noise into the agent's tool set and an evaluator that quantifies how the noise degrades tool selection.

Design contract: **zero change to default DUCX behaviour.** Noise is config-driven; with no config the module is never invoked.

---

## 2. Where it plugs in

From the Task 0 code study, the agent's "environment" is the tool list passed to `Agent` in `main.py:initialize_agent`, between building `tools_dict` and constructing `Agent(...)`. Both the eval (`launch_over_chexbench.py`) and the Gradio demo route through this function, so it is the single clean injection point.

```
tools_dict (real tools)  ──►  apply_noise(tools, cfg)  ──►  NoisyEnv.tools  ──►  Agent(model, tools=...)
                                       │
                                       └─►  ground-truth manifest (JSON)  ──►  evaluator
```

The agent loop (`medrax/agent/agent.py`) is untouched: it consumes `BaseTool` instances and dispatches via `tool.invoke(args)`. Every noise wrapper is a `BaseTool` subclass that preserves `name`/`description`/`args_schema`, so `model.bind_tools(...)` (OpenAI function-calling schema) and dispatch are transparent to the noise.

---

## 3. Task 1 — Noisy Environment

### 3.1 Entry point

`apply_noise(base_env: list[BaseTool], noise_config) -> NoisyEnv`

`NoisyEnv` carries `.tools` (drop-in for the agent) and `.manifest` (`{name: ToolProvenance}` ground truth). The transformation is an ordered pipeline of registered transforms (`registry.PIPELINE_ORDER`).

### 3.2 Noise types

| # | Type | Mechanism | Config block |
|---|------|-----------|--------------|
| 1.1 | **Distractor** | Adds N fake tools (LLM-generated names/descriptions, cached; deterministic fallback). Never calls a real model — returns a plausible-but-useless response. | `distractor` |
| 1.2 | **Unreliable** | Wraps a *real* tool so each call fails with prob `p_fail` (`error`/`empty`/`timeout`). Schema unchanged; only reliability changes. Real-model case (not a mock). | `unreliable` |
| 1.3 | **Redundant** | Adds functional duplicates of real tools — different name/schema, **same backend instance** (no extra weights) — to create selection competition. | `redundant` |
| 1.4 | **Description corruption** | Rewrites descriptions to be `vague`/`ambiguous`/`misleading` (LLM or rule-based). Original preserved in manifest for Φ comparison. | `description_corruption` |
| 1.5 | **Schema noise** | Renames params, injects redundant optional params, shuffles order — all callability-preserving via an exposed→inner arg map. | `schema_noise` |
| (+) | **Drop (D operator)** | Statically removes named tools, runs last. Enables N→D reversibility studies (drop all distractors ⇒ restore E₀). | `drop` |

### 3.3 Mechanics

- `ProxyTool(BaseTool)` — wraps an inner tool (real tool *or* another ProxyTool, so noise composes). Handles arg remapping, description override, and reliability rolls. Reliability is reproducible: per-call RNG from `(seed, tool_name, call_index)`.
- `DistractorTool(BaseTool)` — standalone fake; deterministic canned response keyed by name.
- `make_args_schema(...)` — builds a noised pydantic schema + the exposed→inner mapping (`pydantic.create_model`).

### 3.4 Ground truth (`ToolProvenance`)

Per exposed tool: `source ∈ {real, distractor, redundant, unreliable}`, `noises[]` (every transform applied), `is_noise_tool` (selecting it = mis-selection; true for distractor/redundant), `base_name`, `meta` (p_fail, arg_map, original_description, …). Serialized to a JSON manifest the evaluator joins against.

### 3.5 Reproducibility & cost

- Single top-level `seed` seeds all transforms; YAML/JSON round-trip.
- LLM-backed generation reuses the existing OpenAI-compatible client (no new dependency) and **caches to JSON**; offline/test runs use deterministic fallbacks.
- Sweeps: `count`/`p_fail`/`level` are scalar knobs; ready-made configs for distractor 0/2/5/10/20/50 and combined tables.

---

## 4. Task 2 — Evaluation metrics

`ducx_noise/metrics.py` post-processes the per-question `trace` DUCX already logs + the noise manifest. **The eval loop is not modified**; an opt-in hook (`--noise-metrics`) emits a CSV row after a run.

Per-run metrics: `tool_selection_accuracy` (selections hitting a real tool), `noise_tool_misselection_rate` = p(selected ∈ noise tools), `selection_entropy_bits` (action-space entropy), `avg_tool_calls_per_question`, `task_accuracy`, `unknown_tool_rate` (hallucinated names), and per-tool confusion (`top_confused_noise_tool`, `--confusion-out`).

`aggregate` averages across seeds per condition and emits a base-vs-noisy markdown + CSV summary.

---

## 5. Task 3 — Acceptance

- **One command:** `ducx_noise/run_noise_experiment.sh` — serves vLLM, runs the eval per (config × seed) with `--noise-config`, computes + aggregates metrics, prints selection accuracy and mis-selection.
- **Regression:** `apply_noise` with no/all-disabled config is an identity transform (same objects, same order) — explicit test.
- **Unit tests:** one per noise type + config round-trip + seed reproducibility + metrics + drop reversibility. **23 tests pass** (no GPU/LLM needed).
- **Docs:** `ducx_noise/README.md`.

Integration surface: `launch_over_chexbench.py --noise-config <yaml> [--noise-metrics]`; library hook `initialize_agent(..., noise_config=, noise_manifest_path=)`.

---

## 6. Results

End-to-end on ChestAgentBench (EuroRAD) with Qwen3-VL-8B driving the MedRAX agent (8 real GPU tools), single seed.

| condition | sel_acc | misselect | entropy (bits) | calls/q | task_acc |
|-----------|--------:|----------:|---------------:|--------:|---------:|
| base | 1.000 | 0.000 | 2.32 | 6.12 | 0.62 |
| distractor_2 | 0.814 | 0.186 | 2.44 | 5.80 | 0.66 |
| distractor_5 | 0.691 | 0.309 | 3.03 | 7.32 | 0.68 |
| distractor_10 | 0.768 | 0.232 | 3.22 | 7.58 | 0.64 |
| distractor_10 + unreliable p=0.1 | 0.667 | 0.333 | 3.54 | 8.84 | 0.54 |
| distractor_10 + unreliable p=0.3 | 0.648 | 0.352 | 3.62 | 11.48 | 0.58 |

**Findings**

1. **Distractors induce mis-selection and raise action-space entropy.** mis-selection 0 → 0.19 → 0.31; entropy 2.32 → 3.03 monotonically with distractor count (up to 5).
2. **Unreliable real tools compound the damage.** Layered on distractor_10, raising `p_fail` 0→0.1→0.3 increases mis-selection (0.23→0.33→0.35), inflates tool calls per question (7.58→8.84→**11.48**, retries on failure), and lowers task accuracy (0.64→0.54). Process-level cost is visible even when end-task accuracy moves little.
3. **Caveat (single seed):** distractor_5 mis-selection (0.31) exceeds distractor_10 (0.23) — non-monotone at small sample. Multi-seed averaging (≥3 seeds, larger `--max-cases`) is needed before reading the distractor curve as monotonic.

A separate 2-case smoke confirmed the same direction (base mis-select 0.00 → distractor_5 0.53).

---

## 7. Validation summary

| Check | Result |
|-------|--------|
| Default behaviour unchanged (no config) | ✅ identity-transform regression test |
| Per-noise-type unit tests | ✅ 5 types + drop |
| Config YAML/JSON round-trip + seed reproducibility | ✅ |
| Metrics correctness on synthetic traces | ✅ |
| End-to-end on real agent + real tools | ✅ base vs noisy sweep |
| Full suite | ✅ 23 passed |

Footprint: ~1.9k LOC under `ducx_noise/`; only new dependency is dev-only `pytest`.

---

## 8. Limitations & next steps

- **Statistical power:** current sweeps are single-seed; run 3+ seeds × larger case counts to smooth curves and report mean ± std (the metrics aggregator already computes std).
- **No gold per-question tool label** in ChestAgentBench: "selection accuracy" is defined as *not selecting injected noise tools*, not against an oracle tool. A gold-tool annotation would enable true precision/recall.
- **Compositional environments:** the transform registry is the seam (`registry.compose` is a stub); stacking independently-configured environments is future work.
- **Description/distractor LLM realism:** the deterministic fallback is intentionally generic; enable `llm_generate: true` for harder, more on-topic noise (results cached).
