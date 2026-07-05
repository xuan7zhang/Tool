# Tool-Space Optimization — Multi-Tool Probe + Training-Free Retriever

Goal: optimize the tool space presented to a VLM agent. Built on the distractor
study's pivotal finding — the model **never mis-selects** a distractor, yet
accuracy still drops because irrelevant tool descriptions **pollute context**. So
tool-space optimization is a per-query **pruning / context-hygiene** problem, not
a selection-accuracy problem. Objective: task accuracy. Optimizer: training-free
external retriever.

## Multi-tool probe suite (`ducx_noise/multitool_probe.py`)

Tool-necessary MCQs where **different questions require different real tools**, so
pruning is consequential (the single-tool probe could not exercise selection):

- **CLS** — "which pathology has the HIGHEST predicted probability?" →
  requires `chest_xray_classifier` (GT = classifier argmax).
- **SEG** — "which structure has the LARGEST area?" →
  requires `chest_xray_segmentation` (GT = max area_cm2).

Each record carries `required_tool` = the retrieval ground truth. n=120 (CLS 76 /
SEG 44), Qwen3-VL-8B, fp16, greedy, seed 0, Killarney L40.

## Oracle bounds (`run_multitool_bounds.sh`) — the headroom

| condition | CLS | SEG | overall |
|---|---|---|---|
| no_tool | 0.329 | 0.614 | 0.433 |
| **oracle** (per-query only-needed tool) | 0.829 | 0.977 | **0.883** |
| all_real (both real tools exposed) | 0.711 | 0.932 | 0.792 |
| all_distract (both real + 5 fake) | 0.697 | 0.977 | 0.798 |

- Tool is **necessary**: oracle lifts overall 0.433 → 0.883 (CLS +0.50, SEG +0.36).
- **Irrelevant REAL tools pollute**: all_real is −0.09 below oracle (CLS −0.12,
  SEG −0.05). Merely exposing the unused tool degrades the harder CLS task.
- **First irrelevant tool does the damage**: adding 5 fake distractors on top of
  both real tools moves nothing (0.792 → 0.798) — threshold-like, echoing the
  distractor-count U-shape. `analysis/multitool/bounds.{csv,png}`.

## Training-free retriever (`ducx_noise/retriever.py`)

Select, per query, the relevant tool from a 7-tool registry (2 real + 5
distractors). Two methods:

| retriever | top1 acc | recall@1 | distractor leak | downstream acc | recovery vs polluted |
|---|---|---|---|---|---|
| lexical (TF-IDF) | 1.00 | 1.00 | 0.00 | 0.883 | +0.085 |
| LLM-router (the model itself) | 1.00 | 1.00 | 0.00 | 0.883 | +0.085 |

Both route perfectly and reproduce the oracle grouping exactly (classifier:76 +
segmentation:44), so pruning to the routed tool **recovers the full oracle gap**
(all_distract 0.798 → 0.883). `analysis/multitool/retriever.{csv}`,
`retriever_recovery.png`.

## The key insight (dissociation)

Routing is **trivial** on this suite — the query + option vocabulary
(pathology names vs anatomy names) unambiguously reveal the tool domain, so even a
zero-dependency lexical retriever hits top1=1.0, and **the model itself
(LLM-router) also hits 1.0**. Yet when all tools are present the model degrades
(all_real 0.792 < oracle 0.883).

> The model **knows** which tool it needs but **cannot suppress** the irrelevant
> ones in context. Tool-space optimization's value is therefore not better
> retrieval — it is **enforced external pruning**: removing the tools the model
> itself would deprioritize recovers +0.09 (concentrated on the harder CLS, +0.12).

This argues that self-routing prompting ("pick your tools first") is insufficient;
the pruning must be enforced outside the model's context.

## Limitations / next

- Routing is easy here because the task domain is lexically obvious. A harder
  suite (queries whose required tool is not domain-transparent) would make the
  retriever-quality axis non-trivial; here it is saturated.
- SEG "largest area" is only weakly tool-necessary (no_tool 0.614); CLS is the
  clean probe. A genuinely per-image second tool (e.g. grounding localization)
  would sharpen the SEG arm.
- Single seed, n=120. Downstream retriever accuracy == oracle by construction
  (perfect routing); a harder suite would require an explicit retriever-pruned
  downstream run distinct from oracle.
- Budget/λ (accuracy − cost) axis deferred (objective was pure accuracy).

## Artifacts

- `ducx_noise/multitool_probe.py`, `run_multitool_bounds.sh`
- `ducx_noise/retriever.py`, `run_retriever_eval.sh`
- `analysis/multitool/{bounds,retriever}.csv`, `{bounds,retriever_recovery}.png`
