# Answer-extraction fix — findings

Re-extracted every existing ChestAgentBench rollout with a temp-0 LLM extractor
(Qwen2.5-7B-Instruct, a *different* model from the Qwen3-VL-8B under test) and a
deterministic closed-set (A–F) comparison, replacing the shipped
`extract_choice` line-counting heuristic. **Only answer extraction / `task_acc`
changes** — selection accuracy, entropy, and call counts are untouched (they come
from `trace` tool calls, not the answer text).

Artifacts: `ducx_noise/extract.py`, `extraction_cache/` (per-rollout),
`logs/extract_recompute/{results_extract.csv,summary_extract.md,records_*.jsonl}`.

## 1. The bug

`launch_over_chexbench.py:extract_choice` scans lines bottom-up for a line that is
*exactly* a single letter A–F; failing that it returns the **last**
`\b[A-F]\b` match in the whole text, **case-insensitively**. On this benchmark the
model never emits a bare-letter line — it writes prose like `**C) ...**` — so the
fallback fires **100%** of the time and routinely latches onto the article **"a"**:

| condition | old extracted | latched substring | new (LLM) | GT |
| --- | --- | --- | --- | --- |
| base | A | "…in patients with **a** history of IV drug use." | C | C |
| base | A | "…which is **a** hallmark of septic emboli…" | C | C |
| base | A | "…the presence of **a** cavitary lesion…" | C | C |
| base | A | "…characteristic mass effect of **a** fibrous tumor." | B | B |
| base | A | "…strongly support the diagnosis of **a** localized fibrous tumor" | B | B |

The model answered correctly; the heuristic recorded `A` and scored it wrong.

## 2. Old vs new `task_acc`

Mean over seeds (see `summary_extract.md`; **n=15/seed runs are smoke tests** —
trust the n≥50 rows):

| condition | seeds | n | task_acc_old | task_acc_new | Δ |
| --- | --- | --- | --- | --- | --- |
| **base** | 3 | 530 | 0.505 | **0.824** | +0.32 |
| **distractor_5** | 3 | 530 | 0.575 | **0.869** | +0.29 |
| distractor_2 | 3 | 45 | 0.578 | 0.867 | +0.29 |
| distractor_10 | 3 | 45 | 0.489 | 0.889 | +0.40 |
| distractor_20 | 1 | 50 | 0.680 | 0.920 | +0.24 |
| t2_unrel01 | 1 | 50 | 0.540 | 0.900 | +0.36 |
| t2_unrel03 | 1 | 50 | 0.580 | 0.960 | +0.38 |

The two large anchors (**n=500 each**): `base_seed0` 0.648→**0.94**,
`distractor_5_seed0` 0.658→**0.94**. (`distractor_50_seed0` has 0 scored entries
— empty/failed run, excluded.)

**Cross-check:** the independent deterministic rule reference from the Task-0
audit gave `base_seed0` = 0.930; the LLM extractor gives 0.940 — agreement within
1 pt, so the new number is not an artifact of either method.

### Does the distractor "flat line" survive?

The old sweep (~0.62→0.68→0.64) was an **artifact floor**: the article trap fired
~uniformly and pinned every condition near ~0.5–0.68. Under correct extraction:

- On the **reliable n=500 runs, base = distractor_5 = 0.94** — genuinely flat.
  The "distractors don't tank task accuracy" claim **survives and is now
  credible**, but at a true level of ~0.94, not ~0.65.
- The wider sweep (n=15–50) sits at 0.82–0.92 with no monotonic drop (if anything
  a slight rise), but those runs are underpowered (base seed std 0.11) — treat as
  "flat within noise," not a real upward trend.

Bottom line: the qualitative conclusion holds; the **absolute levels in the old
table were wrong by ~0.3** and must be replaced.

## 3. Differential attribution

Flips between old and new judgement, per condition (all 1300+ rollouts):

| condition | n | old→new **fixed** | **regressed** | abstain | old err-rate |
| --- | --- | --- | --- | --- | --- |
| base | 530 | 156 | 0 | 1 | 29.4% |
| distractor_2 | 45 | 13 | 0 | 0 | 28.9% |
| distractor_5 | 530 | 150 | 0 | 2 | 28.3% |
| distractor_10 | 45 | 18 | 0 | 0 | 40.0% |
| distractor_20 | 50 | 12 | 0 | 1 | 24.0% |
| t2_unrel01 | 50 | 19 | **1** | 1 | 38.0% |
| t2_unrel03 | 50 | 19 | 0 | 0 | 38.0% |

- **387 fixes, 1 regression** — the new extractor is strictly better in all but one
  rollout. Every fix is the article/last-token trap above.
- **Condition correlation:** old extraction error is a large, roughly-uniform
  ~24–40% across *all* conditions. It is mildly higher in the messier conditions
  (distractor_10 40%, unreliable 38%) than in base/distractor_5 (~29%) and
  distractor_20 (24%) — directionally consistent with "noisier reasoning → more
  mis-extraction," but the dominant effect is a near-uniform ~30% error floor, not
  a steep condition gradient. Either way the old cross-condition comparison was
  contaminated and had to be recomputed.

### The one regression (honest accounting)

`t2_unrel01_seed0`, GT=D: the model clearly opens with
`**D) Primary pleuropulmonary synovial sarcoma**`. The old rule returned `D` (its
*last* `\b[A-F]\b` token happened to be D — correct by luck). The LLM extractor
misread the long, heavily-qualified reasoning and returned `B`. This is the LLM
extractor's own error mode (~0.08% here: 1/1300) — long answers that state the
conclusion first and then enumerate/qualify other options. Net effect is still
+387/−1; the extractor need not be perfect, only far better and unbiased, which it
is. A future tweak (ask the extractor for the *first* stated final choice, or
ensemble with the rule backend) could remove it.

## 4. Reproducibility

`EXTRACTOR_VERSION = "extract-v1"`, temp 0 / top_p 1 / seed 0, fixed prompt, results
cached at `extraction_cache/{rollout_id}.json` (keyed on `question_id` + sha1 of the
generation text). Re-running reads the cache; bumping the version invalidates it.
Re-extract with `bash ducx_noise/run_extract_recompute.sh` inside a GPU allocation.
