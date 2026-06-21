# DUCX — Noise Robustness, Reasoning Entropy & Tool Analysis

Summary of the DUCX experiment suite on top of the MedRAX chest-X-ray tool-using
agent. Everything here is **opt-in** and **post-hoc**: none of it changes rollout
generation or the existing eval — the noise environment is applied only when a
config is passed, and all metrics are computed from the saved run-log `trace`.

- **Agent / model:** Qwen3-VL-8B-Instruct served on vLLM (hermes tool-call
  parser), 8 real tools, ChestAgentBench (2,500 questions).
- **Modules:** [`ducx_noise/`](ducx_noise/README.md) (noisy tool environments +
  selection metrics + answer-extraction fix), [`ducx_entropy/`](ducx_entropy/README.md)
  (per-token reasoning likelihood/entropy).

---

## Key findings

### 1. The model is robust to distractor tools (selection is perturbed, task accuracy is not)
On 500 questions, adding 5 fake "distractor" tools heavily perturbs **tool
selection** but leaves **task accuracy unchanged**:

| condition | task_acc | sel_acc | misselect | sel_entropy | calls/q |
|---|---|---|---|---|---|
| base | 0.648 | 1.00 | 0.00 | 2.34 | 5.67 |
| distractor_5 | 0.658 | 0.62 | 0.376 | 3.19 | 7.49 |

Mechanism (from traces): 76% of questions called a distractor, yet questions that
called one scored **0.661** vs **0.647** for those that didn't — calling a fake
tool does **not** hurt. All distractor-callers also called a real tool, and the
synthetic distractors return null ("no actionable information") which the model
simply ignores. The noise is trapped in the *selection* layer and never reaches
the answer.

### 2. The old `task_acc` numbers were an answer-extraction artifact
The shipped scorer (`launch_over_chexbench.py:extract_choice`) falls back to "last
case-insensitive `\b[A-F]\b` in the prose" and latches onto the article **"a"**
("presence of **a** cavitary lesion" → records `A`). This fires ~100% of the time
and mis-scores ~24–40% of rollouts, roughly uniformly across conditions.

Fix ([`ducx_noise/extract.py`](ducx_noise/extract.py)): a temp-0 **Qwen2.5-7B**
extractor (a *different* model from the one under test — no self-grading) reads the
final A–F letter, then a deterministic closed-set exact match decides correctness.
The LLM only *extracts the letter*; it does **not** judge correctness.

Corrected (n=500 anchor): base `0.648 → 0.94`, distractor_5 `0.658 → 0.94`
(387 fixes / 1 regression). **base == distractor_5 == 0.94** — the "distractors
don't tank accuracy" claim holds, just at ~0.94 not ~0.65, and the apparent +0.01
"improvement" from distractors was pure scoring noise.

### 3. Reasoning entropy is *less* sensitive to noise than selection entropy
Per-token likelihood/entropy over the final-answer prose ([`ducx_entropy/`](ducx_entropy/README.md),
path A = vLLM top-K logprobs). Across the distractor sweep, **selection entropy
rises monotonically while reasoning entropy stays flat**:

| level | select_H | reason_H | misselect |
|---|---|---|---|
| base | 2.39 | 0.448 | 0.00 |
| distractor_2 | 2.41 | 0.473 | 0.06 |
| distractor_5 | 2.72 | 0.465 | 0.20 |
| distractor_10 | 3.14 | 0.443 | 0.26 |

This contradicts the original hypothesis (that reasoning entropy would be the
*earlier* signal): the answer-generation layer is decoupled from selection-layer
noise. Caveat: reasoning = final-answer prose, which is templated; "flat entropy"
partly reflects formatting, not necessarily robust reasoning.

### 4. Per-tool profile (8B, base, 500 q)
"Call success" (does the tool throw?) is **100% for every tool — no signal**;
the tools are stable local models. The discriminating axes are **avoidance/coverage**
and **repeat calls**:

| tool | output | coverage | calls | calls/used-q | q with ≥3 calls |
|---|---|---|---|---|---|
| image_visualizer | display/path | 100% | 894 | 1.79 | 97 |
| chest_xray_classifier | probs (readable) | 82% | 664 | 1.61 | 63 |
| chest_xray_report_generator | report (readable) | 73% | 555 | 1.53 | 54 |
| chest_xray_expert | text (readable) | 66% | 343 | 1.04 | 0 |
| chest_xray_segmentation | mask + geometry | 42% | 292 | 1.38 | 20 |
| llava_med_qa | text (readable) | 17% | 88 | 1.04 | 0 |
| dicom_processor | preprocessing | 0% (1) | 1 | 1.00 | 0 |
| **xray_phrase_grounding** | **bbox (non-semantic)** | **0%** | **0** | — | — |

Sharpest signal: the **bbox/non-semantic** `xray_phrase_grounding` tool is
**never used** — consistent with the hypothesis that tools with non-readable
output get avoided. A strong-vs-weak backbone comparison (Qwen3-VL-2B vs 8B,
`run_toolprobe_eval.sh`) to find which tools *amplify* model-quality gaps is in
progress.

---

## Reproduce

```bash
# noisy-environment sweep + selection metrics
CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
SEEDS="0 1 2" MAX_CASES=20 bash ducx_noise/run_noise_experiment.sh

# corrected task_acc (LLM answer-extractor)
bash ducx_noise/run_extract_recompute.sh

# reasoning likelihood/entropy (re-run with logprobs, then extract + compare)
CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
SEEDS="0 1 2" MAX_CASES=20 TOPK=20 bash ducx_entropy/run_entropy_experiment.sh

# strong-vs-weak backbone for the per-tool gap study
SERVED_NAME=qwen3-vl-2b MODEL_ID=Qwen/Qwen3-VL-2B-Instruct MAX_CASES=500 \
VLLM_GPU=0 TOOL_GPU=1 bash run_toolprobe_eval.sh
```

See each module's README for details: [`ducx_noise/README.md`](ducx_noise/README.md),
[`ducx_entropy/README.md`](ducx_entropy/README.md). Extraction-fix details:
[`ducx_noise/extraction_fix_findings.md`](ducx_noise/extraction_fix_findings.md).

## Caveats
- Only the 500-question anchors are comparable; the 15/50-question runs have very
  high variance — don't read gradients off them.
- Reasoning entropy uses the final-answer prose (no `<think>` channel on the
  Instruct model); it is not a true intermediate-reasoning signal.
- All `task_acc` numbers should use the corrected extractor, not the shipped
  `extract_choice`.
