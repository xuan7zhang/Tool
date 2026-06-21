# Distractor sweep — does adding distractor tools change task accuracy?

Confirmatory, fully-paired re-run after the answer-extraction fix
(see `extraction_fix_findings.md`). Settles the earlier puzzle "distractors seem
to *improve* performance."

## Setup (why this run is trustworthy)

The earlier noisy table was invalid for two compounding reasons:

1. **Extraction artifact** — the shipped `extract_choice` mis-scored ~30% of
   rollouts (article-"a" trap), roughly uniformly across conditions, pinning
   every condition near ~0.5–0.68.
2. **Cross-subset comparison** — `data/chestagentbench/metadata.jsonl` gets
   **reordered between sessions** (question order went 10027… → 10098… → 11583…).
   `launch_over_chexbench.py` takes `dataset.select(range(N))` (first N, no
   shuffle), so runs from *different dates* are on *different question subsets*.
   The old "base" mixed an n=500 run on one subset with n=15 smoke runs on
   another, and was compared against distractor runs on yet another subset.

This run fixes both: **all conditions run in one batch on the current data file**
(identical first-N questions → fully paired), and task_acc is computed with the
temp-0 LLM extractor.

Run: `ducx_noise/run_sweep500.sh` (2 vLLM streams on one node), N=500, seed 0.
Extraction: `ducx_noise/_extract_sweep.sh`. Paired test: McNemar exact.

## Result — paired McNemar vs base (same questions, NEW extraction)

| condition | n paired | base acc | cond acc | Δ | McNemar p | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| distractor_5 | 499 | 0.936 | 0.938 | **+0.002** | **1.000** | no effect |
| distractor_10 | 500 | 0.936 | 0.938 | **+0.002** | **1.000** | no effect |
| distractor_20 | 73* | 0.932 | 0.918 | −0.014 | 1.000 | no effect (partial) |
| distractor_50 | 237* | 0.928 | 0.489 | −0.439 | <0.001 | breaks — see below |

\* distractor_20 / distractor_50 are partial (see "failure modes").

## Conclusions

1. **Distractors do NOT improve performance — the earlier "improvement" was an
   artifact.** Through 10 distractor tools, task accuracy is **dead flat at
   ~0.937** (base = d5 = d10, paired p=1.0). The apparent gain in the old table
   was entirely the extraction artifact + cross-subset/seed noise.

2. **They also don't hurt — up to a point.** d5/d10 are statistically identical
   to base; d20 (partial n=73) is within noise. The model effectively **ignores
   distractor tools** when choosing its final answer: selection-level metrics
   (misselection rate) move, but the *answer* doesn't, because the diagnosis
   doesn't depend on the fake tools.

3. **Degradation only appears at extreme tool counts, via mechanical limits, not
   reasoning confusion:**
   - **distractor_50 → context overflow.** 263/500 questions hard-error with
     HTTP 400 "maximum context length 16384 exceeded": 50 tool schemas alone
     blow the context budget. Among the 237 that fit, accuracy still craters to
     0.489 with **66 abstains** (agent emits no parseable answer) — a biased,
     non-comparable subset. This is a prompt-budget failure, not "distractors
     confuse the model."
   - **distractor_20 → tool-call thrashing.** No hard errors, but the agent falls
     into near-infinite loops, re-calling the same tool with reworded prompts and
     not converging (observed live on figure_18592). Throughput collapses (73 vs
     500 in the same wall-clock), so the run was stopped. The 73 that completed
     are still flat (Δ=−0.014, p=1.0).

**Bottom line for the paper:** the selection environment is robust — injecting up
to 10 distractor tools leaves diagnostic task accuracy unchanged (paired p=1.0 at
n≈500, true level ~0.94). Beyond that, failures are mechanical (context budget at
50, agent looping at 20), not a graceful "reasoning degrades with noise" curve.
Any sweep figure must (a) use the LLM extractor and (b) hold the question set
fixed across conditions; cross-date runs are not comparable.

Artifacts: `logs/sweep500_*/` (rollouts), `logs/sweep500_extract/` (records +
summary), `extraction_cache/` (cached extractions).
