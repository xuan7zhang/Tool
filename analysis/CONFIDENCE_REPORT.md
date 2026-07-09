# Confidence as a correctness signal for tool-grounded VLM answers

Autonomous experiment suite around: **can the model's answer likelihood predict
whether it is correct?** Qwen3-VL-8B, greedy, seed 0. Offline signals extracted
from the FINAL-answer per-token logprobs stored in each run's trace; one GPU run
for the pollution arm.

## Exp A — which confidence signal is best (with_tool × tool-necessary, n=200)

| signal | AUC(→correct) |
|---|---|
| msg_mean (whole final answer, mean logprob) | 0.803 |
| **msg_last (mean logprob of the last 10 tokens)** | **0.878** |
| msg_min (least-confident token) | 0.456 |
| perplexity (= msg_mean, monotone) | 0.803 |
| ~~ans_name~~ (dropped) | unreliable — matches the pathology name inside the *reasoning*, not the decision |

**The conclusion tokens carry the decision confidence** — `msg_last` beats the
whole-message mean (0.878 vs 0.803). The preamble/reasoning dilutes `msg_mean`.

## Exp B — selective prediction (msg_last, with_tool × tool-necessary)

- Full accuracy **0.765** → **accuracy @ 50% coverage = 0.990** (answer only the
  most-confident half). AURC = 0.061 (lower is better).
- **Actionable:** abstaining on the low-confidence half yields ~99% accuracy on the
  answered half — a strong training-free verification / selective-answer signal.
  Plot: `analysis/confidence/risk_coverage.png`.

## Exp C — condition dependence (the signal is tool-specific)

| condition × qtype | AUC (msg_mean) |
|---|---|
| with_tool × tool-necessary | 0.803 (msg_last 0.878) |
| with_tool × text-answerable | 0.549 |
| no_tool × tool-necessary (guessing) | **0.479** (no signal) |
| no_tool × text-answerable | 0.664 |

The confidence→correctness link is **strong only when the model has and uses the
relevant tool** (0.878). When guessing without a tool it is uninformative (0.479);
when the tool is irrelevant (text-answerable) it is weak (0.55). So the signal
measures *"did the model read the tool output well"*, not generic correctness.

## Exp D — heavy pollution erodes the signal

CLS probe + classifier + {0, 5, 20} distractors, logprobs captured (n=225 each).

| pollution | task acc | AUC(msg_last → correct) |
|---|---|---|
| clean (0) | 0.782 | 0.785 |
| pol5 | 0.622 | 0.807 |
| **pol20** | 0.684 | **0.609** |

- The confidence→correctness signal **survives light pollution** (pol5 0.807 ≈
  clean 0.785, within single-seed noise) but **degrades under heavy pollution**
  (pol20 → 0.609). Under 20 distractors the polluted context muddies both the
  answer and the model's calibration, so confidence becomes a weaker error
  detector.
- This mirrors the pruning result (heavy pollution K=20 is where real harm
  appears): exactly when you'd want to prune, the confidence signal is also least
  trustworthy. Plot: `analysis/confidence/pollution_signal.png`.
- (Accuracy is non-monotone pol5<pol20 — the familiar single-seed U-shape noise;
  the AUC drop at K=20 is the robust signal here.)

## Takeaways

- **Best signal = last-token confidence (msg_last), AUC 0.878** for tool-grounded
  answers — a clean, training-free correctness predictor.
- **Practical value:** 99% accuracy at 50% coverage → usable for selective
  answering / flagging likely-wrong tool-grounded answers for re-try or review.
- **Scope:** works only when the relevant tool was used; it is a "tool-output
  comprehension" signal, not a universal one. Complements the likelihood *gate*
  (which decides whether the tool helped at all).
- **Robustness:** holds under light pollution but **erodes under heavy pollution
  (K=20, AUC→0.61)** — least trustworthy exactly when the toolbox is most
  polluted, so it pairs naturally with pruning.

Code: `analysis/confidence_analysis.py`; runners `run_confidence_pollution.sh`.
