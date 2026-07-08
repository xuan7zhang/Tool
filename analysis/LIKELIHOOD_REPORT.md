# TODO3 — Response Likelihood: tool-useful vs tool-useless

Does the model's answer confidence (log-prob) differ between tool-useful and
tool-useless questions? Mixed set (200 text-answerable + 200 tool-necessary),
both no_tool and with_tool, per-token logprobs captured at inference. Qwen3-VL-8B,
greedy, seed 0, n=400.

## Mean answer logprob (0 = fully confident)

| condition | text-answerable (useless) | tool-necessary (useful) |
|---|---|---|
| no_tool   | −0.040 | −0.079 |
| with_tool | −0.181 | −0.039 |

AUC of logprob separating useless vs useful:
- **no_tool: AUC 0.67** — weak (model slightly less confident on questions it
  cannot answer without a tool).
- **with_tool: AUC ≈ 0.99** (flipped) — near-perfect.

## Mechanism (a clean flip)

Without a tool the model is a bit more confident on text-answerable questions. With
a tool the pattern flips and sharpens: on tool-necessary questions the tool resolves
the answer → high confidence (−0.039); on text-answerable questions the tool output
conflicts with text reasoning → confidence collapses (−0.181). So **with-tool answer
confidence is a near-perfect detector of whether the tool was actually useful.**

## Training-free GATE (completes the gating story)

Rule: run with the tool; if answer confidence is high keep it, else fall back to the
no-tool answer.

| gate | accuracy (n=400) |
|---|---|
| never_tool | 0.367 |
| always_tool | 0.537 |
| **likelihood gate** (with_tool logprob ≥ −0.10) | **0.630** |
| oracle gate | 0.637 |

- likelihood gate **≈ oracle gate** (0.630 vs 0.637), beats always-tools **+0.09**,
  training-free; qtype classification **97.8%**.
- Caveat: post-hoc (needs one tool-augmented pass) — recovers accuracy, not tool
  cost. A pre-hoc no-tool-only gate exists but is weaker (AUC 0.67).
- Plot: `analysis/multitool/likelihood_gate.png`.

## Answer to the meeting question

**Yes — response likelihood distinguishes tool-useful from tool-useless:** weakly
before the tool (AUC 0.67), almost perfectly after (AUC 0.99), and it is directly
usable as a near-oracle training-free tool gate.
