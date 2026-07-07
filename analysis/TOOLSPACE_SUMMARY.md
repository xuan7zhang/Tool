# Tool-Space Optimization — Consolidated Report

Backbone: Qwen3-VL-8B (vLLM, fp16, greedy, seed 0), MedRAX agent on Killarney L40.
All experiments deterministic (greedy) ⇒ same-question **paired** tests are exact.
Code on branch `toolbias-distractor` (repo xuan7zhang/Tool).

---

## 0. Bottom line (honest)

- **A tool's value flips sign with tool-necessity.** On **tool-necessary** tasks
  (answer determined by the tool) the classifier lifts accuracy from chance
  (~0.23) to ~0.78. On **text-answerable** tasks (ChestAgentBench MCQs, solvable
  from the clinical text) adding tools is net-negative: no_tool 0.52 →
  tool_only 0.48 (−0.04, n=50; and the model is *less* confident with tools,
  logprob −0.15 vs −0.02). So "add tools" is not universally good — the
  first-order tool-space decision is **whether to expose any tools at all for a
  query** (gating), before which tool (pruning).
- **The model routes perfectly**: asked which tool it needs, both a zero-dependency
  lexical retriever and the model itself pick the right tool 100% of the time,
  even among 10 distractors.
- **BUT the headroom for "tool-space optimization by pruning" is much smaller and
  more fragile than it first looked.** A powered paired test (n=300) shows that
  exposing one extra *irrelevant real tool* does **not** degrade accuracy
  (p=1.0) — the earlier "−0.09 pollution" was small-sample noise and is
  **retracted**. Distractor harm shows up only in the many-fake-tool regime and is
  **non-monotone / fragile** even there.
- **Self-routing prompting backfires** (−0.27, p<1e-3): instructing the model to
  "pick one tool and ignore the rest" hurt, even though it still called the right
  tool.
- **External pruning DOES win in the heavy-pollution regime**: with 20 distractors
  in the toolbox, the model still calls the correct tool 225/225 but loses −0.14
  accuracy to context occupancy; a trivial (top1=1.0) retriever prunes the box
  back to the one needed tool and **recovers +0.14, paired p=4e-5**. Below ~20
  distractors the effect is null/non-monotone.

Net: tool-space optimization has provable value, but only at two clear levers —
**gating** (use tools at all? value flips sign with tool-necessity) and **pruning
under heavy pollution** (K≈20+). Routing quality and light-pollution pruning are
non-issues for this model.

---

## 1. Goal & reframing

Ultimate goal: optimize the tool space presented to a VLM agent. The pivotal
finding from the distractor study reframes it: **on crisp tool-necessary tasks**
the model never mis-selects a distractor (1 targeted call/q), yet accuracy can
drop because irrelevant tool descriptions **occupy context**. (On open-ended
tasks the model DOES mis-select — see the mis-selection note below.) So tool-space
optimization is partly a per-query **pruning / context-hygiene** problem.
Objective (this phase): pure task accuracy. Optimizer: training-free retriever.

> **Mis-selection is task-dependent (important correction).** The "never
> mis-selects" claim holds only for the probe task (calls/q≈1.00, 0% distractor
> calls across n=875). On open-ended **ChestAgentBench** (full real toolset + 5
> distractors, 500 q, prior run) the model explores 5.7→7.5 calls/q and
> **mis-selects 38%** (sel_acc 1.00→0.62), yet task_acc barely moves
> (0.648→0.658) because those questions are text-answerable. So pollution has TWO
> channels: passive context-occupancy (probe task, hurts task_acc) and active
> mis-selection (open task, high mis-select but little task_acc harm). calls/q —
> driven by whether the task is tool-necessary with a unique answer — is the
> mediating variable.

## 2. Infrastructure built (`ducx_noise/`, `analysis/`)

- **Config lockdown**: `--decoding greedy` (temp0/top_p1), `--seed` → vLLM+torch,
  fp16 via serve `--dtype`. Byte-exact reproducibility verified.
- **taxonomy.py**: Outcome{SUCCESS,FAILURE,UNRELIABLE} + ToolRole{REAL,
  DISTRACTOR_OBVIOUS,DISTRACTOR_ALIGNED}; thresholds are placeholders pending
  sign-off.
- **Distractor mechanism**: count / position(head|tail|random|index) /
  similarity(obvious|aligned) / tool_order(fixed|shuffle|controlled).
- **Stage-5 schema.py**: per-inference record incl. tool_set/roles/positions,
  resp_logprob, outcome_label.
- **multitool_probe.py**: tool-necessary MCQs where different questions need
  different tools (CLS→classifier, SEG→segmentation), each with `required_tool`.
- **retriever.py**: lexical TF-IDF + LLM-router.
- Runners: `run_toolbias_{experiment,sweep}.sh`, `run_multitool_bounds.sh`,
  `run_retriever_eval.sh`, `run_dissociation.sh`, `run_pollution_recovery.sh`.
- Analysis: `toolbias_analysis.py`, `dissociation_analysis.py` (exact McNemar).

## 3. Distractor study (single-tool probe)

Adding distractor tools to a tool-necessary probe (classifier "highest-prob
pathology", n=500/seed0):

- Clean tool 0.654 vs no_tool 0.268 (tool necessary, +0.39).
- Distractor **count** dose-response is **non-monotone / U-shaped** even at n=500
  (count 1 & 20 ≈ clean 0.75–0.80; counts 2–10 dip 0.41–0.67). Not a dose law;
  confounded with distractor identity (single seed).
- **aligned not worse than obvious** (mixed) — "as-if-real" hypothesis rejected.
- Position: modest ordered **primacy** (head 0.47 → tail 0.53).
- **Model never selects a distractor (0/500) ON THIS PROBE TASK** (calls/q≈1) →
  harm is context-occupancy. NOTE: on open-ended ChestAgentBench the model
  mis-selects 38% (calls/q 5–7); the 0% is task-specific, not universal.

## 4. Multi-tool suite & oracle bounds (n=120)

| condition | CLS | SEG | overall |
|---|---|---|---|
| no_tool | 0.329 | 0.614 | 0.433 |
| oracle (only-needed tool) | 0.829 | 0.977 | 0.883 |
| all_real (both real tools) | 0.711 | 0.932 | 0.792 |
| all_distract (+5 fake) | 0.697 | 0.977 | 0.798 |

Read at the time as "+0.09 recoverable gap". **This was under-powered — see §6.**

## 5. Retriever (training-free)

| retriever | top1 | recall@1 | distractor leak |
|---|---|---|---|
| lexical (TF-IDF) | 1.00 | 1.00 | 0.00 |
| LLM-router (the model) | 1.00 | 1.00 | 0.00 |

Routing is **trivial** here: the query + option vocabulary (pathology vs anatomy
names) reveals the tool domain, so even lexical hits 100%, and so does the model
itself — even with a 10-distractor registry. The retriever reproduces the oracle
grouping exactly ⇒ if there were a gap, pruning would close it fully.

## 6. Dissociation experiment (n=300, PAIRED McNemar) — the correction

Powered follow-up. CLS arm is clean (n=225); SEG used the intensity metric and is
broken (oracle 0.36), excluded.

| condition (CLS, n=225) | accuracy |
|---|---|
| no_tool | 0.231 |
| oracle (prune to needed tool) | 0.782 |
| all_real (both tools) | 0.787 |
| self_route (both tools + "pick one, ignore others") | 0.516 |

- **oracle vs all_real: 12 vs 13 discordant, p=1.0 → NO effect.** The extra
  irrelevant real tool does not degrade accuracy. The n=120 −0.09/−0.12 was noise
  → **retracted**.
- **self_route vs all_real: 9 vs 70 discordant, p<1e-3 → −0.27, self-route
  HURTS.** Not mis-routing: the model called the correct classifier on all 225
  CLS questions. The meta-instruction itself degraded the answer.

Interpretation: with a small tool set the model's default behaviour is already
near-oracle (nothing to prune), and nudging it to self-route backfires.

## 7. Pollution-recovery (many-distractor regime) — the clean positive result

Same 225 CLS questions, paired. pruned/oracle = classifier only (0.782, from §6);
polluted_K = classifier + segmentation + K distractors.

| K distractors | polluted acc | Δ vs oracle | paired McNemar p |
|---|---|---|---|
| 5 | 0.738 | −0.044 | 0.10 (n.s.) |
| 10 | 0.800 | +0.018 | 0.62 (n.s.) |
| **20** | **0.640** | **−0.142** | **4e-5 (significant)** |

- **A powered, mechanistically clean win for pruning — but only at heavy
  pollution (K=20).** Non-monotone below that (K=10 doesn't hurt), echoing the
  count U-shape.
- **Mechanism definitively isolated as context-occupancy:** in polluted_20 the
  model called the correct classifier on **225/225** questions, called a
  distractor **0** times, skipped the tool **0** times — same tool, same tool
  output as oracle — yet made 14% more errors purely because 20 irrelevant tool
  descriptions sat in context.
- Retriever routes at **top1=1.0 even with 20 distractors**, so pruning to the one
  needed tool ≡ oracle and **recovers the full +0.14 (p=4e-5)**, training-free.

This is the first powered evidence that external tool-space pruning helps: on this
probe task, not because the model mis-selects (here it never does — calls/q≈1),
but because context pollution degrades reasoning once the toolbox is large enough.
(On open-ended tasks a second channel — active mis-selection, 38% — coexists.)
Plot: `analysis/multitool/pollution_recovery.png`.

---

## 7b. Gating experiment (mixed set, n=379 paired) — the headline positive

Mixed set: 200 text-answerable (ChestAgentBench) + 200 tool-necessary (CLS probe).
Two runs on the same questions (never_tool, always_tool); `gated` = per-question
oracle gate (text→never, tool→always).

Per-qtype (paired McNemar):
| qtype | never | always | tool effect | p |
|---|---|---|---|---|
| text_answerable (n=191) | 0.497 | 0.304 | **−0.19** | <1e-3 |
| tool_necessary (n=188) | 0.218 | 0.771 | **+0.55** | <1e-3 |

Overall (n=379): never 0.359, always 0.536, **gated 0.633**.
- **gated vs always: +0.10, p<1e-3** (47 vs 10 discordant) — the gate beats
  always-tools, entirely by withholding tools on text-answerable questions.
- gated vs never: +0.27, p<1e-3 (122 vs 18).

Mechanism (active, not passive): on text-answerable questions with tools present,
the model **calls a tool 99%** of the time (189/191) and is then misled — accuracy
0.30 when it calls vs 0.50 when it doesn't. It cannot resist using a diagnostic
tool even when the answer is in the text.

**This is the largest, most robust tool-space lever: GATING (expose tools at all?)
beats always- and never-tools with high significance.** Plot:
`analysis/multitool/gating.png`. (Gate here is oracle; a real training-free gate is
easy since probe vs clinical questions are lexically distinct — same saturation
caveat as routing.)

## 8. What's robust vs retracted

| claim | status |
|---|---|
| Tools help on tool-necessary tasks (chance → ~0.78) | **robust** |
| Tools HURT on text-answerable tasks | **robust** (−0.19, p<1e-3, n=191 paired) |
| Tool value flips sign with tool-necessity → gating is the first lever | **robust** (both signs p<1e-3) |
| Gating beats always- AND never-tools | **robust** (+0.10 / +0.27, p<1e-3, n=379) |
| Model calls a tool even when useless (text-answerable) | **robust** (99% call rate) |
| Model routes to the right tool (lexical & self, ≤10 distractors) | **robust** |
| Model never selects a distractor — **crisp probe task only** (calls/q≈1) | **robust** (0/875) |
| Model mis-selects distractors on open-ended tasks (calls/q 5–7) | **robust** (38%, ChestAgentBench) |
| 1 extra irrelevant *real* tool degrades accuracy (−0.09) | **RETRACTED** (n=300 p=1.0) |
| Distractor *count* has a monotone dose-response | **rejected** (U-shaped) |
| aligned distractors worse than obvious | **rejected** |
| Distractor position: primacy (early worse) | weak, single-seed |
| Self-routing prompting helps | **rejected** (−0.27, it hurts) |
| Pruning recovers accuracy — few distractors (K≤10) | null (n.s.) |
| Pruning recovers accuracy — heavy pollution (K=20) | **robust** (+0.14, p=4e-5, paired) |
| On the probe task that harm is pure context-occupancy (not mis-selection) | **robust** (225/225 correct calls) |

## 9. Methodology notes

- Greedy ⇒ deterministic ⇒ scale **N**, not seeds; use **paired McNemar**
  (same-question, exact binomial on discordant pairs) — far more powerful than
  unpaired accuracy deltas, and it is what flipped the §4 "gap" to a §6 null.
- Correctness: `extract.py` temp-0 LLM extractor (Qwen2.5-7B, not self-grade),
  exact-match to the objective tool-derived ground truth.
- SEG "largest area"/"intensity" probes are only weakly/not tool-necessary;
  **CLS is the clean arm**. A genuine second tool (e.g. grounding localization)
  would sharpen the multi-tool story.

## 10. Next steps

1. Finish §7; if null, the honest headline is "tool-space pollution is fragile and
   task-specific — pruning rarely helps a competent model with a modest toolbox."
2. If §7 shows a gap, quantify retriever recovery (paired) and add a budget axis
   (accuracy − λ·context_cost).
3. **Gating experiment (strongest lever):** on a MIXED set (text-answerable +
   tool-necessary questions), test a per-query gate that decides whether to expose
   tools at all. Because tool value flips sign (§0), a good gate should beat both
   always-tools and never-tools — a powered, paired win that pruning could not
   deliver. Ground truth for the gate = tool-necessity (no_tool vs oracle gap).
4. A harder suite where the required tool is NOT lexically obvious, to make
   retrieval quality a non-saturated axis.
4. Understand *why* self-routing hurts (prompt-wording sensitivity vs a real
   effect) — test alternative routing instructions.

## Artifacts

- Reports: `analysis/TOOLSPACE_REPORT.md` (detailed), this summary.
- Plots: `analysis/multitool/{bounds,retriever_recovery,dissociation}.png`,
  `analysis/toolbias_n500/sweep_n500.png`.
- Tables: `analysis/multitool/{bounds,retriever}.csv`,
  `analysis/toolbias_full/*.csv`.
