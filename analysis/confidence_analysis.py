"""Confidence-as-correctness signal analysis (autonomous experiment suite).

Question: can the model's answer likelihood predict whether it is CORRECT, and
which confidence signal is best? Runs offline on Stage-5 run logs (they store the
final-answer per-token logprobs in the trace).

Signals per answer (from the FINAL-answer message tokens):
  msg_mean   : mean logprob over the whole final answer  (baseline used so far)
  msg_min    : min token logprob (least-confident token)
  msg_last   : mean logprob of the last 10 tokens (the conclusion)
  ans_name   : mean logprob of the tokens spelling the answer (option/pathology)
  perplexity : exp(-msg_mean)  (monotone in msg_mean; sanity)

Outputs: AUC(signal -> correct) per (condition x qtype); risk-coverage curve +
AURC for the best signal; CSV + plots.

    python -m analysis.confidence_analysis --out-dir analysis/confidence
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = "/project/6101776/xzhan576/DUCK/logs"


# ---------------------------------------------------------------------------
# final-answer token logprobs from a record (agent trace or direct)
# ---------------------------------------------------------------------------
def final_tokens(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """List of {token, logprob} for the FINAL answer message (last with logprobs)."""
    chosen = None
    for m in rec.get("trace") or []:
        lp = m.get("logprobs")
        if lp and not m.get("tool_calls"):   # prefer non-tool-call assistant messages
            chosen = lp
    if chosen is None:  # fallback: last message with logprobs at all
        for m in rec.get("trace") or []:
            if m.get("logprobs"):
                chosen = m["logprobs"]
    return chosen


def signals(rec: Dict[str, Any], answer_text: Optional[str]) -> Dict[str, Optional[float]]:
    toks = final_tokens(rec)
    if not toks:
        # no-tool / direct path has no token-level trace; fall back to the stored
        # whole-answer mean so no_tool still gets an msg_mean (Exp C).
        m = rec.get("resp_logprob_mean")
        return {"msg_mean": float(m)} if m is not None else {}
    lps = [float(t["logprob"]) for t in toks if t.get("logprob") is not None]
    if not lps:
        return {}
    out = {
        "msg_mean": sum(lps) / len(lps),
        "msg_min": min(lps),
        "msg_last": sum(lps[-10:]) / len(lps[-10:]),
        "perplexity": -math.exp(-sum(lps) / len(lps)),  # negate so higher = more confident
    }
    # ans_name: tokens whose concatenation covers the answer text (first occurrence)
    if answer_text:
        toks_str = [str(t.get("token", "")) for t in toks]
        joined = ""
        spans = []
        for i, s in enumerate(toks_str):
            spans.append((len(joined), i))
            joined += s
        low = joined.lower()
        pos = low.find(answer_text.lower())
        if pos >= 0:
            end = pos + len(answer_text)
            idxs = [i for (start, i) in spans if pos <= start < end]
            if idxs:
                nl = [float(toks[i]["logprob"]) for i in idxs]
                out["ans_name"] = sum(nl) / len(nl)
    return out


# ---------------------------------------------------------------------------
# AUC + risk-coverage
# ---------------------------------------------------------------------------
def auc(pos: List[float], neg: List[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    c = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return c / (len(pos) * len(neg))


def risk_coverage(items: List[tuple]) -> List[tuple]:
    """items = [(confidence, is_correct)] -> [(coverage, accuracy)] sweeping the
    confidence threshold from all-covered to most-confident-only."""
    items = sorted(items, key=lambda x: -x[0])  # most confident first
    out = []
    correct = 0
    for i, (_, c) in enumerate(items, 1):
        correct += 1 if c else 0
        out.append((i / len(items), correct / i))
    return out


def aurc(items: List[tuple]) -> Optional[float]:
    """Area under the risk (=error) coverage curve; lower is better."""
    rc = risk_coverage(items)
    if not rc:
        return None
    # trapezoid over coverage of error=1-acc
    area, prev_cov, prev_err = 0.0, 0.0, 1.0 - rc[0][1]
    for cov, acc in rc:
        err = 1.0 - acc
        area += (cov - prev_cov) * (err + prev_err) / 2
        prev_cov, prev_err = cov, err
    return round(area, 4)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def latest(prefix):
    fs = sorted(glob.glob(f"{ROOT}/{prefix}/{prefix}_*.json"))
    return fs[-1] if fs else None


def load_recs(prefix):
    f = latest(prefix)
    if not f:
        return []
    return [json.loads(l) for l in open(f) if l.strip() and json.loads(l).get("status") == "ok"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="analysis/confidence")
    ap.add_argument("--gating-mixed", default="logs/toolbias/gating_mixed.jsonl")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # qtype + answer-name per question from the mixed set (+ probe answer text)
    qtype, ans_name = {}, {}
    for l in open(args.gating_mixed):
        if not l.strip():
            continue
        r = json.loads(l)
        qtype[r["question_id"]] = r["qtype"]
        # tool_necessary probes carry the correct option text
        opts = r.get("probe_options")
        ans = r.get("answer")
        if opts and ans and ans in "ABCDEF":
            ans_name[r["question_id"]] = opts["ABCDEF".index(ans)]

    conds = {"with_tool": load_recs("gate_always"), "no_tool": load_recs("gate_never")}
    # ans_name dropped: token-span matching lands on the pathology name inside the
    # reasoning (where the model lists all options), not the final decision.
    SIGNALS = ["msg_mean", "msg_min", "msg_last", "perplexity"]

    rows = []
    print("=== Exp A/C: AUC(signal -> correct) per condition x qtype ===")
    best_items = None  # for risk-coverage (with_tool x tool_necessary, best signal)
    for cond, recs in conds.items():
        for qt in ("tool_necessary", "text_answerable"):
            sub = [r for r in recs if qtype.get(r["sample_id"]) == qt]
            sig_vals = defaultdict(lambda: ([], []))  # signal -> (correct_vals, incorrect_vals)
            items_for_signal = defaultdict(list)
            for r in sub:
                sg = signals(r, ans_name.get(r["sample_id"]))
                for s in SIGNALS:
                    if sg.get(s) is not None:
                        (sig_vals[s][0] if r.get("is_correct") else sig_vals[s][1]).append(sg[s])
                        items_for_signal[s].append((sg[s], bool(r.get("is_correct"))))
            line = {"condition": cond, "qtype": qt, "n": len(sub)}
            for s in SIGNALS:
                a = auc(sig_vals[s][0], sig_vals[s][1])
                line[f"AUC_{s}"] = round(a, 3) if a is not None else None
            rows.append(line)
            print(f"  {cond:9s} {qt:16s} n={len(sub):3d} " +
                  " ".join(f"{s}={line['AUC_'+s]}" for s in SIGNALS))
            if cond == "with_tool" and qt == "tool_necessary":
                # pick best signal by AUC for the risk-coverage analysis
                best_s = max(SIGNALS, key=lambda s: (line[f"AUC_{s}"] or 0))
                best_items = (best_s, items_for_signal[best_s])

    with open(f"{args.out_dir}/auc_by_signal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]

    # Exp B: risk-coverage on with_tool x tool_necessary, best signal
    if best_items:
        best_s, items = best_items
        rc = risk_coverage(items)
        a = aurc(items)
        base_acc = sum(1 for _, c in items if c) / len(items)
        # accuracy at 50% coverage
        cov50 = min(rc, key=lambda x: abs(x[0] - 0.5))
        print(f"\n=== Exp B: selective prediction (with_tool x tool_necessary, best signal={best_s}) ===")
        print(f"  full accuracy={base_acc:.3f}  acc@50%coverage={cov50[1]:.3f}  AURC={a} (lower=better)")
        with open(f"{args.out_dir}/risk_coverage.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["coverage", "accuracy"]); [w.writerow(x) for x in rc]
        if plt:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot([c for c, _ in rc], [a for _, a in rc], lw=2)
            ax.axhline(base_acc, ls="--", c="gray", label=f"full acc {base_acc:.2f}")
            ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("accuracy on covered")
            ax.set_title(f"Selective prediction via {best_s}\n(with_tool, tool-necessary; AURC={a})")
            ax.legend(); fig.tight_layout(); fig.savefig(f"{args.out_dir}/risk_coverage.png", dpi=120)
            print(f"  plot: {args.out_dir}/risk_coverage.png")
    print(f"\nwrote {args.out_dir}/auc_by_signal.csv")

    # Exp D: does distractor pollution degrade the confidence->correctness signal?
    pol = [("clean", "conf_clean"), ("pol5", "conf_pol5"), ("pol20", "conf_pol20")]
    drows = []
    have = any(latest(p) for _, p in pol)
    if have:
        print("\n=== Exp D: pollution x confidence signal (best=msg_last) ===")
        for name, pfx in pol:
            recs = load_recs(pfx)
            if not recs:
                continue
            cor, inc = [], []
            for r in recs:
                s = signals(r, None).get("msg_last")
                if s is None:
                    continue
                (cor if r.get("is_correct") else inc).append(s)
            acc = round(sum(1 for r in recs if r.get("is_correct")) / len(recs), 4)
            a = auc(cor, inc)
            drows.append({"pollution": name, "n": len(recs), "accuracy": acc,
                          "AUC_msg_last": round(a, 3) if a else None})
            print(f"  {name:6s} n={len(recs)} acc={acc}  AUC(msg_last->correct)={round(a,3) if a else None}")
        if drows:
            with open(f"{args.out_dir}/pollution_signal.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(drows[0].keys())); w.writeheader(); [w.writerow(r) for r in drows]
            print(f"  wrote {args.out_dir}/pollution_signal.csv")


if __name__ == "__main__":
    main()
