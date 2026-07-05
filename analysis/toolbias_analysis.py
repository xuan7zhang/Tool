"""Stage 6 analysis: read Stage-5 toolbias logs, emit contribution / combination
/ likelihood tables (+ simple plots). Does NOT touch the inference code.

    python -m analysis.toolbias_analysis \
        --logs-glob 'logs/tb_*/tb_*_*.json' --out-dir analysis/toolbias

Three outputs (CSV + PNG where a plot helps):
  6.1 contribution.csv  -- marginal effect of having tools / of distractors on
                           accuracy and response likelihood (with vs without).
  6.2 combination.csv   -- accuracy across distractor combinations (count x
                           similarity x position) and across called-tool sets;
                           surfaces interaction effects.
  6.3 likelihood.csv    -- resp_logprob distribution with-tool vs no-tool, then
                           split by is_correct (correct / incorrect).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots are optional
    plt = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_records(logs_glob: str) -> List[Dict[str, Any]]:
    """Flatten every Stage-5 record from the matched run logs.

    Keeps records that carry a `condition` (i.e. Stage-4 runner output). Skips
    'error'/'skipped' rows with no answer. A record missing is_correct (invalid
    answer) is kept with is_correct=False so accuracy is not silently inflated.
    """
    records: List[Dict[str, Any]] = []
    files = sorted(glob.glob(logs_glob))
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # a matched non-JSONL file (pretty-printed manifest / tool_calls)
                # can yield a bare str/list per line -> skip anything not a record
                if not isinstance(rec, dict):
                    continue
                if rec.get("status") == "error":
                    continue
                if "condition" not in rec:
                    continue
                if rec.get("is_correct") is None:
                    rec["is_correct"] = False  # invalid answer counts as wrong
                records.append(rec)
    return records


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 6) if xs else None


def _std(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.pstdev(xs), 6) if len(xs) > 1 else 0.0 if xs else None


def _acc(recs: List[Dict[str, Any]]) -> Optional[float]:
    vals = [1.0 if r.get("is_correct") else 0.0 for r in recs]
    return round(statistics.fmean(vals), 6) if vals else None


def _logprobs(recs: List[Dict[str, Any]]) -> List[float]:
    return [r.get("resp_logprob_mean") for r in recs if r.get("resp_logprob_mean") is not None]


# ---------------------------------------------------------------------------
# 6.1 Tool contribution
# ---------------------------------------------------------------------------
def tool_contribution(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Marginal contribution of tools / distractors to acc and likelihood.

    Rows:
      * per (path, condition): acc, mean logprob, n.
      * derived deltas: tool-present vs no_tool (same path); each distractor
        config vs the clean tool_useful baseline (same path).
    """
    by_pc: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_pc[(r.get("path"), r.get("condition"))].append(r)

    rows: List[Dict[str, Any]] = []
    for (path, cond), recs in sorted(by_pc.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rows.append({
            "kind": "condition",
            "path": path, "condition": cond,
            "n": len(recs),
            "accuracy": _acc(recs),
            "logprob_mean": _mean(_logprobs(recs)),
        })

    # deltas vs no_tool baseline (per path)
    for path in sorted({r.get("path") for r in records}, key=str):
        precs = [r for r in records if r.get("path") == path]
        base = [r for r in precs if r.get("condition") == "no_tool"]
        if not base:
            continue
        base_acc, base_lp = _acc(base), _mean(_logprobs(base))
        for cond in sorted({r.get("condition") for r in precs if r.get("condition") != "no_tool"}, key=str):
            recs = [r for r in precs if r.get("condition") == cond]
            rows.append({
                "kind": "delta_vs_no_tool",
                "path": path, "condition": cond,
                "n": len(recs),
                "accuracy": _acc(recs),
                "logprob_mean": _mean(_logprobs(recs)),
                "d_accuracy": _delta(_acc(recs), base_acc),
                "d_logprob": _delta(_mean(_logprobs(recs)), base_lp),
            })

    # distractor marginal: each distractor config vs clean tool_useful
    du = [r for r in records if r.get("path") == "tool_useful_distractor"]
    clean = [r for r in du if r.get("condition") == "tool_useful"]
    if clean:
        clean_acc, clean_lp = _acc(clean), _mean(_logprobs(clean))
        by_cfg: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in du:
            if r.get("condition") == "tool_useful_distractor":
                by_cfg[(r.get("n_distractors"), r.get("distractor_similarity"))].append(r)
        for (nd, sim), recs in sorted(by_cfg.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
            rows.append({
                "kind": "distractor_marginal_vs_clean",
                "path": "tool_useful_distractor",
                "condition": f"n={nd},sim={sim}",
                "n": len(recs),
                "accuracy": _acc(recs),
                "logprob_mean": _mean(_logprobs(recs)),
                "d_accuracy": _delta(_acc(recs), clean_acc),
                "d_logprob": _delta(_mean(_logprobs(recs)), clean_lp),
            })
    return rows


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 6)


# ---------------------------------------------------------------------------
# 6.2 Tool combination
# ---------------------------------------------------------------------------
def tool_combination(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accuracy across distractor combinations and across called-tool sets."""
    rows: List[Dict[str, Any]] = []
    du = [r for r in records if r.get("condition") == "tool_useful_distractor"]

    # (a) count x similarity grid -> interaction on accuracy
    grid: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in du:
        grid[(r.get("n_distractors"), r.get("distractor_similarity"))].append(r)
    for (nd, sim), recs in sorted(grid.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
        rows.append({
            "kind": "distractor_grid",
            "n_distractors": nd, "similarity": sim,
            "n": len(recs),
            "accuracy": _acc(recs),
            "logprob_mean": _mean(_logprobs(recs)),
        })

    # (b) which tools the model CALLED -> accuracy per called-set
    called: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = ",".join(sorted(set(r.get("called_tools") or []))) or "(none)"
        called[key].append(r)
    for key, recs in sorted(called.items(), key=lambda kv: -len(kv[1])):
        rows.append({
            "kind": "called_set",
            "called_tools": key,
            "n": len(recs),
            "accuracy": _acc(recs),
            "logprob_mean": _mean(_logprobs(recs)),
        })
    return rows


# ---------------------------------------------------------------------------
# 6.3 Response likelihood
# ---------------------------------------------------------------------------
def likelihood_comparison(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """with-tool vs no-tool resp_logprob, then split by correctness."""
    def group(recs, name):
        lps = _logprobs(recs)
        return {
            "group": name, "n": len(recs), "n_with_logprob": len(lps),
            "logprob_mean": _mean(lps), "logprob_std": _std(lps),
        }

    with_tool = [r for r in records if r.get("condition") not in (None, "no_tool")]
    no_tool = [r for r in records if r.get("condition") == "no_tool"]

    rows = [group(with_tool, "with_tool"), group(no_tool, "no_tool")]
    for label, recs in (("with_tool", with_tool), ("no_tool", no_tool)):
        corr = [r for r in recs if r.get("is_correct")]
        inc = [r for r in recs if not r.get("is_correct")]
        rows.append(group(corr, f"{label}:correct"))
        rows.append(group(inc, f"{label}:incorrect"))
    return rows


def _plot_likelihood(records, out_dir):
    if plt is None:
        return None
    with_tool = _logprobs([r for r in records if r.get("condition") not in (None, "no_tool")])
    no_tool = _logprobs([r for r in records if r.get("condition") == "no_tool"])
    if not with_tool and not no_tool:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 30
    if with_tool:
        ax.hist(with_tool, bins=bins, alpha=0.6, label=f"with_tool (n={len(with_tool)})")
    if no_tool:
        ax.hist(no_tool, bins=bins, alpha=0.6, label=f"no_tool (n={len(no_tool)})")
    ax.set_xlabel("mean response logprob")
    ax.set_ylabel("count")
    ax.set_title("Response likelihood: with-tool vs no-tool")
    ax.legend()
    path = os.path.join(out_dir, "likelihood_hist.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def _plot_contribution(rows, out_dir):
    if plt is None:
        return None
    marg = [r for r in rows if r.get("kind") == "distractor_marginal_vs_clean"
            and r.get("d_accuracy") is not None]
    if not marg:
        return None
    labels = [r["condition"] for r in marg]
    vals = [r["d_accuracy"] for r in marg]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(vals)), vals, color=["#c44" if v < 0 else "#4a4" for v in vals])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Δ accuracy vs clean tool_useful")
    ax.set_title("Distractor marginal effect on accuracy")
    path = os.path.join(out_dir, "contribution_bar.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        # still emit an empty file with a marker so the pipeline is explicit
        with open(path, "w") as fh:
            fh.write("(no records)\n")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-glob", required=True,
                    help="Glob for Stage-5 run logs, e.g. 'logs/tb_*/tb_*_*.json'")
    ap.add_argument("--out-dir", default="analysis/toolbias")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.logs_glob)
    print(f"Loaded {len(records)} records from {args.logs_glob}")

    contrib = tool_contribution(records)
    combo = tool_combination(records)
    like = likelihood_comparison(records)

    _write_csv(contrib, os.path.join(args.out_dir, "contribution.csv"))
    _write_csv(combo, os.path.join(args.out_dir, "combination.csv"))
    _write_csv(like, os.path.join(args.out_dir, "likelihood.csv"))
    p1 = _plot_likelihood(records, args.out_dir)
    p2 = _plot_contribution(contrib, args.out_dir)

    print(f"Wrote: {args.out_dir}/contribution.csv, combination.csv, likelihood.csv")
    if p1:
        print(f"Plot:  {p1}")
    if p2:
        print(f"Plot:  {p2}")
    if plt is None:
        print("(matplotlib unavailable -> CSV only, no plots)")


if __name__ == "__main__":
    main()
