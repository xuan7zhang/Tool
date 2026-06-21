"""Task 2: compare reasoning entropy/likelihood against selection entropy.

Aggregates the granularity-C ``*_perturn.jsonl`` files (token-weighted over the
reasoning tokens of each question) to per-(condition, seed) reasoning numbers,
then to per-condition mean +/- std across seeds, and -- when given the existing
``noise_metrics.csv`` from ``ducx_noise.metrics`` -- joins the per-condition
``selection_entropy_bits`` / ``task_accuracy`` / ``misselection`` alongside, so
you can see whether reasoning entropy tracks (or pre-empts) the selection signal
as noise rises.

    python -m ducx_entropy.analyze \
        --perturn logs/entropy/*_perturn.jsonl \
        --selection-csv logs/.../noise_metrics.csv \
        --out-md logs/entropy/reasoning_vs_selection.md \
        --out-csv logs/entropy/reasoning_vs_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .entropy import perplexity


def _load_perturn(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def reasoning_by_condition_seed(
    perturn_rows: List[Dict[str, Any]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Token-weighted reasoning entropy/logprob per (condition, seed)."""
    acc: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"sum_h": 0.0, "sum_lp": 0.0, "n_tok": 0, "n_q": 0}
    )
    for row in perturn_rows:
        n = row.get("n_reasoning_tokens") or 0
        if n <= 0:
            continue
        key = (row["condition"], str(row["seed"]))
        a = acc[key]
        a["sum_h"] += float(row["reasoning_mean_entropy_bits"]) * n
        a["sum_lp"] += float(row["reasoning_mean_logprob"]) * n
        a["n_tok"] += n
        a["n_q"] += 1  # one reasoning turn per question

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, a in acc.items():
        if a["n_tok"] == 0:
            continue
        mean_lp = a["sum_lp"] / a["n_tok"]
        out[key] = {
            "reasoning_mean_entropy_bits": a["sum_h"] / a["n_tok"],
            "reasoning_mean_logprob": mean_lp,
            "reasoning_perplexity": perplexity(mean_lp),
            "n_reasoning_tokens": a["n_tok"],
            "n_questions": a["n_q"],
        }
    return out


def _mean_std(vals: List[float]) -> Tuple[Optional[float], float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _load_selection_csv(path: str) -> Dict[str, Dict[str, List[float]]]:
    """Per-condition lists (across seeds) of selection metrics."""
    by_label: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"selection_entropy_bits": [], "noise_tool_misselection_rate": [],
                 "task_accuracy": []}
    )
    with open(path, "r", newline="") as handle:
        for r in csv.DictReader(handle):
            label = r.get("label", "base")
            for col in ("selection_entropy_bits", "noise_tool_misselection_rate", "task_accuracy"):
                v = r.get(col)
                if v not in (None, "", "None"):
                    by_label[label][col].append(float(v))
    return by_label


def build_summary(
    reasoning_cs: Dict[Tuple[str, str], Dict[str, Any]],
    selection: Optional[Dict[str, Dict[str, List[float]]]],
) -> List[Dict[str, Any]]:
    # group reasoning per-seed values by condition
    by_cond: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"ent": [], "lp": [], "seeds": set(), "n_tok": 0}
    )
    for (cond, seed), v in reasoning_cs.items():
        by_cond[cond]["ent"].append(v["reasoning_mean_entropy_bits"])
        by_cond[cond]["lp"].append(v["reasoning_mean_logprob"])
        by_cond[cond]["seeds"].add(seed)
        by_cond[cond]["n_tok"] += v["n_reasoning_tokens"]

    summary: List[Dict[str, Any]] = []
    for cond, d in by_cond.items():
        ent_m, ent_s = _mean_std(d["ent"])
        lp_m, lp_s = _mean_std(d["lp"])
        row: Dict[str, Any] = {
            "condition": cond,
            "n_seeds": len(d["seeds"]),
            "n_reasoning_tokens": d["n_tok"],
            "reasoning_mean_entropy_bits_mean": _round(ent_m),
            "reasoning_mean_entropy_bits_std": round(ent_s, 4),
            "reasoning_mean_logprob_mean": _round(lp_m),
            "reasoning_mean_logprob_std": round(lp_s, 4),
            "reasoning_perplexity_mean": _round(perplexity(lp_m) if lp_m is not None else None),
        }
        if selection is not None and cond in selection:
            sel = selection[cond]
            se_m, se_s = _mean_std(sel["selection_entropy_bits"])
            ms_m, _ = _mean_std(sel["noise_tool_misselection_rate"])
            ta_m, _ = _mean_std(sel["task_accuracy"])
            row["selection_entropy_bits_mean"] = _round(se_m)
            row["selection_entropy_bits_std"] = round(se_s, 4)
            row["noise_misselection_mean"] = _round(ms_m)
            row["task_accuracy_mean"] = _round(ta_m)
        summary.append(row)
    summary.sort(key=lambda r: r["condition"])
    return summary


def _round(v: Optional[float]) -> Optional[float]:
    return round(v, 4) if isinstance(v, (int, float)) else v


def summary_to_markdown(summary: List[Dict[str, Any]]) -> str:
    cols = [
        ("condition", "condition"),
        ("n_seeds", "seeds"),
        ("selection_entropy_bits_mean", "select_H"),
        ("reasoning_mean_entropy_bits_mean", "reason_H"),
        ("reasoning_mean_logprob_mean", "reason_logp"),
        ("reasoning_perplexity_mean", "reason_ppl"),
        ("noise_misselection_mean", "misselect"),
        ("task_accuracy_mean", "task_acc"),
    ]
    header = "| " + " | ".join(lbl for _, lbl in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in summary:
        cells = [("-" if row.get(k) is None else str(row.get(k))) for k, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Reasoning vs selection entropy comparison.")
    parser.add_argument("--perturn", nargs="+", required=True,
                        help="One or more *_perturn.jsonl files (granularity C).")
    parser.add_argument("--selection-csv", default=None,
                        help="ducx_noise.metrics noise_metrics.csv to join selection_entropy.")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)

    perturn_rows = _load_perturn(args.perturn)
    reasoning_cs = reasoning_by_condition_seed(perturn_rows)
    selection = _load_selection_csv(args.selection_csv) if args.selection_csv else None
    summary = build_summary(reasoning_cs, selection)

    table = summary_to_markdown(summary)
    print(table)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w") as handle:
            handle.write(table + "\n")
    if args.out_csv and summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        keys = sorted({k for row in summary for k in row})
        with open(args.out_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)


if __name__ == "__main__":
    main()
