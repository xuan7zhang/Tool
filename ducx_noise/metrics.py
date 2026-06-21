"""Task 2 -- noise-aware selection metrics on top of the existing eval logs.

This does NOT change the eval pipeline; it post-processes the run log (the
per-question JSONL DUCX already writes, with a serialized ``trace``) plus the
ground-truth noise manifest produced by ``apply_noise``.

Metrics per run:
* ``tool_selection_accuracy`` -- fraction of tool selections that hit a real
  (non-noise) tool.
* ``noise_tool_misselection_rate`` -- fraction of selections hitting noise tools
  (distractor/redundant). Complement of the above over calls that target a known
  tool.
* ``selection_entropy_bits`` -- Shannon entropy of the selected-tool distribution
  (action-space entropy).
* ``avg_tool_calls_per_question`` and ``task_accuracy`` -- reused task-level
  signals.
* per-tool confusion -- selection counts for each noise tool.

CLI::

    # one run -> append a row to results.csv
    python -m ducx_noise.metrics row --run-log LOG.json --manifest M.json \
        --label distractor_5 --seed 0 --out results.csv

    # aggregate rows (base vs noisy across seeds) -> summary table
    python -m ducx_noise.metrics aggregate --in results.csv \
        --summary-out summary.md --summary-csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

ROW_FIELDS = [
    "label",
    "seed",
    "n_questions",
    "n_questions_with_tools",
    "total_tool_calls",
    "avg_tool_calls_per_question",
    "tool_selection_accuracy",
    "noise_tool_misselection_rate",
    "unknown_tool_rate",
    "selection_entropy_bits",
    "task_accuracy",
    "n_distractor_tools",
    "n_redundant_tools",
    "top_confused_noise_tool",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    """Return {tool_name: provenance_dict} from a saved noise manifest.

    An empty/missing manifest (base env) yields {} -- every selected tool is then
    treated as a real tool.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    tools = data.get("tools", data) if isinstance(data, dict) else data
    return {t["name"]: t for t in tools}


def load_run_log(path: str) -> List[Dict[str, Any]]:
    """Load the per-question JSONL run log written by launch_over_chexbench.py."""
    entries: List[Dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def extract_selected_tools(entry: Dict[str, Any]) -> List[str]:
    """All tool names the model *selected* in one question (from the trace).

    Reads ``tool_calls`` on the serialized assistant messages, which is what the
    model chose (independent of whether execution later failed).
    """
    selected: List[str] = []
    for msg in entry.get("trace") or []:
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                if "<|channel|>" in name:  # mirror agent.execute_tools cleanup
                    name = name.split("<|channel|>", 1)[0].strip()
                selected.append(name)
    return selected


def selection_entropy_bits(names: List[str]) -> float:
    if not names:
        return 0.0
    counts = Counter(names)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


# --------------------------------------------------------------------------
# Per-run metrics
# --------------------------------------------------------------------------
def compute_run_metrics(
    run_log_path: str,
    manifest_path: Optional[str] = None,
    label: str = "base",
    seed: int = 0,
) -> Dict[str, Any]:
    entries = load_run_log(run_log_path)
    manifest = load_manifest(manifest_path) if manifest_path else {}

    noise_tools: Set[str] = {
        n for n, p in manifest.items() if p.get("is_noise_tool")
    }
    known_tools: Set[str] = set(manifest.keys())
    n_distractor = sum(1 for p in manifest.values() if p.get("source") == "distractor")
    n_redundant = sum(1 for p in manifest.values() if p.get("source") == "redundant")

    all_selected: List[str] = []
    n_with_tools = 0
    n_correct = 0
    n_scored = 0
    for entry in entries:
        if entry.get("status") not in ("ok", "invalid_answer", None):
            # skipped/error questions contribute no tool selections
            pass
        sel = extract_selected_tools(entry)
        if sel:
            n_with_tools += 1
        all_selected.extend(sel)
        if "is_correct" in entry:
            n_scored += 1
            n_correct += 1 if entry.get("is_correct") else 0

    total_calls = len(all_selected)
    # Selections that hit a known tool (real or noise). Unknown = hallucinated names.
    known_selected = [n for n in all_selected if not manifest or n in known_tools]
    noise_hits = sum(1 for n in all_selected if n in noise_tools)
    unknown_hits = sum(1 for n in all_selected if manifest and n not in known_tools)
    denom_known = max(1, len(known_selected))

    confusion = Counter(n for n in all_selected if n in noise_tools)
    top_confused = confusion.most_common(1)[0][0] if confusion else ""

    n_questions = len(entries)
    return {
        "label": label,
        "seed": seed,
        "n_questions": n_questions,
        "n_questions_with_tools": n_with_tools,
        "total_tool_calls": total_calls,
        "avg_tool_calls_per_question": round(total_calls / max(1, n_questions), 4),
        "tool_selection_accuracy": round((len(known_selected) - noise_hits) / denom_known, 4),
        "noise_tool_misselection_rate": round(noise_hits / denom_known, 4),
        "unknown_tool_rate": round(unknown_hits / max(1, total_calls), 4),
        "selection_entropy_bits": round(selection_entropy_bits(all_selected), 4),
        "task_accuracy": round(n_correct / n_scored, 4) if n_scored else None,
        "n_distractor_tools": n_distractor,
        "n_redundant_tools": n_redundant,
        "top_confused_noise_tool": top_confused,
        "_confusion": dict(confusion),
    }


# --------------------------------------------------------------------------
# CSV row I/O + aggregation
# --------------------------------------------------------------------------
def append_row(out_csv: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in ROW_FIELDS})


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average numeric metrics across seeds, grouped by label."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("label", "base")].append(row)

    numeric = [
        "avg_tool_calls_per_question",
        "tool_selection_accuracy",
        "noise_tool_misselection_rate",
        "unknown_tool_rate",
        "selection_entropy_bits",
        "task_accuracy",
    ]
    summary: List[Dict[str, Any]] = []
    for label, items in groups.items():
        agg: Dict[str, Any] = {"label": label, "n_seeds": len(items)}
        for col in numeric:
            vals = [float(r[col]) for r in items if r.get(col) not in (None, "", "None")]
            agg[f"{col}_mean"] = round(sum(vals) / len(vals), 4) if vals else None
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                agg[f"{col}_std"] = round(math.sqrt(var), 4)
            else:
                agg[f"{col}_std"] = 0.0
        summary.append(agg)
    summary.sort(key=lambda r: r["label"])
    return summary


def summary_to_markdown(summary: List[Dict[str, Any]]) -> str:
    cols = [
        ("label", "condition"),
        ("n_seeds", "seeds"),
        ("tool_selection_accuracy_mean", "sel_acc"),
        ("noise_tool_misselection_rate_mean", "misselect"),
        ("selection_entropy_bits_mean", "entropy"),
        ("avg_tool_calls_per_question_mean", "calls/q"),
        ("task_accuracy_mean", "task_acc"),
    ]
    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in summary:
        cells = []
        for key, _ in cols:
            val = row.get(key)
            cells.append("-" if val is None else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DUCX noise selection metrics.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_row = sub.add_parser("row", help="Compute metrics for one run; append a CSV row.")
    p_row.add_argument("--run-log", required=True)
    p_row.add_argument("--manifest", default=None)
    p_row.add_argument("--label", default="base")
    p_row.add_argument("--seed", type=int, default=0)
    p_row.add_argument("--out", required=True, help="CSV file to append the row to.")
    p_row.add_argument("--confusion-out", default=None, help="Optional JSON for per-tool confusion.")

    p_agg = sub.add_parser("aggregate", help="Aggregate rows into a summary table.")
    p_agg.add_argument("--in", dest="in_csv", required=True, nargs="+",
                       help="One or more results CSV files (e.g. per-node outputs).")
    p_agg.add_argument("--summary-out", default=None, help="Markdown summary path.")
    p_agg.add_argument("--summary-csv", default=None, help="CSV summary path.")

    args = parser.parse_args(argv)

    if args.cmd == "row":
        metrics = compute_run_metrics(args.run_log, args.manifest, args.label, args.seed)
        append_row(args.out, metrics)
        if args.confusion_out:
            with open(args.confusion_out, "w") as handle:
                json.dump(metrics["_confusion"], handle, indent=2)
        printable = {k: metrics[k] for k in ROW_FIELDS}
        print(json.dumps(printable, indent=2))
        print(
            f"\nselection_accuracy={metrics['tool_selection_accuracy']}  "
            f"noise_misselection_rate={metrics['noise_tool_misselection_rate']}"
        )
    elif args.cmd == "aggregate":
        rows: List[Dict[str, Any]] = []
        for path in args.in_csv:
            rows.extend(_read_csv(path))
        summary = aggregate(rows)
        table = summary_to_markdown(summary)
        print(table)
        if args.summary_out:
            with open(args.summary_out, "w") as handle:
                handle.write(table + "\n")
        if args.summary_csv and summary:
            with open(args.summary_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
                writer.writeheader()
                writer.writerows(summary)


if __name__ == "__main__":
    main()
