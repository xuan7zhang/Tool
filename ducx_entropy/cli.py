"""One command: extract the three granularities for a condition and dump them.

    # explicit run log
    python -m ducx_entropy.cli extract --run-log LOG.json \
        --label distractor_5 --seed 0 --out-dir logs/entropy

    # or point at a condition log dir and auto-pick the latest run log
    python -m ducx_entropy.cli extract --run-dir logs/noise_distractor_5_seed0 \
        --label distractor_5 --seed 0 --out-dir logs/entropy

Writes (seed in the name so multi-seed runs don't clobber; rows also carry
condition+seed so the analyzer can group them):
  {label}_seed{seed}_pertoken.jsonl -- one line per turn = granularity B
                             (per-turn token sequence) with the per-token
                             granularity-A fields embedded in each line's
                             ``tokens`` array.
  {label}_seed{seed}_perturn.jsonl  -- one line per turn = granularity C
                             aggregates (incl. reasoning-only mean_logprob /
                             mean_entropy).
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import List, Optional

from .extract import (
    has_any_logprobs,
    load_run_log,
    per_turn_agg_rows,
    per_turn_seq_rows,
)


def discover_run_log(run_dir: str) -> Optional[str]:
    """Latest per-question run log in a condition dir (skips tool_calls/manifest)."""
    candidates = [
        p for p in glob(os.path.join(run_dir, "*.json")) + glob(os.path.join(run_dir, "*.jsonl"))
        if not os.path.basename(p).startswith(("tool_calls_", "noise_manifest_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


def cmd_extract(args: argparse.Namespace) -> None:
    run_log = args.run_log
    if not run_log and args.run_dir:
        run_log = discover_run_log(args.run_dir)
        if not run_log:
            raise SystemExit(f"No run log found under {args.run_dir}")
        print(f"Using run log: {run_log}")
    if not run_log:
        raise SystemExit("Provide --run-log or --run-dir.")

    entries = load_run_log(run_log)
    if args.max_cases:
        entries = entries[: args.max_cases]
    if not has_any_logprobs(entries):
        print(
            "WARNING: no captured logprobs in this run log. Re-run "
            "launch_over_chexbench.py with --capture-logprobs K (K>=20). "
            "Emitting rows with empty token data."
        )

    stem = f"{args.label}_seed{args.seed}"
    pertoken_path = os.path.join(args.out_dir, f"{stem}_pertoken.jsonl")
    perturn_path = os.path.join(args.out_dir, f"{stem}_perturn.jsonl")

    n_seq = _write_jsonl(
        pertoken_path,
        (row for entry in entries for row in per_turn_seq_rows(entry, args.label, args.seed)),
    )
    n_agg = _write_jsonl(
        perturn_path,
        (row for entry in entries for row in per_turn_agg_rows(entry, args.label, args.seed)),
    )

    # Quick provenance summary.
    n_tokens = n_reason = 0
    for entry in entries:
        for row in per_turn_agg_rows(entry, args.label, args.seed):
            n_tokens += row["n_tokens"]
            n_reason += row["n_reasoning_tokens"]
    print(
        f"[{args.label} seed={args.seed}] questions={len(entries)} turns={n_seq} "
        f"tokens={n_tokens} reasoning_tokens={n_reason}\n"
        f"  -> {pertoken_path} ({n_seq} turn-seq rows, granularity A+B)\n"
        f"  -> {perturn_path} ({n_agg} turn-agg rows, granularity C)"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DUCX reasoning-entropy extraction.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="Extract granularities A/B/C for one run.")
    p.add_argument("--run-log", default=None, help="Path to a per-question run log JSONL.")
    p.add_argument("--run-dir", default=None, help="Condition log dir; auto-picks latest log.")
    p.add_argument("--label", required=True, help="Condition label (e.g. distractor_5).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True, help="Directory for the two JSONL outputs.")
    p.add_argument("--max-cases", type=int, default=0, help="Limit questions (0 = all).")
    p.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
