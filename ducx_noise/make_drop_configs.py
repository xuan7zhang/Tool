"""Generate Table 3 (N->D reversibility) drop configs from a +distractor run.

Reads a noisy run's log + manifest, ranks tools by how often the agent selected
them, and emits drop configs that progressively remove the most mis-selected
distractors (D^1, D^2, D^5, D^all) plus a D^+ config that additionally drops the
most-selected *real* tool (native high-confusion tool).

Usage:
    python -m ducx_noise.make_drop_configs \
        --run-log logs/.../noise_distractor_10_seed0_*.json \
        --manifest logs/.../noise_manifest_*.json \
        --base-config ducx_noise/configs/distractor_10.yaml \
        --out-dir ducx_noise/configs/drop
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from .config import NoiseConfig
from .metrics import extract_selected_tools, load_manifest, load_run_log


def selection_counts(run_log_path: str) -> Counter:
    counts: Counter = Counter()
    for entry in load_run_log(run_log_path):
        counts.update(extract_selected_tools(entry))
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-config", required=True,
                        help="The noise config used for the run (e.g. distractor_10.yaml).")
    parser.add_argument("--out-dir", default="ducx_noise/configs/drop")
    parser.add_argument("--levels", default="1,2,5",
                        help="Comma-separated D^k levels to emit (besides all + plus).")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    counts = selection_counts(args.run_log)
    base = NoiseConfig.load(args.base_config)

    distractors = [n for n, p in manifest.items() if p.get("source") == "distractor"]
    reals = [n for n, p in manifest.items() if p.get("source") in ("real", "unreliable")]

    # Rank distractors by selection frequency (descending); unseen -> 0.
    ranked_distractors = sorted(distractors, key=lambda n: (-counts.get(n, 0), n))
    ranked_reals = sorted(reals, key=lambda n: (-counts.get(n, 0), n))

    os.makedirs(args.out_dir, exist_ok=True)
    print("Distractor selection ranking (most->least mis-selected):")
    for n in ranked_distractors:
        print(f"  {n:32s} selected {counts.get(n, 0)}x")

    emitted = []

    def emit(label, drop_list):
        cfg = NoiseConfig.from_dict(base.to_dict())
        cfg.label = label
        cfg.drop.enabled = True
        cfg.drop.tools = list(drop_list)
        path = os.path.join(args.out_dir, f"{label}.yaml")
        cfg.save(path)
        emitted.append(path)
        print(f"  -> {path}  (drops {len(drop_list)}: {drop_list})")

    for k in [int(x) for x in args.levels.split(",") if x.strip()]:
        emit(f"drop_top{k}", ranked_distractors[:k])
    emit("drop_all", ranked_distractors)
    if ranked_reals:
        emit("drop_plus", ranked_distractors + ranked_reals[:1])

    print(f"\nEmitted {len(emitted)} drop configs to {args.out_dir}")
    print("Run them like the others (CONFIGS=\"... drop/drop_top1.yaml ...\").")


if __name__ == "__main__":
    main()
