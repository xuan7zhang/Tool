"""Tool-necessary probe task for the compositional experiment.

ChestAgentBench MCQs are answerable from clinical text, so task_acc there does not
isolate the agent's ability to *consume the tool's output* (the model bypasses the
tool). This builds a task where the answer is **determined by the real classifier**
and cannot be read off the text:

    "Which pathology has the HIGHEST model-predicted probability for this X-ray?"
    options = the classifier's TOP-6 pathologies (all plausible) in random order;
    correct = the classifier's argmax.

Ground truth comes from the real DenseNet (the same weights the macro wraps), so it
is objective and free. Because the 6 options are the classifier's own top-6,
eyeballing the image cannot reliably distinguish #1 from #2-6 -> the tool is
necessary, and guessing without it is ~chance (1/6).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from typing import List, Optional


def _collect_images(source_jsonl: str, figures_root: str, limit: int) -> List[str]:
    """Unique existing image paths drawn from a source metadata JSONL."""
    seen, out = set(), []
    with open(source_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for p in ex.get("images") or []:
                cand = os.path.join(figures_root, p) if figures_root else p
                if cand not in seen and os.path.exists(cand):
                    seen.add(cand)
                    out.append(p if not figures_root else p)  # keep relative form
                    if len(out) >= limit:
                        return out
    return out


def build_probe_dataset(
    source_jsonl: str,
    out_file: str,
    n: int,
    device: str = "cuda",
    seed: int = 0,
    figures_root: str = ".",
    n_options: int = 6,
) -> str:
    """Write a tool-necessary MCQ dataset (same schema as ChestAgentBench)."""
    from medrax.tools.classification import ChestXRayClassifierTool

    clf = ChestXRayClassifierTool(device=device)
    rng = random.Random(seed)
    rel_imgs = _collect_images(source_jsonl, figures_root, n * 3)
    rng.shuffle(rel_imgs)

    letters = string.ascii_uppercase
    written = 0
    with open(out_file, "w") as fout:
        for rel in rel_imgs:
            if written >= n:
                break
            img = os.path.join(figures_root, rel) if figures_root else rel
            probs, meta = clf._run(img)
            if not isinstance(probs, dict) or "error" in probs:
                continue
            ranked = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
            top = ranked[:n_options]
            if len(top) < n_options:
                continue
            correct_name = top[0][0]
            options = [name for name, _ in top]
            rng.shuffle(options)
            correct_letter = letters[options.index(correct_name)]
            opt_lines = "\n".join(f"{letters[i]}) {name}" for i, name in enumerate(options))
            question = (
                "A chest X-ray image is provided. Using the chest X-ray pathology "
                "classification model, determine which of the following pathologies has "
                "the HIGHEST predicted probability for this image. Base your answer on the "
                "model's predicted probabilities, not on general reasoning.\n" + opt_lines
            )
            rec = {
                "question_id": f"probe_{written}",
                "case_id": f"probe_{written}",
                "question": question,
                "answer": correct_letter,
                "images": [rel],
                "probe_correct_pathology": correct_name,
                "probe_top": [name for name, _ in top],
            }
            fout.write(json.dumps(rec) + "\n")
            written += 1
    return out_file


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build a tool-necessary probe MCQ dataset.")
    ap.add_argument("--source", default="data/chestagentbench/metadata.jsonl")
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures-root", default=".")
    ap.add_argument("--n-options", type=int, default=6)
    args = ap.parse_args(argv)
    out = build_probe_dataset(args.source, args.out_file, args.n, args.device,
                              args.seed, args.figures_root, args.n_options)
    n = sum(1 for _ in open(out))
    print(f"wrote {n} probe questions -> {out}")


if __name__ == "__main__":
    main()
