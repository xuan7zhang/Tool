"""Multi-tool probe suite: tool-necessary MCQs where DIFFERENT questions require
DIFFERENT real tools.

Motivation (from the distractor study): the model never mis-selects a distractor,
yet accuracy drops purely because irrelevant tool descriptions pollute context. So
"tool space optimization" is a retrieval/pruning problem, and to test it we need a
task where the *correct tool subset varies per query* -- otherwise pruning is
trivial. The single-tool probe (`probe_task.py`) cannot exercise selection.

Two probe types, each objectively answerable ONLY from a specific real tool
(ground truth comes from the tool itself, so it is free and unambiguous):

  * CLS  -- "which pathology has the HIGHEST predicted probability?"
            required tool: chest_xray_classifier   (GT = classifier argmax)
  * SEG  -- "which anatomical structure has the LARGEST area?"
            required tool: chest_xray_segmentation (GT = max area_cm2)

Each record carries ``required_tool`` (the retrieval target) and ``probe_type``.
The eval then compares, per condition, the accuracy under different tool exposures
(no_tool / oracle=only-required / all_real / all_real+distractors) to quantify how
much a tool-space optimizer can recover. Whether each probe is truly tool-necessary
is validated empirically by its no_tool accuracy (~chance).

Same output schema as ChestAgentBench so `launch_over_chexbench.py` consumes it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from typing import Dict, List, Optional

CLS_TOOL = "chest_xray_classifier"
SEG_TOOL = "chest_xray_segmentation"

# Structures excluded from being the SEG answer: the lungs almost always have the
# largest area, so "which is largest?" is guessable without the tool (no_tool acc
# ~0.6). Ranking area AMONG non-lung structures genuinely needs the segmentation
# output, making the probe tool-necessary.
_SEG_EXCLUDE = {"Left Lung", "Right Lung"}

_LETTERS = string.ascii_uppercase


def _collect_images(source_jsonl: str, figures_root: str, limit: int) -> List[str]:
    """Unique existing (relative) image paths from a source metadata JSONL."""
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
                    out.append(p)
                    if len(out) >= limit:
                        return out
    return out


def _mc(options: List[str], correct_name: str, rng: random.Random):
    """Shuffle options, return (option_list, correct_letter, opt_lines)."""
    opts = list(options)
    rng.shuffle(opts)
    letter = _LETTERS[opts.index(correct_name)]
    lines = "\n".join(f"{_LETTERS[i]}) {name}" for i, name in enumerate(opts))
    return opts, letter, lines


def _cls_probe(clf, img_abs, rel, idx, rng, n_options):
    probs, _ = clf._run(img_abs)
    if not isinstance(probs, dict) or "error" in probs:
        return None
    ranked = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
    top = [name for name, _ in ranked[:n_options]]
    if len(top) < n_options:
        return None
    correct = top[0]
    opts, letter, lines = _mc(top, correct, rng)
    q = (
        "A chest X-ray image is provided. Using the chest X-ray pathology "
        "classification model, determine which of the following pathologies has the "
        "HIGHEST predicted probability for this image. Base your answer on the model's "
        "predicted probabilities, not on general reasoning.\n" + lines
    )
    return {
        "question_id": f"mtp_cls_{idx}",
        "case_id": f"mtp_cls_{idx}",
        "question": q,
        "answer": letter,
        "images": [rel],
        "probe_type": "CLS",
        "required_tool": CLS_TOOL,
        "probe_correct": correct,
        "probe_options": opts,
    }


def _seg_probe(seg, img_abs, rel, idx, rng, n_options, metric="intensity"):
    output, _ = seg._run(img_abs)
    metrics = (output or {}).get("metrics") if isinstance(output, dict) else None
    if not metrics:
        return None
    if metric == "area":
        # physical area, lungs excluded (they are near-always largest -> guessable)
        vals = {o: float(m.get("area_cm2", 0.0)) for o, m in metrics.items()
                if float(m.get("area_cm2", 0.0)) > 0 and o not in _SEG_EXCLUDE}
        superl, prop = "LARGEST", "area"
        phrase = "occupies the LARGEST area"
    else:
        # mean pixel intensity: varies per image, NOT predictable from anatomy priors
        # -> genuinely needs the segmentation output (tool-necessary)
        vals = {o: float(m.get("mean_intensity", 0.0)) for o, m in metrics.items()
                if float(m.get("area_cm2", 0.0)) > 0}
        superl, prop = "HIGHEST", "mean intensity"
        phrase = "has the HIGHEST mean pixel intensity (appears brightest)"
    if len(vals) < n_options:
        return None
    ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][0]
    pool = [o for o, _ in ranked]
    chosen = [top] + rng.sample(pool[1:], n_options - 1)
    opts, letter, lines = _mc(chosen, top, rng)
    q = (
        "A chest X-ray image is provided. Using the chest X-ray anatomical "
        f"segmentation model, determine which of the following anatomical structures "
        f"{phrase} in this image. Base your answer on the segmentation model's "
        "measured values, not on general reasoning.\n" + lines
    )
    return {
        "question_id": f"mtp_seg_{idx}",
        "case_id": f"mtp_seg_{idx}",
        "question": q,
        "answer": letter,
        "images": [rel],
        "probe_type": "SEG",
        "required_tool": SEG_TOOL,
        "probe_correct": top,
        "probe_options": opts,
    }


def build_multitool_dataset(source_jsonl, out_file, n, device="cuda", seed=0,
                            figures_root=".", n_options=6, seg_frac=0.5,
                            seg_metric="intensity"):
    """Write a mixed CLS/SEG tool-necessary MCQ dataset (ChestAgentBench schema)."""
    from medrax.tools.classification import ChestXRayClassifierTool
    from medrax.tools.segmentation import ChestXRaySegmentationTool

    clf = ChestXRayClassifierTool(device=device)
    seg = ChestXRaySegmentationTool(device=device)
    rng = random.Random(seed)
    imgs = _collect_images(source_jsonl, figures_root, n * 4)
    rng.shuffle(imgs)

    written = {"CLS": 0, "SEG": 0}
    target_seg = int(round(n * seg_frac))
    target_cls = n - target_seg
    with open(out_file, "w") as fout:
        for rel in imgs:
            if written["CLS"] + written["SEG"] >= n:
                break
            img_abs = os.path.join(figures_root, rel) if figures_root else rel
            # choose which probe this image serves, honoring the target mix
            want_seg = (written["SEG"] < target_seg) and (
                written["CLS"] >= target_cls or rng.random() < seg_frac)
            idx = written["SEG" if want_seg else "CLS"]
            try:
                rec = (_seg_probe(seg, img_abs, rel, idx, rng, n_options, seg_metric) if want_seg
                       else _cls_probe(clf, img_abs, rel, idx, rng, n_options))
            except Exception:
                rec = None
            if rec is None:
                # fall back to the other probe type on this image
                try:
                    rec = (_cls_probe(clf, img_abs, rel, written["CLS"], rng, n_options)
                           if want_seg else
                           _seg_probe(seg, img_abs, rel, written["SEG"], rng, n_options, seg_metric))
                except Exception:
                    rec = None
                if rec is None:
                    continue
            fout.write(json.dumps(rec) + "\n")
            written[rec["probe_type"]] += 1
    return out_file, written


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build a multi-tool tool-necessary MCQ dataset.")
    ap.add_argument("--source", default="data/chestagentbench/metadata.jsonl")
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures-root", default=".")
    ap.add_argument("--n-options", type=int, default=6)
    ap.add_argument("--seg-frac", type=float, default=0.5)
    ap.add_argument("--seg-metric", choices=["intensity", "area"], default="intensity",
                    help="SEG probe target: 'intensity' (per-image, tool-necessary) "
                         "or 'area' (tracks anatomy priors, more guessable).")
    args = ap.parse_args(argv)
    out, written = build_multitool_dataset(
        args.source, args.out_file, args.n, args.device, args.seed,
        args.figures_root, args.n_options, args.seg_frac, args.seg_metric)
    total = sum(1 for _ in open(out))
    print(f"wrote {total} multi-tool probes -> {out}  (CLS={written['CLS']} SEG={written['SEG']})")


if __name__ == "__main__":
    main()
