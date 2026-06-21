"""Task 2 -- automatic pair mining for the compose (K) operator.

Goal: find tool pairs ``(t_a, t_b)`` where ``t_a.output_type`` is compatible with
``t_b.input_type`` AND the intermediate type is **non-semantic** (the model cannot
self-chain it), then rank by a ϕ score = IO-compat x non-readability x low-p_b.

DUCX tools do not declare machine-readable IO types, so we attach a lightweight,
opt-in typed-IO descriptor per tool name (no behaviour change). The descriptor marks
each type as semantic (text/probs the model can read & forward) or non-semantic
(embedding/mask/bbox/tensor it cannot transcribe).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple

# IO type vocabulary. ``semantic`` types are readable/forwardable by the model;
# non-semantic types are the targets for macro-composition.
SEMANTIC_TYPES = {"text", "probs", "report", "path", "image_path", "dicom_path"}
NONSEMANTIC_TYPES = {"embedding", "mask", "bbox", "tensor"}


@dataclass
class ToolIO:
    """Typed-IO descriptor for one tool (keyed by tool name)."""

    name: str
    input_type: str
    output_type: str
    output_semantic: bool  # is the output something the model can read & forward?
    note: str = ""


# Opt-in registry. ``image`` is the VLM-readable pixel input; ``embedding`` is the
# non-semantic intermediate produced by the DenseNet-split encoder.
TOOL_IO: Dict[str, ToolIO] = {
    "chest_xray_classifier": ToolIO("chest_xray_classifier", "image", "probs", True),
    "chest_xray_report_generator": ToolIO("chest_xray_report_generator", "image", "report", True),
    "chest_xray_expert": ToolIO("chest_xray_expert", "image", "text", True),
    "llava_med_qa": ToolIO("llava_med_qa", "image", "text", True),
    "chest_xray_segmentation": ToolIO("chest_xray_segmentation", "image", "path", True,
                                      "exposes a viz path + named metrics, not a raw mask"),
    "xray_phrase_grounding": ToolIO("xray_phrase_grounding", "image", "bbox", True,
                                    "bbox is only 4 numbers -> copyable, treated as semantic"),
    "dicom_processor": ToolIO("dicom_processor", "dicom_path", "image_path", True),
    "image_visualizer": ToolIO("image_visualizer", "image_path", "image_path", True),
    # K-operator atoms (DenseNet split):
    "cxr_encoder": ToolIO("cxr_encoder", "image", "embedding", False,
                          "1024-d raw float vector -> non-semantic, un-transcribable"),
    "cxr_embedding_classifier": ToolIO("cxr_embedding_classifier", "embedding", "probs", True),
}

# Which (producer_type -> consumer_type) hops are IO-compatible.
COMPATIBLE = {
    ("embedding", "embedding"),
    ("image_path", "image"),
    ("image_path", "image_path"),
    ("dicom_path", "image_path"),
    ("mask", "mask"),
    ("bbox", "bbox"),
}


@dataclass
class PairCandidate:
    t_a: str
    t_b: str
    intermediate_type: str
    io_compatible: bool
    intermediate_nonsemantic: bool
    p_b: Optional[float] = None  # probed low-is-good
    phi: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return {
            "t_a": self.t_a,
            "t_b": self.t_b,
            "intermediate_type": self.intermediate_type,
            "io_compatible": self.io_compatible,
            "intermediate_nonsemantic": self.intermediate_nonsemantic,
            "p_b": "" if self.p_b is None else round(self.p_b, 4),
            "phi": round(self.phi, 4),
        }


def _io_compatible(a: ToolIO, b: ToolIO) -> bool:
    if a.output_type == b.input_type:
        return True
    return (a.output_type, b.input_type) in COMPATIBLE


def mine_pairs(
    tool_names: Optional[List[str]] = None,
    io_registry: Optional[Dict[str, ToolIO]] = None,
    p_b_probe: Optional[Callable[[str, str], float]] = None,
    require_nonsemantic: bool = True,
) -> List[PairCandidate]:
    """Rank candidate macro pairs.

    Args:
        tool_names: pool to mine over (default: all in the IO registry).
        io_registry: typed-IO descriptors (default: TOOL_IO).
        p_b_probe: optional fn (t_a, t_b) -> measured p_b in [0,1]; lower ranks higher.
        require_nonsemantic: keep only pairs whose intermediate is non-semantic.

    Returns:
        Candidates sorted by phi descending. phi = io_compat(1) * nonsemantic(1) *
        (1 - p_b) when a probe is given, else io_compat * nonsemantic.
    """
    io = io_registry or TOOL_IO
    names = tool_names or list(io.keys())
    candidates: List[PairCandidate] = []
    for a_name, b_name in permutations(names, 2):
        if a_name not in io or b_name not in io:
            continue
        a, b = io[a_name], io[b_name]
        compat = _io_compatible(a, b)
        if not compat:
            continue
        nonsemantic = not a.output_semantic
        if require_nonsemantic and not nonsemantic:
            continue
        cand = PairCandidate(
            t_a=a_name, t_b=b_name,
            intermediate_type=a.output_type,
            io_compatible=compat,
            intermediate_nonsemantic=nonsemantic,
            meta={"note_a": a.note, "note_b": b.note},
        )
        if p_b_probe is not None:
            cand.p_b = float(p_b_probe(a_name, b_name))
            cand.phi = (1.0 if compat else 0.0) * (1.0 if nonsemantic else 0.0) * (1.0 - cand.p_b)
        else:
            cand.phi = (1.0 if compat else 0.0) * (1.0 if nonsemantic else 0.0)
        candidates.append(cand)
    candidates.sort(key=lambda c: (c.phi, c.intermediate_nonsemantic), reverse=True)
    return candidates


def write_ranked_csv(candidates: List[PairCandidate], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = ["t_a", "t_b", "intermediate_type", "io_compatible",
              "intermediate_nonsemantic", "p_b", "phi"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow(c.to_row())


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mine macro-tool candidate pairs.")
    parser.add_argument("--out", default="ducx_noise/cache/mined_pairs.csv")
    parser.add_argument("--all", action="store_true",
                        help="Include semantic-intermediate pairs too (default: non-semantic only).")
    args = parser.parse_args(argv)
    cands = mine_pairs(require_nonsemantic=not args.all)
    write_ranked_csv(cands, args.out)
    print(f"Mined {len(cands)} candidate pair(s) -> {args.out}\n")
    for c in cands[:10]:
        flag = "non-semantic" if c.intermediate_nonsemantic else "semantic"
        print(f"  phi={c.phi:.2f}  {c.t_a} -> [{c.intermediate_type}:{flag}] -> {c.t_b}")


if __name__ == "__main__":
    main()
