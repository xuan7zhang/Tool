"""Task 3 -- three-condition compositional evaluation.

Measures the three p's for a macro-tool over ONE query batch / ONE backbone, with
identical answer judgment; the only thing that changes is which tools the
environment exposes (fairness invariant). Reuses ``launch_over_chexbench.py`` (the
agent runs are driven by the shell runner) and ``ducx_noise.metrics``.

Conditions (each = one launch run with a compose config):
* ``p_b``    : expose only t_b; the question carries t_a's real precomputed embedding.
* ``p_c``    : expose only the macro t_c (embedding hidden inside).
* ``p_chain``: expose both atoms t_a, t_b; the model must self-chain.
* ``both_macro`` (optional): expose atoms + macro -> measure macro-selection rate.

This module provides: (1) the p_b embedding-precompute harness, (2) condition config
generation, (3) the summary that computes task_acc per condition (the headline p),
Δ_comp, the criteria table, and macro-selection rate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .config import NoiseConfig
from .metrics import compute_run_metrics, extract_selected_tools, load_manifest, load_run_log


# --------------------------------------------------------------------------
# Condition config generation
# --------------------------------------------------------------------------
def condition_config(kind: str, device: str = "cuda", seed: int = 0,
                     macro_name: str = "cxr_macro") -> NoiseConfig:
    """Build the NoiseConfig for one condition (compose block, exclusive env)."""
    base = {
        "seed": seed, "shuffle_tools": False,
        "compose": {"enabled": True, "atom_groups": ["cxr_densenet"],
                    "device": device, "exclusive": True},
    }
    chain = ["cxr_encoder", "cxr_embedding_classifier"]
    if kind == "p_b":
        base["compose"]["expose_atoms"] = ["cxr_embedding_classifier"]
    elif kind == "p_c":
        base["compose"]["macros"] = [{"name": macro_name, "chain": chain}]
    elif kind == "p_chain":
        base["compose"]["expose_atoms"] = chain
    elif kind == "both_macro":
        base["compose"]["expose_atoms"] = chain
        base["compose"]["macros"] = [{"name": macro_name, "chain": chain}]
        base["compose"]["exclusive"] = True
    else:
        raise ValueError(f"unknown condition kind: {kind}")
    base["label"] = kind
    return NoiseConfig.from_dict(base)


def write_condition_configs(out_dir: str, device: str = "cuda", seed: int = 0,
                            kinds: Optional[List[str]] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    kinds = kinds or ["p_b", "p_c", "p_chain", "both_macro"]
    paths = {}
    for kind in kinds:
        path = os.path.join(out_dir, f"{kind}.yaml")
        condition_config(kind, device=device, seed=seed).save(path)
        paths[kind] = path
    return paths


# --------------------------------------------------------------------------
# p_b harness: precompute the real embedding and inject it into each question
# --------------------------------------------------------------------------
def precompute_embeddings(
    data_file: str,
    out_file: str,
    max_cases: Optional[int] = None,
    device: str = "cuda",
    figures_root: str = ".",
) -> str:
    """Write a derived JSONL where each question carries t_a's real embedding.

    Fairness: the embedding is t_a's *actual* output on the real image (not a gold
    label). The agent (with only t_b exposed) must route these floats into t_b.
    """
    from .atoms import build_cxr_atoms

    enc = build_cxr_atoms(device=device)["cxr_encoder"]
    n = 0
    with open(data_file) as fin, open(out_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if max_cases is not None and n >= max_cases:
                break
            ex = json.loads(line)
            imgs = ex.get("images") or []
            img = None
            for p in imgs:
                cand = os.path.join(figures_root, p) if figures_root else p
                if os.path.exists(cand):
                    img = cand
                    break
            if img is None:
                continue
            emb_out, _ = enc._run(img)
            emb = emb_out.get("embedding")
            if emb is None:
                continue
            blob = "[" + ", ".join(f"{v:.5f}" for v in emb) + "]"
            ex["question"] = (
                ex.get("question", "")
                + "\n\nA precomputed CXR feature embedding for this image is available; "
                "pass it to cxr_embedding_classifier to obtain pathology probabilities, "
                f"then answer.\nEMBEDDING ({emb_out.get('dim')}-d): {blob}"
            )
            fout.write(json.dumps(ex) + "\n")
            n += 1
    return out_file


# --------------------------------------------------------------------------
# Selection diagnostics
# --------------------------------------------------------------------------
def macro_selection_rate(run_log_path: str, macro_name: str = "cxr_macro") -> float:
    """Fraction of questions that selected the macro at least once."""
    entries = load_run_log(run_log_path)
    if not entries:
        return 0.0
    hit = 0
    for e in entries:
        if macro_name in set(extract_selected_tools(e)):
            hit += 1
    return round(hit / len(entries), 4)


def valid_intermediate_rate(run_log_path: str, encoder_name: str = "cxr_encoder") -> Optional[float]:
    """Operational p_a: fraction of t_a calls that returned a valid embedding."""
    entries = load_run_log(run_log_path)
    total = good = 0
    for e in entries:
        for msg in e.get("trace") or []:
            if msg.get("name") == encoder_name and msg.get("type") in (None, "tool", "ToolMessage"):
                total += 1
                content = str(msg.get("content", ""))
                if "embedding" in content and "error" not in content.lower():
                    good += 1
    if total == 0:
        return None
    return round(good / total, 4)


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------
def compute_criteria(p_b: float, p_c: float, p_chain: Optional[float],
                     p_a: float = 1.0, n_options: int = 6,
                     strong_margin: float = 0.15) -> Dict[str, Any]:
    """Three-tier criteria.

    p_a is the *operational* valid-intermediate rate (not a task accuracy), so the
    weak tier reduces to ~"chain-error elimination" and the mid tier is evaluated
    against the best *manual* alternative max(p_b, p_chain) -- i.e. the macro must
    beat both using-t_b-alone and self-chaining. Strong = t_b alone ~ chance while
    the macro is high.
    """
    chance = 1.0 / max(2, n_options)
    manual_best = max(p_b, p_chain if p_chain is not None else 0.0)
    weak = p_c > p_a * p_b
    mid = p_c > manual_best
    strong = (p_b <= chance * 1.25) and (p_c >= p_b + strong_margin)
    return {
        "chance_level": round(chance, 4),
        "weak (p_c > p_a*p_b)": weak,
        "mid (p_c > max manual alt)": mid,
        "strong (p_b~chance & p_c high)": strong,
        "delta_comp_vs_chain": None if p_chain is None else round(p_c - p_chain, 4),
        "delta_comp_vs_b": round(p_c - p_b, 4),
    }


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarize(
    runs: Dict[str, Dict[str, str]],  # kind -> {"run_log":..., "manifest":...}
    macro_name: str = "cxr_macro",
    n_options: int = 6,
) -> Dict[str, Any]:
    """Compute per-condition task_acc (the p's), criteria, and macro-selection rate."""
    per_cond: Dict[str, Any] = {}
    for kind, paths in runs.items():
        m = compute_run_metrics(paths["run_log"], paths.get("manifest"), label=kind, seed=0)
        per_cond[kind] = {
            "task_acc": m["task_accuracy"],
            "sel_acc": m["tool_selection_accuracy"],
            "calls_per_q": m["avg_tool_calls_per_question"],
            "n_questions": m["n_questions"],
        }
    p_b = _get(per_cond, "p_b")
    p_c = _get(per_cond, "p_c")
    p_chain = _get(per_cond, "p_chain")
    p_a = 1.0
    if "p_chain" in runs:
        v = valid_intermediate_rate(runs["p_chain"]["run_log"])
        if v is not None:
            p_a = v
    criteria = compute_criteria(p_b or 0.0, p_c or 0.0, p_chain, p_a=p_a, n_options=n_options)
    msr = None
    if "both_macro" in runs:
        msr = macro_selection_rate(runs["both_macro"]["run_log"], macro_name)
    return {
        "p_a_operational": p_a,
        "per_condition": per_cond,
        "p_b": p_b, "p_c": p_c, "p_chain": p_chain,
        "macro_selection_rate": msr,
        "criteria": criteria,
    }


def _get(per_cond: Dict[str, Any], kind: str) -> Optional[float]:
    if kind in per_cond and per_cond[kind]["task_acc"] is not None:
        return float(per_cond[kind]["task_acc"])
    return None


def summary_to_markdown(summary: Dict[str, Any]) -> str:
    lines = ["## Compositional environment -- three-condition result", ""]
    lines.append("| condition | task_acc (p) | sel_acc | calls/q | n |")
    lines.append("| --- | --- | --- | --- | --- |")
    for kind, d in summary["per_condition"].items():
        lines.append(
            f"| {kind} | {d['task_acc']} | {d['sel_acc']} | {d['calls_per_q']} | {d['n_questions']} |"
        )
    c = summary["criteria"]
    lines += [
        "",
        f"- p_b={summary['p_b']}  p_c={summary['p_c']}  p_chain={summary['p_chain']}  "
        f"p_a(op)={summary['p_a_operational']}  chance={c['chance_level']}",
        f"- Δ_comp = p_c − p_chain = {c['delta_comp_vs_chain']} ; p_c − p_b = {c['delta_comp_vs_b']}",
        f"- macro-selection rate (atoms+macro) = {summary['macro_selection_rate']}",
        "",
        "**Criteria:**",
        f"- weak  (p_c > p_a·p_b): {c['weak (p_c > p_a*p_b)']}",
        f"- mid   (p_c > best manual alt max(p_b,p_chain)): {c['mid (p_c > max manual alt)']}",
        f"- strong (p_b≈chance & p_c high): {c['strong (p_b~chance & p_c high)']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compositional 3-condition tools.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("configs", help="Write p_b/p_c/p_chain/both_macro condition configs.")
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--seed", type=int, default=0)

    pp = sub.add_parser("precompute-pb", help="Write p_b data file with embeddings injected.")
    pp.add_argument("--data-file", required=True)
    pp.add_argument("--out-file", required=True)
    pp.add_argument("--max-cases", type=int, default=None)
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--figures-root", default=".")

    ps = sub.add_parser("summarize", help="Summarize conditions into the criteria table.")
    ps.add_argument("--runs", required=True, help="JSON: {kind: {run_log, manifest}}")
    ps.add_argument("--out", default=None, help="Markdown summary path.")
    ps.add_argument("--n-options", type=int, default=6)

    args = parser.parse_args(argv)
    if args.cmd == "configs":
        paths = write_condition_configs(args.out_dir, device=args.device, seed=args.seed)
        print(json.dumps(paths, indent=2))
    elif args.cmd == "precompute-pb":
        out = precompute_embeddings(args.data_file, args.out_file, args.max_cases,
                                    args.device, args.figures_root)
        print(f"wrote {out}")
    elif args.cmd == "summarize":
        runs = json.loads(args.runs) if args.runs.strip().startswith("{") else json.load(open(args.runs))
        summary = summarize(runs, n_options=args.n_options)
        table = summary_to_markdown(summary)
        print(table)
        if args.out:
            with open(args.out, "w") as h:
                h.write(table + "\n")
        with open(os.path.splitext(args.out or "compose_summary")[0] + ".json", "w") as h:
            json.dump(summary, h, indent=2)


if __name__ == "__main__":
    main()
