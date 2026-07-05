"""Paired analysis of the dissociation experiment.

Builds a per-question correctness table across conditions (no_tool / oracle /
all_real / self_route), then runs exact-McNemar (binomial on discordant pairs)
for the key contrasts. Greedy decoding => deterministic => the pairing is exact.

    python -m analysis.dissociation_analysis \
        --logs-root /project/6101776/xzhan576/DUCK/logs \
        --dataset logs/toolbias/multitool_probe_n300.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def _latest(logs_root, prefix):
    fs = sorted(glob.glob(f"{logs_root}/{prefix}/{prefix}_*.json"))
    return fs[-1] if fs else None


def _load_correct(path):
    """question_id -> bool is_correct (ok records only)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if isinstance(r, dict) and r.get("status") == "ok":
            out[r["sample_id"]] = bool(r.get("is_correct"))
    return out


def _binom_two_sided(k, n):
    """Exact two-sided binomial p (p=0.5) for k successes in n discordant pairs."""
    if n == 0:
        return 1.0
    from math import comb
    kk = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, kk + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a_correct, b_correct, ids):
    """Paired A vs B over shared ids. Returns discordant cells + exact p.

    n10 = A right & B wrong ; n01 = A wrong & B right.
    """
    n10 = n01 = both = neither = 0
    for q in ids:
        a, b = a_correct.get(q), b_correct.get(q)
        if a is None or b is None:
            continue
        if a and b:
            both += 1
        elif a and not b:
            n10 += 1
        elif not a and b:
            n01 += 1
        else:
            neither += 1
    p = _binom_two_sided(min(n10, n01), n10 + n01)
    return {"n10(A>B)": n10, "n01(B>A)": n01, "both": both, "neither": neither, "p_exact": round(p, 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", default="/project/6101776/xzhan576/DUCK/logs")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)

    ptype = {}
    for l in open(args.dataset):
        if l.strip():
            r = json.loads(l)
            ptype[r["question_id"]] = r["probe_type"]

    root = args.logs_root
    no_tool = _load_correct(_latest(root, "ds_no_tool"))
    all_real = _load_correct(_latest(root, "ds_all_real"))
    self_route = _load_correct(_latest(root, "ds_self_route"))
    # oracle = per-question union of the two split runs
    oracle = {}
    oracle.update(_load_correct(_latest(root, "ds_oracle_cls")))
    oracle.update(_load_correct(_latest(root, "ds_oracle_seg")))

    conds = {"no_tool": no_tool, "oracle": oracle, "all_real": all_real, "self_route": self_route}

    def acc(d, sub=None):
        vals = [v for q, v in d.items() if sub is None or ptype.get(q) == sub]
        return (round(sum(vals) / len(vals), 4), len(vals)) if vals else (None, 0)

    print("=== accuracy by condition (overall / CLS / SEG) ===")
    for name, d in conds.items():
        o, no = acc(d); c, nc = acc(d, "CLS"); s, ns = acc(d, "SEG")
        print(f"  {name:11s} overall={o} (n={no})  CLS={c} (n={nc})  SEG={s} (n={ns})")

    ids = set(all_real) & set(oracle) & set(self_route)
    cls_ids = [q for q in ids if ptype.get(q) == "CLS"]
    print(f"\n=== paired McNemar (shared n={len(ids)}, CLS n={len(cls_ids)}) ===")
    for label, A, B in [("oracle vs all_real", oracle, all_real),
                        ("oracle vs self_route", oracle, self_route),
                        ("self_route vs all_real", self_route, all_real)]:
        print(f"  [{label}] overall: {mcnemar(A, B, ids)}")
        print(f"  [{label}] CLS    : {mcnemar(A, B, cls_ids)}")


if __name__ == "__main__":
    main()
