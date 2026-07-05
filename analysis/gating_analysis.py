"""Gating analysis: does a per-query tool gate beat always/never-tools?

Reads gate_never / gate_always run logs, joins by question_id + qtype from the
mixed dataset, and reports (paired, exact-McNemar):
  * per qtype: never vs always (does the tool help / hurt on this type?)
  * overall: never / always / gated(oracle) accuracy
  * gated vs always, gated vs never (the gating win)

gated(oracle) per question: text_answerable -> never's result, tool_necessary ->
always's result (expose tools only where they are necessary).

    python -m analysis.gating_analysis --dataset logs/toolbias/gating_mixed.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb


def _latest(root, prefix):
    fs = sorted(glob.glob(f"{root}/{prefix}/{prefix}_*.json"))
    return fs[-1] if fs else None


def _load(path):
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
    if n == 0:
        return 1.0
    kk = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / (2 ** n))


def mcnemar(A, B, ids):
    n10 = n01 = 0
    for q in ids:
        a, b = A.get(q), B.get(q)
        if a is None or b is None:
            continue
        if a and not b:
            n10 += 1
        elif not a and b:
            n01 += 1
    return {"A>B": n10, "B>A": n01, "p": round(_binom_two_sided(min(n10, n01), n10 + n01), 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", default="/project/6101776/xzhan576/DUCK/logs")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)

    qtype = {}
    for l in open(args.dataset):
        if l.strip():
            r = json.loads(l)
            qtype[r["question_id"]] = r["qtype"]

    never = _load(_latest(args.logs_root, "gate_never"))
    always = _load(_latest(args.logs_root, "gate_always"))
    ids = set(never) & set(always)
    text_ids = [q for q in ids if qtype.get(q) == "text_answerable"]
    tool_ids = [q for q in ids if qtype.get(q) == "tool_necessary"]

    def acc(d, sub):
        v = [d[q] for q in sub if q in d]
        return (round(sum(v) / len(v), 4), len(v)) if v else (None, 0)

    print("=== per-qtype accuracy (paired) ===")
    for name, sub in [("text_answerable", text_ids), ("tool_necessary", tool_ids)]:
        na, nn = acc(never, sub); al, an = acc(always, sub)
        mc = mcnemar(always, never, sub)  # always vs never (A=always)
        print(f"  {name:16s} never={na} always={al} (n={nn})  "
              f"always-vs-never McNemar {mc}")

    # gated(oracle): text -> never, tool -> always
    gated = {}
    for q in ids:
        gated[q] = never[q] if qtype.get(q) == "text_answerable" else always[q]

    def acc_all(d):
        v = [d[q] for q in ids]
        return round(sum(v) / len(v), 4)

    print("\n=== overall (n={}) ===".format(len(ids)))
    print(f"  never_tool  = {acc_all(never)}")
    print(f"  always_tool = {acc_all(always)}")
    print(f"  gated       = {acc_all(gated)}   (oracle gate)")
    print(f"  gated vs always: {mcnemar(gated, always, ids)}  (win concentrated on text-answerable)")
    print(f"  gated vs never : {mcnemar(gated, never, ids)}   (win concentrated on tool-necessary)")


if __name__ == "__main__":
    main()
