"""Training-free tool retriever for tool-space optimization.

Given a query and a tool REGISTRY (real tools + injected distractors, each with a
description), select the small relevant subset to expose -- pruning the polluted
tool space back toward the per-query oracle. Two training-free methods:

  * ``lexical`` -- TF-IDF token overlap between the query and each tool
    description (zero dependency, no GPU/model).
  * ``llm``     -- ask the served model to name the relevant tool(s) (a router).

Outputs, per question: the ranked tool selection + retrieval metrics vs the
``required_tool`` ground truth (top-1 accuracy, recall@k, distractor leakage),
and DATASET GROUPS (questions bucketed by their selected real tool) so the
existing fixed-tool-set launcher can measure downstream task accuracy exactly
like the oracle condition.

    python -m ducx_noise.retriever eval \
        --dataset logs/toolbias/multitool_probe.jsonl \
        --registry logs/toolbias/registry.json \
        --method lexical --topk 1 --out-dir logs/toolbias/retriever_lexical
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_WORD = re.compile(r"[a-z0-9]+")
# generic tokens that carry no tool-discriminative signal (appear in every query)
_STOP = {
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "this", "which", "following", "image", "chest", "x", "ray", "xray", "model",
    "determine", "base", "answer", "your", "not", "general", "reasoning", "using",
    "provided", "has", "have", "with", "that", "these", "from", "predicted",
}


def _tok(text: str) -> List[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def load_registry(path: str) -> List[Dict[str, Any]]:
    """Registry = list of {name, description, is_real}."""
    with open(path) as fh:
        reg = json.load(fh)
    return reg["tools"] if isinstance(reg, dict) else reg


# ---------------------------------------------------------------------------
# Lexical (TF-IDF) scorer
# ---------------------------------------------------------------------------
class LexicalScorer:
    """IDF-weighted token-overlap of query vs each tool description."""

    def __init__(self, registry: List[Dict[str, Any]]):
        self.registry = registry
        self.doc_tokens = [set(_tok(t.get("description", "") + " " + t.get("name", "")))
                           for t in registry]
        n = len(registry)
        df = Counter()
        for toks in self.doc_tokens:
            for w in toks:
                df[w] += 1
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> List[float]:
        q = _tok(query)
        out = []
        for toks in self.doc_tokens:
            out.append(sum(self.idf.get(w, 0.0) for w in q if w in toks))
        return out


# ---------------------------------------------------------------------------
# LLM router scorer
# ---------------------------------------------------------------------------
def llm_select(query: str, registry, client, model, topk: int) -> List[str]:
    listing = "\n".join(f"- {t['name']}: {t.get('description','')[:160]}" for t in registry)
    prompt = (
        "You route a query to the RIGHT tools. Given the tool list below and the "
        f"question, return ONLY a JSON list of the {topk} most relevant tool "
        "name(s) (exact names), most relevant first. Pick tools that are actually "
        "needed to answer; ignore irrelevant ones.\n\n"
        f"Tools:\n{listing}\n\nQuestion:\n{query[:600]}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You output only valid JSON."},
                      {"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.0, seed=0,
        )
        txt = resp.choices[0].message.content if resp.choices else "[]"
        m = re.search(r"\[.*\]", txt or "", re.DOTALL)
        names = json.loads(m.group(0)) if m else []
        valid = {t["name"] for t in registry}
        return [n for n in names if n in valid][:topk]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Selection + metrics
# ---------------------------------------------------------------------------
def rank_lexical(query, registry, scorer) -> List[str]:
    sc = scorer.scores(query)
    order = sorted(range(len(registry)), key=lambda i: sc[i], reverse=True)
    return [registry[i]["name"] for i in order]


def evaluate(dataset_path, registry, method, topk, client=None, model=None):
    """Return per-question selections + aggregate retrieval metrics."""
    recs = [json.loads(l) for l in open(dataset_path) if l.strip()]
    real_names = {t["name"] for t in registry if t.get("is_real")}
    scorer = LexicalScorer(registry) if method == "lexical" else None

    per_q = []
    top1_hit = 0
    recall_k = 0
    leaked = 0  # distractors that appear in the top-k selection
    for r in recs:
        q = r["question"]
        if method == "lexical":
            ranked = rank_lexical(q, registry, scorer)
        else:
            ranked = llm_select(q, registry, client, model, topk)
            if not ranked:  # fallback to lexical if the router failed
                ranked = rank_lexical(q, registry, LexicalScorer(registry))
        sel = ranked[:topk]
        req = r.get("required_tool")
        top1_hit += int(bool(ranked) and ranked[0] == req)
        recall_k += int(req in sel)
        leaked += sum(1 for n in sel if n not in real_names)
        # the real tool(s) the retriever would EXPOSE (drop selected distractors)
        sel_real = [n for n in sel if n in real_names]
        per_q.append({**r, "selected": sel, "selected_real": sel_real,
                      "top1": ranked[0] if ranked else None})

    n = len(recs)
    metrics = {
        "method": method, "topk": topk, "n": n,
        "top1_accuracy": round(top1_hit / n, 4) if n else None,
        f"recall@{topk}": round(recall_k / n, 4) if n else None,
        "distractor_leak_per_q": round(leaked / n, 4) if n else None,
    }
    return per_q, metrics


def write_groups(per_q, out_dir):
    """Bucket questions by their selected real tool-set -> one file per bucket,
    so the launcher can run each with `--tools <set>` (oracle-style). Questions
    whose retriever kept NO real tool go to an empty-set bucket (-> no_tool)."""
    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(list)
    for r in per_q:
        key = ",".join(sorted(r["selected_real"]))  # "" = no real tool kept
        groups[key].append(r)
    manifest = {}
    for key, rows in groups.items():
        tag = key.replace(",", "+") if key else "none"
        path = os.path.join(out_dir, f"group_{tag}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if k not in ("selected", "selected_real", "top1")}) + "\n")
        manifest[key] = {"file": path, "n": len(rows), "tools": key}
    with open(os.path.join(out_dir, "groups.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--method", choices=["lexical", "llm"], default="lexical")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--model", default="qwen3-vl-8b")
    args = ap.parse_args(argv)

    registry = load_registry(args.registry)
    client = None
    if args.method == "llm":
        import openai
        client = openai.OpenAI(base_url=args.base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
    per_q, metrics = evaluate(args.dataset, registry, args.method, args.topk, client, args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "selections.jsonl"), "w") as f:
        for r in per_q:
            f.write(json.dumps({"question_id": r["question_id"], "required_tool": r.get("required_tool"),
                                "selected": r["selected"], "top1": r["top1"]}) + "\n")
    manifest = write_groups(per_q, args.out_dir)
    with open(os.path.join(args.out_dir, "retrieval_metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "groups": {k: v["n"] for k, v in manifest.items()}}, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("groups:", {k or "(none)": v["n"] for k, v in manifest.items()})


if __name__ == "__main__":
    main()
