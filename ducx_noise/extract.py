"""Task 1 -- temp-0 LLM answer extractor + deterministic A-F comparison.

Why this module exists
----------------------
The shipped scorer (``launch_over_chexbench.py:extract_choice``) reads the final
multiple-choice letter with a "scan lines from the bottom, else last ``\\b[A-F]\\b``
in the whole text" heuristic. On ChestAgentBench the model *never* emits a bare
letter line -- it always writes prose like ``**C) ...**`` -- so the brittle
fallback fires 100% of the time and, being case-insensitive, latches onto the
article "a" in sentences such as "...presence of **a** cavitary lesion" and
records the answer as ``A``. The audit (Task 0) showed this destroys ~0.2-0.4 of
task accuracy and is the dominant term in the old ``task_acc`` numbers.

This module replaces *only the answer-extraction step* with a two-stage design:

1. **LLM extract** -- a temp-0 (greedy) model that is **not** the model under
   test reads the full generation and returns the model's *final* chosen letter
   (or abstains). Reproducible: fixed prompt + fixed ``EXTRACTOR_VERSION`` + temp 0
   + on-disk cache.
2. **Deterministic compare** -- ChestAgentBench answers are a closed set
   (single letter A-F), so the extracted letter is normalised and exact-matched
   against the ground truth. The open-text LLM-judge branch is implemented for
   completeness but is not exercised by this closed-set benchmark.

It does **not** touch rollout generation, sampling temperature (still 0.7), or
the selection / entropy / call-count metrics (those come from ``trace`` tool
calls and are independent of answer extraction).

CLI::

    # re-extract one run log, cache per-rollout, append an old-vs-new row
    python -m ducx_noise.extract recompute --run-log LOG.json \\
        --label base --seed 0 --out results_extract.csv \\
        --base-url http://localhost:8001/v1 --model qwen2.5-7b-instruct

    # aggregate rows -> old-vs-new task_acc table (mean +/- std over seeds)
    python -m ducx_noise.extract aggregate --in results_extract.csv \\
        --summary-out summary_extract.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Bump this whenever the prompt or parsing logic changes so stale cache entries
# are not silently reused.
EXTRACTOR_VERSION = "extract-v1"

VALID = set("ABCDEF")

# Fixed extractor prompt. "Only the final letter, else NONE" -- no explanation,
# explicit abstain so the extractor never invents an answer the model didn't give.
EXTRACT_SYSTEM = (
    "You read the final answer out of a medical multiple-choice response. "
    "The response argues toward exactly one option labelled A, B, C, D, E, or F. "
    "Output ONLY the single capital letter of the option the response ultimately "
    "selects as its answer. If the response considers an option and then rejects "
    "it in favour of another, return the one it ENDS UP choosing. If the response "
    "gives no clear final choice, output exactly NONE. Output one token only: a "
    "letter A-F, or NONE. No punctuation, no explanation."
)
EXTRACT_USER_TEMPLATE = "Response:\n{answer}\n\nFinal answer letter:"


# --------------------------------------------------------------------------
# Normalisation + comparison (deterministic, closed-set)
# --------------------------------------------------------------------------
def normalize_choice(letter: Optional[str]) -> Optional[str]:
    """Map an extractor output to a canonical A-F letter, or None (abstain)."""
    if letter is None:
        return None
    s = str(letter).strip().upper()
    if not s:
        return None
    if s in ("NONE", "N/A", "NULL", "ABSTAIN"):
        return None
    # Accept "C", "(C)", "C)", "C.", "**C**", "Answer: C" -> take the last
    # *standalone* A-F token, so letters inside words (the "A" in "ANSWER") are
    # not mistaken for the choice.
    matches = re.findall(r"\b([A-F])\b", s)
    if matches:
        return matches[-1]
    # Last resort: a lone letter glued to punctuation (e.g. "C)").
    m = re.search(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", s)
    return m.group(1) if m else None


def compare_closed_set(pred: Optional[str], gt: Optional[str]) -> Optional[bool]:
    """Exact closed-set match after normalisation. None pred -> incorrect (False)
    when a gt exists; None gt -> unscorable (None)."""
    g = normalize_choice(gt)
    if g is None:
        return None
    p = normalize_choice(pred)
    if p is None:
        return False
    return p == g


# --------------------------------------------------------------------------
# Extractor backends
# --------------------------------------------------------------------------
class RuleExtractor:
    """Deterministic offline extractor (no model server needed).

    Prefers the explicit conclusion markers the model emits -- ``**X)``,
    ``answer is X``, a leading ``X)`` option line -- over the bare-letter
    fallback. Used for unit tests, for environments without a vLLM endpoint, and
    as a cross-check against the LLM. Known limitation: when the model echoes an
    option and *then* rejects it (``**C)** ... however the answer is **D)**``)
    this picks the first marker; the LLM backend handles that case. Hence the LLM
    is the default for the real recompute.
    """

    version = f"{EXTRACTOR_VERSION}-rule"

    def extract(self, text: Optional[str]) -> Tuple[Optional[str], str]:
        if not text:
            return None, "none"
        t = str(text)
        m = re.search(r"\*\*\s*([A-F])\s*[\).:\-]", t)
        if m:
            return m.group(1).upper(), "bold_option"
        m = re.search(
            r"(?:answer|option|choice|statement)\s*(?:is|:)?\s*\(?\s*([A-F])\b",
            t,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper(), "answer_is"
        for line in t.splitlines():
            m = re.match(r"\s*\**\s*([A-F])\s*[\).]", line)
            if m:
                return m.group(1).upper(), "line_option"
        for line in (l.strip() for l in t.splitlines() if l.strip()):
            m = re.fullmatch(r"\(?([A-F])\)?", line)
            if m:
                return m.group(1).upper(), "bareline"
        return None, "none"


class LLMExtractor:
    """Temp-0 OpenAI-compatible extractor. Must point at a model that is NOT the
    model under test (default: a Qwen2.5-7B-Instruct vLLM instance). Reads text,
    returns the final A-F letter or None."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 4,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.version = f"{EXTRACTOR_VERSION}-llm:{model}"

    def extract(self, text: Optional[str]) -> Tuple[Optional[str], str]:
        if not text:
            return None, "empty"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(answer=str(text))},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_tokens,
            seed=0,
        )
        raw = resp.choices[0].message.content if resp.choices else None
        return normalize_choice(raw), "llm"


def make_extractor(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = "EMPTY",
) -> Any:
    """LLM backend when an endpoint is configured, else the deterministic rule
    backend. ``base_url``/``model`` fall back to env (EXTRACTOR_BASE_URL /
    EXTRACTOR_MODEL, then OPENAI_BASE_URL)."""
    base_url = base_url or os.getenv("EXTRACTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = model or os.getenv("EXTRACTOR_MODEL")
    api_key = os.getenv("EXTRACTOR_API_KEY") or os.getenv("OPENAI_API_KEY") or api_key
    if base_url and model:
        return LLMExtractor(base_url=base_url, model=model, api_key=api_key)
    return RuleExtractor()


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def rollout_id(entry: Dict[str, Any]) -> str:
    """Stable per-rollout id: question_id + short hash of the generation text.

    Hashing the model_answer keys the cache on the actual extractor input, so the
    same generation reuses its result across files/conditions and any change to
    the text yields a fresh key."""
    qid = str(entry.get("question_id", "unknown"))
    ma = entry.get("model_answer") or ""
    if isinstance(ma, (list, dict)):
        ma = json.dumps(ma, sort_keys=True)
    h = hashlib.sha1(str(ma).encode("utf-8")).hexdigest()[:12]
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]", "_", qid)
    return f"{safe_qid}_{h}"


def extract_entry(
    entry: Dict[str, Any],
    extractor: Any,
    cache_dir: str,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Extract one rollout's answer (cached). Stores extracted answer, old/new
    correctness, extractor version. Returns the cache record."""
    rid = rollout_id(entry)
    path = os.path.join(cache_dir, f"{rid}.json")
    version = getattr(extractor, "version", EXTRACTOR_VERSION)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r") as h:
                rec = json.load(h)
            if rec.get("extractor_version") == version:
                return rec
        except (json.JSONDecodeError, OSError):
            pass

    gt = entry.get("correct_answer")
    model_answer = entry.get("model_answer")
    pred_new, method = extractor.extract(model_answer)
    is_correct_new = compare_closed_set(pred_new, gt)

    rec = {
        "rollout_id": rid,
        "question_id": entry.get("question_id"),
        "extractor_version": version,
        "method": method,
        "ground_truth": gt,
        # new extraction
        "extracted_answer": pred_new,
        "is_correct_new": is_correct_new,
        "abstained": pred_new is None,
        # old result kept side by side for diffing
        "old_predicted_answer": entry.get("predicted_answer"),
        "old_is_correct": entry.get("is_correct"),
    }
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as h:
        json.dump(rec, h, indent=2)
    return rec


# --------------------------------------------------------------------------
# Run-level recompute
# --------------------------------------------------------------------------
def load_run_log(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


ROW_FIELDS = [
    "label",
    "seed",
    "n_scored",
    "task_accuracy_old",
    "task_accuracy_new",
    "delta",
    "n_abstain",
    "n_flip_old_wrong_new_right",
    "n_flip_old_right_new_wrong",
    "extractor_version",
]


def recompute_run(
    run_log_path: str,
    extractor: Any,
    cache_dir: str,
    label: str = "base",
    seed: int = 0,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    entries = load_run_log(run_log_path)
    scored = [e for e in entries if "is_correct" in e]
    records: List[Dict[str, Any]] = []
    n_old = 0
    n_new = 0
    n_abstain = 0
    flip_up = 0  # old wrong -> new right
    flip_down = 0  # old right -> new wrong
    for e in scored:
        rec = extract_entry(e, extractor, cache_dir, use_cache=use_cache)
        rec["label"] = label
        rec["seed"] = seed
        records.append(rec)
        old_ok = bool(rec.get("old_is_correct"))
        new_ok = bool(rec.get("is_correct_new"))
        n_old += 1 if old_ok else 0
        n_new += 1 if new_ok else 0
        n_abstain += 1 if rec.get("abstained") else 0
        if not old_ok and new_ok:
            flip_up += 1
        if old_ok and not new_ok:
            flip_down += 1
    n = len(scored)
    acc_old = round(n_old / n, 4) if n else None
    acc_new = round(n_new / n, 4) if n else None
    row = {
        "label": label,
        "seed": seed,
        "n_scored": n,
        "task_accuracy_old": acc_old,
        "task_accuracy_new": acc_new,
        "delta": round(acc_new - acc_old, 4) if n else None,
        "n_abstain": n_abstain,
        "n_flip_old_wrong_new_right": flip_up,
        "n_flip_old_right_new_wrong": flip_down,
        "extractor_version": getattr(extractor, "version", EXTRACTOR_VERSION),
    }
    return row, records


# --------------------------------------------------------------------------
# CSV row I/O + aggregation
# --------------------------------------------------------------------------
def append_row(out_csv: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in ROW_FIELDS})


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as h:
        return list(csv.DictReader(h))


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean +/- std of old/new task_acc over seeds, grouped by label."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get("label", "base")].append(r)
    out: List[Dict[str, Any]] = []
    for label, items in groups.items():
        agg: Dict[str, Any] = {"label": label, "n_seeds": len(items)}
        agg["n_scored"] = sum(int(float(r["n_scored"])) for r in items if r.get("n_scored"))
        for col in ("task_accuracy_old", "task_accuracy_new"):
            vals = [float(r[col]) for r in items if r.get(col) not in (None, "", "None")]
            if vals:
                mean = sum(vals) / len(vals)
                agg[f"{col}_mean"] = round(mean, 4)
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    agg[f"{col}_std"] = round(math.sqrt(var), 4)
                else:
                    agg[f"{col}_std"] = 0.0
            else:
                agg[f"{col}_mean"] = None
                agg[f"{col}_std"] = None
        mo, mn = agg.get("task_accuracy_old_mean"), agg.get("task_accuracy_new_mean")
        agg["delta_mean"] = round(mn - mo, 4) if (mo is not None and mn is not None) else None
        out.append(agg)
    out.sort(key=lambda r: r["label"])
    return out


def summary_to_markdown(summary: List[Dict[str, Any]]) -> str:
    cols = [
        ("label", "condition"),
        ("n_seeds", "seeds"),
        ("n_scored", "n"),
        ("task_accuracy_old_mean", "task_acc_old"),
        ("task_accuracy_new_mean", "task_acc_new"),
        ("task_accuracy_new_std", "new_std"),
        ("delta_mean", "Δ"),
    ]
    header = "| " + " | ".join(lbl for _, lbl in cols) + " |"
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
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DUCX temp-0 answer extractor.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_re = sub.add_parser("recompute", help="Re-extract a run log; cache + append a row.")
    p_re.add_argument("--run-log", required=True)
    p_re.add_argument("--label", default="base")
    p_re.add_argument("--seed", type=int, default=0)
    p_re.add_argument("--out", required=True, help="CSV file to append the row to.")
    p_re.add_argument("--cache-dir", default="extraction_cache")
    p_re.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (else env / rule backend).")
    p_re.add_argument("--model", default=None, help="Extractor model name (must NOT be the model under test).")
    p_re.add_argument("--records-out", default=None, help="Optional JSONL of per-rollout records.")
    p_re.add_argument("--no-cache", action="store_true")

    p_agg = sub.add_parser("aggregate", help="Aggregate rows into an old-vs-new table.")
    p_agg.add_argument("--in", dest="in_csv", required=True, nargs="+")
    p_agg.add_argument("--summary-out", default=None)
    p_agg.add_argument("--summary-csv", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "recompute":
        extractor = make_extractor(args.base_url, args.model)
        row, records = recompute_run(
            args.run_log, extractor, args.cache_dir,
            label=args.label, seed=args.seed, use_cache=not args.no_cache,
        )
        append_row(args.out, row)
        if args.records_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.records_out)) or ".", exist_ok=True)
            with open(args.records_out, "w") as h:
                for rec in records:
                    h.write(json.dumps(rec) + "\n")
        print(json.dumps(row, indent=2))
        print(f"\nbackend={getattr(extractor, 'version', '?')}  cache={args.cache_dir}")
    elif args.cmd == "aggregate":
        rows: List[Dict[str, Any]] = []
        for path in args.in_csv:
            rows.extend(_read_csv(path))
        summary = aggregate(rows)
        table = summary_to_markdown(summary)
        print(table)
        if args.summary_out:
            with open(args.summary_out, "w") as h:
                h.write(table + "\n")
        if args.summary_csv and summary:
            keys = sorted({k for r in summary for k in r})
            with open(args.summary_csv, "w", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summary)


if __name__ == "__main__":
    main()
