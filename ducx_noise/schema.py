"""Stage 5 log schema: the per-inference record the runner must persist.

This is the contract Stage 6 analysis reads. `build_record` assembles every
required field from the pieces the eval loop already has, derives the
`outcome_label` from the central taxonomy, and computes the response-likelihood
aggregates from per-token logprobs captured at generation time.

Keeping the schema here (not inline in launch_over_chexbench.py) makes it
testable offline and gives one authoritative field list (`RECORD_FIELDS`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import taxonomy

#: Ordered, authoritative field list for a Stage-5 record.
RECORD_FIELDS = [
    "sample_id",
    "query",
    "image_ref",
    "path",
    "condition",
    "tool_set",
    "tool_roles",
    "n_distractors",
    "distractor_positions",
    "distractor_similarity",
    "raw_output",
    "called_tools",
    "final_answer",
    "is_correct",
    "outcome_label",
    "resp_logprob_sum",
    "resp_logprob_mean",
    "per_token_logprob",
    "seed",
    "model_dtype",
    "decoding",
    "timestamp",
    "run_id",
]

#: Sources whose selection is a distractor (mirrors provenance NOISE, distractor only).
_DISTRACTOR_SOURCE = "distractor"


def tool_roles_from_manifest(manifest_tools: List[Dict[str, Any]]) -> Dict[str, str]:
    """{tool_name -> ToolRole value} from a saved manifest's ``tools`` list.

    A manifest entry is a ToolProvenance dict ({name, source, meta{similarity}}).
    Uses the taxonomy so REAL/DISTRACTOR_OBVIOUS/DISTRACTOR_ALIGNED stay in sync.
    """
    roles: Dict[str, str] = {}
    for prov in manifest_tools or []:
        name = prov.get("name")
        if name is None:
            continue
        roles[name] = taxonomy.classify_tool_role(prov).value
    return roles


def distractor_positions(tool_set: List[str], tool_roles: Dict[str, str]) -> List[int]:
    """Indices in ``tool_set`` (exposed order) whose role is a distractor."""
    out = []
    for i, name in enumerate(tool_set):
        role = tool_roles.get(name)
        if role in ("DISTRACTOR_OBVIOUS", "DISTRACTOR_ALIGNED"):
            out.append(i)
    return out


def answer_logprob_stats(per_token_logprob: Optional[List[float]]) -> Dict[str, Optional[float]]:
    """Sum / mean of the answer-token logprobs (None-safe)."""
    lps = [float(x) for x in (per_token_logprob or []) if x is not None]
    if not lps:
        return {"resp_logprob_sum": None, "resp_logprob_mean": None}
    total = sum(lps)
    return {"resp_logprob_sum": round(total, 6), "resp_logprob_mean": round(total / len(lps), 6)}


def final_answer_logprobs_from_trace(trace: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Per-token logprobs of the FINAL assistant message in an agent trace.

    The serialized trace stores ``logprobs`` (a list of {token, logprob, ...})
    on messages when --capture-logprobs was on. The response likelihood is the
    model's *final answer*, so we take the last message that carries logprobs.
    Returns None when none present (logprobs not captured).
    """
    chosen = None
    for msg in trace or []:
        lp = msg.get("logprobs")
        if lp:
            chosen = lp
    return chosen


def _logprob_values(per_token: Optional[List[Dict[str, Any]]]) -> Optional[List[float]]:
    if not per_token:
        return None
    vals = []
    for item in per_token:
        if isinstance(item, dict) and item.get("logprob") is not None:
            vals.append(float(item["logprob"]))
    return vals or None


def build_record(
    *,
    sample_id: Any,
    query: str,
    image_ref: Any,
    path: Optional[str],
    condition: Optional[str],
    tool_set: List[str],
    tool_roles: Dict[str, str],
    distractor_similarity: Optional[str],
    raw_output: Any,
    called_tools: List[str],
    final_answer: Any,
    is_correct: Optional[bool],
    per_token_logprob: Optional[List[Dict[str, Any]]],
    seed: Optional[int],
    model_dtype: Optional[str],
    decoding: Optional[str],
    timestamp: str,
    run_id: Optional[str],
    malformed_tool_calls: int = 0,
    tool_errors: int = 0,
) -> Dict[str, Any]:
    """Assemble a full Stage-5 record and derive outcome + likelihood fields."""
    positions = distractor_positions(tool_set, tool_roles)
    n_distractors = len(positions)

    # Response likelihood from the captured per-token logprobs (answer segment).
    lp_values = _logprob_values(per_token_logprob)
    lp_stats = answer_logprob_stats(lp_values)

    # Outcome via the central taxonomy. Feed it exactly the signals it reads.
    called_roles_map = {name: tool_roles.get(name) for name in tool_set}
    tax_input = {
        "is_correct": is_correct,
        "n_distractors": n_distractors,
        "distractor_similarity": distractor_similarity,
        "tool_roles": tool_roles,
        "called_tools": called_tools,
        "malformed_tool_calls": malformed_tool_calls,
        "tool_errors": tool_errors,
    }
    outcome_label = taxonomy.classify_outcome(tax_input).value

    record = {
        "sample_id": sample_id,
        "query": query,
        "image_ref": image_ref,
        "path": path,
        "condition": condition,
        "tool_set": tool_set,
        "tool_roles": tool_roles,
        "n_distractors": n_distractors,
        "distractor_positions": positions,
        "distractor_similarity": distractor_similarity,
        "raw_output": raw_output,
        "called_tools": called_tools,
        "final_answer": final_answer,
        "is_correct": is_correct,
        "outcome_label": outcome_label,
        "resp_logprob_sum": lp_stats["resp_logprob_sum"],
        "resp_logprob_mean": lp_stats["resp_logprob_mean"],
        "per_token_logprob": lp_values,
        "seed": seed,
        "model_dtype": model_dtype,
        "decoding": decoding,
        "timestamp": timestamp,
        "run_id": run_id,
    }
    # keep the map used by the taxonomy for behaviour axis (harmless extra info)
    record["_called_roles"] = called_roles_map
    return record


def validate_record(record: Dict[str, Any]) -> List[str]:
    """Return the list of REQUIRED fields missing/None (empty list == valid).

    per_token_logprob is optional-by-default in the spec, so None there is OK;
    likelihood aggregates may be None when logprobs were not captured.
    """
    optional_none_ok = {
        "per_token_logprob",
        "resp_logprob_sum",
        "resp_logprob_mean",
        "path",
        "condition",
        "distractor_similarity",
        "run_id",
    }
    missing = []
    for field in RECORD_FIELDS:
        if field not in record:
            missing.append(field)
        elif record[field] is None and field not in optional_none_ok:
            missing.append(field)
    return missing
