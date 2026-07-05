"""Central label taxonomy for the VLM tool-use / distractor experiment.

This is the single place that defines the three experiment labels and how to
derive them from a run record. Downstream analysis (Stage 6) must import from
here rather than re-implementing any judgment, so a definition change lands in
exactly one file.

Three deliberately-separate axes (per the confirmed draft; the two are NOT
mutually exclusive with each other, see below):

* **Distractor** -- an *input-level* annotation: the tool list presented to the
  model contained at least one tool that should not have been called for this
  query. This says nothing about what the model did; it is a property of the
  environment. Derived by :func:`is_distractor_present` / :func:`distractor_kind`.

* **Outcome** -- a single mutually-exclusive label over {SUCCESS, FAILURE,
  UNRELIABLE} describing what happened:
    - ``FAILURE``     -- outcome level: the final answer is wrong.
    - ``UNRELIABLE``  -- behaviour level: the tool interaction was malformed /
      wrong-tool / untrustworthy-output, *even if the final answer is right*.
    - ``SUCCESS``     -- answer right and no unreliable behaviour.
  Derived by :func:`classify_outcome`.

A record can be BOTH "distractor-present" (input axis) and ``FAILURE`` /
``UNRELIABLE`` (outcome axis) at once -- those are different axes. Within the
outcome axis exactly one label is returned (mutually exclusive); when a record
is both wrong AND unreliable, ``OUTCOME_PRECEDENCE`` decides which wins.

NOTE: every judgment threshold / precedence below is a PLACEHOLDER pending final
sign-off. They are gathered at the top of the file on purpose -- change them
here and every classifier and every downstream table updates.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from .provenance import (
    SOURCE_DISTRACTOR,
    SOURCE_REDUNDANT,
    SOURCE_UNRELIABLE,
)

# ===========================================================================
# PLACEHOLDER judgment constants -- FINAL VALUES PENDING SIGN-OFF.
# ===========================================================================

# --- Distractor (input axis) ------------------------------------------------
# A record counts as "distractor-present" iff its tool_set contains at least
# this many distractor tools.
DISTRACTOR_MIN_COUNT: int = 1  # PLACEHOLDER

# --- Unreliable (behaviour axis) -------------------------------------------
# Which behaviours mark a record UNRELIABLE. Each is an independent switch so
# the final definition can turn any of them on/off without touching logic.
UNRELIABLE_ON_MALFORMED_CALL: bool = True  # PLACEHOLDER: tool-call JSON/schema malformed
UNRELIABLE_ON_WRONG_TOOL: bool = True      # PLACEHOLDER: model called a distractor / non-applicable tool
UNRELIABLE_ON_TOOL_ERROR: bool = True      # PLACEHOLDER: a called tool errored / returned empty / untrusted

# --- Outcome precedence -----------------------------------------------------
# When a record is BOTH wrong (would be FAILURE) and exhibits unreliable
# behaviour (would be UNRELIABLE), which label wins. Default: FAILURE dominates
# (a wrong answer is the headline result regardless of how it got there).
OUTCOME_PRECEDENCE: tuple = ("FAILURE", "UNRELIABLE")  # PLACEHOLDER


# ===========================================================================
# Enums
# ===========================================================================


class Outcome(str, Enum):
    """Mutually-exclusive outcome label for one run record."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNRELIABLE = "UNRELIABLE"


class ToolRole(str, Enum):
    """Ground-truth role of a single tool presented to the model."""

    REAL = "REAL"
    DISTRACTOR_OBVIOUS = "DISTRACTOR_OBVIOUS"  # description obviously unrelated
    DISTRACTOR_ALIGNED = "DISTRACTOR_ALIGNED"  # description mimics a real tool (hard)


#: Roles whose selection counts as a distractor mis-selection.
DISTRACTOR_ROLES = frozenset({ToolRole.DISTRACTOR_OBVIOUS, ToolRole.DISTRACTOR_ALIGNED})


# ===========================================================================
# Tool-role mapping (provenance -> ToolRole)
# ===========================================================================


def classify_tool_role(provenance: Dict[str, Any]) -> ToolRole:
    """Map one manifest ``ToolProvenance`` dict to a :class:`ToolRole`.

    Judgment condition: distractor tools split into OBVIOUS vs ALIGNED by the
    ``meta.similarity`` tag set at generation time (Stage 3); everything else
    (real / redundant / unreliable / macro / atom) is REAL for the purpose of
    the distractor axis (those are legitimate or real-backed selections).

    +example: ``{"source": "distractor", "meta": {"similarity": "aligned"}}`` -> DISTRACTOR_ALIGNED
    -example: ``{"source": "real"}``                                          -> REAL   (a genuine tool is never a distractor)
    """
    source = (provenance or {}).get("source")
    if source == SOURCE_DISTRACTOR:
        similarity = ((provenance or {}).get("meta") or {}).get("similarity")
        if similarity == "aligned":
            return ToolRole.DISTRACTOR_ALIGNED
        return ToolRole.DISTRACTOR_OBVIOUS
    return ToolRole.REAL


# ===========================================================================
# Distractor axis (input level)
# ===========================================================================


def _record_tool_roles(record: Dict[str, Any]) -> List[ToolRole]:
    """Resolve the roles of the tools presented in one record.

    Prefers an explicit ``tool_roles`` field (name->role string, from Stage 5);
    falls back to a manifest-style ``tool_set``/``manifest`` list of provenance
    dicts. Returns [] when neither is present.
    """
    roles = record.get("tool_roles")
    if isinstance(roles, dict):
        return [ToolRole(v) if not isinstance(v, ToolRole) else v for v in roles.values()]
    if isinstance(roles, list):
        return [ToolRole(v) if not isinstance(v, ToolRole) else v for v in roles]
    manifest = record.get("manifest") or record.get("tool_set_provenance")
    if isinstance(manifest, list):
        return [classify_tool_role(p) for p in manifest]
    return []


def is_distractor_present(record: Dict[str, Any]) -> bool:
    """Input-axis label: does the presented tool list contain distractor tools?

    Judgment condition: ``#distractor tools >= DISTRACTOR_MIN_COUNT``. Uses the
    explicit ``n_distractors`` count when present, else counts distractor roles.

    +example: tool_set = [real, real, distractor_obvious]  -> True
    -example: tool_set = [real, real, redundant_copy]       -> False (redundant is not a distractor)
    """
    n = record.get("n_distractors")
    if isinstance(n, int):
        return n >= DISTRACTOR_MIN_COUNT
    roles = _record_tool_roles(record)
    n_distractor = sum(1 for r in roles if r in DISTRACTOR_ROLES)
    return n_distractor >= DISTRACTOR_MIN_COUNT


def distractor_kind(record: Dict[str, Any]) -> Optional[ToolRole]:
    """The dominant distractor kind present, or None if no distractor.

    ALIGNED dominates OBVIOUS when both are present (report the harder kind).
    """
    roles = _record_tool_roles(record)
    if any(r == ToolRole.DISTRACTOR_ALIGNED for r in roles):
        return ToolRole.DISTRACTOR_ALIGNED
    if any(r == ToolRole.DISTRACTOR_OBVIOUS for r in roles):
        return ToolRole.DISTRACTOR_OBVIOUS
    # Fall back to the config-level similarity tag if roles are absent.
    if is_distractor_present(record):
        sim = record.get("distractor_similarity")
        if sim == "aligned":
            return ToolRole.DISTRACTOR_ALIGNED
        if sim == "obvious":
            return ToolRole.DISTRACTOR_OBVIOUS
    return None


# ===========================================================================
# Outcome axis (mutually exclusive)
# ===========================================================================


def _called_tool_roles(record: Dict[str, Any]) -> List[ToolRole]:
    """Roles of the tools the model actually CALLED (needs a name->role map)."""
    called = record.get("called_tools") or []
    roles_map = record.get("tool_roles")
    if not isinstance(roles_map, dict):
        return []
    out = []
    for name in called:
        role = roles_map.get(name)
        if role is None:
            continue
        out.append(role if isinstance(role, ToolRole) else ToolRole(role))
    return out


def is_unreliable(record: Dict[str, Any]) -> bool:
    """Behaviour-axis predicate: was the tool interaction untrustworthy?

    True (subject to the switches at the top of the file) when any of:
      * a tool call was malformed (``malformed_tool_calls`` > 0), or
      * the model called a distractor / non-applicable tool
        (``wrong_tool_calls`` > 0, or a called tool has a distractor role), or
      * a called tool errored / returned empty / untrusted (``tool_errors`` > 0).

    Independent of correctness -- a record can be UNRELIABLE with a right answer.

    +example: answer correct but the model invoked ``distractor_ct_planner`` -> True
    -example: answer correct, one clean call to a real classifier            -> False
    """
    if UNRELIABLE_ON_MALFORMED_CALL and int(record.get("malformed_tool_calls", 0) or 0) > 0:
        return True
    if UNRELIABLE_ON_TOOL_ERROR and int(record.get("tool_errors", 0) or 0) > 0:
        return True
    if UNRELIABLE_ON_WRONG_TOOL:
        if int(record.get("wrong_tool_calls", 0) or 0) > 0:
            return True
        if any(r in DISTRACTOR_ROLES for r in _called_tool_roles(record)):
            return True
    return False


def _is_correct(record: Dict[str, Any]) -> Optional[bool]:
    """Correctness per decision-point 2 (extract.py exact-match writes is_correct)."""
    if "is_correct" in record and record["is_correct"] is not None:
        return bool(record["is_correct"])
    return None


def classify_outcome(record: Dict[str, Any]) -> Outcome:
    """Return the single mutually-exclusive outcome label for a record.

    Precedence (with default ``OUTCOME_PRECEDENCE = ("FAILURE","UNRELIABLE")``):
      1. wrong answer   -> FAILURE
      2. else unreliable-> UNRELIABLE
      3. else           -> SUCCESS
    If ``OUTCOME_PRECEDENCE`` is flipped, an unreliable+wrong record reports
    UNRELIABLE instead. A record with unknown correctness and no unreliable
    signal defaults to SUCCESS only if explicitly correct; unknown -> UNRELIABLE
    is NOT assumed here (unknown correctness returns FAILURE-safe? see below).

    +example: is_correct=False, clean tools           -> FAILURE
    -example: is_correct=True,  clean tools            -> SUCCESS
    """
    correct = _is_correct(record)
    unreliable = is_unreliable(record)
    wrong = correct is False  # unknown correctness is treated as "not wrong" here

    if wrong and unreliable:
        return Outcome.FAILURE if OUTCOME_PRECEDENCE[0] == "FAILURE" else Outcome.UNRELIABLE
    if wrong:
        return Outcome.FAILURE
    if unreliable:
        return Outcome.UNRELIABLE
    return Outcome.SUCCESS


def classify(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: all axes for one record in a single dict.

    Returns ``{"outcome_label", "distractor_present", "distractor_kind"}`` --
    outcome_label / distractor_kind are enum *values* (strings) for JSON/CSV.
    """
    kind = distractor_kind(record)
    return {
        "outcome_label": classify_outcome(record).value,
        "distractor_present": is_distractor_present(record),
        "distractor_kind": kind.value if kind is not None else None,
    }
