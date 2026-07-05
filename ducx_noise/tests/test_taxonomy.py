"""Stage 2 acceptance: the three labels resolve unambiguously and correctly.

Covers the two axes:
  * outcome axis {SUCCESS, FAILURE, UNRELIABLE} is mutually exclusive.
  * distractor axis is an independent input flag that coexists with any outcome.
"""

from ducx_noise.taxonomy import (
    Outcome,
    ToolRole,
    classify,
    classify_outcome,
    classify_tool_role,
    distractor_kind,
    is_distractor_present,
    is_unreliable,
)


def _clean_correct():
    return {
        "is_correct": True,
        "n_distractors": 0,
        "tool_roles": {"cxr_classifier": "REAL"},
        "called_tools": ["cxr_classifier"],
    }


def test_success_record():
    rec = _clean_correct()
    assert classify_outcome(rec) is Outcome.SUCCESS
    assert not is_distractor_present(rec)
    assert not is_unreliable(rec)


def test_failure_wrong_answer():
    rec = _clean_correct()
    rec["is_correct"] = False
    assert classify_outcome(rec) is Outcome.FAILURE


def test_unreliable_right_answer_but_called_distractor():
    # answer correct, but the model invoked a distractor tool -> UNRELIABLE
    rec = {
        "is_correct": True,
        "n_distractors": 1,
        "tool_roles": {"real_clf": "REAL", "fake_ct": "DISTRACTOR_OBVIOUS"},
        "called_tools": ["fake_ct"],
    }
    assert is_unreliable(rec)
    assert classify_outcome(rec) is Outcome.UNRELIABLE


def test_outcome_is_mutually_exclusive():
    # exactly one outcome label per record, always a valid enum member
    for rec in (
        _clean_correct(),
        {"is_correct": False},
        {"is_correct": True, "malformed_tool_calls": 2},
    ):
        out = classify_outcome(rec)
        assert isinstance(out, Outcome)
        assert sum(out is o for o in Outcome) == 1


def test_distractor_coexists_with_failure():
    # distractor axis (input) and FAILURE outcome coexist on the same record
    rec = {
        "is_correct": False,
        "n_distractors": 3,
        "distractor_similarity": "aligned",
    }
    res = classify(rec)
    assert res["distractor_present"] is True
    assert res["outcome_label"] == "FAILURE"
    assert res["distractor_kind"] == "DISTRACTOR_ALIGNED"


def test_tool_role_mapping():
    assert classify_tool_role({"source": "real"}) is ToolRole.REAL
    assert classify_tool_role({"source": "redundant"}) is ToolRole.REAL
    assert (
        classify_tool_role({"source": "distractor", "meta": {"similarity": "aligned"}})
        is ToolRole.DISTRACTOR_ALIGNED
    )
    assert (
        classify_tool_role({"source": "distractor", "meta": {"similarity": "obvious"}})
        is ToolRole.DISTRACTOR_OBVIOUS
    )
    # distractor with no similarity tag defaults to OBVIOUS
    assert classify_tool_role({"source": "distractor"}) is ToolRole.DISTRACTOR_OBVIOUS


def test_distractor_kind_none_when_absent():
    assert distractor_kind(_clean_correct()) is None
    assert not is_distractor_present({"n_distractors": 0})
