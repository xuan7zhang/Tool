"""Stage 5 acceptance: a built record carries every field, correct types."""

from ducx_noise import schema


def _mock_record():
    tool_roles = {
        "chest_xray_classifier": "REAL",
        "fake_ct_planner": "DISTRACTOR_OBVIOUS",
        "chest_xray_classifier_v2": "DISTRACTOR_ALIGNED",
    }
    tool_set = ["chest_xray_classifier", "fake_ct_planner", "chest_xray_classifier_v2"]
    per_token = [{"token": "B", "logprob": -0.12}, {"token": ".", "logprob": -0.30}]
    return schema.build_record(
        sample_id="11583_x",
        query="Which finding is present?",
        image_ref=["figures/11583/figure_1.jpg"],
        path="tool_useful_distractor",
        condition="tool_useful_distractor",
        tool_set=tool_set,
        tool_roles=tool_roles,
        distractor_similarity="aligned",
        raw_output="The answer is B.",
        called_tools=["chest_xray_classifier"],
        final_answer="B",
        is_correct=True,
        per_token_logprob=per_token,
        seed=0,
        model_dtype="fp16",
        decoding="greedy",
        timestamp="2026-07-03T00:00:00",
        run_id="run_abc",
    )


def test_all_required_fields_present():
    rec = _mock_record()
    assert schema.validate_record(rec) == []


def test_distractor_positions_and_count():
    rec = _mock_record()
    # indices 1 and 2 are distractors in the exposed order
    assert rec["distractor_positions"] == [1, 2]
    assert rec["n_distractors"] == 2


def test_likelihood_values_reasonable():
    rec = _mock_record()
    # sum of negative logprobs, mean between them
    assert rec["resp_logprob_sum"] == -0.42
    assert rec["resp_logprob_mean"] == -0.21
    assert rec["per_token_logprob"] == [-0.12, -0.30]


def test_outcome_label_derived():
    rec = _mock_record()
    # correct answer + only a REAL tool was called -> SUCCESS
    assert rec["outcome_label"] == "SUCCESS"


def test_no_logprobs_yields_none_but_still_valid():
    rec = schema.build_record(
        sample_id="q", query="?", image_ref=[], path=None, condition="no_tool",
        tool_set=[], tool_roles={}, distractor_similarity=None,
        raw_output="A", called_tools=[], final_answer="A", is_correct=False,
        per_token_logprob=None, seed=1, model_dtype="fp16", decoding="greedy",
        timestamp="t", run_id="r",
    )
    assert rec["resp_logprob_sum"] is None
    assert schema.validate_record(rec) == []  # likelihood is optional-None
    assert rec["outcome_label"] == "FAILURE"


def test_roles_from_manifest():
    manifest_tools = [
        {"name": "real_a", "source": "real"},
        {"name": "d1", "source": "distractor", "meta": {"similarity": "aligned"}},
        {"name": "d2", "source": "distractor", "meta": {"similarity": "obvious"}},
    ]
    roles = schema.tool_roles_from_manifest(manifest_tools)
    assert roles == {
        "real_a": "REAL",
        "d1": "DISTRACTOR_ALIGNED",
        "d2": "DISTRACTOR_OBVIOUS",
    }
    assert schema.distractor_positions(["real_a", "d1", "d2"], roles) == [1, 2]
