"""Unit tests for noise selection metrics (Task 2)."""

import json

from ducx_noise.metrics import (
    aggregate,
    compute_run_metrics,
    extract_selected_tools,
    selection_entropy_bits,
)


def _write_run(path, questions):
    with open(path, "w") as handle:
        handle.write("\n".join(json.dumps(q) for q in questions))


def _question(qid, tool_names, correct):
    return {
        "question_id": qid,
        "status": "ok",
        "is_correct": correct,
        "trace": [{"type": "ai", "tool_calls": [{"name": n} for n in tool_names]}],
    }


def _manifest(path):
    data = {
        "config": None,
        "tools": [
            {"name": "chest_xray_classifier", "source": "real", "is_noise_tool": False},
            {"name": "chest_xray_segmentation", "source": "real", "is_noise_tool": False},
            {"name": "bone_density_estimator", "source": "distractor", "is_noise_tool": True},
            {"name": "chest_xray_classifier_alt", "source": "redundant", "is_noise_tool": True},
        ],
    }
    with open(path, "w") as handle:
        json.dump(data, handle)


def test_extract_selected_tools():
    entry = {"trace": [
        {"tool_calls": [{"name": "a"}, {"name": "b"}]},
        {"tool_calls": [{"name": "a<|channel|>extra"}]},
        {"tool_calls": []},
    ]}
    assert extract_selected_tools(entry) == ["a", "b", "a"]


def test_selection_entropy():
    assert selection_entropy_bits([]) == 0.0
    assert selection_entropy_bits(["a", "a", "a"]) == 0.0
    assert abs(selection_entropy_bits(["a", "b"]) - 1.0) < 1e-9


def test_run_metrics_misselection(tmp_path):
    mp = tmp_path / "m.json"
    lp = tmp_path / "log.json"
    _manifest(mp)
    _write_run(lp, [
        _question("a", ["chest_xray_classifier", "bone_density_estimator"], True),
        _question("b", ["chest_xray_classifier_alt", "chest_xray_segmentation"], False),
    ])
    m = compute_run_metrics(str(lp), str(mp), label="t", seed=0)
    # 4 known selections, 2 noise (distractor + redundant)
    assert m["total_tool_calls"] == 4
    assert m["noise_tool_misselection_rate"] == 0.5
    assert m["tool_selection_accuracy"] == 0.5
    assert m["task_accuracy"] == 0.5
    assert m["n_distractor_tools"] == 1 and m["n_redundant_tools"] == 1


def test_base_env_no_manifest(tmp_path):
    lp = tmp_path / "log.json"
    _write_run(lp, [_question("a", ["chest_xray_classifier"], True)])
    m = compute_run_metrics(str(lp), None, label="base", seed=0)
    assert m["noise_tool_misselection_rate"] == 0.0
    assert m["tool_selection_accuracy"] == 1.0


def test_aggregate_across_seeds():
    rows = [
        {"label": "d5", "tool_selection_accuracy": "0.5", "noise_tool_misselection_rate": "0.5",
         "selection_entropy_bits": "1.0", "avg_tool_calls_per_question": "2.0", "task_accuracy": "0.4",
         "unknown_tool_rate": "0.0"},
        {"label": "d5", "tool_selection_accuracy": "0.7", "noise_tool_misselection_rate": "0.3",
         "selection_entropy_bits": "1.2", "avg_tool_calls_per_question": "2.4", "task_accuracy": "0.6",
         "unknown_tool_rate": "0.0"},
    ]
    summary = aggregate(rows)
    assert len(summary) == 1
    assert summary[0]["tool_selection_accuracy_mean"] == 0.6
    assert summary[0]["n_seeds"] == 2
