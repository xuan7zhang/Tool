"""Unit tests: one per noise type + identity/regression, config, reproducibility."""

import json

from ducx_noise import NoiseConfig, apply_noise
from ducx_noise.provenance import (
    SOURCE_DISTRACTOR,
    SOURCE_REDUNDANT,
    SOURCE_UNRELIABLE,
)


# --------------------------------------------------------------------------
# Regression: default / disabled config is an identity transform
# --------------------------------------------------------------------------
def test_identity_no_config(base_tools):
    env = apply_noise(base_tools, None)
    assert env.tool_names == [t.name for t in base_tools]
    assert all(a is b for a, b in zip(env.tools, base_tools))  # same objects
    assert all(p.source == "real" for p in env.manifest.values())


def test_identity_all_disabled(base_tools):
    env = apply_noise(base_tools, NoiseConfig())
    assert env.tool_names == [t.name for t in base_tools]
    assert all(a is b for a, b in zip(env.tools, base_tools))


# --------------------------------------------------------------------------
# 1.1 Distractor
# --------------------------------------------------------------------------
def test_distractor_count_and_tags(base_tools):
    cfg = NoiseConfig.from_dict(
        {"seed": 1, "shuffle_tools": False, "distractor": {"enabled": True, "count": 5}}
    )
    env = apply_noise(base_tools, cfg)
    distractors = [n for n, p in env.manifest.items() if p.source == SOURCE_DISTRACTOR]
    assert len(distractors) == 5
    assert len(env.tools) == len(base_tools) + 5
    # distractors return a canned, non-real response
    d_tool = next(t for t in env.tools if t.name in distractors)
    out = d_tool.invoke({"image_path": "x.jpg"})
    assert isinstance(out, tuple) and out[1].get("distractor") is True


def test_distractor_sweep_levels(base_tools):
    for level in (0, 2, 5, 10):
        cfg = NoiseConfig.from_dict({"distractor": {"enabled": level > 0, "count": level}})
        env = apply_noise(base_tools, cfg)
        n_d = sum(1 for p in env.manifest.values() if p.source == SOURCE_DISTRACTOR)
        assert n_d == level


# --------------------------------------------------------------------------
# 1.2 Unreliable
# --------------------------------------------------------------------------
def test_unreliable_always_fails(base_tools):
    cfg = NoiseConfig.from_dict(
        {"unreliable": {"enabled": True, "p_fail": 1.0, "failure_mode": "error"}}
    )
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_classifier")
    result = tool.invoke({"image_path": "x.jpg"})
    assert isinstance(result, str) and "tool_execution_error" in result
    assert env.manifest["chest_xray_classifier"].source == SOURCE_UNRELIABLE


def test_unreliable_never_fails_passthrough(base_tools):
    cfg = NoiseConfig.from_dict({"unreliable": {"enabled": True, "p_fail": 0.0}})
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_classifier")
    out = tool.invoke({"image_path": "x.jpg"})
    assert out[0] == {"Pneumonia": 0.7}  # real output preserved


def test_unreliable_empty_mode(base_tools):
    cfg = NoiseConfig.from_dict(
        {"unreliable": {"enabled": True, "p_fail": 1.0, "failure_mode": "empty"}}
    )
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_segmentation")
    assert tool.invoke({"image_path": "x.jpg"}) == ""


def test_unreliable_is_reproducible(base_tools):
    cfg = NoiseConfig.from_dict({"seed": 42, "unreliable": {"enabled": True, "p_fail": 0.5}})
    seq_a = _failure_sequence(base_tools, cfg, n=20)
    seq_b = _failure_sequence(base_tools, cfg, n=20)
    assert seq_a == seq_b


def _failure_sequence(base_tools, cfg, n):
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_classifier")
    return [isinstance(tool.invoke({"image_path": "x.jpg"}), str) for _ in range(n)]


# --------------------------------------------------------------------------
# 1.3 Redundant
# --------------------------------------------------------------------------
def test_redundant_equivalent_output(base_tools):
    cfg = NoiseConfig.from_dict(
        {
            "shuffle_tools": False,
            "redundant": {"enabled": True, "copies_per_tool": 1, "tools": ["chest_xray_classifier"]},
        }
    )
    env = apply_noise(base_tools, cfg)
    copies = [n for n, p in env.manifest.items() if p.source == SOURCE_REDUNDANT]
    assert len(copies) == 1
    copy_tool = next(t for t in env.tools if t.name == copies[0])
    real_tool = next(t for t in base_tools if t.name == "chest_xray_classifier")
    assert copy_tool.invoke({"image_path": "y.jpg"}) == real_tool.invoke({"image_path": "y.jpg"})
    assert env.manifest[copies[0]].is_noise_tool is True
    assert env.manifest[copies[0]].base_name == "chest_xray_classifier"


# --------------------------------------------------------------------------
# 1.4 Description corruption
# --------------------------------------------------------------------------
def test_description_corruption_changes_and_preserves_original(base_tools):
    cfg = NoiseConfig.from_dict(
        {"description_corruption": {"enabled": True, "style": "vague", "level": 0.9}}
    )
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_classifier")
    prov = env.manifest["chest_xray_classifier"]
    assert "description" in prov.noises
    assert prov.meta["original_description"] != tool.description
    assert prov.meta["corrupted_description"] == tool.description


# --------------------------------------------------------------------------
# 1.5 Schema noise
# --------------------------------------------------------------------------
def test_schema_noise_renames_but_stays_callable(base_tools):
    cfg = NoiseConfig.from_dict(
        {
            "schema_noise": {
                "enabled": True,
                "rename_params": True,
                "inject_optional": 2,
                "shuffle_order": True,
            }
        }
    )
    env = apply_noise(base_tools, cfg)
    tool = next(t for t in env.tools if t.name == "chest_xray_classifier")
    params = list(tool.args_schema.model_fields.keys())
    assert "image_path" not in params  # renamed
    arg_map = env.manifest["chest_xray_classifier"].meta["arg_map"]
    exposed = next(k for k, v in arg_map.items() if v == "image_path")
    # still callable, injected optionals ignored, real output preserved
    out = tool.invoke({exposed: "z.jpg"})
    assert out[0] == {"Pneumonia": 0.7}


# --------------------------------------------------------------------------
# Drop operator D
# --------------------------------------------------------------------------
def test_drop_removes_distractors_restores_base(base_tools):
    # N: add 3 distractors; D: drop all of them -> should restore E0 tool set.
    cfg_n = NoiseConfig.from_dict(
        {"seed": 1, "shuffle_tools": False, "distractor": {"enabled": True, "count": 3}}
    )
    env_n = apply_noise(base_tools, cfg_n)
    distractors = [n for n, p in env_n.manifest.items() if p.source == SOURCE_DISTRACTOR]
    assert len(distractors) == 3

    cfg_dn = NoiseConfig.from_dict(
        {
            "seed": 1,
            "shuffle_tools": False,
            "distractor": {"enabled": True, "count": 3},
            "drop": {"enabled": True, "tools": distractors},
        }
    )
    env_dn = apply_noise(base_tools, cfg_dn)
    assert sorted(env_dn.tool_names) == sorted(t.name for t in base_tools)
    assert all(not p.is_noise_tool for p in env_dn.manifest.values())


def test_drop_native_real_tool(base_tools):
    cfg = NoiseConfig.from_dict(
        {"drop": {"enabled": True, "tools": ["chest_xray_segmentation"]}}
    )
    env = apply_noise(base_tools, cfg)
    assert "chest_xray_segmentation" not in env.tool_names
    assert "chest_xray_classifier" in env.tool_names


# --------------------------------------------------------------------------
# Config round-trip + reproducibility
# --------------------------------------------------------------------------
def test_config_json_roundtrip(tmp_path):
    cfg = NoiseConfig.from_dict(
        {"seed": 9, "distractor": {"enabled": True, "count": 3}}
    )
    path = tmp_path / "cfg.json"
    cfg.save(str(path))
    loaded = NoiseConfig.load(str(path))
    assert loaded.seed == 9
    assert loaded.distractor.enabled and loaded.distractor.count == 3


def test_config_yaml_roundtrip(tmp_path):
    cfg = NoiseConfig.from_dict({"seed": 3, "unreliable": {"enabled": True, "p_fail": 0.25}})
    path = tmp_path / "cfg.yaml"
    cfg.save(str(path))
    loaded = NoiseConfig.load(str(path))
    assert loaded.unreliable.p_fail == 0.25


def test_same_seed_same_env(base_tools):
    cfg = NoiseConfig.from_dict({"seed": 5, "distractor": {"enabled": True, "count": 4}})
    a = apply_noise(base_tools, cfg)
    b = apply_noise(base_tools, cfg)
    assert a.tool_names == b.tool_names


def test_different_seed_differs(base_tools):
    c1 = NoiseConfig.from_dict({"seed": 1, "distractor": {"enabled": True, "count": 4}})
    c2 = NoiseConfig.from_dict({"seed": 2, "distractor": {"enabled": True, "count": 4}})
    a = apply_noise(base_tools, c1)
    b = apply_noise(base_tools, c2)
    # Order and/or names should differ across seeds (shuffle + generation).
    assert a.tool_names != b.tool_names


def test_manifest_saves(base_tools, tmp_path):
    cfg = NoiseConfig.from_dict({"distractor": {"enabled": True, "count": 2}})
    env = apply_noise(base_tools, cfg)
    path = tmp_path / "manifest.json"
    env.save_manifest(str(path))
    data = json.loads(path.read_text())
    assert "tools" in data and len(data["tools"]) == len(env.tools)
