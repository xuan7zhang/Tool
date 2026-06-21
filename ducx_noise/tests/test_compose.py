"""Unit tests for the compositional (K-operator) environment.

Covers: DenseNet-split atom parity, MacroTool intermediate-hiding + pipe parity,
the three condition tool-sets, pair-mining filter, identity regression, and the
criteria logic. Uses CPU + the real (cached) DenseNet weights; no GPU/LLM needed.
"""

import numpy as np
import pytest
from PIL import Image

from ducx_noise import NoiseConfig, apply_noise


@pytest.fixture(scope="module")
def cxr_image(tmp_path_factory):
    # Any image works for parity (split == original holds for all inputs).
    path = tmp_path_factory.mktemp("img") / "x.png"
    arr = (np.random.RandomState(0).rand(224, 224) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture(scope="module")
def atoms():
    from ducx_noise.atoms import build_cxr_atoms

    return build_cxr_atoms(device="cpu")


# --------------------------------------------------------------------------
# Atom split parity
# --------------------------------------------------------------------------
def test_atom_split_parity(atoms, cxr_image):
    from medrax.tools.classification import ChestXRayClassifierTool

    enc, head = atoms["cxr_encoder"], atoms["cxr_embedding_classifier"]
    emb_out, _ = enc._run(cxr_image)
    assert emb_out["dim"] == 1024
    split, _ = head._run(emb_out["embedding"])
    ref, _ = ChestXRayClassifierTool(device="cpu")._run(cxr_image)
    diff = max(abs(float(split[k]) - float(ref[k])) for k in ref)
    assert diff < 1e-5  # split == original real model


def test_embedding_classifier_rejects_bad_dim(atoms):
    out, meta = atoms["cxr_embedding_classifier"]._run([0.0, 1.0, 2.0])
    assert "error" in out and meta["analysis_status"] == "failed"


# --------------------------------------------------------------------------
# MacroTool: hides the intermediate, pipes correctly
# --------------------------------------------------------------------------
def test_macro_hides_intermediate_and_parity(atoms, cxr_image):
    from ducx_noise.macro import build_macro_tool
    from medrax.tools.classification import ChestXRayClassifierTool

    macro = build_macro_tool(
        "cxr_macro", [atoms["cxr_encoder"], atoms["cxr_embedding_classifier"]]
    )
    # exposed schema is image-only; embedding never surfaced
    assert "embedding" not in macro.args_schema.model_fields
    assert list(macro.args_schema.model_fields.keys()) == ["image_path"]
    out = macro.invoke({"image_path": cxr_image})
    output = out[0] if isinstance(out, tuple) else out
    assert "embedding" not in output  # intermediate hidden in agent-facing output
    ref, _ = ChestXRayClassifierTool(device="cpu")._run(cxr_image)
    diff = max(abs(float(output[k]) - float(ref[k])) for k in ref)
    assert diff < 1e-5  # in-memory pipe reproduces the original


# --------------------------------------------------------------------------
# Compose conditions
# --------------------------------------------------------------------------
def _compose_cfg(**compose):
    return NoiseConfig.from_dict({
        "seed": 0, "shuffle_tools": False,
        "compose": {"enabled": True, "atom_groups": ["cxr_densenet"], "device": "cpu", **compose},
    })


def test_condition_p_c_only_macro():
    env = apply_noise([], _compose_cfg(
        macros=[{"name": "cxr_macro", "chain": ["cxr_encoder", "cxr_embedding_classifier"]}],
        exclusive=True))
    assert env.tool_names == ["cxr_macro"]
    assert env.manifest["cxr_macro"].source == "macro"


def test_condition_p_b_only_head():
    env = apply_noise([], _compose_cfg(expose_atoms=["cxr_embedding_classifier"], exclusive=True))
    assert env.tool_names == ["cxr_embedding_classifier"]
    assert env.manifest["cxr_embedding_classifier"].source == "atom"


def test_condition_p_chain_both_atoms():
    env = apply_noise([], _compose_cfg(
        expose_atoms=["cxr_encoder", "cxr_embedding_classifier"], exclusive=True))
    assert set(env.tool_names) == {"cxr_encoder", "cxr_embedding_classifier"}


def test_compose_identity_when_disabled():
    # No compose block -> identity (default DUCX behaviour preserved).
    env = apply_noise([], NoiseConfig())
    assert env.tool_names == []


# --------------------------------------------------------------------------
# Pair mining
# --------------------------------------------------------------------------
def test_mine_pairs_surfaces_densenet_chain():
    from ducx_noise.mine_pairs import mine_pairs

    cands = mine_pairs(require_nonsemantic=True)
    assert any(c.t_a == "cxr_encoder" and c.t_b == "cxr_embedding_classifier" for c in cands)
    # all kept pairs have a non-semantic intermediate
    assert all(c.intermediate_nonsemantic for c in cands)


def test_mine_pairs_filters_semantic_image_chains():
    from ducx_noise.mine_pairs import mine_pairs

    cands = mine_pairs(require_nonsemantic=True)
    names = {(c.t_a, c.t_b) for c in cands}
    # an image-path producer -> image consumer is semantic (copyable) -> excluded
    assert ("dicom_processor", "chest_xray_classifier") not in names


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------
def test_criteria_strong_case():
    from ducx_noise.eval_compose import compute_criteria

    c = compute_criteria(p_b=0.15, p_c=0.70, p_chain=0.18, p_a=1.0, n_options=6)
    assert c["weak (p_c > p_a*p_b)"] is True
    assert c["mid (p_c > max manual alt)"] is True
    assert c["strong (p_b~chance & p_c high)"] is True
    assert c["delta_comp_vs_chain"] == pytest.approx(0.52)


def test_criteria_negative_case():
    from ducx_noise.eval_compose import compute_criteria

    c = compute_criteria(p_b=0.60, p_c=0.60, p_chain=0.62, p_a=1.0, n_options=6)
    assert c["mid (p_c > max manual alt)"] is False
    assert c["strong (p_b~chance & p_c high)"] is False
