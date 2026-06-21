"""Entry point: turn a base tool environment into a noisy one.

``apply_noise(base_env, noise_config) -> NoisyEnv``

The result carries both the transformed tool list (drop-in for the agent) and a
ground-truth manifest mapping every exposed tool name to its provenance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import BaseTool

from .config import NoiseConfig, load_noise_config
from .provenance import ToolProvenance, manifest_to_list
from .registry import PIPELINE_ORDER, TRANSFORM_REGISTRY, TransformContext
from . import transforms as _transforms  # noqa: F401  (registers transforms)


@dataclass
class NoisyEnv:
    """A (possibly) noisy environment: tools + ground-truth provenance."""

    tools: List[BaseTool]
    manifest: Dict[str, ToolProvenance] = field(default_factory=dict)
    config: Optional[NoiseConfig] = None

    @property
    def tool_names(self) -> List[str]:
        return [t.name for t in self.tools]

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict() if self.config else None,
            "tools": manifest_to_list(self.manifest),
        }

    def save_manifest(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.manifest_dict(), handle, indent=2)


def apply_noise(
    base_env: List[BaseTool],
    noise_config: Union[NoiseConfig, dict, str, None],
    client: Any = None,
    model: Optional[str] = None,
) -> NoisyEnv:
    """Transform ``base_env`` (a list of BaseTools) into a :class:`NoisyEnv`.

    With ``noise_config=None`` or an all-disabled config this is an identity
    transform: the same tool objects, in the same order, each tagged ``real``.

    Args:
        base_env: the real tools (drop-in from ``initialize_agent``).
        noise_config: NoiseConfig, dict, path to YAML/JSON, or None.
        client: optional OpenAI-compatible client for LLM-backed generation.
        model: model name for the client.
    """
    base_tools = list(base_env)
    cfg = load_noise_config(noise_config)

    # Start: every real tool tagged 'real'.
    manifest: Dict[str, ToolProvenance] = {
        t.name: ToolProvenance(name=t.name, source="real", base_name=t.name)
        for t in base_tools
    }

    if cfg is None or not cfg.any_enabled:
        return NoisyEnv(tools=base_tools, manifest=manifest, config=cfg)

    ctx = TransformContext(
        seed=cfg.seed, client=client, model=model, base_real_tools=base_tools
    )

    tools = base_tools
    for name in PIPELINE_ORDER:
        transform = TRANSFORM_REGISTRY[name]
        tools, manifest = transform(tools, manifest, cfg, ctx)

    if cfg.shuffle_tools:
        order_rng = ctx.rng("final_order")
        order_rng.shuffle(tools)

    # Keep manifest ordered like the exposed tool list for readability.
    ordered = {t.name: manifest[t.name] for t in tools}
    return NoisyEnv(tools=tools, manifest=ordered, config=cfg)
