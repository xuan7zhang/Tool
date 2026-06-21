"""Transform registry.

Each noise type is registered here as a named transform with a common
signature, so the environment pipeline is just an ordered list of registered
transforms. This is the seam left open for *compositional* environments
(composing/stacking several transforms or whole environments): new transforms
register the same way and ``PIPELINE_ORDER`` (or a future ``compose``) decides
how they combine. Multi-environment composition itself is intentionally not
implemented in this task -- see :func:`compose`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

from .config import NoiseConfig
from .provenance import ToolProvenance


@dataclass
class TransformContext:
    """Shared state passed to every transform."""

    seed: int = 0
    client: Any = None  # OpenAI-compatible client for LLM-backed transforms
    model: Optional[str] = None
    base_real_tools: List[BaseTool] = field(default_factory=list)  # pristine real tools

    def rng(self, *salt: Any) -> random.Random:
        return random.Random("|".join([str(self.seed)] + [str(s) for s in salt]))


# transform(tools, manifest, cfg, ctx) -> (tools, manifest)
Transform = Callable[
    [List[BaseTool], Dict[str, ToolProvenance], NoiseConfig, TransformContext],
    Tuple[List[BaseTool], Dict[str, ToolProvenance]],
]

TRANSFORM_REGISTRY: Dict[str, Transform] = {}

# Fixed application order. In-place modifiers run first (so redundant copies and
# distractors are added afterwards and are not themselves re-noised), then
# tool-adding transforms.
PIPELINE_ORDER: List[str] = [
    "schema_noise",
    "description_corruption",
    "unreliable",
    "redundant",
    "distractor",
    "compose_macro",  # K operator: add composed macro-tools / exposed atoms
    "drop",  # D operator: remove tools (runs last, after all noise added)
]


def register_transform(name: str) -> Callable[[Transform], Transform]:
    def decorator(func: Transform) -> Transform:
        TRANSFORM_REGISTRY[name] = func
        return func

    return decorator


def compose(*env_specs: Any) -> Any:  # pragma: no cover - interface placeholder
    """Compose multiple environments/transform-pipelines into one.

    Reserved for compositional environments (e.g. stacking independently
    configured noisy envs). Not implemented in this task; the single-config
    pipeline is driven by :data:`PIPELINE_ORDER` instead.
    """
    raise NotImplementedError(
        "Compositional environment composition is not implemented yet; "
        "use a single NoiseConfig with PIPELINE_ORDER."
    )
