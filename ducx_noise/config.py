"""Noise configuration schema for the DUCX noisy-environment module.

Configs are plain dataclasses that round-trip to YAML or JSON. Every noise type
is an independent block with its own ``enabled`` switch, strength parameter(s),
and an optional ``tools`` whitelist. The top-level ``seed`` makes the whole
transformation reproducible.

Nothing here touches DUCX behaviour: an all-disabled config (the default) is an
identity transform, and the integration points skip the module entirely when no
config is supplied.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

try:  # YAML is optional; JSON always works.
    import yaml
except Exception:  # pragma: no cover - environment without pyyaml
    yaml = None


@dataclass
class DistractorConfig:
    """1.1 Distractor (fake) tools that look relevant but do nothing useful."""

    enabled: bool = False
    count: int = 0
    llm_generate: bool = False
    cache_path: str = "ducx_noise/cache/distractors.json"
    style: str = "plausible"  # plausible | misleading  (runtime RESPONSE style)
    tools: Optional[List[str]] = None  # real tools to imitate; None = all
    # --- Stage 3: description-alignment tier -------------------------------
    # obvious  -> description is clearly unrelated to any real tool.
    # aligned  -> description mimics a specific real tool's wording/structure
    #             (LLM-rewritten when a client is available; template fallback).
    similarity: str = "obvious"  # obvious | aligned
    # --- Stage 3: insertion position (used when tool_order != 'shuffle') ----
    # head | tail | random | index  (with `index` giving the absolute slot when
    # position == 'index'). Under tool_order='shuffle' position is irrelevant.
    position: str = "tail"
    index: int = 0


@dataclass
class UnreliableConfig:
    """1.2 Wrap real tool calls so they fail with probability ``p_fail``."""

    enabled: bool = False
    p_fail: float = 0.0
    failure_mode: str = "error"  # error | empty | timeout
    timeout_seconds: float = 0.0  # real sleep before failing when mode == timeout
    tools: Optional[List[str]] = None  # None = all real tools


@dataclass
class RedundantConfig:
    """1.3 Functionally-equivalent duplicates of real tools (same backend)."""

    enabled: bool = False
    copies_per_tool: int = 0
    rename_via_llm: bool = False
    tools: Optional[List[str]] = None  # None = all real tools


@dataclass
class DescriptionConfig:
    """1.4 Corrupt tool descriptions (vague / ambiguous / misleading)."""

    enabled: bool = False
    level: float = 0.0  # 0..1 intensity
    style: str = "vague"  # vague | ambiguous | misleading
    llm_generate: bool = False
    cache_path: str = "ducx_noise/cache/descriptions.json"
    tools: Optional[List[str]] = None  # None = all real tools


@dataclass
class SchemaConfig:
    """1.5 Parameter-level noise that preserves callability."""

    enabled: bool = False
    rename_params: bool = False
    inject_optional: int = 0  # number of redundant optional params to add
    shuffle_order: bool = False
    tools: Optional[List[str]] = None  # None = all real tools


@dataclass
class DropConfig:
    """Drop operator D: statically remove tools from the environment.

    Applied after all noise (last pipeline stage). Used to test N->D
    reversibility: dropping every distractor restores E0, and dropping a native
    high-confusion real tool can net-improve J.
    """

    enabled: bool = False
    tools: List[str] = field(default_factory=list)  # exact tool names to remove


@dataclass
class MacroSpec:
    """One macro-tool: an ordered chain of atom/tool names composed by the K operator."""

    name: str = "cxr_macro"
    chain: List[str] = field(default_factory=list)  # ordered atom/tool names
    description: Optional[str] = None  # hand override; None -> auto (LLM/template, cached)
    field_maps: Optional[List[Optional[Dict[str, str]]]] = None  # per-hop consumer_field->producer_key


@dataclass
class ComposeConfig:
    """K operator: build atoms and macro-tools, optionally exposing them exclusively.

    Used both for the macro-tool itself and for the three-condition compositional eval
    (expose only t_b, only the macro, or both atoms).
    """

    enabled: bool = False
    atom_groups: List[str] = field(default_factory=list)  # e.g. ["cxr_densenet"]
    expose_atoms: List[str] = field(default_factory=list)  # atom names exposed standalone
    macros: List[MacroSpec] = field(default_factory=list)
    exclusive: bool = False  # if True, keep ONLY the compose-produced tools (drop base)
    llm_describe: bool = False
    cache_path: str = "ducx_noise/cache/macro_descriptions.json"
    device: str = "cuda"


@dataclass
class NoiseConfig:
    """Top-level reproducible noise specification."""

    seed: int = 0
    shuffle_tools: bool = True  # shuffle final tool order (seeded) to avoid position cues
    # Final tool ordering policy (Stage 3). None -> derived from `shuffle_tools`
    # for backward compat (True->'shuffle', False->'fixed').
    #   fixed      -> keep the real-tool order; distractors placed per position.
    #   shuffle    -> seeded shuffle of the WHOLE list (decouples position from
    #                 role so distractors never sit at a fixed slot).
    #   controlled -> honor each distractor's explicit position/index; no shuffle
    #                 (for the Task 5 position-ablation sweep).
    tool_order: Optional[str] = None  # fixed | shuffle | controlled
    label: Optional[str] = None  # free-form label for sweeps / bookkeeping
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    unreliable: UnreliableConfig = field(default_factory=UnreliableConfig)
    redundant: RedundantConfig = field(default_factory=RedundantConfig)
    description_corruption: DescriptionConfig = field(default_factory=DescriptionConfig)
    schema_noise: SchemaConfig = field(default_factory=SchemaConfig)
    drop: DropConfig = field(default_factory=DropConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)

    @property
    def resolved_tool_order(self) -> str:
        """Effective ordering policy: explicit ``tool_order`` or derived from
        the legacy ``shuffle_tools`` flag."""
        if self.tool_order in ("fixed", "shuffle", "controlled"):
            return self.tool_order
        return "shuffle" if self.shuffle_tools else "fixed"

    @property
    def any_enabled(self) -> bool:
        noise_on = any(
            getattr(self, name).enabled
            for name in (
                "distractor",
                "unreliable",
                "redundant",
                "description_corruption",
                "schema_noise",
            )
        )
        drop_on = self.drop.enabled and len(self.drop.tools) > 0
        compose_on = self.compose.enabled
        return noise_on or drop_on or compose_on

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseConfig":
        data = dict(data or {})
        block_types = {
            "distractor": DistractorConfig,
            "unreliable": UnreliableConfig,
            "redundant": RedundantConfig,
            "description_corruption": DescriptionConfig,
            "schema_noise": SchemaConfig,
            "drop": DropConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key in ("seed", "shuffle_tools", "tool_order", "label"):
            if key in data:
                kwargs[key] = data[key]
        for name, block_cls in block_types.items():
            if name in data and data[name] is not None:
                kwargs[name] = _build_block(block_cls, data[name])
        if data.get("compose") is not None:
            kwargs["compose"] = _build_compose(data["compose"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "NoiseConfig":
        with open(path, "r") as handle:
            text = handle.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("pyyaml is required to load YAML noise configs.")
            data = yaml.safe_load(text)
        elif ext == ".json":
            data = json.loads(text)
        else:  # best effort: try YAML then JSON
            data = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return cls.from_dict(data or {})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = self.to_dict()
        ext = os.path.splitext(path)[1].lower()
        with open(path, "w") as handle:
            if ext in (".yaml", ".yml") and yaml is not None:
                yaml.safe_dump(data, handle, sort_keys=False)
            else:
                json.dump(data, handle, indent=2)


def _build_block(block_cls: Any, data: Any) -> Any:
    """Construct a config block dataclass from a dict, ignoring unknown keys."""
    if is_dataclass(data):
        return data
    valid = {f.name for f in fields(block_cls)}
    filtered = {k: v for k, v in dict(data or {}).items() if k in valid}
    return block_cls(**filtered)


def _build_compose(data: Any) -> "ComposeConfig":
    """Build ComposeConfig, converting the nested macros list into MacroSpec objects."""
    if is_dataclass(data):
        return data
    data = dict(data or {})
    macros = [
        m if is_dataclass(m) else _build_block(MacroSpec, m)
        for m in (data.get("macros") or [])
    ]
    cfg = _build_block(ComposeConfig, data)
    cfg.macros = macros
    return cfg


def load_noise_config(source: Optional[Any]) -> Optional[NoiseConfig]:
    """Resolve a noise config from a path, dict, NoiseConfig, or None.

    Returns None when ``source`` is None so callers can cheaply detect the
    "no noise" case and skip the module altogether.
    """
    if source is None:
        return None
    if isinstance(source, NoiseConfig):
        return source
    if isinstance(source, dict):
        return NoiseConfig.from_dict(source)
    if isinstance(source, str):
        return NoiseConfig.load(source)
    raise TypeError(f"Unsupported noise config source: {type(source)!r}")
