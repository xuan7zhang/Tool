"""Ground-truth provenance tags for tools in a (possibly noisy) environment.

Every tool exposed to the agent carries a :class:`ToolProvenance` record so the
evaluator can tell real tools apart from injected noise and compute selection
metrics (mis-selection rate, per-tool confusion, etc.).

Axes are deliberately separated:

* ``source`` -- origin category, one of {real, distractor, redundant, unreliable}
  (the four categories requested for ground truth). ``unreliable`` is used when
  a real tool's defining injected modification is the reliability wrapper.
* ``noises`` -- the full list of transforms applied (description, schema,
  unreliable, redundant, distractor), so stacked noise stays auditable.
* ``is_noise_tool`` -- whether *selecting* this tool counts as a mis-selection
  (True for injected extra tools: distractor + redundant). Real and unreliable
  tools are legitimate selections (unreliable is still backed by a real model;
  it just sometimes fails).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SOURCE_REAL = "real"
SOURCE_DISTRACTOR = "distractor"
SOURCE_REDUNDANT = "redundant"
SOURCE_UNRELIABLE = "unreliable"
SOURCE_MACRO = "macro"  # K operator: composed macro-tool (real-backed)
SOURCE_ATOM = "atom"  # K operator: an exposed sub-tool / atom (real-backed)

VALID_SOURCES = {
    SOURCE_REAL,
    SOURCE_DISTRACTOR,
    SOURCE_REDUNDANT,
    SOURCE_UNRELIABLE,
    SOURCE_MACRO,
    SOURCE_ATOM,
}

# Selecting a tool from these sources is counted as a "noise-tool mis-selection".
NOISE_SOURCES = {SOURCE_DISTRACTOR, SOURCE_REDUNDANT}


@dataclass
class ToolProvenance:
    name: str
    source: str = SOURCE_REAL
    base_name: Optional[str] = None  # underlying real tool (redundant/unreliable copies)
    noises: List[str] = field(default_factory=list)
    real_backed: bool = True  # True if a real model call ultimately runs
    meta: Dict = field(default_factory=dict)

    @property
    def is_noise_tool(self) -> bool:
        return self.source in NOISE_SOURCES

    def add_noise(self, kind: str) -> None:
        if kind not in self.noises:
            self.noises.append(kind)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["is_noise_tool"] = self.is_noise_tool
        return data


def manifest_to_list(manifest: Dict[str, ToolProvenance]) -> List[Dict]:
    return [manifest[name].to_dict() for name in manifest]


def noise_tool_names(manifest: Dict[str, ToolProvenance]) -> List[str]:
    return [name for name, prov in manifest.items() if prov.is_noise_tool]


def real_tool_names(manifest: Dict[str, ToolProvenance]) -> List[str]:
    return [name for name, prov in manifest.items() if not prov.is_noise_tool]
