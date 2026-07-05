"""Print the actual tool-list order a NoiseConfig produces (Stage 3 check).

Offline verification tool: builds placeholder tools whose *names* mirror the real
MedRAX tools, applies the noise config, and prints the exposed order with each
tool's role / similarity / position. No GPU or LLM needed (aligned distractors
fall back to the deterministic template when no client is supplied).

    python -m ducx_noise.preview --config ducx_noise/configs/distractor_5_aligned.yaml
    python -m ducx_noise.preview --config <cfg> --seed 3
"""

from __future__ import annotations

import argparse
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .apply import apply_noise
from .config import NoiseConfig

# Placeholder stand-ins for the real MedRAX tools (names/descriptions only).
_REAL_TOOLS = [
    ("chest_xray_classifier", "Classifies chest X-ray images for 18 pathologies."),
    ("chest_xray_segmentation", "Segments anatomical structures in a chest X-ray."),
    ("chest_xray_report_generator", "Generates a radiology report from a chest X-ray."),
    ("xray_vqa", "Answers free-form visual questions about a chest X-ray."),
    ("image_visualizer", "Renders and displays a medical image."),
]


class _ImageInput(BaseModel):
    image_path: str = Field(..., description="Path to the radiology image file")


def _make_tool(tool_name: str, desc: str) -> BaseTool:
    class _Placeholder(BaseTool):
        name: str = tool_name
        description: str = desc
        args_schema: Type[BaseModel] = _ImageInput

        def _run(self, image_path: str, run_manager=None):
            return {"ok": True}

    return _Placeholder()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a ducx_noise YAML/JSON config.")
    ap.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = ap.parse_args(argv)

    cfg = NoiseConfig.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    base = [_make_tool(n, d) for n, d in _REAL_TOOLS]
    env = apply_noise(base, cfg)

    print(f"config={args.config}  seed={cfg.seed}  order={cfg.resolved_tool_order}")
    print(f"tools exposed: {len(env.tools)}  "
          f"(distractors={sum(p.is_noise_tool for p in env.manifest.values())})")
    print("-" * 78)
    print(f"{'idx':>3}  {'role':<20} {'similarity':<9} {'imitates':<26} name")
    for i, tool in enumerate(env.tools):
        prov = env.manifest[tool.name]
        meta = prov.meta or {}
        print(f"{i:>3}  {prov.source:<20} {str(meta.get('similarity','-')):<9} "
              f"{str(meta.get('imitates','-')):<26} {tool.name}")


if __name__ == "__main__":
    main()
