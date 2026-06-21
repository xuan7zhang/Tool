"""DUCX noisy-environment module.

Opt-in tool-environment noise for tool-using chest X-ray agents. Nothing here
runs unless a noise config is supplied; an all-disabled config is an identity
transform.

Public API::

    from ducx_noise import apply_noise, NoiseConfig, load_noise_config
    env = apply_noise(base_tools, "ducx_noise/configs/distractor_5.yaml")
    agent_tools = env.tools          # drop-in for the agent
    env.save_manifest("manifest.json")  # ground truth for the evaluator
"""

from .apply import NoisyEnv, apply_noise
from .config import NoiseConfig, load_noise_config
from .provenance import ToolProvenance

__all__ = [
    "apply_noise",
    "NoisyEnv",
    "NoiseConfig",
    "load_noise_config",
    "ToolProvenance",
]
