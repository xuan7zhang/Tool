"""Tool wrappers used to build a noisy environment.

Two BaseTool subclasses cover every noise type:

* :class:`ProxyTool` wraps an inner tool (real tool or another ProxyTool) and can
  remap parameter names, override the description, and inject reliability
  failures -- all while preserving the BaseTool contract the agent relies on
  (``name``, ``description``, ``args_schema``, ``invoke``). It is used both for
  in-place noising of real tools and for redundant duplicates (a ProxyTool whose
  inner is the shared real tool instance, so no extra model weights are loaded).
* :class:`DistractorTool` is a standalone fake tool that never touches a real
  model; it returns a plausible-but-useless response.

Reliability rolls are reproducible: each ProxyTool derives a per-call RNG from
``(seed, name, call_index)`` so the same config + seed yields the same failure
pattern given the same selection sequence.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool


def _seeded_random(*parts: Any) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


class ProxyTool(BaseTool):
    """Wrap an inner tool, optionally remapping args, description and reliability.

    Attributes:
        inner: the wrapped tool (BaseTool); ``inner._run`` performs the real call.
        arg_map: mapping from *exposed* parameter name -> *inner* parameter name.
            Identity ({}) when no schema renaming is applied.
        unreliable: optional dict with keys ``p_fail`` (float), ``failure_mode``
            ("error"|"empty"|"timeout"), ``timeout_seconds`` (float), ``seed``.
        call_index: internal counter for reproducible per-call reliability rolls.
    """

    inner: Any = None
    arg_map: Dict[str, str] = {}
    unreliable: Optional[Dict[str, Any]] = None
    call_index: int = 0

    def _maybe_fail(self) -> Optional[Any]:
        """Return a failure payload if this call should fail, else None."""
        spec = self.unreliable
        if not spec:
            return None
        p_fail = float(spec.get("p_fail", 0.0))
        if p_fail <= 0.0:
            self.call_index += 1
            return None
        rng = _seeded_random(spec.get("seed", 0), self.name, self.call_index)
        self.call_index += 1
        if rng.random() >= p_fail:
            return None
        mode = spec.get("failure_mode", "error")
        if mode == "empty":
            return ""
        if mode == "timeout":
            delay = float(spec.get("timeout_seconds", 0.0))
            if delay > 0:
                time.sleep(delay)
            return f"tool_timeout: '{self.name}' did not return in time (simulated)"
        # default: error -- mimic the agent's own tool_execution_error surface
        return f"tool_execution_error: '{self.name}' failed (simulated unreliable tool)"

    def _to_inner_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {self.arg_map.get(k, k): v for k, v in kwargs.items()}
        # Drop injected/redundant params the inner schema does not accept.
        inner_fields = set(self.inner.args_schema.model_fields.keys()) if getattr(
            self.inner, "args_schema", None
        ) else None
        if inner_fields is not None:
            mapped = {k: v for k, v in mapped.items() if k in inner_fields}
        return mapped

    def _run(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        failure = self._maybe_fail()
        if failure is not None:
            return failure
        inner_args = self._to_inner_args(kwargs)
        return self.inner._run(**inner_args)

    async def _arun(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._run(**kwargs)


class DistractorTool(BaseTool):
    """A fake tool that imitates a real one but returns a useless response.

    Never calls a real model. The response is deterministic given ``name`` so
    repeated runs are reproducible. ``style`` controls how misleading it is.
    """

    response_style: str = "plausible"  # plausible | misleading
    imitates: Optional[str] = None  # name of the real tool it paraphrases

    def _run(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        if self.response_style == "misleading":
            output = {
                "result": "analysis complete",
                "findings": "no abnormality of interest detected",
                "confidence": 0.92,
            }
        else:
            output = {
                "result": "no actionable information",
                "note": "this tool did not produce findings relevant to the question",
            }
        metadata = {
            "analysis_status": "completed",
            "distractor": True,
            "tool": self.name,
        }
        return output, metadata

    async def _arun(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._run(**kwargs)


def make_args_schema(
    base_schema: Type[BaseModel],
    rename: Optional[Dict[str, str]] = None,
    inject_optional: int = 0,
    shuffle_order: bool = False,
    rng: Optional[random.Random] = None,
    model_name: str = "NoisedSchema",
) -> "tuple[Type[BaseModel], Dict[str, str]]":
    """Build a noised pydantic schema and the exposed->original arg map.

    Renaming, optional-parameter injection and field reordering are all
    callability-preserving: every exposed required field maps back to an inner
    field, and injected optionals default to None and are dropped before the
    inner call.
    """
    rename = rename or {}
    rng = rng or random.Random(0)
    field_items: List[tuple] = []  # (exposed_name, (type, FieldInfo))
    arg_map: Dict[str, str] = {}

    for orig_name, info in base_schema.model_fields.items():
        exposed = rename.get(orig_name, orig_name)
        arg_map[exposed] = orig_name
        annotation = info.annotation if info.annotation is not None else Any
        default = ... if info.is_required() else info.default
        field_items.append(
            (exposed, (annotation, Field(default, description=info.description or "")))
        )

    decoys = ["mode", "verbosity", "extra_options", "context_hint", "format", "max_items"]
    for i in range(max(0, inject_optional)):
        decoy = decoys[i % len(decoys)]
        if decoy in dict(field_items):
            decoy = f"{decoy}_{i}"
        field_items.append(
            (decoy, (Optional[str], Field(None, description="optional parameter")))
        )

    if shuffle_order:
        rng.shuffle(field_items)

    schema = create_model(model_name, **dict(field_items))
    return schema, arg_map
