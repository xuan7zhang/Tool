"""MacroTool -- the K (compose) operator.

A MacroTool wraps an ordered chain ``[t_1, ..., t_k]`` and exposes it to the agent
as a **single** tool: its input schema is ``t_1``'s, its output is ``t_k``'s. The
intermediate outputs are piped between sub-tools **in memory** (by calling each
sub-tool's ``_run`` directly, bypassing the agent and pydantic coercion), so a
non-semantic intermediate (e.g. a 1024-d embedding) is consumed internally and
**never exposed** to the model. This is the mechanism behind the compositional
unfamiliarity hypothesis: t_b alone / manual chaining force the model to route the
raw intermediate (which it cannot transcribe), while the macro hides it.

Piping: between hop i and i+1 the producer's output dict is mapped to the
consumer's input fields. By default, output keys that match the consumer's input
field names are forwarded; an explicit per-hop ``field_map`` (consumer_field ->
producer_key) can override this.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from langchain_core.tools import BaseTool


def _as_output(result: Any) -> Any:
    """Sub-tools return (output, metadata); take the output part."""
    if isinstance(result, tuple) and len(result) >= 1:
        return result[0]
    return result


def _map_inputs(
    producer_output: Any,
    consumer: BaseTool,
    field_map: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Build the consumer's kwargs from the producer's output."""
    consumer_fields = (
        list(consumer.args_schema.model_fields.keys())
        if getattr(consumer, "args_schema", None) is not None
        else []
    )
    if not isinstance(producer_output, dict):
        # single positional value -> first required consumer field
        if consumer_fields:
            return {consumer_fields[0]: producer_output}
        return {}
    if field_map:
        return {cf: producer_output[pk] for cf, pk in field_map.items() if pk in producer_output}
    # default: forward matching keys
    return {cf: producer_output[cf] for cf in consumer_fields if cf in producer_output}


class MacroTool(BaseTool):
    """Single tool exposing an in-memory composition of sub-tools."""

    chain: List[Any] = []
    field_maps: List[Optional[Dict[str, str]]] = []  # one per hop (len = len(chain)-1)

    def _run(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        if not self.chain:
            return {"error": "empty macro chain"}, {"analysis_status": "failed"}
        # First sub-tool: agent-facing inputs (validated against MacroTool.args_schema).
        result = self.chain[0]._run(**kwargs)
        for i, tool in enumerate(self.chain[1:]):
            output = _as_output(result)
            # propagate soft failures instead of crashing the chain
            if isinstance(output, dict) and "error" in output:
                return result
            fmap = self.field_maps[i] if i < len(self.field_maps) else None
            next_kwargs = _map_inputs(output, tool, fmap)
            result = tool._run(**next_kwargs)
        return result  # last sub-tool's (output, metadata) -- semantic, agent-facing

    async def _arun(self, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._run(**kwargs)


def build_macro_tool(
    name: str,
    chain: List[BaseTool],
    description: Optional[str] = None,
    field_maps: Optional[List[Optional[Dict[str, str]]]] = None,
) -> MacroTool:
    """Construct a MacroTool exposing t_1's input schema and t_k's output.

    Args:
        name: exposed tool name.
        chain: ordered sub-tools [t_1, ..., t_k].
        description: composed end-to-end description (auto-generated upstream if None).
        field_maps: optional per-hop mapping consumer_field -> producer_key.
    """
    if not chain:
        raise ValueError("MacroTool requires a non-empty chain")
    head_schema: Type[BaseModel] = chain[0].args_schema
    desc = description or _default_description(name, chain)
    return MacroTool(
        name=name,
        description=desc,
        args_schema=head_schema,
        chain=list(chain),
        field_maps=list(field_maps or []),
    )


def _default_description(name: str, chain: List[BaseTool]) -> str:
    first, last = chain[0], chain[-1]
    steps = " -> ".join(t.name for t in chain)
    return (
        f"End-to-end tool ({steps}). Takes the same input as '{first.name}' and returns "
        f"the final result of '{last.name}', handling all intermediate steps internally. "
        "Use this single tool instead of chaining the steps manually."
    )
