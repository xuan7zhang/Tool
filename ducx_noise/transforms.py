"""The five noise transforms, each registered in the transform registry.

All transforms take and return ``(tools, manifest)`` and never mutate the
pristine real tools in place -- in-place noising is done by *replacing* a slot
with a :class:`ProxyTool` that wraps the original, so provenance and the real
backend stay intact.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.tools import BaseTool

from .config import NoiseConfig
from .generation import (
    corrupt_description,
    generate_aligned_distractors,
    generate_distractors,
)
from .provenance import (
    SOURCE_ATOM,
    SOURCE_DISTRACTOR,
    SOURCE_MACRO,
    SOURCE_REDUNDANT,
    SOURCE_UNRELIABLE,
    ToolProvenance,
)
from .registry import TransformContext, register_transform
from .wrappers import DistractorTool, ProxyTool, make_args_schema


def _selected(names: List[str], whitelist) -> List[str]:
    if whitelist is None:
        return list(names)
    wl = set(whitelist)
    return [n for n in names if n in wl]


def _real_slots(tools: List[BaseTool], manifest: Dict[str, ToolProvenance]) -> List[str]:
    """Names of tools that originate from real tools (real or unreliable source)."""
    return [t.name for t in tools if manifest[t.name].source not in
            (SOURCE_DISTRACTOR, SOURCE_REDUNDANT)]


def _place(existing: List[BaseTool], extra: List[BaseTool], position: str,
           index: int, rng) -> List[BaseTool]:
    """Insert ``extra`` tools into ``existing`` at the requested position.

    position: head | tail | random | index (index -> absolute slot `index`).
    'random' inserts each extra tool at an independent seeded random slot so the
    distractors are scattered rather than clustered. Under tool_order='shuffle'
    this placement is irrelevant (the whole list is reshuffled downstream), but
    for 'fixed'/'controlled' it is the final layout.
    """
    if not extra:
        return existing
    if position == "head":
        return list(extra) + list(existing)
    if position == "index":
        k = max(0, min(int(index), len(existing)))
        return list(existing[:k]) + list(extra) + list(existing[k:])
    if position == "random":
        out = list(existing)
        for tool in extra:
            j = rng.randint(0, len(out))
            out.insert(j, tool)
        return out
    # default: tail
    return list(existing) + list(extra)


# --------------------------------------------------------------------------
# 1.5 Schema noise
# --------------------------------------------------------------------------
@register_transform("schema_noise")
def schema_noise(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    sc = cfg.schema_noise
    if not sc.enabled:
        return tools, manifest
    targets = set(_selected(_real_slots(tools, manifest), sc.tools))
    new_tools: List[BaseTool] = []
    for tool in tools:
        if tool.name not in targets:
            new_tools.append(tool)
            continue
        rng = ctx.rng("schema", tool.name)
        rename = {}
        if sc.rename_params and getattr(tool, "args_schema", None) is not None:
            decoy_names = ["scan_path", "img", "file_ref", "target_image", "input_path"]
            for i, field_name in enumerate(tool.args_schema.model_fields.keys()):
                rename[field_name] = decoy_names[i % len(decoy_names)]
        schema, arg_map = make_args_schema(
            tool.args_schema,
            rename=rename,
            inject_optional=sc.inject_optional,
            shuffle_order=sc.shuffle_order,
            rng=rng,
            model_name=f"{tool.name}_schema",
        )
        proxy = ProxyTool(
            name=tool.name,
            description=tool.description,
            args_schema=schema,
            inner=tool,
            arg_map=arg_map,
        )
        new_tools.append(proxy)
        manifest[tool.name].add_noise("schema")
        manifest[tool.name].meta["arg_map"] = arg_map
    return new_tools, manifest


# --------------------------------------------------------------------------
# 1.4 Description corruption
# --------------------------------------------------------------------------
@register_transform("description_corruption")
def description_corruption(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    dc = cfg.description_corruption
    if not dc.enabled:
        return tools, manifest
    targets = set(_selected(_real_slots(tools, manifest), dc.tools))
    new_tools: List[BaseTool] = []
    for tool in tools:
        if tool.name not in targets:
            new_tools.append(tool)
            continue
        original = tool.description
        corrupted = corrupt_description(
            original, dc.style, dc.level, ctx.seed,
            client=ctx.client if dc.llm_generate else None,
            model=ctx.model if dc.llm_generate else None,
            cache_path=dc.cache_path,
            tool_name=tool.name,
        )
        proxy = ProxyTool(
            name=tool.name,
            description=corrupted,
            args_schema=getattr(tool, "args_schema", None),
            inner=tool,
            arg_map={},
        )
        new_tools.append(proxy)
        prov = manifest[tool.name]
        prov.add_noise("description")
        prov.meta["original_description"] = original
        prov.meta["corrupted_description"] = corrupted
    return new_tools, manifest


# --------------------------------------------------------------------------
# 1.2 Unreliable tools
# --------------------------------------------------------------------------
@register_transform("unreliable")
def unreliable(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    uc = cfg.unreliable
    if not uc.enabled:
        return tools, manifest
    targets = set(_selected(_real_slots(tools, manifest), uc.tools))
    new_tools: List[BaseTool] = []
    for tool in tools:
        if tool.name not in targets:
            new_tools.append(tool)
            continue
        spec = {
            "p_fail": uc.p_fail,
            "failure_mode": uc.failure_mode,
            "timeout_seconds": uc.timeout_seconds,
            "seed": ctx.seed,
        }
        proxy = ProxyTool(
            name=tool.name,
            description=tool.description,
            args_schema=getattr(tool, "args_schema", None),
            inner=tool,
            arg_map={},
            unreliable=spec,
        )
        new_tools.append(proxy)
        prov = manifest[tool.name]
        prov.add_noise("unreliable")
        prov.meta["p_fail"] = uc.p_fail
        prov.meta["failure_mode"] = uc.failure_mode
        # Promote source to "unreliable" when reliability is the defining noise.
        if prov.source not in (SOURCE_DISTRACTOR, SOURCE_REDUNDANT):
            prov.source = SOURCE_UNRELIABLE
    return new_tools, manifest


# --------------------------------------------------------------------------
# 1.3 Redundant tools
# --------------------------------------------------------------------------
@register_transform("redundant")
def redundant(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    rc = cfg.redundant
    if not rc.enabled or rc.copies_per_tool <= 0:
        return tools, manifest
    # Duplicate the pristine real tools (shared backend, no extra weights).
    base_by_name = {t.name: t for t in ctx.base_real_tools}
    targets = _selected(list(base_by_name.keys()), rc.tools)
    extra: List[BaseTool] = []
    for name in targets:
        inner = base_by_name[name]
        for c in range(rc.copies_per_tool):
            copy_name = f"{name}_alt{c + 1}" if rc.copies_per_tool > 1 else f"{name}_alt"
            while copy_name in manifest:
                copy_name = f"{copy_name}_x"
            proxy = ProxyTool(
                name=copy_name,
                description=inner.description,
                args_schema=getattr(inner, "args_schema", None),
                inner=inner,
                arg_map={},
            )
            extra.append(proxy)
            manifest[copy_name] = ToolProvenance(
                name=copy_name,
                source=SOURCE_REDUNDANT,
                base_name=name,
                noises=["redundant"],
                real_backed=True,
                meta={"equivalent_to": name},
            )
    return tools + extra, manifest


# --------------------------------------------------------------------------
# Drop operator D -- statically remove tools (runs last)
# --------------------------------------------------------------------------
@register_transform("drop")
def drop(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    dc = cfg.drop
    if not dc.enabled or not dc.tools:
        return tools, manifest
    remove = set(dc.tools)
    kept = [t for t in tools if t.name not in remove]
    for name in list(manifest.keys()):
        if name in remove:
            manifest.pop(name, None)
    return kept, manifest


# --------------------------------------------------------------------------
# 1.1 Distractor tools
# --------------------------------------------------------------------------
@register_transform("distractor")
def distractor(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    dc = cfg.distractor
    if not dc.enabled or dc.count <= 0:
        return tools, manifest
    real_specs = [
        {"name": t.name, "description": t.description}
        for t in ctx.base_real_tools
        if dc.tools is None or t.name in set(dc.tools)
    ]
    # Similarity tier (Stage 3): 'aligned' distractors mimic a real tool's
    # wording (LLM-rewritten, cached); 'obvious' are clearly-unrelated tools.
    similarity = getattr(dc, "similarity", "obvious")
    gen = generate_aligned_distractors if similarity == "aligned" else generate_distractors
    specs = gen(
        real_specs, dc.count, ctx.seed,
        client=ctx.client if dc.llm_generate else None,
        model=ctx.model if dc.llm_generate else None,
        cache_path=dc.cache_path,
    )
    # Reuse a generic single-image schema for distractors.
    from .wrappers import make_args_schema as _mk  # local import keeps deps tidy
    template_schema = None
    for t in ctx.base_real_tools:
        if getattr(t, "args_schema", None) is not None:
            template_schema, _ = _mk(t.args_schema, model_name="distractor_schema")
            break
    extra: List[BaseTool] = []
    for spec in specs:
        name = spec["name"]
        while name in manifest:
            name = f"{name}_x"
        kwargs = dict(name=name, description=spec["description"],
                      response_style=dc.style, imitates=spec.get("imitates"))
        if template_schema is not None:
            kwargs["args_schema"] = template_schema
        extra.append(DistractorTool(**kwargs))
        manifest[name] = ToolProvenance(
            name=name,
            source=SOURCE_DISTRACTOR,
            noises=["distractor"],
            real_backed=False,
            meta={
                "imitates": spec.get("imitates"),
                "style": dc.style,
                "similarity": similarity,
                "position": getattr(dc, "position", "tail"),
            },
        )
    # Place the distractors per the configured position. Under 'shuffle' ordering
    # this is overridden by the final reshuffle in apply_noise; under
    # 'fixed'/'controlled' it is the final layout used for position ablation.
    place_rng = ctx.rng("distractor_position")
    tools = _place(
        tools, extra,
        position=getattr(dc, "position", "tail"),
        index=getattr(dc, "index", 0),
        rng=place_rng,
    )
    return tools, manifest


# --------------------------------------------------------------------------
# K operator -- compose atoms into macro-tools (+ expose atoms standalone)
# --------------------------------------------------------------------------
@register_transform("compose_macro")
def compose_macro(tools, manifest, cfg: NoiseConfig, ctx: TransformContext):
    cc = cfg.compose
    if not cc.enabled:
        return tools, manifest

    # Lazy imports: only load real-model atoms when compose is actually used.
    from .atoms import ATOM_GROUPS
    from .macro import build_macro_tool
    from .generation import generate_macro_description

    # Build requested atom groups (each builder returns {atom_name: tool}, shared backend).
    built: Dict[str, BaseTool] = {}
    atom_group_of: Dict[str, str] = {}
    for group in cc.atom_groups:
        if group not in ATOM_GROUPS:
            raise KeyError(f"Unknown atom group '{group}'. Known: {list(ATOM_GROUPS)}")
        for aname, tool in ATOM_GROUPS[group](device=cc.device).items():
            built[aname] = tool
            atom_group_of[aname] = group

    # Resolution pool for macro chains: built atoms + base real tools by name.
    pool: Dict[str, BaseTool] = dict(built)
    for t in ctx.base_real_tools:
        pool.setdefault(t.name, t)

    new_tools: List[BaseTool] = list(tools)
    added: List[str] = []

    # Expose selected atoms standalone (for the p_b / p_chain conditions).
    for aname in cc.expose_atoms:
        if aname not in built:
            raise KeyError(f"expose_atoms '{aname}' not in built atoms {list(built)}")
        if aname in manifest:
            continue
        new_tools.append(built[aname])
        manifest[aname] = ToolProvenance(
            name=aname, source=SOURCE_ATOM, base_name=aname,
            noises=["atom"], real_backed=True,
            meta={"atom_group": atom_group_of.get(aname)},
        )
        added.append(aname)

    # Build macro-tools.
    for spec in cc.macros:
        missing = [n for n in spec.chain if n not in pool]
        if missing:
            raise KeyError(f"macro '{spec.name}' chain references unknown tools {missing}")
        chain = [pool[n] for n in spec.chain]
        desc = spec.description
        if desc is None and cc.llm_describe:
            desc = generate_macro_description(
                spec.name, [t.description for t in chain], ctx.seed,
                client=ctx.client, model=ctx.model, cache_path=cc.cache_path,
            )
        macro = build_macro_tool(spec.name, chain, description=desc, field_maps=spec.field_maps)
        if spec.name in manifest:
            continue
        new_tools.append(macro)
        manifest[spec.name] = ToolProvenance(
            name=spec.name, source=SOURCE_MACRO, noises=["macro"], real_backed=True,
            meta={"chain": [t.name for t in chain]},
        )
        added.append(spec.name)

    # Exclusive: keep ONLY the compose-produced tools (= N(compose) + D(all base)).
    if cc.exclusive:
        keep = set(added)
        new_tools = [t for t in new_tools if t.name in keep]
        manifest = {n: manifest[n] for n in manifest if n in keep}

    return new_tools, manifest
