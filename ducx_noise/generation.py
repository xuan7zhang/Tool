"""Distractor / description generation with on-disk caching.

The LLM path reuses the codebase's existing OpenAI-compatible client (the same
endpoint that drives the agent) -- no new dependency. Results are cached to a
JSON file keyed by the inputs + seed, so repeated runs never re-call the model.

When no client is available (tests, offline) or the call fails, a deterministic
rule-based fallback is used so behaviour stays reproducible without a live LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional


def _cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _load_cache(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        try:
            with open(path, "r") as handle:
                return json.load(handle)
        except Exception:
            return {}
    return {}


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(cache, handle, indent=2)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "tool"


# --------------------------------------------------------------------------
# Distractor tool specs
# --------------------------------------------------------------------------
_DISTRACTOR_PREFIXES = ["advanced", "auxiliary", "legacy", "secondary", "experimental",
                        "rapid", "alternative", "supplementary", "extended", "beta"]
_DISTRACTOR_DOMAINS = [
    ("bone_density_estimator", "Estimates skeletal bone density from a radiograph."),
    ("lung_volume_calculator", "Calculates approximate lung volume from a chest image."),
    ("image_metadata_reader", "Reads acquisition metadata embedded in the image file."),
    ("contrast_enhancer", "Returns a contrast-enhanced view of the radiograph."),
    ("rib_counter", "Counts visible ribs in a chest radiograph."),
    ("patient_positioning_checker", "Assesses patient positioning quality in the scan."),
    ("artifact_detector", "Flags imaging artifacts unrelated to pathology."),
    ("dicom_tag_summarizer", "Summarizes DICOM tags that are not clinically relevant."),
    ("histogram_analyzer", "Computes pixel-intensity histograms of the image."),
    ("scan_quality_scorer", "Scores generic acquisition quality of the scan."),
]


def _fallback_distractors(real_names: List[str], count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(_cache_key("distractor", sorted(real_names), seed))
    pool = list(_DISTRACTOR_DOMAINS)
    rng.shuffle(pool)
    specs: List[Dict[str, str]] = []
    used = set(real_names)
    i = 0
    while len(specs) < count:
        base_name, base_desc = pool[i % len(pool)]
        prefix = _DISTRACTOR_PREFIXES[(i // len(pool)) % len(_DISTRACTOR_PREFIXES)]
        name = base_name if i < len(pool) else f"{prefix}_{base_name}"
        if name in used:
            name = f"{name}_{i}"
        used.add(name)
        specs.append({
            "name": name,
            "description": base_desc,
            "imitates": real_names[i % len(real_names)] if real_names else None,
        })
        i += 1
    return specs


def generate_distractors(
    real_specs: List[Dict[str, str]],
    count: int,
    seed: int,
    client: Any = None,
    model: Optional[str] = None,
    cache_path: str = "",
) -> List[Dict[str, str]]:
    """Return ``count`` distractor specs ({name, description, imitates}).

    Uses ``client`` (OpenAI-compatible) when provided, caching results; otherwise
    a deterministic rule-based fallback.
    """
    if count <= 0:
        return []
    real_names = [s["name"] for s in real_specs]
    key = _cache_key("distractors", real_names, count, seed, model or "fallback")
    cache = _load_cache(cache_path)
    if key in cache:
        return cache[key][:count]

    specs: Optional[List[Dict[str, str]]] = None
    if client is not None and model:
        try:
            specs = _llm_distractors(real_specs, count, seed, client, model)
        except Exception:
            specs = None
    if not specs:
        specs = _fallback_distractors(real_names, count, seed)

    # Guarantee uniqueness and no clash with real names.
    cleaned: List[Dict[str, str]] = []
    used = set(real_names)
    for spec in specs:
        name = _slugify(spec.get("name", "distractor"))
        while name in used:
            name = f"{name}_x"
        used.add(name)
        cleaned.append({
            "name": name,
            "description": spec.get("description", "An auxiliary imaging tool."),
            "imitates": spec.get("imitates"),
        })
    cleaned = cleaned[:count]
    cache[key] = cleaned
    _save_cache(cache_path, cache)
    return cleaned


def _llm_distractors(real_specs, count, seed, client, model) -> List[Dict[str, str]]:
    listing = "\n".join(f"- {s['name']}: {s['description'][:160]}" for s in real_specs)
    prompt = (
        "You are designing DISTRACTOR tools for a chest X-ray agent benchmark. "
        "Given the real tools below, invent {n} fake tools whose names and "
        "descriptions sound plausible and topically related, but which are NOT "
        "functionally useful for answering chest X-ray questions. Return ONLY a "
        "JSON list of objects with keys 'name' (snake_case) and 'description'.\n\n"
        "Real tools:\n{listing}\n"
    ).format(n=count, listing=listing)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=900,
        temperature=0.7,
        seed=seed,
    )
    text = response.choices[0].message.content if response.choices else "[]"
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    data = json.loads(match.group(0)) if match else []
    return [{"name": d["name"], "description": d.get("description", "")} for d in data]


# --------------------------------------------------------------------------
# Description corruption
# --------------------------------------------------------------------------
def _fallback_corrupt(description: str, style: str, level: float, seed: int) -> str:
    rng = random.Random(_cache_key("desc", description, style, round(level, 3), seed))
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    if style == "vague":
        keep = max(1, int(round(len(sentences) * (1.0 - min(1.0, level)))))
        head = " ".join(sentences[:keep])
        return (head + " It may be helpful in some imaging situations.").strip()
    if style == "ambiguous":
        return (
            "A general-purpose imaging utility. " + (sentences[0] if sentences else "")
            + " Its exact behaviour depends on context."
        ).strip()
    if style == "misleading":
        swaps = {
            "chest": "abdominal", "x-ray": "ultrasound", "classif": "segment",
            "segment": "classif", "report": "caption", "probabilit": "coordinat",
        }
        out = description
        for src, dst in swaps.items():
            if rng.random() < min(1.0, level + 0.3):
                out = re.sub(src, dst, out, flags=re.IGNORECASE)
        return out
    return description


def corrupt_description(
    description: str,
    style: str,
    level: float,
    seed: int,
    client: Any = None,
    model: Optional[str] = None,
    cache_path: str = "",
    tool_name: str = "",
) -> str:
    """Return a corrupted description; cached + deterministic fallback."""
    key = _cache_key("corrupt", tool_name, description, style, round(level, 3), seed,
                     model or "fallback")
    cache = _load_cache(cache_path)
    if key in cache:
        return cache[key]

    result: Optional[str] = None
    if client is not None and model:
        try:
            result = _llm_corrupt(description, style, level, seed, client, model)
        except Exception:
            result = None
    if not result:
        result = _fallback_corrupt(description, style, level, seed)
    cache[key] = result
    _save_cache(cache_path, cache)
    return result


def generate_macro_description(
    name: str,
    sub_descriptions: List[str],
    seed: int,
    client: Any = None,
    model: Optional[str] = None,
    cache_path: str = "",
) -> Optional[str]:
    """End-to-end description for a macro-tool from its sub-tool descriptions.

    Returns None when no client (caller then uses the template default). Cached.
    """
    if client is None or not model:
        return None
    key = _cache_key("macro_desc", name, sub_descriptions, seed, model)
    cache = _load_cache(cache_path)
    if key in cache:
        return cache[key]
    listing = "\n".join(f"- step {i+1}: {d[:200]}" for i, d in enumerate(sub_descriptions))
    prompt = (
        "Write ONE concise end-to-end description (1-2 sentences) for a single tool that "
        "internally runs these steps in order and exposes only the first step's input and the "
        "last step's output. Do not mention intermediate representations. Steps:\n" + listing
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise tool descriptions."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
            temperature=0.5,
            seed=seed,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        return None
    cache[key] = text
    _save_cache(cache_path, cache)
    return text


def _llm_corrupt(description, style, level, seed, client, model) -> str:
    prompt = (
        f"Rewrite the following tool description to be more {style} (intensity "
        f"{level:.2f} on 0..1). Keep it one short paragraph. Do not add quotes.\n\n"
        f"{description}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You rewrite text as instructed."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.7,
        seed=seed,
    )
    return (response.choices[0].message.content or "").strip()
