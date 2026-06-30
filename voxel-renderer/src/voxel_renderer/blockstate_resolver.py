"""
Blockstate Resolver — maps block ID + properties to model(s).

Handles both Minecraft blockstate formats:
  - **Variants**: a dict of ``"prop=val,prop=val" → model_ref``.
  - **Multipart**: a list of ``{when: {prop: val}, apply: model_ref}``
    entries that compose additively (fence post + sides, etc.).

The resolver trusts the properties stored in the blueprint rather than
re-deriving adjacency from the grid.

Output is a list of ``ModelApplication`` descriptors, each with a model
name and optional x/y rotation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

from voxel_renderer.assets import get_asset_root

_BLOCKSTATES_DIR = str(get_asset_root() / "blockstates")


def _normalise_known_default_properties(
    block_id: str,
    props: dict[str, str],
) -> dict[str, str]:
    """Fill renderer defaults for block families with mandatory variant axes.

    Stairs are the current high-yield case: Minecraft blockstates always key on
    ``facing``, ``half``, and ``shape``. When semantic payloads omit one of
    those axes, the generic variant resolver would otherwise fall back to the
    first listed variant, which is commonly a corner stair model.
    """

    out = dict(props)
    bare = block_id.removeprefix("minecraft:")
    if bare.endswith("_stairs"):
        out.setdefault("facing", "north")
        out.setdefault("half", "bottom")
        out.setdefault("shape", "straight")
    return out


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelApplication:
    """A model to render, with optional rotation."""

    model: str  # e.g. "minecraft:block/oak_fence_post"
    y_rotation: float = 0.0
    x_rotation: float = 0.0
    uvlock: bool = False


# ---------------------------------------------------------------------------
# Blockstate JSON loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _load_blockstate(block_name: str) -> dict[str, Any] | None:
    """
    Load a blockstate JSON.

    ``block_name`` is the bare name, e.g. "oak_fence" (no "minecraft:" prefix).
    """
    path = os.path.join(_BLOCKSTATES_DIR, f"{block_name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to load blockstate '%s': %s", block_name, e)
        return None


# ---------------------------------------------------------------------------
# Variants resolver
# ---------------------------------------------------------------------------


def _props_to_variant_key(props: dict[str, str]) -> str:
    """
    Convert a properties dict to the variant key format used in blockstates.

    e.g. {"facing": "east", "half": "bottom"} → "facing=east,half=bottom"
    """
    if not props:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(props.items()))


def _parse_model_ref(ref: dict[str, Any] | list) -> list[ModelApplication]:
    """Parse a model reference (single dict or weighted list)."""
    if isinstance(ref, list):
        # Weighted random — just take the first entry (deterministic render)
        if ref:
            ref = ref[0]
        else:
            return []

    return [
        ModelApplication(
            model=ref["model"],
            y_rotation=float(ref.get("y", 0)),
            x_rotation=float(ref.get("x", 0)),
            uvlock=ref.get("uvlock", False),
        )
    ]


def _resolve_variants(variants: dict[str, Any], props: dict[str, str]) -> list[ModelApplication]:
    """
    Resolve a variants-style blockstate.

    Tries exact match first, then progressively relaxed matching.
    """
    key = _props_to_variant_key(props)

    # Exact match
    if key in variants:
        return _parse_model_ref(variants[key])

    # Empty-key default (e.g. simple blocks like planks)
    if "" in variants:
        return _parse_model_ref(variants[""])

    # Partial match: find the variant whose properties are a subset of ours.
    # This handles cases where the blockstate has fewer property axes than
    # the blueprint stores (e.g. waterlogged is often not in blockstates).
    for variant_key, model_ref in variants.items():
        if not variant_key:
            continue
        variant_props = dict(kv.split("=", 1) for kv in variant_key.split(",") if "=" in kv)
        if all(props.get(k) == v for k, v in variant_props.items()):
            return _parse_model_ref(model_ref)

    # Fallback: return first variant
    first = next(iter(variants.values()))
    return _parse_model_ref(first)


# ---------------------------------------------------------------------------
# Multipart resolver
# ---------------------------------------------------------------------------


def _when_matches(when: dict[str, Any], props: dict[str, str]) -> bool:
    """
    Check if a multipart ``when`` condition matches the block properties.

    Handles:
      - Simple: ``{"north": "true"}``
      - OR: ``{"OR": [{"north": "true"}, {"south": "true"}]}``
      - Value alternatives: ``{"north": "true|low"}`` (pipe-separated)
    """
    if "OR" in when:
        return any(_when_matches(sub, props) for sub in when["OR"])

    for key, expected in when.items():
        if key == "OR":
            continue

        actual = props.get(key, "")
        # Handle pipe-separated alternatives: "true|low"
        expected_str = str(expected)
        alternatives = expected_str.split("|")
        if actual not in alternatives:
            return False

    return True


def _resolve_multipart(
    parts: list[dict[str, Any]], props: dict[str, str]
) -> list[ModelApplication]:
    """
    Resolve a multipart blockstate.

    Each part with a matching ``when`` (or unconditional) contributes
    its model(s) to the result.
    """
    result: list[ModelApplication] = []

    for part in parts:
        when = part.get("when")
        apply = part.get("apply")

        if when is None or _when_matches(when, props):
            if apply is not None:
                result.extend(_parse_model_ref(apply))

    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def resolve_block_models(
    block_id: str,
    properties: dict[str, str] | None = None,
) -> list[ModelApplication] | None:
    """
    Resolve the model(s) to render for a given block ID and properties.

    Parameters
    ----------
    block_id : str
        Full Minecraft block ID, e.g. "minecraft:oak_fence".
    properties : dict
        Block state properties, e.g. {"north": "true", "south": "false"}.

    Returns
    -------
    list[ModelApplication] or None
        Model(s) to render, or None if no blockstate found.
    """
    # Strip namespace
    bare = block_id.removeprefix("minecraft:")
    props = _normalise_known_default_properties(block_id, properties or {})

    bs = _load_blockstate(bare)
    if bs is None:
        return None

    if "variants" in bs:
        return _resolve_variants(bs["variants"], props)
    elif "multipart" in bs:
        return _resolve_multipart(bs["multipart"], props)

    return None
