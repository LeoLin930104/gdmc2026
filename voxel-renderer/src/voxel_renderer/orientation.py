"""Block-property rotation rules.

Minecraft directional blocks carry orientation in their `properties` dict.
This module rotates those properties around the vertical (Y) axis to match a
prefab/tile rotation applied by ``voxel_renderer.prefab.rotate_y``.

Design
------
Pure data tables. No object graph. The whitelist is intentionally finite — any
property name not in the whitelist passes through unchanged. Unknown values for
*known* properties are also passed through (with a soft note retained on a
per-call accumulator the caller may inspect via ``rotate_block_properties``).

The intended call-site is the WFC orientation-fixup stage and any prefab
rotation pipeline that wants Minecraft-faithful orientation, not the renderer
hot path.

Public API
----------
- ``rotate_y_property(name, value, degrees) -> str | None``
- ``rotate_block_properties(properties, degrees) -> dict[str, str]``
- ``rotate_block(block, degrees) -> SemanticBlockDict``  (positions and props)
- ``KNOWN_ROTATABLE_PROPERTIES``  (frozenset)
"""

from __future__ import annotations

from typing import Any, Final, Literal

YRotation = Literal[0, 90, 180, 270]

# Cardinal facing chain north -> east -> south -> west, clockwise viewed from above.
# `rotate_y(90)` of a block facing north should now face east.
_CARDINALS: Final[tuple[str, str, str, str]] = ("north", "east", "south", "west")
_CARDINAL_INDEX: Final[dict[str, int]] = {d: i for i, d in enumerate(_CARDINALS)}

# Steps for each rotation value.
_STEPS: Final[dict[int, int]] = {0: 0, 90: 1, 180: 2, 270: 3}

KNOWN_ROTATABLE_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "facing",   # stairs, doors, ladders, furnaces, ...
        "axis",     # logs, basalt, hay
        "rotation", # signs, banners; 0..15 in 22.5 deg increments
    }
)


def _rotate_facing(value: str, steps: int) -> str | None:
    if value not in _CARDINAL_INDEX:
        # `up` / `down` for piston-style facings: vertical, unaffected by Y rotation.
        if value in {"up", "down"}:
            return value
        return None
    return _CARDINALS[(_CARDINAL_INDEX[value] + steps) % 4]


def _rotate_axis(value: str, steps: int) -> str | None:
    # 90 / 270 swap x and z; 180 keeps original orientation; y axis unaffected.
    if value not in {"x", "y", "z"}:
        return None
    if value == "y":
        return "y"
    if steps % 2 == 0:
        return value
    return "z" if value == "x" else "x"


def _rotate_rotation_16(value: str, steps: int) -> str | None:
    # signs/banners encode 0..15 in 22.5 deg steps; 90 deg = 4 steps.
    try:
        n = int(value)
    except ValueError:
        return None
    if not 0 <= n < 16:
        return None
    return str((n + 4 * steps) % 16)


def rotate_y_property(name: str, value: str, degrees: YRotation) -> str | None:
    """Return the rotated property value, or ``None`` if the rule does not apply.

    ``None`` indicates the caller should leave the property untouched. This is
    distinct from a successful rotation that happens to be a no-op (e.g. the
    Y-axis under any rotation).
    """
    if degrees not in _STEPS:
        raise ValueError(f"degrees must be one of 0/90/180/270, got {degrees!r}")
    steps = _STEPS[degrees]
    if steps == 0:
        return value
    if name == "facing":
        return _rotate_facing(value, steps)
    if name == "axis":
        return _rotate_axis(value, steps)
    if name == "rotation":
        return _rotate_rotation_16(value, steps)
    return None


def rotate_block_properties(
    properties: dict[str, str] | None,
    degrees: YRotation,
) -> dict[str, str]:
    """Apply Y-rotation to every recognised property.

    Returns a new dict; never mutates the input. Unknown property names and
    unparseable values pass through untouched.
    """
    if not properties:
        return {}
    out: dict[str, str] = {}
    for name, value in properties.items():
        rotated = rotate_y_property(name, str(value), degrees)
        out[name] = rotated if rotated is not None else str(value)
    return out


def rotate_block(block: dict[str, Any], degrees: YRotation) -> dict[str, Any]:
    """Return a copy of ``block`` with its ``properties`` rotated around Y.

    Coordinates are *not* touched here — this is solely a property transformer.
    Coordinate rotation lives in ``voxel_renderer.prefab.rotate_y``; this
    function is the missing complement.
    """
    rotated_props = rotate_block_properties(block.get("properties"), degrees)
    new_block = dict(block)
    if rotated_props:
        new_block["properties"] = rotated_props
    elif "properties" in new_block:
        # Strip an empty properties dict to keep canonical form tight.
        del new_block["properties"]
    return new_block


__all__ = [
    "KNOWN_ROTATABLE_PROPERTIES",
    "YRotation",
    "rotate_block",
    "rotate_block_properties",
    "rotate_y_property",
]
