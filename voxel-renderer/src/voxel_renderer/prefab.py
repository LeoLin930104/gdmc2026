"""Prefab utilities for module-based voxel assembly.

These helpers keep WFC/container-housing experiments out of the renderer hot
path while exposing deterministic geometry operations the renderer consumers
need: bounds, translation, Y-axis rotation, normalisation to local origin, and
face signatures for adjacency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voxel_renderer.orientation import rotate_block_properties
from voxel_renderer.state import canonicalise_block_array
from voxel_renderer.types import SemanticBlockDict

FaceName = Literal["north", "south", "east", "west", "up", "down"]
Rotation = Literal[0, 90, 180, 270]


@dataclass(frozen=True, slots=True)
class Bounds:
    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int

    @property
    def size(self) -> tuple[int, int, int]:
        return (
            self.max_x - self.min_x + 1,
            self.max_y - self.min_y + 1,
            self.max_z - self.min_z + 1,
        )


def get_bounds(blocks: list[SemanticBlockDict]) -> Bounds | None:
    non_air = [b for b in blocks if b.get("id") != "minecraft:air"]
    if not non_air:
        return None
    xs = [int(b["x"]) for b in non_air]
    ys = [int(b["y"]) for b in non_air]
    zs = [int(b["z"]) for b in non_air]
    return Bounds(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def translate_blocks(
    blocks: list[SemanticBlockDict],
    dx: int,
    dy: int,
    dz: int,
) -> list[SemanticBlockDict]:
    translated: list[SemanticBlockDict] = []
    for block in blocks:
        entry = dict(block)
        entry["x"] = int(entry["x"]) + dx
        entry["y"] = int(entry["y"]) + dy
        entry["z"] = int(entry["z"]) + dz
        translated.append(entry)
    return translated


def normalise_to_origin(blocks: list[SemanticBlockDict]) -> list[SemanticBlockDict]:
    bounds = get_bounds(blocks)
    if bounds is None:
        return []
    return translate_blocks(blocks, -bounds.min_x, -bounds.min_y, -bounds.min_z)


def rotate_y(
    blocks: list[SemanticBlockDict],
    degrees: Rotation,
    *,
    transform_properties: bool = True,
) -> list[SemanticBlockDict]:
    """Rotate blocks around the local Y axis and renormalise to origin.

    Rotation operates on integer grid coordinates. When
    ``transform_properties`` is true (the default since the orientation
    transformer landed), Minecraft directional property bits such as
    ``facing``/``axis``/``rotation`` are rotated to match. Pass
    ``transform_properties=False`` to preserve the legacy coordinate-only
    behaviour for tests or callers that handle property rotation themselves.
    """
    if degrees not in (0, 90, 180, 270):
        raise ValueError("degrees must be one of 0, 90, 180, 270")
    normalised = normalise_to_origin(blocks)
    if degrees == 0:
        if transform_properties:
            # Even at 0 deg we run the transformer so callers get a uniform
            # output dict shape (empty properties stripped, etc.). It is a
            # no-op on values.
            return [
                _apply_property_rotation(block, 0) for block in normalised
            ]
        return normalised

    rotated: list[SemanticBlockDict] = []
    for block in normalised:
        x = int(block["x"])
        z = int(block["z"])
        if degrees == 90:
            rx, rz = z, -x
        elif degrees == 180:
            rx, rz = -x, -z
        else:  # 270
            rx, rz = -z, x
        entry = (
            _apply_property_rotation(block, degrees)
            if transform_properties
            else dict(block)
        )
        entry["x"] = rx
        entry["z"] = rz
        rotated.append(entry)
    return normalise_to_origin(rotated)


def _apply_property_rotation(
    block: SemanticBlockDict, degrees: Rotation
) -> SemanticBlockDict:
    """Internal helper: clone block with rotated property bits."""
    new_block = dict(block)
    rotated = rotate_block_properties(new_block.get("properties"), degrees)
    if rotated:
        new_block["properties"] = rotated
    elif "properties" in new_block:
        del new_block["properties"]
    return new_block


def merge_prefabs(*prefabs: list[SemanticBlockDict]) -> list[SemanticBlockDict]:
    """Merge block arrays using last-write-wins coordinate semantics."""
    merged: list[SemanticBlockDict] = []
    for prefab in prefabs:
        merged.extend(prefab)
    canonical, _ = canonicalise_block_array(merged)
    return canonical


def face_signature(blocks: list[SemanticBlockDict], face: FaceName) -> frozenset[tuple[int, int, str]]:
    """Return occupied cells on a prefab boundary face.

    The signature is intentionally simple: `(u, v, id)` tuples on the requested
    face after normalisation to local origin.  WFC callers can compare opposite
    faces directly or map IDs into coarser categories (wall/open/window/door)
    before comparison.
    """
    normalised = normalise_to_origin(blocks)
    bounds = get_bounds(normalised)
    if bounds is None:
        return frozenset()

    signature: set[tuple[int, int, str]] = set()
    for block in normalised:
        if block.get("id") == "minecraft:air":
            continue
        x, y, z = int(block["x"]), int(block["y"]), int(block["z"])
        bid = str(block["id"])
        if face == "north" and z == bounds.min_z:
            signature.add((x, y, bid))
        elif face == "south" and z == bounds.max_z:
            signature.add((x, y, bid))
        elif face == "west" and x == bounds.min_x:
            signature.add((z, y, bid))
        elif face == "east" and x == bounds.max_x:
            signature.add((z, y, bid))
        elif face == "down" and y == bounds.min_y:
            signature.add((x, z, bid))
        elif face == "up" and y == bounds.max_y:
            signature.add((x, z, bid))
    return frozenset(signature)


def opposite_face(face: FaceName) -> FaceName:
    mapping: dict[FaceName, FaceName] = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
    }
    return mapping[face]
