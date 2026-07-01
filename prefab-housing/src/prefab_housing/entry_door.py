"""Entrance-door module for assembled prefab houses."""

from __future__ import annotations

from typing import Final

from prefab_housing.grid import EAST, NORTH, SOUTH, WEST, CellGrid
from prefab_housing.palette import SLOT_DOOR_FRAME, SLOT_FRAME_BLOCK, resolve_palette
from prefab_housing.types import FaceName, SemanticBlockDict, SemanticCell


ENTRY_DOOR_MODULE_VERSION: Final[str] = "entry_door_module_v2"
ENTRY_DOOR_BLOCK: Final[str] = "minecraft:dark_oak_door"
FALLBACK_DOOR_FRAME_BLOCK: Final[str] = "minecraft:stripped_spruce_log"

_FACE_TO_INDEX: Final[dict[FaceName, int]] = {
    "north": NORTH,
    "east": EAST,
    "south": SOUTH,
    "west": WEST,
}


def _block(
    x: int,
    y: int,
    z: int,
    block_id: str,
    properties: dict[str, str] | None = None,
) -> SemanticBlockDict:
    block: SemanticBlockDict = {"x": x, "y": y, "z": z, "id": block_id}
    if properties:
        block["properties"] = dict(properties)
    return block


def _axis_span_for_face(
    cell: SemanticCell,
    face: FaceName,
) -> tuple[list[int], int, int, str]:
    (x0, y0, z0), (x1, _, z1) = cell.voxel_bbox
    if face in {"north", "south"}:
        axis_values = list(range(x0 + 2, x1 - 1))
        fixed = z0 if face == "north" else z1
        return axis_values, fixed, y0, "x"
    axis_values = list(range(z0 + 2, z1 - 1))
    fixed = x0 if face == "west" else x1
    return axis_values, fixed, y0, "z"


def _position(axis: str, axis_value: int, fixed: int, y: int) -> tuple[int, int, int]:
    if axis == "x":
        return axis_value, y, fixed
    return fixed, y, axis_value


def _door_columns(axis_values: list[int]) -> set[int]:
    mid = len(axis_values) // 2
    if len(axis_values) <= 2:
        return set(axis_values)
    return {axis_values[mid - 1], axis_values[mid]}


def _door_properties(face: FaceName, half: str, hinge: str) -> dict[str, str]:
    return {
        "facing": face,
        "half": half,
        "hinge": hinge,
        "open": "false",
        "powered": "false",
    }


def _outer_hinge(axis_value: int, door_columns: set[int]) -> str:
    # Minecraft's hinge property is inverted from the aperture-axis ordering
    # for paired doors: this keeps hinges on the outer frame and handles in the
    # centre, so opening the doors clears the entrance rather than folding into it.
    return "right" if axis_value == min(door_columns) else "left"


def _entry_face_blocks(
    cell: SemanticCell,
    face: FaceName,
    *,
    frame_block: str,
) -> list[SemanticBlockDict]:
    axis_values, fixed, y0, axis = _axis_span_for_face(cell, face)
    if len(axis_values) < 2:
        return []

    door_columns = _door_columns(axis_values)
    blocks: list[SemanticBlockDict] = []
    for axis_value in axis_values:
        if axis_value in door_columns:
            hinge = _outer_hinge(axis_value, door_columns)
            for dy, half in ((1, "lower"), (2, "upper")):
                blocks.append(
                    _block(
                        *_position(axis, axis_value, fixed, y0 + dy),
                        ENTRY_DOOR_BLOCK,
                        _door_properties(face, half, hinge),
                    )
                )
        else:
            for dy in (1, 2):
                blocks.append(
                    _block(
                        *_position(axis, axis_value, fixed, y0 + dy),
                        frame_block,
                    )
                )

    for axis_value in axis_values:
        blocks.append(
            _block(
                *_position(axis, axis_value, fixed, y0 + 3),
                frame_block,
            )
        )
    return blocks


def generate_entry_door_blocks(
    semantic_cells: list[SemanticCell],
    grid: CellGrid,
    *,
    material_theme: str | None = None,
) -> list[SemanticBlockDict]:
    """Emit door modules for entry-room exterior door faces."""
    palette = resolve_palette(material_theme)
    frame_block = (
        palette.get(SLOT_DOOR_FRAME)
        or palette.get(SLOT_FRAME_BLOCK)
        or FALLBACK_DOOR_FRAME_BLOCK
    )

    blocks: list[SemanticBlockDict] = []
    for cell in semantic_cells:
        if cell.label != "entry":
            continue
        for face in cell.door_faces:
            face_index = _FACE_TO_INDEX.get(face)
            if face_index is None:
                continue
            if grid.neighbour(*cell.cell_index, face_index) is not None:
                continue
            blocks.extend(_entry_face_blocks(cell, face, frame_block=frame_block))
    return blocks


__all__ = [
    "ENTRY_DOOR_MODULE_VERSION",
    "ENTRY_DOOR_BLOCK",
    "FALLBACK_DOOR_FRAME_BLOCK",
    "generate_entry_door_blocks",
]
