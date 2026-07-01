"""Modular per-cell placement helpers.

Cells are boxed in all directions first. Connection cuts are resolved later as a
separate stage, so every placed cell begins as a fully enclosed module with:

- four walls
- floor
- ceiling
- exterior face texture only on faces later exposed to outer air
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from prefab_housing.grid import EAST, NORTH, SOUTH, WEST
from prefab_housing.palette import SLOT_FRAME_BLOCK, SLOT_WALL_EXTERIOR
from prefab_housing.types import SemanticBlockDict
from prefab_housing.wallface import (
    base_wall_block,
    emit_wall_face_blocks,
    load_wall_face_design,
)

_ACTIVE_WALL_FACE_DESIGN_PATH: Path | None = None
_WALL_FACE_DESIGN_DIR = Path(__file__).resolve().parents[3] / "designs"
_DEFAULT_WALL_FACE_DESIGN_NAME = "modular_default.wallface"
_WALL_FACE_PRESET_GLOB = "modular_*.wallface"
_DEFAULT_WALL_FACE_DESIGN_PATH = _WALL_FACE_DESIGN_DIR / _DEFAULT_WALL_FACE_DESIGN_NAME
_DEFAULT_FACE_INSET_BLOCK = "minecraft:glass_pane"


def _block(x: int, y: int, z: int, block_id: str) -> SemanticBlockDict:
    return {"x": x, "y": y, "z": z, "id": block_id}


def _emit_rect_ring(
    blocks: list[SemanticBlockDict],
    *,
    axis: str,
    fixed: int,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
    block_id: str,
) -> None:
    if a1 < a0 or y1 < y0:
        return
    for a in range(a0, a1 + 1):
        if axis == "x":
            blocks.append(_block(a, y0, fixed, block_id))
            blocks.append(_block(a, y1, fixed, block_id))
        else:
            blocks.append(_block(fixed, y0, a, block_id))
            blocks.append(_block(fixed, y1, a, block_id))
    for y in range(y0, y1 + 1):
        if axis == "x":
            blocks.append(_block(a0, y, fixed, block_id))
            blocks.append(_block(a1, y, fixed, block_id))
        else:
            blocks.append(_block(fixed, y, a0, block_id))
            blocks.append(_block(fixed, y, a1, block_id))


def _emit_rect_fill(
    blocks: list[SemanticBlockDict],
    *,
    axis: str,
    fixed: int,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
    block_id: str,
) -> None:
    if a1 < a0 or y1 < y0:
        return
    for a in range(a0, a1 + 1):
        for y in range(y0, y1 + 1):
            if axis == "x":
                blocks.append(_block(a, y, fixed, block_id))
            else:
                blocks.append(_block(fixed, y, a, block_id))


def _inner_rect_inset(span: int) -> int:
    if span < 6:
        return 0
    return max(2, min(math.floor(span * 0.20), max(2, span // 3)))


def _adaptive_inner_rect(
    *,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
) -> tuple[int, int, int, int] | None:
    span_axis = a1 - a0 + 1
    span_y = y1 - y0 + 1
    if span_axis < 6 or span_y < 6:
        return None
    # Reserve: outer rectangle, at least one air block, then the filled inner
    # rectangle. Faces smaller than 6 on either axis cannot satisfy that rule.
    inset_axis = _inner_rect_inset(span_axis)
    inset_y = _inner_rect_inset(span_y)
    inner_a0 = a0 + inset_axis
    inner_a1 = a1 - inset_axis
    inner_y0 = y0 + inset_y
    inner_y1 = y1 - inset_y
    if inner_a0 > inner_a1 or inner_y0 > inner_y1:
        return None
    return inner_a0, inner_a1, inner_y0, inner_y1


def _dedupe(blocks: list[SemanticBlockDict]) -> list[SemanticBlockDict]:
    by_pos: dict[tuple[int, int, int], SemanticBlockDict] = {}
    for block in blocks:
        by_pos[(block["x"], block["y"], block["z"])] = block
    return list(by_pos.values())


def set_active_wall_face_design(path: str | Path | None) -> None:
    global _ACTIVE_WALL_FACE_DESIGN_PATH
    _ACTIVE_WALL_FACE_DESIGN_PATH = None if path is None else Path(path)


def get_active_wall_face_design_path() -> Path | None:
    return _ACTIVE_WALL_FACE_DESIGN_PATH


def get_default_wall_face_design_path() -> Path:
    return _DEFAULT_WALL_FACE_DESIGN_PATH


def list_wall_face_design_paths() -> tuple[Path, ...]:
    """Return available modular wallface presets in stable selection order."""
    paths = tuple(sorted(_WALL_FACE_DESIGN_DIR.glob(_WALL_FACE_PRESET_GLOB)))
    if not paths:
        return ()
    if _DEFAULT_WALL_FACE_DESIGN_PATH not in paths:
        return paths
    return (
        _DEFAULT_WALL_FACE_DESIGN_PATH,
        *(path for path in paths if path != _DEFAULT_WALL_FACE_DESIGN_PATH),
    )


def choose_wall_face_design_path(seed: int, *, salt: str = "") -> Path | None:
    """Choose one wallface preset deterministically for a whole house."""
    paths = list_wall_face_design_paths()
    if not paths:
        return None
    key = f"{seed}:{salt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return paths[int.from_bytes(digest, byteorder="big") % len(paths)]


def _resolve_wall_face_design_path() -> Path | None:
    if _ACTIVE_WALL_FACE_DESIGN_PATH is not None:
        return _ACTIVE_WALL_FACE_DESIGN_PATH
    if _DEFAULT_WALL_FACE_DESIGN_PATH.exists():
        return _DEFAULT_WALL_FACE_DESIGN_PATH
    paths = list_wall_face_design_paths()
    if paths:
        return paths[0]
    return None


def _emit_solid_face(
    blocks: list[SemanticBlockDict],
    *,
    axis: str,
    fixed: int,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
    block_id: str,
) -> None:
    _emit_rect_fill(
        blocks,
        axis=axis,
        fixed=fixed,
        a0=a0,
        a1=a1,
        y0=y0,
        y1=y1,
        block_id=block_id,
    )


def _emit_floor_and_ceiling(
    blocks: list[SemanticBlockDict],
    *,
    vx: int,
    vy: int,
    vz: int,
    block_id: str,
) -> None:
    for x in range(vx):
        for z in range(vz):
            blocks.append(_block(x, 0, z, block_id))
            blocks.append(_block(x, vy - 1, z, block_id))


# Per-biome interior shell material, keyed by the wallface base wall block (the
# narrative bake sets that from the biome family). The boxed cell's floor,
# ceiling and walls use this stone/brick block so interior surfaces read as
# masonry rather than the timber/concrete exterior; sandstone biomes keep a
# cut-stone variant. Exterior faces exposed to air are repainted on top by the
# wallface overlay (build_exterior_face_overlay), so the outside walls keep their
# own per-biome material and are unaffected. The sci_fi default (white_concrete)
# is left untouched so the standalone module demo is unchanged. Anything unmapped
# falls back to stone bricks.
_DEFAULT_SHELL_BLOCK = "minecraft:stone_bricks"
_SHELL_BY_WALL_BASE: dict[str, str] = {
    "minecraft:white_concrete": "minecraft:white_concrete",       # sci_fi default
    "minecraft:oak_planks": "minecraft:stone_bricks",             # temperate / default
    "minecraft:birch_planks": "minecraft:stone_bricks",          # birch
    "minecraft:jungle_planks": "minecraft:mossy_stone_bricks",   # jungle (overgrown)
    "minecraft:spruce_planks": "minecraft:polished_diorite",     # snowy (cold, pale)
    "minecraft:acacia_planks": "minecraft:polished_granite",     # savanna (warm)
    "minecraft:mangrove_planks": "minecraft:mossy_cobblestone",  # swamp (wet)
    "minecraft:dark_oak_planks": "minecraft:deepslate_bricks",   # dark forest (dark)
    "minecraft:smooth_sandstone": "minecraft:cut_sandstone",     # desert
    "minecraft:smooth_red_sandstone": "minecraft:cut_red_sandstone",  # badlands
}


def _resolve_shell_block(palette: dict[str, str]) -> str:
    """Choose a biome-appropriate stone material for the interior shell.

    The exterior walls are rebaked per-biome from the active wallface design's
    base plane; the interior shell (floor, ceiling and walls) maps that biome to
    a stone/brick material (see ``_SHELL_BY_WALL_BASE``) so interior surfaces read
    as masonry rather than the timber/concrete exterior. Falls back to stone
    bricks for an unmapped design base, and to the ``wall_exterior`` palette slot
    when no wallface design is resolvable.
    """
    design_path = _resolve_wall_face_design_path()
    if design_path is not None:
        try:
            base = base_wall_block(load_wall_face_design(design_path))
        except (OSError, ValueError):
            return palette[SLOT_WALL_EXTERIOR]
        return _SHELL_BY_WALL_BASE.get(base, _DEFAULT_SHELL_BLOCK)
    return palette[SLOT_WALL_EXTERIOR]


def build_face_texture_panel(
    *,
    axis: str,
    fixed: int,
    outward_sign: int,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
    palette: dict[str, str],
    pod_name: str,
) -> list[SemanticBlockDict]:
    """Build one modular face panel.

    The overlay contains a neutral glass inset on the base wall plane plus an
    exterior outline frame. Room type is now communicated by interiors rather
    than coloured exterior placeholder panels.
    """
    design_path = _resolve_wall_face_design_path()
    if design_path is not None:
        return _dedupe(
            emit_wall_face_blocks(
                load_wall_face_design(design_path),
                axis=axis,
                fixed=fixed,
                outward_sign=outward_sign,
                a0=a0,
                a1=a1,
                y0=y0,
                y1=y1,
            )
        )

    wall_id = palette[SLOT_WALL_EXTERIOR]
    rim_id = palette[SLOT_FRAME_BLOCK]
    del pod_name
    blocks: list[SemanticBlockDict] = []

    _emit_rect_fill(
        blocks,
        axis=axis,
        fixed=fixed,
        a0=a0,
        a1=a1,
        y0=y0,
        y1=y1,
        block_id=wall_id,
    )

    outer_fixed = fixed + outward_sign
    _emit_rect_ring(
        blocks,
        axis=axis,
        fixed=outer_fixed,
        a0=a0,
        a1=a1,
        y0=y0,
        y1=y1,
        block_id=rim_id,
    )

    inner_rect = _adaptive_inner_rect(a0=a0, a1=a1, y0=y0, y1=y1)
    if inner_rect is None:
        return _dedupe(blocks)
    inner_a0, inner_a1, inner_y0, inner_y1 = inner_rect

    _emit_rect_fill(
        blocks,
        axis=axis,
        fixed=fixed,
        a0=inner_a0,
        a1=inner_a1,
        y0=inner_y0,
        y1=inner_y1,
        block_id=_DEFAULT_FACE_INSET_BLOCK,
    )

    return _dedupe(blocks)


def build_placeholder_cell(
    *,
    cell_voxel_size: tuple[int, int, int],
    palette: dict[str, str],
    pod_name: str,
) -> list[SemanticBlockDict]:
    """Build a modular boxed cell placeholder with all walls present.

    This stage intentionally ignores adjacency and connectivity. It places the
    base module only; later stages cut connections.
    """
    vx, vy, vz = cell_voxel_size
    # Floor, ceiling and walls of the boxed cell all use the biome shell stone.
    # Exposed exterior faces are repainted afterwards by the wallface overlay, so
    # only the interior surfaces (and the floor/ceiling) end up showing this.
    shell_id = _resolve_shell_block(palette)
    blocks: list[SemanticBlockDict] = []

    _emit_floor_and_ceiling(
        blocks,
        vx=vx,
        vy=vy,
        vz=vz,
        block_id=shell_id,
    )

    faces = (
        (NORTH, "x", 0, -1, 0, vz - 1),
        (SOUTH, "x", vx - 1, 1, 0, vz - 1),
        (WEST, "z", 0, -1, 0, vx - 1),
        (EAST, "z", vz - 1, 1, 0, vx - 1),
    )
    for _face, axis, fixed, outward_sign, a0, a1 in faces:
        _emit_solid_face(
            blocks,
            axis=axis,
            fixed=fixed,
            a0=a0,
            a1=a1,
            y0=0,
            y1=vy - 1,
            block_id=shell_id,
        )

    return _dedupe(blocks)


def build_exterior_face_overlay(
    *,
    face: int,
    cell_voxel_size: tuple[int, int, int],
    palette: dict[str, str],
    pod_name: str,
) -> list[SemanticBlockDict]:
    """Build only the exterior texture appendage for one face.

    This overlay is applied after boxed-cell placement and only on faces exposed
    to outer air.
    """
    vx, vy, vz = cell_voxel_size
    if face == NORTH:
        return build_face_texture_panel(
            axis="x",
            fixed=0,
            outward_sign=-1,
            a0=0,
            a1=vx - 1,
            y0=0,
            y1=vy - 1,
            palette=palette,
            pod_name=pod_name,
        )
    if face == SOUTH:
        return build_face_texture_panel(
            axis="x",
            fixed=vz - 1,
            outward_sign=1,
            a0=0,
            a1=vx - 1,
            y0=0,
            y1=vy - 1,
            palette=palette,
            pod_name=pod_name,
        )
    if face == WEST:
        return build_face_texture_panel(
            axis="z",
            fixed=0,
            outward_sign=-1,
            a0=0,
            a1=vz - 1,
            y0=0,
            y1=vy - 1,
            palette=palette,
            pod_name=pod_name,
        )
    if face == EAST:
        return build_face_texture_panel(
            axis="z",
            fixed=vx - 1,
            outward_sign=1,
            a0=0,
            a1=vz - 1,
            y0=0,
            y1=vy - 1,
            palette=palette,
            pod_name=pod_name,
        )
    return []


__all__ = [
    "build_exterior_face_overlay",
    "build_face_texture_panel",
    "build_placeholder_cell",
    "choose_wall_face_design_path",
    "get_active_wall_face_design_path",
    "get_default_wall_face_design_path",
    "list_wall_face_design_paths",
    "set_active_wall_face_design",
]
