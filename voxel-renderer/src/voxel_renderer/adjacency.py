"""
Adjacency Resolver — derives block connection properties from spatial neighbours.

Minecraft fence/wall/pane/redstone blocks connect to adjacent solid blocks
or same-type connectable blocks.  This module provides:

  - ``BlockGrid``: spatial index storing block data by (x, y, z).
  - ``resolve_connections()``: derives connection properties for a single block.
  - ``get_affected_neighbours()``: returns grid positions whose connections
    may change when a block is placed/removed at a given position.

The resolver is designed for incremental use during animation: when a new
block is placed, only it and its horizontal neighbours need re-evaluation.

Connection rules follow Minecraft 1.21 behaviour:
  - Fences connect horizontally to same-material fences, fence gates, or
    solid full-cube blocks.
  - Walls connect horizontally to walls or solid blocks; "up" is true when
    the wall has no straight-through connection or has blocks above.
  - Glass panes / iron bars connect to panes, bars, or solid blocks.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block classification helpers
# ---------------------------------------------------------------------------

# Suffixes that identify connectable block families
_FENCE_SUFFIXES = ("_fence",)
_FENCE_GATE_SUFFIXES = ("_fence_gate",)
_WALL_SUFFIXES = ("_wall",)
_PANE_IDS = frozenset({"minecraft:glass_pane", "minecraft:iron_bars"})
_PANE_SUFFIXES = ("_stained_glass_pane",)

# Blocks that are never solid connectors (transparent / non-full-cube)
_NON_SOLID_SUFFIXES = (
    "_slab",
    "_stairs",
    "_fence",
    "_fence_gate",
    "_wall",
    "_pane",
    "_bars",
    "_door",
    "_trapdoor",
    "_button",
    "_pressure_plate",
    "_sign",
    "_wall_sign",
    "_torch",
    "_wall_torch",
    "_carpet",
    "_chain",
    "_lantern",
    "_banner",
    "_wall_banner",
    "_skull",
    "_head",
    "_candle",
    "_flower_pot",
    "_sapling",
)
_NON_SOLID_IDS = frozenset(
    {
        "minecraft:air",
        "minecraft:cave_air",
        "minecraft:void_air",
        "minecraft:water",
        "minecraft:lava",
        "minecraft:short_grass",
        "minecraft:tall_grass",
        "minecraft:fern",
        "minecraft:large_fern",
        "minecraft:torch",
        "minecraft:wall_torch",
        "minecraft:soul_torch",
        "minecraft:soul_wall_torch",
        "minecraft:redstone_wire",
        "minecraft:ladder",
        "minecraft:glass",
        "minecraft:glass_pane",
        "minecraft:iron_bars",
    }
)


def _is_fence(block_id: str) -> bool:
    return any(block_id.endswith(s) for s in _FENCE_SUFFIXES)


def _is_fence_gate(block_id: str) -> bool:
    return any(block_id.endswith(s) for s in _FENCE_GATE_SUFFIXES)


def _is_wall(block_id: str) -> bool:
    return any(block_id.endswith(s) for s in _WALL_SUFFIXES)


def _is_pane_or_bars(block_id: str) -> bool:
    if block_id in _PANE_IDS:
        return True
    return any(block_id.endswith(s) for s in _PANE_SUFFIXES)


def _is_connectable(block_id: str) -> bool:
    """Return True if this block type uses connection properties."""
    return _is_fence(block_id) or _is_wall(block_id) or _is_pane_or_bars(block_id)


def _is_solid_full_cube(block_id: str) -> bool:
    """
    Heuristic: a block is a solid full cube if it's not air and not
    one of the known non-full-cube types.

    This is intentionally conservative — false positives (treating a
    non-solid as solid) cause extra connections which look slightly
    wrong; false negatives (treating a solid as non-solid) cause
    missing connections which look very wrong.
    """
    if block_id in _NON_SOLID_IDS:
        return False
    if any(block_id.endswith(s) for s in _NON_SOLID_SUFFIXES):
        return False
    # Assume solid if not explicitly excluded
    return block_id != "minecraft:air"


def _fence_connects_to(neighbour_id: str, source_id: str) -> bool:
    """Does a fence block connect to this neighbour?

    In Minecraft, fences connect to solid full cubes, other fences, fence
    gates, and stairs (which have at least one full face).  We approximate
    stair connectivity as always-true — the per-face check is too complex
    for the offline heuristic.
    """
    if _is_fence(neighbour_id) or _is_fence_gate(neighbour_id):
        return True
    if any(neighbour_id.endswith(s) for s in ("_stairs",)):
        return True
    return _is_solid_full_cube(neighbour_id)


def _wall_connects_to(neighbour_id: str) -> str:
    """
    Wall connection value for a neighbour.

    Returns "none", "low", or "tall".
    Minecraft 1.16+ walls use "none"/"low"/"tall" rather than boolean.
    Simplified: connect as "low" to walls and solid blocks.
    """
    if _is_wall(neighbour_id) or _is_solid_full_cube(neighbour_id):
        return "low"
    return "none"


def _pane_connects_to(neighbour_id: str) -> bool:
    """Does a pane/bars block connect to this neighbour?"""
    if _is_pane_or_bars(neighbour_id):
        return True
    return _is_solid_full_cube(neighbour_id)


# ---------------------------------------------------------------------------
# BlockGrid — spatial index
# ---------------------------------------------------------------------------

# Horizontal neighbour offsets: north(-Z), south(+Z), west(-X), east(+X)
_H_NEIGHBOURS: dict[str, tuple[int, int, int]] = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
}


class BlockGrid:
    """
    Sparse spatial index of blocks keyed by integer (x, y, z).

    Each entry stores ``{"id": str, "props": dict[str, str]}``.
    Supports O(1) lookup, insert, and removal.
    """

    def __init__(self) -> None:
        self._grid: dict[tuple[int, int, int], dict[str, Any]] = {}

    def get(self, x: int, y: int, z: int) -> dict[str, Any] | None:
        return self._grid.get((x, y, z))

    def set(self, x: int, y: int, z: int, block_id: str, props: dict[str, str]) -> None:
        self._grid[(x, y, z)] = {"id": block_id, "props": dict(props)}

    def remove(self, x: int, y: int, z: int) -> None:
        self._grid.pop((x, y, z), None)

    def has(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) in self._grid

    def __len__(self) -> int:
        return len(self._grid)

    def positions(self):
        """Iterate over all occupied positions."""
        return self._grid.keys()


# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------


def resolve_connections(
    grid: BlockGrid,
    x: int,
    y: int,
    z: int,
) -> dict[str, str] | None:
    """
    Derive connection properties for the block at (x, y, z).

    Returns an updated properties dict with connection keys set, or None
    if the block is not a connectable type.

    The original non-connection properties (e.g. ``waterlogged``) are
    preserved unchanged.
    """
    entry = grid.get(x, y, z)
    if entry is None:
        return None

    block_id = entry["id"]
    props = dict(entry["props"])  # copy — don't mutate grid entry directly

    if _is_fence(block_id):
        for direction, (dx, dy, dz) in _H_NEIGHBOURS.items():
            nb = grid.get(x + dx, y + dy, z + dz)
            nb_id = nb["id"] if nb else "minecraft:air"
            props[direction] = str(_fence_connects_to(nb_id, block_id)).lower()

    elif _is_wall(block_id):
        any_connection = False
        for direction, (dx, dy, dz) in _H_NEIGHBOURS.items():
            nb = grid.get(x + dx, y + dy, z + dz)
            nb_id = nb["id"] if nb else "minecraft:air"
            val = _wall_connects_to(nb_id)
            props[direction] = val
            if val != "none":
                any_connection = True

        # "up" is true if: no straight-through, block above, or no connections
        above = grid.get(x, y + 1, z)
        n_s_straight = (
            props.get("north", "none") != "none"
            and props.get("south", "none") != "none"
            and props.get("east", "none") == "none"
            and props.get("west", "none") == "none"
        )
        e_w_straight = (
            props.get("east", "none") != "none"
            and props.get("west", "none") != "none"
            and props.get("north", "none") == "none"
            and props.get("south", "none") == "none"
        )
        is_straight = n_s_straight or e_w_straight

        if above is not None or not is_straight or not any_connection:
            props["up"] = "true"
        else:
            props["up"] = "false"

    elif _is_pane_or_bars(block_id):
        for direction, (dx, dy, dz) in _H_NEIGHBOURS.items():
            nb = grid.get(x + dx, y + dy, z + dz)
            nb_id = nb["id"] if nb else "minecraft:air"
            props[direction] = str(_pane_connects_to(nb_id)).lower()

    else:
        return None  # Not a connectable block

    return props


def get_affected_neighbours(
    x: int,
    y: int,
    z: int,
    grid: BlockGrid,
) -> list[tuple[int, int, int]]:
    """
    Return grid positions that may need connection re-evaluation when the
    block at (x, y, z) changes.

    Only returns positions that actually contain connectable blocks.
    Includes the position itself if it's connectable.
    """
    affected = []

    # The block itself
    entry = grid.get(x, y, z)
    if entry and _is_connectable(entry["id"]):
        affected.append((x, y, z))

    # Horizontal neighbours
    for dx, dy, dz in _H_NEIGHBOURS.values():
        nx, ny, nz = x + dx, y + dy, z + dz
        nb = grid.get(nx, ny, nz)
        if nb and _is_connectable(nb["id"]):
            affected.append((nx, ny, nz))

    # Block below (for wall "up" computation)
    below = grid.get(x, y - 1, z)
    if below and _is_wall(below["id"]):
        affected.append((x, y - 1, z))

    return affected


def resolve_all_connections(grid: BlockGrid) -> dict[tuple[int, int, int], dict[str, str]]:
    """
    Resolve connections for every connectable block in the grid.

    Returns a dict mapping position → updated properties for blocks
    whose connections were computed.
    """
    updates: dict[tuple[int, int, int], dict[str, str]] = {}
    for pos in grid.positions():
        x, y, z = pos
        entry = grid.get(x, y, z)
        if entry and _is_connectable(entry["id"]):
            new_props = resolve_connections(grid, x, y, z)
            if new_props is not None:
                updates[pos] = new_props
                # Update grid entry in-place for subsequent resolutions
                entry["props"] = new_props
    return updates
