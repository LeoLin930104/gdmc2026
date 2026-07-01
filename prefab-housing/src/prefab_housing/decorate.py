"""Swappable whole-house decorative stages: foundation + inter-storey trim.

These helpers emit features that span multiple cells and therefore do not fit
cleanly inside per-cell shell synthesis. They are late detail stages in the
exterior pipeline and are intentionally excluded from reusable structure
template baking.

Layers
------
1. **Foundation course** — one voxel-thick ring of ``foundation`` block
   directly beneath storey-0 occupied cells (one row below each cell's
   AABB minimum y).  Provides a visual base and grounds the house.
2. **Inter-storey trim band** — at each storey seam, paint a 1-voxel band
   of ``trim_band`` block on every exterior boundary face of an occupied
   cell.  Visually breaks the slab look and signals storey count.

Vulnerability: these stages may collide with shell/facade blocks where trim
covers the corner of a ceiling slab. The exterior stage compositor handles those
collisions by deterministic later-stage priority.
"""

from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import EAST, NORTH, NUM_FACES, SOUTH, WEST
from prefab_housing.layout import SpatialLayout
from prefab_housing.palette import (
    SLOT_FOUNDATION,
    SLOT_TRIM_BAND,
)
from prefab_housing.types import SemanticBlockDict
from prefab_housing.wfc.solver import SolverState


def _block(x: int, y: int, z: int, block_id: str) -> SemanticBlockDict:
    return {"x": x, "y": y, "z": z, "id": block_id}


def _is_occupied(state: SolverState, ix: int, iy: int, iz: int) -> bool:
    grid = state.grid
    if not grid.in_bounds(ix, iy, iz):
        return False
    flat = grid.flat_index(ix, iy, iz)
    tid = int(state.assignment[flat])
    if tid < 0:
        return False
    return not pt.is_void_pod_index(int(state.tiles.pod_index[tid]))


def _exterior_faces(state: SolverState, ix: int, iy: int, iz: int) -> tuple[bool, ...]:
    """A face is 'exterior' for decoration purposes if there is no occupied
    cell on the other side.  This includes grid-boundary faces *and* faces
    adjacent to EMPTY cells (so the decoration follows the actual silhouette,
    not the bounding box)."""
    out: list[bool] = []
    for f in range(NUM_FACES):
        n = state.grid.neighbour(ix, iy, iz, f)
        if n is None:
            out.append(True)
        else:
            out.append(not _is_occupied(state, *n))
    return tuple(out)


def _foundation_blocks(
    state: SolverState, layout: SpatialLayout, foundation_id: str
) -> list[SemanticBlockDict]:
    grid = state.grid
    out: list[SemanticBlockDict] = []
    for iz in range(grid.cz):
        for ix in range(grid.cx):
            if not _is_occupied(state, ix, 0, iz):
                continue
            (x0, y0, z0), (x1, _, z1) = layout.bbox(ix, 0, iz)
            fy = y0 - 1
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    out.append(_block(x, fy, z, foundation_id))
    return out


def _trim_band_blocks(
    state: SolverState, layout: SpatialLayout, trim_id: str
) -> list[SemanticBlockDict]:
    """Single-voxel horizontal trim at each inter-storey seam, painted on
    cells whose face is exterior (boundary or against EMPTY)."""
    grid = state.grid
    out: list[SemanticBlockDict] = []
    if grid.cy < 2:
        return out
    for iy in range(grid.cy - 1):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if not _is_occupied(state, ix, iy, iz):
                    continue
                (x0, _, z0), (x1, y1, z1) = layout.bbox(ix, iy, iz)
                ext = _exterior_faces(state, ix, iy, iz)
                trim_y = y1  # top voxel row of this storey (= ceiling level)
                if ext[NORTH]:
                    for x in range(x0, x1 + 1):
                        out.append(_block(x, trim_y, z0, trim_id))
                if ext[SOUTH]:
                    for x in range(x0, x1 + 1):
                        out.append(_block(x, trim_y, z1, trim_id))
                if ext[WEST]:
                    for z in range(z0, z1 + 1):
                        out.append(_block(x0, trim_y, z, trim_id))
                if ext[EAST]:
                    for z in range(z0, z1 + 1):
                        out.append(_block(x1, trim_y, z, trim_id))
    return out


def decorate(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit foundation + storey-trim blocks for the solved state.

    Pure function: safe to call multiple times. Returned list is intended
    for compatibility callers that still want the combined decorative output.
    """
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    out: list[SemanticBlockDict] = []
    out.extend(generate_foundation_blocks(state, layout, palette))
    out.extend(generate_trim_band_blocks(state, layout, palette))
    return out


def generate_foundation_blocks(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit the swappable foundation detail layer."""
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    foundation_id = palette.get(SLOT_FOUNDATION)
    if foundation_id is None:
        return []
    return _foundation_blocks(state, layout, foundation_id)


def generate_trim_band_blocks(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit the swappable inter-storey trim layer."""
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    trim_id = palette.get(SLOT_TRIM_BAND)
    if trim_id is None:
        return []
    return _trim_band_blocks(state, layout, trim_id)


__all__ = ["decorate", "generate_foundation_blocks", "generate_trim_band_blocks"]
