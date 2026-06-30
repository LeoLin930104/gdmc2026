"""Voxel-space geometry layer for a solved (or to-be-solved) cell grid.

Layer 2 of the three-layer architecture:

  Layer 1 — :mod:`prefab_housing.grid` — pure topology (cell counts, faces).
  Layer 2 — :mod:`prefab_housing.layout` — per-cell AABBs in world voxels.
  Layer 3 — :mod:`prefab_housing.exterior` — staged voxel emission.

The split exists so that WFC, programme planning and scoring can operate on
the abstract topology while geometry varies independently per cell. Two
factories ship in v1:

- :func:`uniform_layout` — every cell shares ``cell_voxel_size``. Used by
  init-time consumers that don't yet have a solved state.
- :func:`banded_layout` — reads the solver assignment and applies per-pod
  horizontal multipliers (:data:`POD_SIZE_MULTIPLIER`). Sizes are aggregated
  *per ix-column / iz-row* (banded) so shared faces between neighbouring
  cells keep matching areas — this is the only mechanism in v1 that prevents
  ragged seams when multipliers diverge.

Vulnerability: banding sacrifices per-cell freedom; the maximum multiplier
in any column/row inflates every cell in that column/row. A pod with
multiplier 2.0 in a single cell expands its entire ix-column. The current
greedy aggregation is acceptable for the v1 scope (≤4 storeys, <6×6 grids);
larger grids should switch to a brick-bonded layout that allows independent
adjacency per storey.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import CellGrid

if TYPE_CHECKING:
    from prefab_housing.wfc.solver import SolverState


# A cell's voxel-space bounding box: ``(min_corner, max_corner)`` inclusive.
CellAABB = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class SpatialLayout:
    """Per-cell voxel AABBs anchored in a world frame.

    ``cell_bbox`` maps a flat cell index (see :meth:`CellGrid.flat_index`) to
    an inclusive AABB ``((x0,y0,z0), (x1,y1,z1))`` in world voxel coords.
    Empty cells may be omitted; consumers must therefore guard lookups.
    """

    grid: CellGrid
    origin_world: tuple[int, int, int]
    cell_bbox: Mapping[int, CellAABB]

    def bbox(self, ix: int, iy: int, iz: int) -> CellAABB:
        return self.cell_bbox[self.grid.flat_index(ix, iy, iz)]

    def has_cell(self, ix: int, iy: int, iz: int) -> bool:
        return self.grid.flat_index(ix, iy, iz) in self.cell_bbox

    def cell_size(self, ix: int, iy: int, iz: int) -> tuple[int, int, int]:
        (x0, y0, z0), (x1, y1, z1) = self.bbox(ix, iy, iz)
        return (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def uniform_layout(
    grid: CellGrid,
    cell_voxel_size: tuple[int, int, int],
    origin_world: tuple[int, int, int] = (0, 0, 0),
) -> SpatialLayout:
    """Produce a layout where every cell has the same voxel dimensions.

    The layout includes *all* cells (occupied or not) — exterior/materialisation
    stages still consult the WFC assignment to decide whether to emit voxels.
    """
    vx, vy, vz = cell_voxel_size
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError("cell voxel size must be positive")
    ox, oy, oz = origin_world
    bbox: dict[int, CellAABB] = {}
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                x0 = ox + ix * vx
                y0 = oy + iy * vy
                z0 = oz + iz * vz
                bbox[flat] = ((x0, y0, z0), (x0 + vx - 1, y0 + vy - 1, z0 + vz - 1))
    return SpatialLayout(grid=grid, origin_world=origin_world, cell_bbox=bbox)


def _column_widths(
    state: "SolverState",
    base: int,
    axis: str,
) -> list[int]:
    """Aggregate per-column (axis ∈ {'x','z'}) horizontal voxel widths from
    pod multipliers in the solved assignment.

    For each slot along ``axis`` we walk every cell in the orthogonal plane
    and take ``ceil(base * max(multiplier))`` so that shared horizontal
    faces between adjacent storeys/rows have matching extents. Empty/un
    assigned cells contribute multiplier 1.0 (the base).
    """
    grid = state.grid
    asg = state.assignment
    pod_index = state.tiles.pod_index
    if axis == "x":
        outer, inner_y, inner_z = grid.cx, grid.cy, grid.cz
    elif axis == "z":
        outer, inner_y, inner_z = grid.cz, grid.cy, grid.cx
    else:
        raise ValueError(f"unsupported axis {axis!r}")

    widths: list[int] = []
    for slot in range(outer):
        m_max = 1.0
        for iy in range(inner_y):
            for j in range(inner_z):
                if axis == "x":
                    flat = grid.flat_index(slot, iy, j)
                else:
                    flat = grid.flat_index(j, iy, slot)
                tid = int(asg[flat])
                if tid < 0:
                    continue
                pod_idx = int(pod_index[tid])
                m = pt.POD_SIZE_MULTIPLIER[pod_idx]
                if m > m_max:
                    m_max = m
        # ceil to int voxel count; 0.5 rounds away from zero.
        widths.append(int(base * m_max + 0.999_999))
    return widths


def banded_layout(
    state: "SolverState",
    base_voxel_size: tuple[int, int, int],
    origin_world: tuple[int, int, int] = (0, 0, 0),
) -> SpatialLayout:
    """Solver-aware layout with per-column / per-row banded sizing.

    Reads :data:`POD_SIZE_MULTIPLIER` from the assigned pod of every cell
    and aggregates a single width per ix-column and per iz-row (taking the
    maximum). Storey height (y) is uniform — varying it would require
    storey-aware roof and stair geometry not present in v1.

    With all multipliers equal to 1.0 (v1 default) the output is byte-for
    byte identical to :func:`uniform_layout` with the same base size.

    Pre: ``state`` is fully assigned (calling on partial states is supported
    but unassigned cells contribute multiplier 1.0).
    """
    grid = state.grid
    base_x, base_y, base_z = base_voxel_size
    if base_x <= 0 or base_y <= 0 or base_z <= 0:
        raise ValueError("base voxel size must be positive")

    widths_x = _column_widths(state, base_x, "x")
    widths_z = _column_widths(state, base_z, "z")

    # Cumulative offsets along each axis from origin.
    ox, oy, oz = origin_world
    x_offsets: list[int] = [ox]
    for w in widths_x:
        x_offsets.append(x_offsets[-1] + w)
    z_offsets: list[int] = [oz]
    for w in widths_z:
        z_offsets.append(z_offsets[-1] + w)

    bbox: dict[int, CellAABB] = {}
    for iy in range(grid.cy):
        y0 = oy + iy * base_y
        y1 = y0 + base_y - 1
        for iz in range(grid.cz):
            z0 = z_offsets[iz]
            z1 = z_offsets[iz + 1] - 1
            for ix in range(grid.cx):
                x0 = x_offsets[ix]
                x1 = x_offsets[ix + 1] - 1
                bbox[grid.flat_index(ix, iy, iz)] = ((x0, y0, z0), (x1, y1, z1))
    return SpatialLayout(grid=grid, origin_world=origin_world, cell_bbox=bbox)


__all__ = [
    "CellAABB",
    "SpatialLayout",
    "banded_layout",
    "uniform_layout",
]
