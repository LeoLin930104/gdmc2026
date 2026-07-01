"""Cell-grid coordinate system and face direction tables.

Data-oriented design: cells are addressed by integer indices `(cx, cy, cz)`
and packed into 1D arrays via :func:`flat_index`. Face directions and their
rotation chain are precomputed constants — no objects in hot paths.

Topology-only
-------------
This module describes the **abstract topology** of the cell grid: how many
cells exist, how they are addressed, and which faces neighbour which. It is
deliberately decoupled from voxel-space geometry — voxel sizes, world
origin, and per-cell AABBs live in :mod:`prefab_housing.layout`. WFC,
programme planning and scoring consume the grid; materialisation and
annotation consume both grid and layout.

Coordinate convention
---------------------
- ``+x`` east, ``+y`` up, ``+z`` south. ``-z`` is north.
- Cell-grid faces, ordered:
    0 = north (-z), 1 = east (+x), 2 = south (+z), 3 = west (-x),
    4 = up (+y),    5 = down (-y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- Face index tables -------------------------------------------------------

NORTH: Final[int] = 0
EAST: Final[int] = 1
SOUTH: Final[int] = 2
WEST: Final[int] = 3
UP: Final[int] = 4
DOWN: Final[int] = 5

NUM_FACES: Final[int] = 6
HORIZONTAL_FACES: Final[tuple[int, int, int, int]] = (NORTH, EAST, SOUTH, WEST)

FACE_NAMES: Final[tuple[str, ...]] = ("north", "east", "south", "west", "up", "down")
FACE_INDEX: Final[dict[str, int]] = {n: i for i, n in enumerate(FACE_NAMES)}

# Unit step from cell A to its neighbour through face F.
FACE_DELTA: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, -1),   # NORTH
    (1, 0, 0),    # EAST
    (0, 0, 1),    # SOUTH
    (-1, 0, 0),   # WEST
    (0, 1, 0),    # UP
    (0, -1, 0),   # DOWN
)

OPPOSITE_FACE: Final[tuple[int, ...]] = (SOUTH, WEST, NORTH, EAST, DOWN, UP)


def rotate_face_y(face: int, steps: int) -> int:
    """Rotate a face index by ``steps * 90`` degrees around Y (clockwise from above)."""
    if face >= UP:
        return face  # up/down unchanged by Y rotation
    return (face + steps) % 4


# --- Cell grid (topology only) ----------------------------------------------


@dataclass(frozen=True, slots=True)
class CellGrid:
    """Static topological description of the cell grid for a single house.

    Carries cell counts only; voxel-space geometry lives in
    :class:`prefab_housing.layout.SpatialLayout`. Hot loops should index into
    ``numpy`` arrays of length ``cells_total`` using :meth:`flat_index`; this
    dataclass is the value object that defines the index space.
    """

    cx: int                       # number of cells along +x
    cy: int                       # number of cells along +y (storeys)
    cz: int                       # number of cells along +z

    @property
    def cells_total(self) -> int:
        return self.cx * self.cy * self.cz

    def flat_index(self, ix: int, iy: int, iz: int) -> int:
        # Layout: x fastest, then z, then y. y last keeps storey slabs contiguous-friendly
        # for most score components that walk one storey at a time.
        return (iy * self.cz + iz) * self.cx + ix

    def from_flat(self, flat: int) -> tuple[int, int, int]:
        ix = flat % self.cx
        rest = flat // self.cx
        iz = rest % self.cz
        iy = rest // self.cz
        return ix, iy, iz

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.cx and 0 <= iy < self.cy and 0 <= iz < self.cz

    def neighbour(self, ix: int, iy: int, iz: int, face: int) -> tuple[int, int, int] | None:
        dx, dy, dz = FACE_DELTA[face]
        nx, ny, nz = ix + dx, iy + dy, iz + dz
        if not self.in_bounds(nx, ny, nz):
            return None
        return nx, ny, nz


def design_grid(
    footprint_xz: tuple[int, int],
    max_storeys: int,
    cell_voxel_size: tuple[int, int, int],
) -> CellGrid:
    """Compute the cell grid that fits inside ``footprint_xz`` voxels.

    ``cell_voxel_size`` is consumed only to size the topology — the resulting
    grid carries no voxel information. Storey count is bounded above by
    ``max_storeys`` and is left as the cap; the solver decides actual
    occupancy via the ``EMPTY`` sentinel tile.
    """
    fx, fz = footprint_xz
    vx, vy, vz = cell_voxel_size
    if fx <= 0 or fz <= 0:
        raise ValueError("footprint dimensions must be positive")
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError("cell voxel size must be positive")
    cx = max(1, fx // vx)
    cz = max(1, fz // vz)
    cy = max(1, max_storeys)
    return CellGrid(cx=cx, cy=cy, cz=cz)


__all__ = [
    "CellGrid",
    "DOWN",
    "EAST",
    "FACE_DELTA",
    "FACE_INDEX",
    "FACE_NAMES",
    "HORIZONTAL_FACES",
    "NORTH",
    "NUM_FACES",
    "OPPOSITE_FACE",
    "SOUTH",
    "UP",
    "WEST",
    "design_grid",
    "rotate_face_y",
]
