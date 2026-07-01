"""Per-cell programme-prior pruning.

Hard masks applied *after* :func:`prefab_housing.wfc.solver.init_state` and
*before* MCTS. These complement the categorical compat table by encoding
positional semantics that aren't expressible as pairwise face constraints:

R1. **Entry placement.** An ``entry`` tile is allowed only on a ground-floor
    (``iy == 0``) cell that has at least one horizontal boundary face, AND
    only those rotation variants whose ``DOOR`` face is aligned with one of
    those boundary faces. Result: front doors face outward.

R2. **Windows on boundary faces only.** A tile whose face profile carries
    ``WINDOW`` on a non-boundary face is forbidden at that cell. Windows
    therefore only ever look outside — never into another cell.

R3. **External doors are entry-exclusive.** For any non-entry tile, ``DOOR``
    on a boundary face is forbidden. Only the front door is a front door.

These are *hard* restrictions: domains are masked then AC-3 is re-run. If the
mask kills every tile in a cell, the state is contradicted and the search
layer is responsible for handling it (in practice only on degenerate inputs).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import HORIZONTAL_FACES, NUM_FACES
from prefab_housing.programme import Programme
from prefab_housing.wfc.solver import SolverState, propagate


def _required_count(programme: Programme, label: str) -> int:
    return programme.required_counter().get(label, 0)


def _boundary_face_mask(state: SolverState, ix: int, iy: int, iz: int) -> np.ndarray:
    """Return ``bool[NUM_FACES]`` — True for faces that point outside the grid."""
    grid = state.grid
    out = np.zeros(NUM_FACES, dtype=bool)
    for f in range(NUM_FACES):
        if grid.neighbour(ix, iy, iz, f) is None:
            out[f] = True
    return out


def _build_per_cell_allow_mask(state: SolverState) -> np.ndarray:
    """Compute ``bool[C, T]`` allowed-by-position mask for the whole grid."""
    grid = state.grid
    tiles = state.tiles
    T = tiles.num_tiles
    C = grid.cells_total

    # Tile metadata pulled into local arrays for hot loop.
    pod_index = tiles.pod_index            # int8[T]
    faces = tiles.faces                    # int8[T, 6]
    entry_pod = pt.POD_INDEX[pt.POD_ENTRY]

    # Per-tile: which faces are DOOR / WINDOW.
    door_face_mask = faces == pt.DOOR      # bool[T, 6]
    window_face_mask = faces == pt.WINDOW  # bool[T, 6]
    is_entry = pod_index == entry_pod      # bool[T]

    allow = np.ones((C, T), dtype=bool)

    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                bnd = _boundary_face_mask(state, ix, iy, iz)
                horiz_bnd = bnd.copy()
                horiz_bnd[4:] = False  # discard up/down for "perimeter" semantics
                is_ground = (iy == 0)
                has_horiz_boundary = bool(horiz_bnd.any())

                # R2: WINDOW faces must lie on boundary faces.
                # tile fails if any WINDOW face of the tile is NOT a boundary.
                # i.e. window_face_mask[t] AND NOT bnd has any True → forbid.
                non_bnd = ~bnd                                   # bool[6]
                window_violates = (window_face_mask & non_bnd).any(axis=1)  # bool[T]
                allow[flat] &= ~window_violates

                # R3: non-entry pods may not have DOOR on a boundary face.
                door_on_boundary = (door_face_mask & bnd).any(axis=1)  # bool[T]
                non_entry_violates = door_on_boundary & ~is_entry
                allow[flat] &= ~non_entry_violates

                # R1: entry only on ground-floor with a horizontal boundary,
                # AND its DOOR face must align with one of those boundaries.
                if not (is_ground and has_horiz_boundary):
                    allow[flat] &= ~is_entry
                else:
                    door_on_horiz_boundary = (door_face_mask & horiz_bnd).any(axis=1)
                    entry_misaligned = is_entry & ~door_on_horiz_boundary
                    allow[flat] &= ~entry_misaligned

    return allow


def apply_position_priors(
    state: SolverState,
    programme: Programme,
    *,
    utility_type: str = "residential",
    public_storey_max: int = 0,
    private_storey_min: int = 0,
    terrace_start_storey: int | None = None,
    tower_core_cell: tuple[int, int] | None = None,
) -> None:
    """Prune ``state.domain`` by per-cell positional priors and re-propagate.

    Mutates ``state``. Sets ``state.contradicted=True`` if any cell empties.

    The programme-derived stage outputs let the planner keep public/service uses
    low, private uses high, and upper-storey carve-outs reachable without
    hard-coding building archetypes into the tile catalogue.
    """
    if state.contradicted:
        return
    allow = _build_per_cell_allow_mask(state)

    # Stage-derived storey constraints: keep public/service uses low and private
    # uses higher when the utility policy expects vertical separation.
    grid = state.grid
    tiles = state.tiles
    pod_index = tiles.pod_index
    role_required = programme.required_counter()
    has_vertical_programme = programme.target_min_cells > max(1, grid.cx * grid.cz)
    upper_terrace_storeys = terrace_start_storey if terrace_start_storey is not None else grid.cy
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                is_tower_core = tower_core_cell == (ix, iz)
                for tid in range(tiles.num_tiles):
                    pod = pt.POD_LABELS[int(pod_index[tid])]
                    if is_tower_core and grid.cy >= 4:
                        # Tall grids need a guaranteed anchored vertical spine.
                        # Without this, search can spend most of its budget on
                        # sparse upper-storey voiding that later fails structural
                        # validation.
                        if pod != pt.POD_STAIRWELL:
                            allow[flat, tid] = False
                            continue
                    if pod == pt.POD_STRUCTURAL_VOID:
                        if is_tower_core:
                            allow[flat, tid] = False
                        continue
                    if pod == pt.POD_TERRACE_VOID:
                        if is_tower_core:
                            allow[flat, tid] = False
                            continue
                        if iy < upper_terrace_storeys:
                            allow[flat, tid] = False
                        continue
                    if not has_vertical_programme:
                        continue
                    # Bedrooms above ground when possible.
                    if pod == pt.POD_BEDROOM and iy < private_storey_min and grid.cy >= 2 and role_required.get(pt.POD_BEDROOM, 0) >= 2:
                        allow[flat, tid] = False
                    # Entry stays grounded.
                    if pod == pt.POD_ENTRY and iy > 0:
                        allow[flat, tid] = False
                    # Public-facing uses stay low when the planner requested split stacking.
                    if pod in (pt.POD_ENTRY, pt.POD_LIVING, pt.POD_KITCHEN) and iy > public_storey_max:
                        allow[flat, tid] = False
                    # Taller programmes should keep the stair core off the outer skin
                    # when a more central slot exists.
                    if pod == pt.POD_STAIRWELL and grid.cy >= 3 and grid.cx >= 3 and grid.cz >= 3:
                        bnd = _boundary_face_mask(state, ix, iy, iz)
                        if bool(bnd[:4].any()):
                            allow[flat, tid] = False
                    # Upper-storey service pods should stay off exposed corners in
                    # taller buildings so stacks consolidate around an internal core.
                    if pod in (pt.POD_BATHROOM, pt.POD_KITCHEN) and grid.cy >= 3 and iy > 0:
                        horiz_boundary_count = 0
                        for f in HORIZONTAL_FACES:
                            if grid.neighbour(ix, iy, iz, f) is None:
                                horiz_boundary_count += 1
                        if horiz_boundary_count >= 2:
                            allow[flat, tid] = False
    new_domain = state.domain & allow
    changed = np.any(new_domain != state.domain, axis=1)  # bool[C]
    if not changed.any():
        return
    state.domain = new_domain
    state.entropy_count = new_domain.sum(axis=1, dtype=np.int32)
    if (state.entropy_count == 0).any():
        state.contradicted = True
        return
    dirty_cells = np.flatnonzero(changed).tolist()
    propagate(state, dirty=deque(dirty_cells))


__all__ = [
    "apply_position_priors",
]
