"""WFC solver: domain bitsets, AC-3 propagation, single-step collapse.

Data layout (DOD)
-----------------
- ``domain : bool[C, T]`` — per-cell tile mask. True ⇔ tile still legal here.
- ``assignment : int16[C]`` — chosen tile per cell, or ``-1`` if undecided.
- ``entropy_count : int32[C]`` — number of legal tiles per cell (sum of mask).

The solver is *not* responsible for picking which tile to collapse to next —
that is the search layer's job (random for vanilla WFC, MCTS for the real
pipeline). This module exposes:

- :func:`init_state` — build initial domain incorporating boundary EXTERIOR.
- :func:`propagate` — AC-3 propagation given a starting set of dirty cells.
- :func:`collapse_to` — assign a tile and propagate.
- :func:`is_solved` / :func:`is_contradicted`.

The state is held in a mutable :class:`SolverState`. AC-3 mutates in place;
copy via :func:`copy_state` for backtracking / MCTS rollouts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from prefab_housing.grid import (
    CellGrid,
    FACE_DELTA,
    NUM_FACES,
    OPPOSITE_FACE,
)
from prefab_housing.wfc.tiles import TileSet


@dataclass
class SolverState:
    grid: CellGrid
    tiles: TileSet
    domain: np.ndarray              # bool[C, T]
    assignment: np.ndarray          # int16[C]; -1 = undecided
    entropy_count: np.ndarray       # int32[C]
    contradicted: bool = False

    def copy(self) -> "SolverState":
        return SolverState(
            grid=self.grid,
            tiles=self.tiles,
            domain=self.domain.copy(),
            assignment=self.assignment.copy(),
            entropy_count=self.entropy_count.copy(),
            contradicted=self.contradicted,
        )


def init_state(grid: CellGrid, tiles: TileSet) -> SolverState:
    C = grid.cells_total
    T = tiles.num_tiles
    domain = np.ones((C, T), dtype=bool)
    assignment = np.full(C, -1, dtype=np.int16)
    entropy_count = np.full(C, T, dtype=np.int32)

    state = SolverState(grid, tiles, domain, assignment, entropy_count)

    # Apply boundary constraints: at any face of any cell that points outside
    # the grid, only tiles whose face is EXTERIOR-compatible may remain.
    EXT = T  # column index for EXTERIOR sentinel in compat table
    compat = tiles.compat

    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                cell_mask = state.domain[flat]
                for f in range(NUM_FACES):
                    if grid.neighbour(ix, iy, iz, f) is None:
                        # Boundary face: keep only tiles compatible with EXTERIOR.
                        legal_against_ext = compat[:, f, EXT]   # bool[T]
                        cell_mask &= legal_against_ext
                state.domain[flat] = cell_mask
                state.entropy_count[flat] = int(cell_mask.sum())
                if state.entropy_count[flat] == 0:
                    state.contradicted = True

    if not state.contradicted:
        # AC-3 from every cell since boundary constraints reshape the lattice.
        propagate(state, dirty=range(C))

    return state


def propagate(state: SolverState, dirty: "deque[int] | range | list[int]") -> None:
    """Arc-consistency propagation: shrink neighbours' domains to those
    supported by ``cell``'s current domain across the shared face.

    Mutates ``state``. Sets ``state.contradicted`` if any domain empties.
    """
    grid = state.grid
    tiles = state.tiles
    compat = tiles.compat
    domain = state.domain
    entropy = state.entropy_count
    T = tiles.num_tiles

    queue: deque[int] = deque(dirty) if not isinstance(dirty, deque) else dirty

    while queue:
        flat = queue.popleft()
        if state.contradicted:
            return
        ix, iy, iz = grid.from_flat(flat)
        cell_mask = domain[flat]                         # bool[T]
        if not cell_mask.any():
            state.contradicted = True
            return

        for f in range(NUM_FACES):
            n = grid.neighbour(ix, iy, iz, f)
            if n is None:
                continue
            n_flat = grid.flat_index(*n)
            opp = OPPOSITE_FACE[f]
            # For each candidate tile in neighbour's domain, it survives iff
            # at least one tile in cell_mask is compatible across face f.
            # support[b] = OR over a in cell_mask of compat[a, f, b]
            # Vectorise: legal_pairs = compat[:, f, :] (T, T); restrict rows to cell_mask.
            legal_pairs = compat[:, f, :T]                # bool[T, T]
            # For neighbour face index opp, we need compat from *neighbour's*
            # perspective. But compat is symmetric in the sense that
            # compat[a, f, b] holds iff cat[a,f] is compatible with cat[b,opp(f)].
            # So domain[n_flat] should be reduced to those b for which exists
            # a in cell_mask with compat[a, f, b]=True. Equivalently:
            support = (legal_pairs[cell_mask, :]).any(axis=0)  # bool[T]
            new_neighbour_mask = domain[n_flat] & support
            if not np.array_equal(new_neighbour_mask, domain[n_flat]):
                domain[n_flat] = new_neighbour_mask
                entropy[n_flat] = int(new_neighbour_mask.sum())
                if entropy[n_flat] == 0:
                    state.contradicted = True
                    return
                if state.assignment[n_flat] == -1:
                    queue.append(n_flat)


def collapse_to(state: SolverState, flat: int, tile_id: int) -> None:
    """Assign ``tile_id`` to cell ``flat`` and propagate."""
    if state.contradicted:
        return
    if not state.domain[flat, tile_id]:
        state.contradicted = True
        return
    new_mask = np.zeros(state.tiles.num_tiles, dtype=bool)
    new_mask[tile_id] = True
    state.domain[flat] = new_mask
    state.entropy_count[flat] = 1
    state.assignment[flat] = tile_id
    propagate(state, dirty=deque([flat]))


def is_solved(state: SolverState) -> bool:
    return bool(np.all(state.assignment >= 0)) and not state.contradicted


def is_contradicted(state: SolverState) -> bool:
    return state.contradicted


def lowest_entropy_cell(state: SolverState) -> int | None:
    """Return the flat index of the undecided cell with the smallest domain.

    Ties broken by ascending flat index for statistical determinism.
    """
    if state.contradicted:
        return None
    undecided = state.assignment < 0
    if not undecided.any():
        return None
    masked = np.where(undecided, state.entropy_count, np.iinfo(np.int32).max)
    idx = int(np.argmin(masked))
    return idx if undecided[idx] else None


def candidate_tiles(state: SolverState, flat: int) -> np.ndarray:
    """Return the int array of tile_ids still legal at ``flat`` (sorted)."""
    return np.flatnonzero(state.domain[flat]).astype(np.int32)


__all__ = [
    "SolverState",
    "candidate_tiles",
    "collapse_to",
    "init_state",
    "is_contradicted",
    "is_solved",
    "lowest_entropy_cell",
    "propagate",
]
