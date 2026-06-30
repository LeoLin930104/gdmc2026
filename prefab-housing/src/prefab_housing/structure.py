"""Structural integrity analysis for solved 3D cell plans.

This is the planning-stage physics layer: it reasons about support chains,
cantilever depth, altitude penalties, and clustering. The current system uses
cell-level rather than voxel-level mechanics, which is sufficient for early
settlement-scale massing and is cheap enough to run in scoring and reporting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependency sync
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.wfc.solver import SolverState


@dataclass(frozen=True, slots=True)
class StructuralReport:
    occupied_mask: np.ndarray
    supported_mask: np.ndarray
    support_depth: np.ndarray
    cantilever_distance: np.ndarray
    anchored_mask: np.ndarray
    max_altitude: int
    largest_cluster: int
    occupied_cells: int
    supported_cells: int
    unsupported_cells: int
    anchored_cells: int
    support_ratio: float
    anchored_ratio: float
    overhang_ratio: float
    unsupported_indices: tuple[tuple[int, int, int], ...]


@njit(cache=True)
def _build_occupied_mask_numba(
    assignment: np.ndarray,
    pod_idx: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
    void_a: int,
    void_b: int,
) -> np.ndarray:
    occupied = np.zeros((cx, cy, cz), dtype=np.bool_)
    for iy in range(cy):
        for iz in range(cz):
            for ix in range(cx):
                flat = (iy * cz + iz) * cx + ix
                tid = int(assignment[flat])
                if tid < 0:
                    continue
                pod = int(pod_idx[tid])
                if pod == void_a or pod == void_b:
                    continue
                occupied[ix, iy, iz] = True
    return occupied


@njit(cache=True)
def _analyse_support_numba(
    occupied: np.ndarray,
    allow_lateral_support: bool,
    lateral_support_quorum: int,
    lateral_max_span: int,
    diagonal_support: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = occupied.shape
    supported = np.zeros((cx, cy, cz), dtype=np.bool_)
    support_depth = np.full((cx, cy, cz), -1, dtype=np.int32)
    cantilever_distance = np.full((cx, cy, cz), -1, dtype=np.int32)

    if cy > 0:
        for ix in range(cx):
            for iz in range(cz):
                if occupied[ix, 0, iz]:
                    supported[ix, 0, iz] = True
                    support_depth[ix, 0, iz] = 0
                    cantilever_distance[ix, 0, iz] = 0

    for iy in range(1, cy):
        for iz in range(cz):
            for ix in range(cx):
                if not occupied[ix, iy, iz]:
                    continue
                if supported[ix, iy - 1, iz]:
                    supported[ix, iy, iz] = True
                    support_depth[ix, iy, iz] = support_depth[ix, iy - 1, iz] + 1
                    cantilever_distance[ix, iy, iz] = 0

        if not allow_lateral_support:
            continue

        changed = True
        while changed:
            changed = False
            for iz in range(cz):
                for ix in range(cx):
                    if not occupied[ix, iy, iz] or supported[ix, iy, iz]:
                        continue
                    votes = 0
                    best_depth = -1
                    best_cantilever = 1_000_000
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = ix + dx
                        nz = iz + dz
                        if nx < 0 or nx >= cx or nz < 0 or nz >= cz:
                            continue
                        if not supported[nx, iy, nz]:
                            continue
                        neighbour_cantilever = int(cantilever_distance[nx, iy, nz])
                        if neighbour_cantilever < 0 or neighbour_cantilever >= lateral_max_span:
                            continue
                        votes += 1
                        neighbour_depth = int(support_depth[nx, iy, nz])
                        if best_depth < 0 or neighbour_depth < best_depth:
                            best_depth = neighbour_depth
                        if neighbour_cantilever < best_cantilever:
                            best_cantilever = neighbour_cantilever
                    if diagonal_support:
                        for dx, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            nx = ix + dx
                            nz = iz + dz
                            if nx < 0 or nx >= cx or nz < 0 or nz >= cz:
                                continue
                            if not supported[nx, iy, nz]:
                                continue
                            neighbour_cantilever = int(cantilever_distance[nx, iy, nz])
                            if neighbour_cantilever < 0 or neighbour_cantilever >= lateral_max_span:
                                continue
                            votes += 1
                            neighbour_depth = int(support_depth[nx, iy, nz])
                            if best_depth < 0 or neighbour_depth < best_depth:
                                best_depth = neighbour_depth
                            if neighbour_cantilever < best_cantilever:
                                best_cantilever = neighbour_cantilever
                    if votes < lateral_support_quorum:
                        continue
                    supported[ix, iy, iz] = True
                    support_depth[ix, iy, iz] = 0 if best_depth < 0 else best_depth
                    cantilever_distance[ix, iy, iz] = best_cantilever + 1
                    changed = True

    return supported, support_depth, cantilever_distance


def analyse_structure(
    state: SolverState,
    *,
    allow_lateral_support: bool = True,
    lateral_support_quorum: int = 2,
    lateral_max_span: int = 1,
    diagonal_support: bool = False,
) -> StructuralReport:
    grid = state.grid
    pod_idx = state.tiles.pod_index
    asg = state.assignment

    occupied = _build_occupied_mask_numba(
        asg,
        pod_idx,
        grid.cx,
        grid.cy,
        grid.cz,
        pt.POD_INDEX[pt.POD_STRUCTURAL_VOID],
        pt.POD_INDEX[pt.POD_TERRACE_VOID],
    )
    supported, support_depth, cantilever_distance = _analyse_support_numba(
        occupied,
        allow_lateral_support,
        lateral_support_quorum,
        lateral_max_span,
        diagonal_support,
    )
    anchored = occupied & supported & (cantilever_distance == 0)

    largest_cluster = 0
    seen = np.zeros_like(occupied, dtype=bool)
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if not occupied[ix, iy, iz] or seen[ix, iy, iz]:
                    continue
                size = 0
                q_cluster: deque[tuple[int, int, int]] = deque([(ix, iy, iz)])
                seen[ix, iy, iz] = True
                while q_cluster:
                    cx, cy, cz = q_cluster.popleft()
                    size += 1
                    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)):
                        nx, ny, nz = cx + dx, cy + dy, cz + dz
                        if not (0 <= nx < grid.cx and 0 <= ny < grid.cy and 0 <= nz < grid.cz):
                            continue
                        if not occupied[nx, ny, nz] or seen[nx, ny, nz]:
                            continue
                        seen[nx, ny, nz] = True
                        q_cluster.append((nx, ny, nz))
                if size > largest_cluster:
                    largest_cluster = size

    occupied_cells = int(occupied.sum())
    supported_cells = int((occupied & supported).sum())
    anchored_cells = int(anchored.sum())
    unsupported_indices: list[tuple[int, int, int]] = []
    max_altitude = 0
    for iy in range(grid.cy):
        if occupied[:, iy, :].any():
            max_altitude = iy
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if occupied[ix, iy, iz] and not supported[ix, iy, iz]:
                    unsupported_indices.append((ix, iy, iz))
    unsupported_cells = len(unsupported_indices)
    return StructuralReport(
        occupied_mask=occupied,
        supported_mask=supported,
        support_depth=support_depth,
        cantilever_distance=cantilever_distance,
        anchored_mask=anchored,
        max_altitude=max_altitude,
        largest_cluster=largest_cluster,
        occupied_cells=occupied_cells,
        supported_cells=supported_cells,
        unsupported_cells=unsupported_cells,
        anchored_cells=anchored_cells,
        support_ratio=(supported_cells / occupied_cells) if occupied_cells else 1.0,
        anchored_ratio=(anchored_cells / occupied_cells) if occupied_cells else 1.0,
        overhang_ratio=((supported_cells - anchored_cells) / occupied_cells) if occupied_cells else 0.0,
        unsupported_indices=tuple(unsupported_indices),
    )


def assigned_cells_have_support_potential(
    state: SolverState,
    *,
    allow_lateral_support: bool = True,
    lateral_support_quorum: int = 2,
    lateral_max_span: int = 1,
    diagonal_support: bool = False,
) -> bool:
    """Return True iff every currently assigned occupied cell can still acquire support.

    This is an optimistic partial-state check for search-time pruning. Unassigned
    cells count as potentially occupied when their domain still contains at least
    one non-void tile, so we only reject a branch when an assigned occupied cell
    has no remaining route to ground support even under best-case future fills.
    """
    grid = state.grid
    pod_idx = state.tiles.pod_index
    assignment = state.assignment
    occupied_tile = np.ones(state.tiles.num_tiles, dtype=np.bool_)
    occupied_tile[pt.POD_INDEX[pt.POD_STRUCTURAL_VOID]] = False
    occupied_tile[pt.POD_INDEX[pt.POD_TERRACE_VOID]] = False

    possible_occupied = np.zeros((grid.cx, grid.cy, grid.cz), dtype=np.bool_)
    assigned_occupied = np.zeros((grid.cx, grid.cy, grid.cz), dtype=np.bool_)
    for flat in range(grid.cells_total):
        ix, iy, iz = grid.from_flat(flat)
        tid = int(assignment[flat])
        if tid >= 0:
            if occupied_tile[tid]:
                possible_occupied[ix, iy, iz] = True
                assigned_occupied[ix, iy, iz] = True
            continue
        if bool((state.domain[flat] & occupied_tile).any()):
            possible_occupied[ix, iy, iz] = True

    potentially_supported, _, _ = _analyse_support_numba(
        possible_occupied,
        allow_lateral_support,
        lateral_support_quorum,
        lateral_max_span,
        diagonal_support,
    )
    unsupported_assigned = assigned_occupied & ~potentially_supported
    return not bool(unsupported_assigned.any())


__all__ = [
    "StructuralReport",
    "analyse_structure",
    "assigned_cells_have_support_potential",
]
