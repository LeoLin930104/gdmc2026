"""Build :class:`SemanticCell` records from a solved ``SolverState``.

Output is the contract handed to the interior-decorating team: one record
per *occupied* (non-EMPTY) cell, with privacy depth and daylight score
already computed.
"""

from __future__ import annotations

from collections import deque

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import (
    FACE_NAMES,
    NUM_FACES,
    OPPOSITE_FACE,
)
from prefab_housing.layout import SpatialLayout
from prefab_housing.types import ConnectionPolicy, FaceName, SemanticCell
from prefab_housing.wfc.solver import SolverState


def _door_open_graph(state: SolverState, connection_policy: ConnectionPolicy) -> tuple[list[list[int]], list[int]]:
    """Return adjacency list for the door/open connectivity graph and the
    list of cells whose tile is the ``entry`` pod (BFS sources)."""
    grid = state.grid
    tiles = state.tiles
    asg = state.assignment
    pod_index = tiles.pod_index

    entry_pod = pt.POD_INDEX[pt.POD_ENTRY]
    C = grid.cells_total
    adj: list[list[int]] = [[] for _ in range(C)]
    sources: list[int] = []

    for flat in range(C):
        tid = int(asg[flat])
        if tid < 0:
            continue
        if int(pod_index[tid]) == entry_pod:
            sources.append(flat)
        ix, iy, iz = grid.from_flat(flat)
        policy = connection_policy.for_cell((ix, iy, iz))
        allowed = frozenset((*policy.door_faces, *policy.open_faces)) if policy is not None else frozenset()
        for f in range(NUM_FACES):
            n = grid.neighbour(ix, iy, iz, f)
            if n is None:
                continue
            name: FaceName = FACE_NAMES[f]  # type: ignore[assignment]
            if name not in allowed:
                continue
            n_flat = grid.flat_index(*n)
            n_tid = int(asg[n_flat])
            if n_tid < 0:
                continue
            nb_policy = connection_policy.for_cell(n)
            if nb_policy is None:
                continue
            opposite_name: FaceName = FACE_NAMES[OPPOSITE_FACE[f]]  # type: ignore[assignment]
            nb_allowed = frozenset((*nb_policy.door_faces, *nb_policy.open_faces))
            if opposite_name in nb_allowed:
                adj[flat].append(n_flat)

    return adj, sources


def _bfs_depths(adj: list[list[int]], sources: list[int]) -> list[int]:
    INF = -1
    depths = [INF] * len(adj)
    q: deque[int] = deque()
    for s in sources:
        depths[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depths[v] == -1:
                depths[v] = depths[u] + 1
                q.append(v)
    return depths


def _daylight_for_cell(
    state: SolverState, flat: int
) -> float:
    """Fraction of WINDOW faces on this tile that align with a grid boundary."""
    tiles = state.tiles
    asg = state.assignment
    grid = state.grid
    tid = int(asg[flat])
    if tid < 0:
        return 0.0
    faces_t = tiles.faces[tid]
    ix, iy, iz = grid.from_flat(flat)
    boundary = [grid.neighbour(ix, iy, iz, f) is None for f in range(NUM_FACES)]
    window_faces = [int(faces_t[f]) == pt.WINDOW for f in range(NUM_FACES)]
    n_windows = sum(window_faces)
    if n_windows == 0:
        return 0.0
    n_aligned = sum(1 for f in range(NUM_FACES) if window_faces[f] and boundary[f])
    return n_aligned / n_windows


def _interior_volume(cell_voxel_size: tuple[int, int, int]) -> int:
    """Heuristic interior volume after the 1-voxel shell is subtracted."""
    vx, vy, vz = cell_voxel_size
    inner_x = max(0, vx - 2)
    inner_y = max(0, vy - 2)
    inner_z = max(0, vz - 2)
    return inner_x * inner_y * inner_z


def _opening_pattern(door_faces: list[FaceName], open_faces: list[FaceName]) -> str:
    directional = tuple(dict.fromkeys((*door_faces, *open_faces)))
    if not directional:
        return "sealed"
    if len(directional) == 1 and not open_faces:
        return "edge_only"
    return "multi_direction_open"


def annotate(state: SolverState, layout: SpatialLayout, connection_policy: ConnectionPolicy) -> list[SemanticCell]:
    """Build one :class:`SemanticCell` per occupied cell of ``state``."""
    grid = state.grid
    if layout.grid is not grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    tiles = state.tiles
    asg = state.assignment
    pod_index = tiles.pod_index
    adj, sources = _door_open_graph(state, connection_policy)
    depths = _bfs_depths(adj, sources)

    out: list[SemanticCell] = []
    for flat in range(grid.cells_total):
        tid = int(asg[flat])
        if tid < 0:
            continue
        pod_idx = int(pod_index[tid])
        label = pt.POD_LABELS[pod_idx]
        if pt.is_void_pod_index(pod_idx):
            continue
        ix, iy, iz = grid.from_flat(flat)
        bbox = layout.bbox(ix, iy, iz)
        cell_size = layout.cell_size(ix, iy, iz)
        role = pt.POD_ROLE[pod_idx]
        occupancy = int(pt.POD_OCCUPANCY[pod_idx])

        policy = connection_policy.for_cell((ix, iy, iz))
        if policy is None:
            door_faces = []
            window_faces = []
            open_faces = []
            opening_pattern = "sealed"
        else:
            door_faces = list(policy.door_faces)
            window_faces = list(policy.window_faces)
            open_faces = list(policy.open_faces)
            opening_pattern = policy.opening_pattern

        privacy_depth = depths[flat] if depths[flat] >= 0 else -1

        out.append(
            SemanticCell(
                cell_index=(ix, iy, iz),
                voxel_bbox=bbox,
                label=label,
                role=role,  # type: ignore[arg-type]
                occupancy_capacity=occupancy,
                daylight_score=_daylight_for_cell(state, flat),
                privacy_depth=privacy_depth,
                door_faces=tuple(door_faces),
                window_faces=tuple(window_faces),
                interior_volume_voxels=_interior_volume(cell_size),
                pod_template_id=tiles.tile_label[tid],
                open_faces=tuple(open_faces),
                opening_pattern=opening_pattern,
            )
        )

    return out


__all__ = [
    "annotate",
]
