"""Modular whole-house materialisation.

Pipeline stages:

1. solve topology
2. place boxed placeholder cells
3. resolve connections between adjacent cells
4. add swappable facade/detail stages later
"""

from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.catalogue.shell import build_exterior_face_overlay, build_placeholder_cell
from prefab_housing.grid import DOWN, EAST, FACE_NAMES, HORIZONTAL_FACES, NORTH, OPPOSITE_FACE, SOUTH, UP, WEST
from prefab_housing.layout import SpatialLayout
from prefab_housing.stairwell import stairwell_opening_rect
from prefab_housing.types import ConnectionPolicy, FaceName, SemanticBlockDict
from prefab_housing.wfc.solver import SolverState


def _dedupe(blocks: list[SemanticBlockDict]) -> list[SemanticBlockDict]:
    by_pos: dict[tuple[int, int, int], SemanticBlockDict] = {}
    for block in blocks:
        by_pos[(block["x"], block["y"], block["z"])] = block
    return list(by_pos.values())


def _translate_blocks(
    local_blocks: list[SemanticBlockDict],
    origin: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    ox, oy, oz = origin
    out: list[SemanticBlockDict] = []
    for block in local_blocks:
        out.append(
            {
                "x": block["x"] + ox,
                "y": block["y"] + oy,
                "z": block["z"] + oz,
                "id": block["id"],
            }
        )
    return out


def _air_exposed_faces(state: SolverState, ix: int, iy: int, iz: int) -> tuple[int, ...]:
    grid = state.grid
    asg = state.assignment
    pod_index = state.tiles.pod_index
    out: list[int] = []
    for face in (NORTH, EAST, SOUTH, WEST):
        nb = grid.neighbour(ix, iy, iz, face)
        if nb is None:
            out.append(face)
            continue
        nflat = grid.flat_index(*nb)
        ntid = int(asg[nflat])
        if ntid < 0 or pt.is_void_pod_index(int(pod_index[ntid])):
            out.append(face)
    return tuple(out)


def _connection_opening(
    *,
    face: int,
    bbox: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    (x0, y0, z0), (x1, y1, z1) = bbox
    y_mid0 = y0 + 1
    y_mid1 = y1 - 2
    if face == NORTH:
        return {(x, y, z0) for x in range(x0 + 2, x1 - 1) for y in range(y_mid0, y_mid1 + 1)}
    if face == SOUTH:
        return {(x, y, z1) for x in range(x0 + 2, x1 - 1) for y in range(y_mid0, y_mid1 + 1)}
    if face == WEST:
        return {(x0, y, z) for z in range(z0 + 2, z1 - 1) for y in range(y_mid0, y_mid1 + 1)}
    if face == EAST:
        return {(x1, y, z) for z in range(z0 + 2, z1 - 1) for y in range(y_mid0, y_mid1 + 1)}
    return set()


def _vertical_opening(
    *,
    face: int,
    bbox: tuple[tuple[int, int, int], tuple[int, int, int]],
    cell_index: tuple[int, int, int],
) -> set[tuple[int, int, int]]:
    (x0, y0, z0), (x1, y1, z1) = bbox
    vx = x1 - x0 + 1
    vy = y1 - y0 + 1
    vz = z1 - z0 + 1
    direction = "up" if face == UP else "down"
    rx0, rx1, rz0, rz1 = stairwell_opening_rect((vx, vy, vz), cell_index, direction=direction)
    y = y1 if face == UP else y0
    return {
        (x0 + x, y, z0 + z)
        for x in range(rx0, rx1 + 1)
        for z in range(rz0, rz1 + 1)
    }


def _is_occupied_cell(state: SolverState, ix: int, iy: int, iz: int) -> bool:
    flat = state.grid.flat_index(ix, iy, iz)
    tid = int(state.assignment[flat])
    if tid < 0:
        return False
    return not pt.is_void_pod_index(int(state.tiles.pod_index[tid]))


def _should_carve_connection(
    connection_policy: ConnectionPolicy,
    ix: int,
    iy: int,
    iz: int,
    face: int,
    nb: tuple[int, int, int],
) -> bool:
    current = connection_policy.for_cell((ix, iy, iz))
    other = connection_policy.for_cell(nb)
    if current is None or other is None:
        return False
    face_name: FaceName = FACE_NAMES[face]  # type: ignore[assignment]
    opposite_name: FaceName = FACE_NAMES[OPPOSITE_FACE[face]]  # type: ignore[assignment]
    current_allowed = frozenset((*current.door_faces, *current.open_faces))
    other_allowed = frozenset((*other.door_faces, *other.open_faces))
    return face_name in current_allowed and opposite_name in other_allowed


def _position(block: SemanticBlockDict) -> tuple[int, int, int]:
    return (int(block["x"]), int(block["y"]), int(block["z"]))


def _place_structural_shells(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    grid = state.grid
    pod_index = state.tiles.pod_index
    out: list[SemanticBlockDict] = []
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                tid = int(state.assignment[flat])
                if tid < 0:
                    continue
                pod_idx = int(pod_index[tid])
                if pt.is_void_pod_index(pod_idx):
                    continue
                pod_name = pt.POD_LABELS[pod_idx]
                local_blocks = build_placeholder_cell(
                    cell_voxel_size=layout.cell_size(ix, iy, iz),
                    palette=palette,
                    pod_name=pod_name,
                )
                origin, _ = layout.bbox(ix, iy, iz)
                out.extend(_translate_blocks(local_blocks, origin))
    return out


def _place_facade_overlays(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
    connection_policy: ConnectionPolicy | None = None,
) -> list[SemanticBlockDict]:
    grid = state.grid
    pod_index = state.tiles.pod_index
    out: list[SemanticBlockDict] = []
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                flat = grid.flat_index(ix, iy, iz)
                tid = int(state.assignment[flat])
                if tid < 0:
                    continue
                pod_idx = int(pod_index[tid])
                if pt.is_void_pod_index(pod_idx):
                    continue
                pod_name = pt.POD_LABELS[pod_idx]
                origin, _ = layout.bbox(ix, iy, iz)
                policy = connection_policy.for_cell((ix, iy, iz)) if connection_policy is not None else None
                blocked_faces = frozenset(
                    (*policy.door_faces, *policy.open_faces)
                ) if policy is not None else frozenset()
                for face in _air_exposed_faces(state, ix, iy, iz):
                    face_name: FaceName = FACE_NAMES[face]  # type: ignore[assignment]
                    if face_name in blocked_faces:
                        continue
                    overlay_blocks = build_exterior_face_overlay(
                        face=face,
                        cell_voxel_size=layout.cell_size(ix, iy, iz),
                        palette=palette,
                        pod_name=pod_name,
                    )
                    out.extend(_translate_blocks(overlay_blocks, origin))
    return out


def carve_connection_openings(
    state: SolverState,
    layout: SpatialLayout,
    connection_policy: ConnectionPolicy,
    placed_blocks: list[SemanticBlockDict],
) -> tuple[list[SemanticBlockDict], tuple[tuple[int, int, int], ...]]:
    blocked: set[tuple[int, int, int]] = set()
    grid = state.grid
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if not _is_occupied_cell(state, ix, iy, iz):
                    continue
                bbox = layout.bbox(ix, iy, iz)
                policy = connection_policy.for_cell((ix, iy, iz))
                boundary_open_faces = frozenset(
                    (*policy.door_faces, *policy.open_faces)
                ) if policy is not None else frozenset()
                for face in HORIZONTAL_FACES:
                    face_name: FaceName = FACE_NAMES[face]  # type: ignore[assignment]
                    if grid.neighbour(ix, iy, iz, face) is None:
                        if face_name in boundary_open_faces:
                            blocked.update(_connection_opening(face=face, bbox=bbox))
                        continue
                    if face not in (EAST, SOUTH):
                        continue
                    nb = grid.neighbour(ix, iy, iz, face)
                    if nb is None or not _is_occupied_cell(state, *nb):
                        continue
                    if not _should_carve_connection(connection_policy, ix, iy, iz, face, nb):
                        continue
                    blocked.update(_connection_opening(face=face, bbox=bbox))
                    blocked.update(
                        _connection_opening(face=WEST if face == EAST else NORTH, bbox=layout.bbox(*nb))
                    )

                for face in (UP,):
                    nb = grid.neighbour(ix, iy, iz, face)
                    if nb is None or not _is_occupied_cell(state, *nb):
                        continue
                    if not _should_carve_connection(connection_policy, ix, iy, iz, face, nb):
                        continue
                    blocked.update(
                        _vertical_opening(face=UP, bbox=bbox, cell_index=(ix, iy, iz))
                    )
                    blocked.update(
                        _vertical_opening(face=DOWN, bbox=layout.bbox(*nb), cell_index=nb)
                    )

    removed = tuple(
        sorted({_position(block) for block in placed_blocks if _position(block) in blocked})
    )
    out = [block for block in placed_blocks if _position(block) not in blocked]
    return _dedupe(out), removed


def place_structural_shells(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit boxed cell shells before connection carving or facade decoration."""
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    return _place_structural_shells(state, layout, palette)


def place_facade_overlays(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
    connection_policy: ConnectionPolicy | None = None,
) -> list[SemanticBlockDict]:
    """Emit swappable exterior wall-face overlays for exposed faces only."""
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    return _place_facade_overlays(state, layout, palette, connection_policy)


def materialise(
    state: SolverState,
    layout: SpatialLayout,
    connection_policy: ConnectionPolicy,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit world-coord blocks for the solved state.

    Stage 1 places every occupied cell as a boxed modular placeholder. Stage 2
    resolves topology-driven connections by carving openings between neighbours.
    """
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    placed_blocks = place_structural_shells(state, layout, palette)
    structural_blocks, _removed = carve_connection_openings(
        state,
        layout,
        connection_policy,
        placed_blocks,
    )
    return _dedupe(
        structural_blocks
        + place_facade_overlays(state, layout, palette, connection_policy)
    )


__all__ = [
    "carve_connection_openings",
    "materialise",
    "place_facade_overlays",
    "place_structural_shells",
]
