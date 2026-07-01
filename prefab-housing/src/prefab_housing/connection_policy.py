"""Sequential post-plan connection policy.

The topology search still chooses occupied cells and room labels. This module
then derives the opening policy afterwards from room adjacency and room type,
so shell carving and semantic door/open annotations no longer depend on raw
tile-face categories.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from prefab_housing.grid import (
    DOWN,
    EAST,
    FACE_NAMES,
    HORIZONTAL_FACES,
    NORTH,
    OPPOSITE_FACE,
    SOUTH,
    UP,
    WEST,
)
from prefab_housing.types import CellConnectionPolicy, ConnectionPolicy, FaceName

if TYPE_CHECKING:
    from prefab_housing.housing_plan import HousingPlan, HousingPlanCell


def _opening_pattern(door_faces: tuple[FaceName, ...], open_faces: tuple[FaceName, ...]) -> str:
    directional = tuple(dict.fromkeys((*door_faces, *open_faces)))
    if not directional:
        return "sealed"
    if len(directional) == 1:
        return "edge_only"
    return "multi_direction_open"


def _face_name(face: int) -> FaceName:
    return FACE_NAMES[face]  # type: ignore[return-value]


_APPEND_ONLY_ENTRY_BOUNDARY_FACE_PREFERENCE = (WEST, SOUTH, NORTH, EAST)


def _preferred_entry_boundary_face(plan: HousingPlan, boundary_faces: list[int]) -> int:
    if plan.metadata.scale_class != "append_only_residential_upgrade":
        return boundary_faces[0]
    for face in _APPEND_ONLY_ENTRY_BOUNDARY_FACE_PREFERENCE:
        if face in boundary_faces:
            return face
    return boundary_faces[0]


def _cell_lookup(plan: HousingPlan) -> dict[tuple[int, int, int], HousingPlanCell]:
    return {cell.cell_index: cell for cell in plan.cells if not cell.is_empty}


def _is_window_candidate(label: str) -> bool:
    return label in {"living", "kitchen", "bathroom", "bedroom"}


def _is_primary_connector(label: str) -> bool:
    return label in {"entry", "corridor", "stairwell"}


def _is_public_room(label: str) -> bool:
    return label in {"entry", "living", "kitchen", "corridor"}


def _preferred_connection_kind(label_a: str, label_b: str) -> str | None:
    pair = frozenset((label_a, label_b))
    if pair in ({"corridor"}, {"stairwell"}, {"corridor", "stairwell"}):
        return "open"
    if pair == {"living", "kitchen"}:
        return "open"
    if _is_primary_connector(label_a) and _is_primary_connector(label_b):
        return "open"
    if _is_primary_connector(label_a) or _is_primary_connector(label_b):
        return "door"
    return None


def _fallback_connection_score(label_a: str, label_b: str) -> int:
    if _is_primary_connector(label_b):
        return 100
    if label_a == "bathroom" and label_b in {"entry", "living", "corridor"}:
        return 85
    if label_a == "bedroom" and label_b in {"corridor", "living", "entry"}:
        return 80
    if frozenset((label_a, label_b)) == {"living", "kitchen"}:
        return 75
    if _is_public_room(label_b):
        return 60
    if label_b == "bathroom":
        return 20
    if label_b == "bedroom":
        return 10
    return 0


def _register_pair(
    mapping: dict[tuple[int, int, int], set[FaceName]],
    a: tuple[int, int, int],
    face: int,
    b: tuple[int, int, int],
) -> None:
    mapping[a].add(_face_name(face))
    mapping[b].add(_face_name(OPPOSITE_FACE[face]))


def _initialise_face_sets(
    plan: HousingPlan,
) -> tuple[
    dict[tuple[int, int, int], set[FaceName]],
    dict[tuple[int, int, int], set[FaceName]],
    dict[tuple[int, int, int], set[FaceName]],
]:
    cells = [cell for cell in plan.cells if not cell.is_empty]
    return (
        {cell.cell_index: set() for cell in cells},
        {cell.cell_index: set() for cell in cells},
        {cell.cell_index: set() for cell in cells},
    )


def _assign_internal_connections(
    plan: HousingPlan,
    cells_by_index: dict[tuple[int, int, int], HousingPlanCell],
    door_faces: dict[tuple[int, int, int], set[FaceName]],
    open_faces: dict[tuple[int, int, int], set[FaceName]],
) -> None:
    grid = plan.state.grid

    for cell in plan.cells:
        if cell.is_empty:
            continue
        ix, iy, iz = cell.cell_index
        for face in HORIZONTAL_FACES:
            neighbour = grid.neighbour(ix, iy, iz, face)
            if neighbour is None or neighbour not in cells_by_index:
                continue
            if neighbour < cell.cell_index:
                continue
            nb_cell = cells_by_index[neighbour]
            kind = _preferred_connection_kind(cell.label, nb_cell.label)
            if kind == "open":
                _register_pair(open_faces, cell.cell_index, face, neighbour)
            elif kind == "door":
                _register_pair(door_faces, cell.cell_index, face, neighbour)

    for cell in plan.cells:
        if cell.is_empty or cell.label == "entry":
            continue
        if door_faces[cell.cell_index] or open_faces[cell.cell_index]:
            continue
        ix, iy, iz = cell.cell_index
        best: tuple[int, int, tuple[int, int, int]] | None = None
        for face in HORIZONTAL_FACES:
            neighbour = grid.neighbour(ix, iy, iz, face)
            if neighbour is None or neighbour not in cells_by_index:
                continue
            nb_cell = cells_by_index[neighbour]
            candidate = (_fallback_connection_score(cell.label, nb_cell.label), face, neighbour)
            if best is None or candidate > best:
                best = candidate
        if best is not None and best[0] > 0:
            _, face, neighbour = best
            _register_pair(door_faces, cell.cell_index, face, neighbour)

    for cell in plan.cells:
        if cell.is_empty or cell.label != "stairwell":
            continue
        ix, iy, iz = cell.cell_index
        for face in (UP, DOWN):
            neighbour = grid.neighbour(ix, iy, iz, face)
            if neighbour is None or neighbour not in cells_by_index:
                continue
            if cells_by_index[neighbour].label != "stairwell":
                continue
            _register_pair(open_faces, cell.cell_index, face, neighbour)


def _allowed_faces(
    cell_index: tuple[int, int, int],
    door_faces: dict[tuple[int, int, int], set[FaceName]],
    open_faces: dict[tuple[int, int, int], set[FaceName]],
) -> frozenset[FaceName]:
    return frozenset((*door_faces[cell_index], *open_faces[cell_index]))


def _reachable_from_entries(
    plan: HousingPlan,
    cells_by_index: dict[tuple[int, int, int], HousingPlanCell],
    door_faces: dict[tuple[int, int, int], set[FaceName]],
    open_faces: dict[tuple[int, int, int], set[FaceName]],
) -> set[tuple[int, int, int]]:
    sources = [
        cell.cell_index
        for cell in cells_by_index.values()
        if cell.label == "entry"
    ]
    if not sources:
        return set()

    grid = plan.state.grid
    seen: set[tuple[int, int, int]] = set(sources)
    queue: deque[tuple[int, int, int]] = deque(sources)
    while queue:
        ix, iy, iz = queue.popleft()
        current = (ix, iy, iz)
        allowed = _allowed_faces(current, door_faces, open_faces)
        for face in (*HORIZONTAL_FACES, UP, DOWN):
            face_name = _face_name(face)
            if face_name not in allowed:
                continue
            neighbour = grid.neighbour(ix, iy, iz, face)
            if neighbour is None or neighbour not in cells_by_index:
                continue
            opposite_name = _face_name(OPPOSITE_FACE[face])
            if opposite_name not in _allowed_faces(neighbour, door_faces, open_faces):
                continue
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen


def _assign_connectivity_repairs(
    plan: HousingPlan,
    cells_by_index: dict[tuple[int, int, int], HousingPlanCell],
    door_faces: dict[tuple[int, int, int], set[FaceName]],
    open_faces: dict[tuple[int, int, int], set[FaceName]],
) -> None:
    reachable = _reachable_from_entries(plan, cells_by_index, door_faces, open_faces)
    if not reachable:
        return

    grid = plan.state.grid
    while len(reachable) < len(cells_by_index):
        best: tuple[int, int, int, tuple[int, int, int], tuple[int, int, int]] | None = None
        for cell in cells_by_index.values():
            if cell.cell_index in reachable:
                continue
            ix, iy, iz = cell.cell_index
            for face in HORIZONTAL_FACES:
                neighbour = grid.neighbour(ix, iy, iz, face)
                if neighbour is None or neighbour not in reachable:
                    continue
                nb_cell = cells_by_index[neighbour]
                score = _fallback_connection_score(cell.label, nb_cell.label)
                candidate = (score, -iy, -iz, cell.cell_index, neighbour)
                if best is None or candidate > best:
                    best = candidate

        if best is None:
            return

        _, _, _, cell_index, neighbour = best
        face = next(
            face
            for face in HORIZONTAL_FACES
            if grid.neighbour(*cell_index, face) == neighbour
        )
        cell = cells_by_index[cell_index]
        nb_cell = cells_by_index[neighbour]
        kind = _preferred_connection_kind(cell.label, nb_cell.label)
        if kind == "open":
            _register_pair(open_faces, cell_index, face, neighbour)
        else:
            _register_pair(door_faces, cell_index, face, neighbour)
        reachable = _reachable_from_entries(plan, cells_by_index, door_faces, open_faces)


def _assign_boundary_faces(
    plan: HousingPlan,
    door_faces: dict[tuple[int, int, int], set[FaceName]],
    open_faces: dict[tuple[int, int, int], set[FaceName]],
    window_faces: dict[tuple[int, int, int], set[FaceName]],
) -> None:
    grid = plan.state.grid
    for cell in plan.cells:
        if cell.is_empty:
            continue
        ix, iy, iz = cell.cell_index
        boundary_faces = [
            face
            for face in HORIZONTAL_FACES
            if grid.neighbour(ix, iy, iz, face) is None
        ]
        if cell.label == "entry" and boundary_faces:
            door_faces[cell.cell_index].add(
                _face_name(_preferred_entry_boundary_face(plan, boundary_faces))
            )
        if not _is_window_candidate(cell.label):
            continue
        for face in boundary_faces:
            name = _face_name(face)
            if name in door_faces[cell.cell_index] or name in open_faces[cell.cell_index]:
                continue
            window_faces[cell.cell_index].add(name)


def derive_connection_policy(plan: HousingPlan) -> ConnectionPolicy:
    cells_by_index = _cell_lookup(plan)
    door_faces, open_faces, window_faces = _initialise_face_sets(plan)
    _assign_internal_connections(plan, cells_by_index, door_faces, open_faces)
    _assign_connectivity_repairs(plan, cells_by_index, door_faces, open_faces)
    _assign_boundary_faces(plan, door_faces, open_faces, window_faces)

    out: list[CellConnectionPolicy] = []
    for cell in plan.cells:
        if cell.is_empty:
            continue
        doors = tuple(sorted(door_faces[cell.cell_index], key=FACE_NAMES.index))
        opens = tuple(sorted(open_faces[cell.cell_index], key=FACE_NAMES.index))
        windows = tuple(sorted(window_faces[cell.cell_index], key=FACE_NAMES.index))
        out.append(
            CellConnectionPolicy(
                cell_index=cell.cell_index,
                door_faces=doors,
                open_faces=opens,
                window_faces=windows,
                opening_pattern=_opening_pattern(doors, opens),
            )
        )
    return ConnectionPolicy(cells=tuple(out))


__all__ = ["derive_connection_policy"]
