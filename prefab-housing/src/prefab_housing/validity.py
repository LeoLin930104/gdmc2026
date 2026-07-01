"""Hard validity checks for solved housing plans."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import FACE_NAMES, NUM_FACES, OPPOSITE_FACE
from prefab_housing.programme import validate_pod_counts
from prefab_housing.structure import analyse_structure
from prefab_housing.types import FaceName

if TYPE_CHECKING:
    from prefab_housing.housing_plan import HousingPlan


@dataclass(frozen=True, slots=True)
class PlanValidityReport:
    """Post-search acceptance report for gameplay-facing generated buildings."""

    is_valid: bool
    errors: tuple[str, ...]
    missing_required: tuple[tuple[str, int], ...]
    excess_capped: tuple[tuple[str, int], ...]
    unreachable_cells: tuple[tuple[int, int, int], ...]
    unsupported_cells: tuple[tuple[int, int, int], ...]
    sleeping_capacity: int
    required_sleeping_capacity: int


class NoValidPlanError(RuntimeError):
    """Raised when generation cannot produce a valid plan within budget."""

    def __init__(self, message: str, report: PlanValidityReport | None = None) -> None:
        super().__init__(message)
        self.report = report


def _cell_counts(plan: HousingPlan) -> Counter[str]:
    return Counter(cell.label for cell in plan.cells if not cell.is_empty)


def _allowed_faces(plan: HousingPlan, cell_index: tuple[int, int, int]) -> frozenset[FaceName]:
    policy = plan.connection_policy.for_cell(cell_index)
    if policy is None:
        return frozenset()
    return frozenset((*policy.door_faces, *policy.open_faces))


def _reachable_cells(plan: HousingPlan) -> set[tuple[int, int, int]]:
    cells_by_index = {cell.cell_index: cell for cell in plan.cells if not cell.is_empty}
    sources = [cell.cell_index for cell in cells_by_index.values() if cell.label == pt.POD_ENTRY]
    if not sources:
        return set()

    seen: set[tuple[int, int, int]] = set(sources)
    queue: deque[tuple[int, int, int]] = deque(sources)
    while queue:
        ix, iy, iz = queue.popleft()
        allowed = _allowed_faces(plan, (ix, iy, iz))
        for face in range(NUM_FACES):
            face_name: FaceName = FACE_NAMES[face]  # type: ignore[assignment]
            if face_name not in allowed:
                continue
            neighbour = plan.state.grid.neighbour(ix, iy, iz, face)
            if neighbour is None or neighbour not in cells_by_index:
                continue
            opposite_name: FaceName = FACE_NAMES[OPPOSITE_FACE[face]]  # type: ignore[assignment]
            if opposite_name not in _allowed_faces(plan, neighbour):
                continue
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen


def validate_housing_plan(plan: HousingPlan) -> PlanValidityReport:
    """Validate the generated topology as a usable building, not just a score."""

    errors: list[str] = []
    counts = _cell_counts(plan)
    programme_validation = validate_pod_counts(counts, plan.programme)
    if programme_validation.missing_pods:
        errors.append(f"missing required rooms: {dict(programme_validation.missing_pods)}")
    if programme_validation.excess_pods:
        errors.append(f"rooms exceed caps: {dict(programme_validation.excess_pods)}")

    structure = analyse_structure(plan.state)
    unsupported = tuple(structure.unsupported_indices)
    if unsupported:
        errors.append(f"unsupported cells: {unsupported}")

    reachable = _reachable_cells(plan)
    occupied_indices = tuple(cell.cell_index for cell in plan.cells if not cell.is_empty)
    unreachable = tuple(cell_index for cell_index in occupied_indices if cell_index not in reachable)
    if unreachable:
        errors.append(f"unreachable occupied cells: {unreachable}")

    sleeping_capacity = sum(
        cell.occupancy_capacity
        for cell in plan.cells
        if not cell.is_empty and cell.label == pt.POD_BEDROOM
    )
    required_sleeping_capacity = int(plan.metadata.occupant_count)
    if sleeping_capacity < required_sleeping_capacity:
        errors.append(
            f"sleeping capacity {sleeping_capacity} below occupants {required_sleeping_capacity}"
        )

    return PlanValidityReport(
        is_valid=not errors,
        errors=tuple(errors),
        missing_required=programme_validation.missing_pods,
        excess_capped=programme_validation.excess_pods,
        unreachable_cells=unreachable,
        unsupported_cells=unsupported,
        sleeping_capacity=sleeping_capacity,
        required_sleeping_capacity=required_sleeping_capacity,
    )


def ensure_valid_housing_plan(plan: HousingPlan) -> PlanValidityReport:
    report = validate_housing_plan(plan)
    if not report.is_valid:
        raise NoValidPlanError("; ".join(report.errors), report)
    return report


__all__ = [
    "NoValidPlanError",
    "PlanValidityReport",
    "ensure_valid_housing_plan",
    "validate_housing_plan",
]
