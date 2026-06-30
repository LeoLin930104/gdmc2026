"""Programme resolver: Brief × utility_type → required-pod multiset.

Pure function. Deterministic from inputs alone (no RNG). Used by:

- The scorer (functional-adequacy hard floor).
- The MCTS prior (bias tile choice towards required pods we don't yet have).
- The grid sizer (storey count must accommodate required-pod count).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.types import Brief, UtilityType


@dataclass(frozen=True, slots=True)
class Programme:
    """Resolved spatial programme.

    ``required_pods`` is a multiset (Counter) of pod-label → minimum count.
    ``max_pods`` is a sparse cap table for pod-label → maximum count.
    ``optional_pods`` is similarly a Counter but the score does not enforce a
    hard floor on these — MCTS may include them when capacity allows.
    """

    required_pods: tuple[tuple[str, int], ...]
    max_pods: tuple[tuple[str, int], ...]
    optional_pods: tuple[tuple[str, int], ...]
    target_min_cells: int

    def required_counter(self) -> Counter[str]:
        return Counter(dict(self.required_pods))

    def optional_counter(self) -> Counter[str]:
        return Counter(dict(self.optional_pods))

    def max_counter(self) -> Counter[str]:
        return Counter(dict(self.max_pods))


@dataclass(frozen=True, slots=True)
class ProgrammeValidation:
    missing_pods: tuple[tuple[str, int], ...]
    excess_pods: tuple[tuple[str, int], ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_pods and not self.excess_pods

    @property
    def total_missing(self) -> int:
        return sum(count for _, count in self.missing_pods)

    @property
    def total_excess(self) -> int:
        return sum(count for _, count in self.excess_pods)


def validate_pod_counts(counts: Counter[str], programme: Programme) -> ProgrammeValidation:
    required = programme.required_counter()
    capped = programme.max_counter()
    missing: list[tuple[str, int]] = []
    excess: list[tuple[str, int]] = []
    for pod, need in required.items():
        deficit = need - counts.get(pod, 0)
        if deficit > 0:
            missing.append((pod, deficit))
    for pod, cap in capped.items():
        overflow = counts.get(pod, 0) - cap
        if overflow > 0:
            excess.append((pod, overflow))
    return ProgrammeValidation(
        missing_pods=tuple(sorted(missing)),
        excess_pods=tuple(sorted(excess)),
    )


def exceeds_pod_caps(counts: Counter[str], programme: Programme) -> bool:
    capped = programme.max_counter()
    for pod, cap in capped.items():
        if counts.get(pod, 0) > cap:
            return True
    return False


def resolve_programme(brief: Brief, utility_type: UtilityType) -> Programme:
    occ = brief.occupant_count

    required: Counter[str] = Counter()
    max_pods: Counter[str] = Counter()
    if utility_type == "residential":
        required[pt.POD_ENTRY] = 1
        required[pt.POD_KITCHEN] = 1
        required[pt.POD_BATHROOM] = 1
        max_pods[pt.POD_ENTRY] = 1
        max_pods[pt.POD_KITCHEN] = 1
        max_pods[pt.POD_BATHROOM] = 1
        if brief.household_type != "solo":
            required[pt.POD_LIVING] = 1
            max_pods[pt.POD_LIVING] = 1

        if brief.household_type == "solo":
            bedrooms = 1
        elif brief.household_type == "couple":
            bedrooms = 1
        elif brief.household_type == "single_family":
            bedrooms = max(1, math.ceil((occ - 2) / 2) + 1)
        elif brief.household_type == "shared":
            bedrooms = max(1, occ)
        elif brief.household_type == "multi_family":
            bedrooms = max(2, math.ceil(occ / 2))
        else:
            bedrooms = max(1, math.ceil(occ / 2))
        required[pt.POD_BEDROOM] = bedrooms
        max_pods[pt.POD_BEDROOM] = bedrooms + 2
        if occ >= 5:
            required[pt.POD_CORRIDOR] = 1

    elif utility_type == "commercial":
        required[pt.POD_ENTRY] = 1
        max_pods[pt.POD_ENTRY] = 1
        required[pt.POD_LIVING] = max(1, math.ceil(occ / 6))
        required[pt.POD_KITCHEN] = max(1, math.ceil(occ / 12))
        required[pt.POD_BATHROOM] = max(1, math.ceil(occ / 8))
        required[pt.POD_CORRIDOR] = max(1, math.ceil(occ / 8))

    elif utility_type == "service_building":
        required[pt.POD_ENTRY] = 1
        max_pods[pt.POD_ENTRY] = 1
        required[pt.POD_LIVING] = max(1, math.ceil(occ / 8))
        required[pt.POD_KITCHEN] = max(1, math.ceil(occ / 10))
        required[pt.POD_BATHROOM] = max(2, math.ceil(occ / 3))
        required[pt.POD_BEDROOM] = max(2, math.ceil(occ / 2))
        required[pt.POD_CORRIDOR] = max(1, math.ceil(occ / 10))
        required[pt.POD_STAIRWELL] = 1

    elif utility_type == "storage_utility":
        required[pt.POD_ENTRY] = 1
        max_pods[pt.POD_ENTRY] = 1
        required[pt.POD_CORRIDOR] = 1
        required[pt.POD_LIVING] = max(1, math.ceil(occ / 8))
        if occ >= 4:
            required[pt.POD_BATHROOM] = 1

    # Extra rooms requested explicitly.
    for extra in brief.required_extra_rooms:
        required[extra] = required.get(extra, 0) + 1

    if required.get(pt.POD_BEDROOM, 0) > 0:
        max_pods[pt.POD_BEDROOM] = max(
            max_pods.get(pt.POD_BEDROOM, 0),
            required[pt.POD_BEDROOM] + 2,
        )
    if required.get(pt.POD_CORRIDOR, 0) > 0:
        max_pods[pt.POD_CORRIDOR] = max(
            max_pods.get(pt.POD_CORRIDOR, 0),
            required[pt.POD_CORRIDOR] + 4,
        )

    for pod, need in required.items():
        if max_pods.get(pod, 0) < need:
            max_pods[pod] = need

    optional: Counter[str] = Counter()
    optional[pt.POD_CORRIDOR] = max(1, optional.get(pt.POD_CORRIDOR, 0))
    if brief.outdoor_living_priority > 0.4:
        # Balcony is M2; for v1 we mark it as optional but no tile exists, so it is
        # silently dropped in scoring.
        optional["balcony"] = 1

    # Stairwells are added below if multi-storey is required by capacity.
    target_min_cells = sum(required.values())

    return Programme(
        required_pods=tuple(required.items()),
        max_pods=tuple(max_pods.items()),
        optional_pods=tuple(optional.items()),
        target_min_cells=target_min_cells,
    )


def needs_multi_storey(programme: Programme, footprint_cells_xz: int) -> bool:
    """Return True iff required-pod count cannot fit in a single storey."""
    return programme.target_min_cells > footprint_cells_xz


__all__ = [
    "ProgrammeValidation",
    "Programme",
    "exceeds_pod_caps",
    "needs_multi_storey",
    "resolve_programme",
    "validate_pod_counts",
]
