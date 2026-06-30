"""Household-upgrade planning helpers.

This module does not mutate geometry directly. It computes how the utility
coverage should change when an outside controller upgrades a brief — for
example, a solo house being expanded toward a family programme.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from prefab_housing.housing_plan import HousingPlanTuning, HousingRequest
from prefab_housing.programme import Programme, resolve_programme
from prefab_housing.search.score import PlanFitPolicy
from prefab_housing.types import Brief, HouseholdType, UtilityType

MAX_RESIDENTIAL_LEVEL = 3


@dataclass(frozen=True, slots=True)
class HousingUpgradePlan:
    current_brief: Brief
    target_brief: Brief
    current_programme: Programme
    target_programme: Programme
    additional_required_pods: tuple[tuple[str, int], ...]
    additional_optional_pods: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ResidentialLevelSpec:
    level: int
    name: str
    occupant_count: int
    household_type: HouseholdType
    footprint_xz: tuple[int, int]
    max_storeys: int
    search_iterations: int
    tuning: HousingPlanTuning


RESIDENTIAL_LEVEL_SPECS: dict[int, ResidentialLevelSpec] = {
    1: ResidentialLevelSpec(
        level=1,
        name="basic_shelter",
        occupant_count=1,
        household_type="solo",
        footprint_xz=(20, 20),
        max_storeys=1,
        search_iterations=128,
        tuning=HousingPlanTuning(
            quirkiness=0.10,
            allow_floor_empty=False,
            min_storeys=1,
            preferred_storeys=1,
            vertical_bias=0.0,
            ground_floor_empty_factor=0.10,
            massing_slack_ratio=0.0,
            storey_bias_strength=0.15,
            fit_policy=PlanFitPolicy(
                ground_fill_min=0.85,
                ground_fill_target=1.0,
                ground_fill_max=1.0,
                storeys_min=1,
                storeys_target=1,
                storeys_max=1,
                occupied_cells_min=4,
                occupied_cells_target=4,
                occupied_cells_max=4,
            ),
        ),
    ),
    2: ResidentialLevelSpec(
        level=2,
        name="small_home",
        occupant_count=2,
        household_type="couple",
        footprint_xz=(30, 20),
        max_storeys=2,
        search_iterations=224,
        tuning=HousingPlanTuning(
            quirkiness=0.35,
            allow_floor_empty=False,
            min_storeys=2,
            preferred_storeys=1,
            vertical_bias=0.70,
            terrace_void_bias=0.70,
            ground_floor_empty_factor=0.18,
            massing_slack_ratio=0.30,
            massing_slack_cells=1,
            storey_bias_strength=0.45,
            fit_policy=PlanFitPolicy(
                ground_fill_min=0.75,
                ground_fill_target=0.90,
                ground_fill_max=1.0,
                storeys_min=2,
                storeys_target=2,
                storeys_max=2,
                occupied_cells_min=7,
                occupied_cells_target=9,
                occupied_cells_max=11,
            ),
        ),
    ),
    3: ResidentialLevelSpec(
        level=3,
        name="family_home",
        occupant_count=3,
        household_type="single_family",
        footprint_xz=(30, 30),
        max_storeys=3,
        search_iterations=320,
        tuning=HousingPlanTuning(
            quirkiness=0.45,
            allow_floor_empty=False,
            min_storeys=3,
            preferred_storeys=2,
            vertical_bias=0.80,
            terrace_void_bias=0.80,
            ground_floor_empty_factor=0.16,
            massing_slack_ratio=0.45,
            massing_slack_cells=2,
            storey_bias_strength=0.60,
            fit_policy=PlanFitPolicy(
                ground_fill_min=0.65,
                ground_fill_target=0.80,
                ground_fill_max=0.95,
                storeys_min=2,
                storeys_target=3,
                storeys_max=3,
                occupied_cells_min=10,
                occupied_cells_target=12,
                occupied_cells_max=14,
            ),
        ),
    ),
}


def _counter_diff(target: Counter[str], current: Counter[str]) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for label, need in target.items():
        delta = need - current.get(label, 0)
        if delta > 0:
            out.append((label, delta))
    return tuple(out)


def residential_level_spec(level: int) -> ResidentialLevelSpec:
    try:
        return RESIDENTIAL_LEVEL_SPECS[level]
    except KeyError as exc:
        raise ValueError(f"residential level must be in 1..{MAX_RESIDENTIAL_LEVEL}") from exc


def brief_for_residential_level(
    level: int,
    *,
    material_theme: str | None = "sci_fi_modular",
    seed: int = 0,
) -> Brief:
    spec = residential_level_spec(level)
    return Brief(
        occupant_count=spec.occupant_count,
        household_type=spec.household_type,
        max_storeys=spec.max_storeys,
        material_theme=material_theme,
        seed=seed,
    )


def request_for_residential_level(
    level: int,
    *,
    material_theme: str | None = "sci_fi_modular",
    seed: int = 0,
) -> HousingRequest:
    spec = residential_level_spec(level)
    return HousingRequest(
        footprint_xz=spec.footprint_xz,
        utility_type="residential",
        capacity_override=spec.occupant_count,
        max_storeys=spec.max_storeys,
        material_theme=material_theme,
        seed=seed,
    )


def plan_house_upgrade(
    current_brief: Brief,
    *,
    target_occupant_count: int,
    target_household_type: HouseholdType | None = None,
    utility_type: UtilityType = "residential",
) -> HousingUpgradePlan:
    """Compute the utility delta between the current and target households."""
    target_brief = Brief(
        occupant_count=target_occupant_count,
        household_type=target_household_type or current_brief.household_type,
        outdoor_living_priority=current_brief.outdoor_living_priority,
        max_storeys=current_brief.max_storeys,
        material_theme=current_brief.material_theme,
        seed=current_brief.seed,
        required_extra_rooms=current_brief.required_extra_rooms,
    )
    current_programme = resolve_programme(current_brief, utility_type)
    target_programme = resolve_programme(target_brief, utility_type)
    return HousingUpgradePlan(
        current_brief=current_brief,
        target_brief=target_brief,
        current_programme=current_programme,
        target_programme=target_programme,
        additional_required_pods=_counter_diff(
            target_programme.required_counter(),
            current_programme.required_counter(),
        ),
        additional_optional_pods=_counter_diff(
            target_programme.optional_counter(),
            current_programme.optional_counter(),
        ),
    )


__all__ = [
    "HousingUpgradePlan",
    "MAX_RESIDENTIAL_LEVEL",
    "RESIDENTIAL_LEVEL_SPECS",
    "ResidentialLevelSpec",
    "brief_for_residential_level",
    "plan_house_upgrade",
    "request_for_residential_level",
    "residential_level_spec",
]
