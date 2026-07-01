"""Housing-plan stage: request -> solved 3D utility-marked cell topology.

This module extracts the planning/search segment from ``build_house`` so the
topology can be iterated independently from facade, roof, and interior work.
The output ``HousingPlan`` intentionally stops at square-cell occupancy and
utility assignment; downstream surface systems may re-render the same plan many
times without re-running WFC/MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
import time

from prefab_housing.connection_policy import derive_connection_policy
from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import CellGrid, design_grid
from prefab_housing.layout import uniform_layout
from prefab_housing.programme import Programme, resolve_programme
from prefab_housing.search.mcts import MCTSConfig, mcts_search
from prefab_housing.search.priors import apply_position_priors
from prefab_housing.search.score import PlanFitPolicy, ScoreWeights
from prefab_housing.types import Brief, CellRole, ConnectionPolicy, SemanticBlockDict, UtilityType
from prefab_housing.validity import NoValidPlanError, PlanValidityReport, validate_housing_plan
from prefab_housing.wfc.solver import SolverState, init_state, is_solved
from prefab_housing.wfc.tiles import build_tile_set

logger = logging.getLogger(__name__)
_MAX_UNSUPPORTED_RETRIES = 3
_MAX_VALIDITY_ITERATION_MULTIPLIER = 4
_MAX_VALIDITY_ITERATIONS = 2048


DEFAULT_CELL_VOXEL_SIZE: tuple[int, int, int] = (10, 6, 10)
DEFAULT_MAX_STOREYS: int = 4

# Topology-only preview uses cubic cells so the planning loop ignores facade
# aspect ratio and focuses purely on massing / occupancy / utility spread.
DEFAULT_PLAN_PREVIEW_CELL_VOXEL_SIZE: tuple[int, int, int] = (6, 6, 6)


@dataclass(frozen=True, slots=True)
class HousingPlanTuning:
    """Search-shape controls for the planning stage.

    ``quirkiness`` is the high-level dial exposed to the outside controller.
    Under the current system it mainly drives EMPTY-perimeter pressure: lower
    values produce more grounded box-like massing; higher values promote more
    stepped / carved silhouettes while structural viability remains constrained
    by the compatibility table and the scorer.
    """

    quirkiness: float = 0.5
    expansion_prior_strength: float = 2.0
    rollout_prior_strength: float = 4.0
    empty_perimeter_strength: float | None = None
    ground_floor_empty_factor: float | None = None
    allow_floor_empty: bool = False
    min_storeys: int | None = None
    preferred_storeys: int | None = None
    vertical_bias: float = 0.5
    terrace_void_bias: float = 0.6
    massing_slack_ratio: float = 0.0
    massing_slack_cells: int = 0
    storey_bias_strength: float = 0.0
    fit_policy: PlanFitPolicy = field(default_factory=PlanFitPolicy)

    def __post_init__(self) -> None:
        if not 0.0 <= self.quirkiness <= 1.0:
            raise ValueError("quirkiness must be in [0, 1]")
        if self.expansion_prior_strength <= 0.0:
            raise ValueError("expansion_prior_strength must be > 0")
        if self.rollout_prior_strength <= 0.0:
            raise ValueError("rollout_prior_strength must be > 0")
        if (
            self.empty_perimeter_strength is not None
            and self.empty_perimeter_strength < 0.0
        ):
            raise ValueError("empty_perimeter_strength must be >= 0 when provided")
        if (
            self.ground_floor_empty_factor is not None
            and not 0.0 <= self.ground_floor_empty_factor <= 1.0
        ):
            raise ValueError("ground_floor_empty_factor must be in [0, 1] when provided")
        if self.min_storeys is not None and self.min_storeys < 1:
            raise ValueError("min_storeys must be >= 1 when provided")
        if self.preferred_storeys is not None and self.preferred_storeys < 1:
            raise ValueError("preferred_storeys must be >= 1 when provided")
        if not 0.0 <= self.vertical_bias <= 1.0:
            raise ValueError("vertical_bias must be in [0, 1]")
        if not 0.0 <= self.terrace_void_bias <= 1.0:
            raise ValueError("terrace_void_bias must be in [0, 1]")
        if self.massing_slack_ratio < 0.0:
            raise ValueError("massing_slack_ratio must be >= 0")
        if self.massing_slack_cells < 0:
            raise ValueError("massing_slack_cells must be >= 0")
        if not 0.0 <= self.storey_bias_strength <= 1.0:
            raise ValueError("storey_bias_strength must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class HousingPlanCell:
    cell_index: tuple[int, int, int]
    label: str
    role: CellRole
    tile_id: int
    tile_label: str
    rotation_quarters: int
    occupancy_capacity: int
    is_empty: bool


@dataclass(frozen=True, slots=True)
class HousingPlanMetadata:
    seed: int
    utility_type: UtilityType
    occupant_count: int
    scale_class: str
    site_footprint_xz: tuple[int, int]
    cell_grid_size: tuple[int, int, int]
    cell_voxel_size: tuple[int, int, int]
    score_total: float
    score_breakdown: dict[str, float]
    stage_timings_ms: dict[str, float]
    rollouts: int
    tuning: HousingPlanTuning
    storey_distribution: "StoreyDistributionPlan"
    massing_profile: "MassingProfile"


@dataclass(frozen=True, slots=True)
class HousingPlanProfile:
    name: str
    footprint_xz: tuple[int, int]
    capacity_override: int | None
    max_storeys: int
    search_iterations: int
    utility_type: UtilityType
    tuning: HousingPlanTuning


@dataclass(frozen=True, slots=True)
class HousingRequest:
    footprint_xz: tuple[int, int]
    utility_type: UtilityType = "residential"
    capacity_override: int | None = None
    max_storeys: int | None = None
    material_theme: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.footprint_xz[0] < 1 or self.footprint_xz[1] < 1:
            raise ValueError("footprint_xz must be positive")
        if self.capacity_override is not None and self.capacity_override < 1:
            raise ValueError("capacity_override must be >= 1 when provided")
        if self.max_storeys is not None and self.max_storeys < 1:
            raise ValueError("max_storeys must be >= 1 when provided")


@dataclass(frozen=True, slots=True)
class UtilitySizingPolicy:
    min_occupants: int
    max_occupants: int
    cells_per_occupant_limit: int


@dataclass(frozen=True, slots=True)
class StoreyDistributionPlan:
    min_storeys: int
    target_storeys: int
    public_storey_max: int
    private_storey_min: int


@dataclass(frozen=True, slots=True)
class MassingProfile:
    terrace_start_storey: int
    terrace_axis: str
    terrace_direction: int
    asymmetry_strength: float
    terrace_void_strength: float
    preferred_storeys: int = 1
    occupancy_storey_bias: tuple[float, ...] = ()
    void_storey_bias: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningStageResult:
    programme: Programme
    grid: CellGrid
    scale_class: str
    storey_distribution: StoreyDistributionPlan
    massing_profile: MassingProfile
    tower_core_cell: tuple[int, int] | None
    timings_ms: dict[str, float]


PLAN_PROFILES: dict[str, HousingPlanProfile] = {
    "small_house": HousingPlanProfile(
        name="small_house",
        footprint_xz=(20, 20),
        capacity_override=2,
        max_storeys=2,
        search_iterations=96,
        utility_type="residential",
        tuning=HousingPlanTuning(
            quirkiness=0.20,
            allow_floor_empty=False,
            min_storeys=2,
            preferred_storeys=1,
            vertical_bias=0.65,
            terrace_void_bias=0.65,
            massing_slack_ratio=0.25,
            massing_slack_cells=1,
            storey_bias_strength=0.40,
        ),
    ),
    "townhouse": HousingPlanProfile(
        name="townhouse",
        footprint_xz=(30, 20),
        capacity_override=3,
        max_storeys=2,
        search_iterations=128,
        utility_type="residential",
        tuning=HousingPlanTuning(
            quirkiness=0.35,
            allow_floor_empty=False,
            min_storeys=2,
            preferred_storeys=1,
            vertical_bias=0.60,
            terrace_void_bias=0.70,
            massing_slack_ratio=0.30,
            massing_slack_cells=1,
            storey_bias_strength=0.45,
        ),
    ),
    "courtyard_family": HousingPlanProfile(
        name="courtyard_family",
        footprint_xz=(30, 30),
        capacity_override=4,
        max_storeys=2,
        search_iterations=160,
        utility_type="residential",
        tuning=HousingPlanTuning(
            quirkiness=0.45,
            allow_floor_empty=False,
            min_storeys=2,
            preferred_storeys=2,
            vertical_bias=0.45,
            terrace_void_bias=0.75,
            massing_slack_ratio=0.35,
            massing_slack_cells=1,
            storey_bias_strength=0.35,
        ),
    ),
    "quirky_stack": HousingPlanProfile(
        name="quirky_stack",
        footprint_xz=(30, 30),
        capacity_override=3,
        max_storeys=3,
        search_iterations=160,
        utility_type="residential",
        tuning=HousingPlanTuning(
            quirkiness=0.70,
            allow_floor_empty=True,
            min_storeys=3,
            preferred_storeys=2,
            vertical_bias=0.80,
            terrace_void_bias=0.85,
            massing_slack_ratio=0.45,
            massing_slack_cells=1,
            storey_bias_strength=0.65,
        ),
    ),
    "grand_mansion": HousingPlanProfile(
        name="grand_mansion",
        footprint_xz=(40, 40),
        capacity_override=6,
        max_storeys=3,
        search_iterations=192,
        utility_type="residential",
        tuning=HousingPlanTuning(
            quirkiness=0.45,
            allow_floor_empty=False,
            min_storeys=3,
            preferred_storeys=2,
            vertical_bias=0.35,
            terrace_void_bias=0.75,
            massing_slack_ratio=0.30,
            massing_slack_cells=2,
            storey_bias_strength=0.45,
        ),
    ),
    "sky_scraper": HousingPlanProfile(
        name="sky_scraper",
        footprint_xz=(40, 40),
        capacity_override=12,
        max_storeys=6,
        search_iterations=224,
        utility_type="service_building",
        tuning=HousingPlanTuning(quirkiness=0.70, allow_floor_empty=True, min_storeys=4, vertical_bias=1.0),
    ),
}


UTILITY_SIZING_POLICIES: dict[UtilityType, UtilitySizingPolicy] = {
    "residential": UtilitySizingPolicy(
        min_occupants=1,
        max_occupants=8,
        cells_per_occupant_limit=2,
    ),
    "commercial": UtilitySizingPolicy(
        min_occupants=2,
        max_occupants=12,
        cells_per_occupant_limit=2,
    ),
    "service_building": UtilitySizingPolicy(
        min_occupants=4,
        max_occupants=24,
        cells_per_occupant_limit=1,
    ),
    "storage_utility": UtilitySizingPolicy(
        min_occupants=1,
        max_occupants=6,
        cells_per_occupant_limit=3,
    ),
}


@dataclass(slots=True)
class HousingPlan:
    state: SolverState
    programme: Programme
    cells: tuple[HousingPlanCell, ...]
    metadata: HousingPlanMetadata
    connection_policy: ConnectionPolicy


_PLAN_PREVIEW_BLOCKS: dict[str, str] = {
    pt.POD_ENTRY: "minecraft:orange_concrete",
    pt.POD_LIVING: "minecraft:red_concrete",
    pt.POD_KITCHEN: "minecraft:yellow_concrete",
    pt.POD_BATHROOM: "minecraft:light_blue_concrete",
    pt.POD_BEDROOM: "minecraft:purple_concrete",
    pt.POD_CORRIDOR: "minecraft:gray_concrete",
    pt.POD_STAIRWELL: "minecraft:lime_concrete",
    pt.POD_STRUCTURAL_VOID: "minecraft:black_concrete",
    pt.POD_TERRACE_VOID: "minecraft:white_concrete",
}


def _resolved_empty_perimeter_strength(tuning: HousingPlanTuning) -> float:
    if tuning.empty_perimeter_strength is not None:
        return tuning.empty_perimeter_strength
    # q=0.5 reproduces the current default (10.0).
    return 2.0 + 16.0 * tuning.quirkiness


def _resolved_ground_floor_empty_factor(tuning: HousingPlanTuning) -> float:
    if tuning.ground_floor_empty_factor is not None:
        return tuning.ground_floor_empty_factor
    # q=0.5 reproduces the current default (0.25).
    return 0.05 + 0.40 * tuning.quirkiness


def _resolved_terrace_void_bias(tuning: HousingPlanTuning) -> float:
    return 0.20 + 0.60 * tuning.terrace_void_bias


def _ease_out_quadratic(value: float) -> float:
    clamped = min(1.0, max(0.0, value))
    return 1.0 - (1.0 - clamped) * (1.0 - clamped)


def _infer_scale_class(
    utility_type: UtilityType,
    occupant_count: int,
    cap: CellGrid,
) -> str:
    footprint_cells = cap.cx * cap.cz
    if utility_type == "residential":
        if cap.cy >= 5 or occupant_count >= 9:
            return "vertical"
        if footprint_cells <= 4 and occupant_count <= 2:
            return "compact"
        if cap.cy >= 3 or occupant_count >= 6:
            return "stacked"
        return "family"
    if utility_type == "commercial":
        return "compact" if footprint_cells <= 6 else "frontage" if cap.cy <= 2 else "stacked"
    if utility_type == "service_building":
        return "civic" if cap.cy <= 3 else "campus"
    return "compact" if footprint_cells <= 6 else "industrial"


def _plan_storey_distribution(
    programme: Programme,
    *,
    occupant_count: int,
    footprint_cells_xz: int,
    utility_type: UtilityType,
    max_storeys: int,
    tuning: HousingPlanTuning,
) -> StoreyDistributionPlan:
    utility_min = 1 if programme.target_min_cells <= 4 else 2 if programme.target_min_cells <= 12 else 3 if programme.target_min_cells <= 24 else 4
    if utility_type == "residential":
        utility_min = max(2, utility_min)
    requested_min = tuning.min_storeys or 1
    min_storeys = min(max_storeys, max(requested_min, utility_min))
    if utility_type == "residential":
        target_storeys = min_storeys
        # Larger family housing was underweighted and collapsed into shallow
        # boxes. Force a 3-storey baseline when family load or programme spill
        # suggests stacking, then only use tuning to decide the remaining lift.
        if max_storeys >= 3 and (
            occupant_count >= 6 or programme.target_min_cells > footprint_cells_xz
        ):
            target_storeys = max(target_storeys, 3)
        remaining_storeys = max(0, max_storeys - target_storeys)
        residential_lift = 0.75 * tuning.vertical_bias
        if programme.target_min_cells > footprint_cells_xz:
            overflow = programme.target_min_cells - footprint_cells_xz
            residential_lift += 0.25 * min(1.0, overflow / max(1, footprint_cells_xz))
        if occupant_count <= 4 and programme.target_min_cells <= footprint_cells_xz:
            residential_lift *= 0.35
        target_storeys = min(
            max_storeys,
            target_storeys + round(_ease_out_quadratic(residential_lift) * remaining_storeys),
        )
    else:
        capacity_pressure = min(1.0, max(0.0, (programme.target_min_cells - 8) / 10.0))
        vertical_preference = tuning.vertical_bias + 0.15 * capacity_pressure
        eased_preference = _ease_out_quadratic(vertical_preference)
        extra_storeys = max(0, max_storeys - min_storeys)
        target_storeys = min(
            max_storeys,
            min_storeys + round(eased_preference * extra_storeys),
        )
    if utility_type == "residential":
        public_storey_max = 0
        private_storey_min = 1 if target_storeys >= 2 else 0
    elif utility_type == "service_building":
        public_storey_max = min(target_storeys - 1, 1) if target_storeys >= 2 else 0
        private_storey_min = 1 if target_storeys >= 2 else 0
    elif utility_type == "commercial":
        public_storey_max = min(target_storeys - 1, 1) if target_storeys >= 2 else 0
        private_storey_min = 0
    else:
        public_storey_max = 0
        private_storey_min = 0
    return StoreyDistributionPlan(
        min_storeys=min_storeys,
        target_storeys=target_storeys,
        public_storey_max=public_storey_max,
        private_storey_min=private_storey_min,
    )


def _with_vertical_circulation(programme: Programme, target_storeys: int) -> Programme:
    """Require one stairwell cell per occupied storey for multi-storey plans."""
    required = programme.required_counter()
    max_pods = programme.max_counter()
    optional = programme.optional_counter()
    if target_storeys < 2:
        required_stairwells = required.get(pt.POD_STAIRWELL, 0)
        max_pods[pt.POD_STAIRWELL] = max(max_pods.get(pt.POD_STAIRWELL, 0), required_stairwells)
        return Programme(
            required_pods=tuple(required.items()),
            max_pods=tuple(max_pods.items()),
            optional_pods=tuple(optional.items()),
            target_min_cells=sum(required.values()),
        )
    stairwells = max(target_storeys, required.get(pt.POD_STAIRWELL, 0))
    required[pt.POD_STAIRWELL] = stairwells
    max_pods[pt.POD_STAIRWELL] = max(max_pods.get(pt.POD_STAIRWELL, 0), stairwells)
    return Programme(
        required_pods=tuple(required.items()),
        max_pods=tuple(max_pods.items()),
        optional_pods=tuple(optional.items()),
        target_min_cells=sum(required.values()),
    )


def _plan_massing_profile(
    *,
    utility_type: UtilityType,
    grid: CellGrid,
    storey_distribution: StoreyDistributionPlan,
    tuning: HousingPlanTuning,
    seed: int,
) -> MassingProfile:
    terrace_start_storey = grid.cy
    if utility_type == "residential" and grid.cy >= 2:
        terrace_start_storey = max(1, storey_distribution.private_storey_min)
    elif utility_type in ("commercial", "service_building") and grid.cy >= 3:
        terrace_start_storey = max(1, grid.cy - 2)

    terrace_axis = "x" if grid.cx >= grid.cz else "z"
    terrace_direction = 1 if ((seed + grid.cx + grid.cz + grid.cy) % 2 == 0) else -1
    asymmetry_strength = 0.25 + 0.75 * tuning.quirkiness
    if utility_type == "residential":
        asymmetry_strength += 0.15
    terrace_void_strength = _resolved_terrace_void_bias(tuning)
    if utility_type == "residential" and grid.cy >= 2:
        terrace_void_strength *= 1.25
    preferred_storeys = min(
        grid.cy,
        max(1, tuning.preferred_storeys or storey_distribution.target_storeys),
    )
    occupancy_storey_bias, void_storey_bias = _storey_bias_curves(
        grid=grid,
        preferred_storeys=preferred_storeys,
        tuning=tuning,
    )
    return MassingProfile(
        terrace_start_storey=min(grid.cy, terrace_start_storey),
        terrace_axis=terrace_axis,
        terrace_direction=terrace_direction,
        asymmetry_strength=min(1.5, asymmetry_strength),
        terrace_void_strength=terrace_void_strength,
        preferred_storeys=preferred_storeys,
        occupancy_storey_bias=occupancy_storey_bias,
        void_storey_bias=void_storey_bias,
    )


def _plan_tower_core_cell(grid: CellGrid) -> tuple[int, int] | None:
    if grid.cy < 4 or grid.cx < 3 or grid.cz < 3:
        return None
    return (grid.cx // 2, grid.cz // 2)


def _storey_bias_curves(
    *,
    grid: CellGrid,
    preferred_storeys: int,
    tuning: HousingPlanTuning,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return occupancy/void multipliers per storey.

    Positive occupancy bias applies through ``preferred_storeys``. Above that
    band, occupancy is discouraged and void/terrace choices are favoured. This
    gives the search a soft taper instead of a hard no-build rule.
    """
    strength = tuning.storey_bias_strength
    if strength <= 0.0 or grid.cy <= 0:
        return (), ()
    preferred_top = min(grid.cy - 1, max(0, preferred_storeys - 1))
    occupancy: list[float] = []
    void: list[float] = []
    for iy in range(grid.cy):
        if iy <= preferred_top:
            t = iy / max(1, preferred_top)
            occupancy.append(1.0 + strength * (0.10 + 0.25 * t))
            void.append(max(0.20, 1.0 - strength * (0.25 + 0.20 * t)))
            continue
        over_t = (iy - preferred_top) / max(1, grid.cy - 1 - preferred_top)
        occupancy.append(max(0.25, 1.0 - strength * (0.65 + 0.25 * over_t)))
        void.append(1.0 + strength * (0.85 + 0.65 * over_t))
    return tuple(occupancy), tuple(void)


def _band_error_float(
    value: float,
    *,
    minimum: float | None,
    target: float | None,
    maximum: float | None,
) -> float:
    if minimum is None and target is None and maximum is None:
        return 0.0
    if minimum is None:
        minimum = target if target is not None else 0.0
    if maximum is None:
        maximum = target if target is not None else 1.0
    if target is None:
        target = (minimum + maximum) * 0.5
    if minimum <= value <= maximum:
        return 0.05 * abs(value - target)
    if value < minimum:
        return 1.0 + minimum - value
    return 1.0 + value - maximum


def _planning_capacity_budget(
    programme: Programme,
    target_storeys: int,
    tuning: HousingPlanTuning,
) -> int:
    """Estimate useful planning capacity from utility load.

    The footprint is only an upper bound; however exact-fit grids produce low-
    yield plans with no space for stairs/circulation and tend to overfit the
    solver into degenerate needles. We therefore budget a small amount of slack
    proportional to programme size and vertical complexity.
    """
    base = int(programme.target_min_cells)
    overhead = 0
    optional = programme.optional_counter()
    required = programme.required_counter()
    if target_storeys > 1 and required.get(pt.POD_STAIRWELL, 0) < target_storeys:
        overhead += 1  # stair / vertical circulation budget
    if optional.get(pt.POD_CORRIDOR, 0) > 0 and base >= 8:
        overhead += 1
    overhead += max(0, base // 8)
    slack = math.ceil(base * tuning.massing_slack_ratio) + tuning.massing_slack_cells
    return base + overhead + slack


def _household_type_for_capacity(utility_type: UtilityType, occupant_count: int) -> str:
    if utility_type in ("commercial", "service_building", "storage_utility"):
        return "shared" if occupant_count <= 4 else "multi_family"
    if occupant_count <= 1:
        return "solo"
    if occupant_count <= 2:
        return "couple"
    if occupant_count <= 6:
        return "single_family"
    if occupant_count <= 8:
        return "shared"
    return "multi_family"


def _select_planning_grid(
    *,
    footprint_xz: tuple[int, int],
    max_storeys: int,
    cell_voxel_size: tuple[int, int, int],
    programme: Programme,
    storey_distribution: StoreyDistributionPlan,
    tuning: HousingPlanTuning,
) -> CellGrid | None:
    cap = design_grid(
        footprint_xz=footprint_xz,
        max_storeys=max_storeys,
        cell_voxel_size=cell_voxel_size,
    )
    min_storeys = min(cap.cy, storey_distribution.min_storeys)
    target_storeys = min(cap.cy, storey_distribution.target_storeys)
    if programme.target_min_cells > cap.cells_total:
        return None
    target_capacity = min(
        cap.cells_total,
        _planning_capacity_budget(programme, target_storeys, tuning),
    )
    fit_policy = tuning.fit_policy
    footprint_ratio = cap.cx / max(1, cap.cz)
    min_span = 1 if target_capacity <= 4 else 2 if target_capacity <= 18 else 3
    best: tuple[tuple[float, ...], CellGrid] | None = None
    for cy in range(1, cap.cy + 1):
        for cx in range(1, cap.cx + 1):
            for cz in range(1, cap.cz + 1):
                capacity = cx * cy * cz
                if capacity < target_capacity:
                    continue
                if min(cx, cz) < min_span and cap.cx >= min_span and cap.cz >= min_span:
                    continue
                below_min = max(0, min_storeys - cy)
                vertical_shortfall = max(0, target_storeys - cy)
                vertical_overshoot = max(0, cy - target_storeys)
                excess = capacity - target_capacity
                footprint_area = cx * cz
                footprint_per_storey = footprint_area / max(1, cy)
                ground_fill_ratio = footprint_area / max(1, cap.cx * cap.cz)
                fill_error = _band_error_float(
                    ground_fill_ratio,
                    minimum=fit_policy.ground_fill_min,
                    target=fit_policy.ground_fill_target,
                    maximum=fit_policy.ground_fill_max,
                )
                aspect_error = abs((cx / max(1, cz)) - footprint_ratio)
                thinness = max(0, 2 - min(cx, cz))
                key = (
                    float(below_min),
                    float(vertical_shortfall),
                    float(thinness),
                    float(vertical_overshoot),
                    float(aspect_error),
                    float(fill_error),
                    float(abs(cap.cy - cy)) if tuning.storey_bias_strength > 0.0 else 0.0,
                    float(footprint_per_storey),
                    float(abs(cx - cz)),
                    float(footprint_area),
                    float(excess),
                )
                candidate = CellGrid(cx=cx, cy=cy, cz=cz)
                if best is None or key < best[0]:
                    best = (key, candidate)
    if best is None:
        return None
    return best[1]


def _resolve_planning_stages(
    brief: Brief,
    *,
    footprint_xz: tuple[int, int],
    utility_type: UtilityType,
    cell_voxel_size: tuple[int, int, int],
    tuning: HousingPlanTuning,
) -> PlanningStageResult:
    timings_ms: dict[str, float] = {}
    max_storeys = brief.max_storeys or DEFAULT_MAX_STOREYS
    cap = design_grid(
        footprint_xz=footprint_xz,
        max_storeys=max_storeys,
        cell_voxel_size=cell_voxel_size,
    )

    t0 = time.perf_counter()
    programme = resolve_programme(brief, utility_type)
    timings_ms["plan_programme_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scale_class = _infer_scale_class(utility_type, brief.occupant_count, cap)
    timings_ms["plan_scale_class_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    storey_distribution = _plan_storey_distribution(
        programme,
        occupant_count=brief.occupant_count,
        footprint_cells_xz=cap.cx * cap.cz,
        utility_type=utility_type,
        max_storeys=cap.cy,
        tuning=tuning,
    )
    timings_ms["plan_storey_distribution_ms"] = (time.perf_counter() - t0) * 1000.0
    programme = _with_vertical_circulation(programme, storey_distribution.target_storeys)

    t0 = time.perf_counter()
    grid = _select_planning_grid(
        footprint_xz=footprint_xz,
        max_storeys=max_storeys,
        cell_voxel_size=cell_voxel_size,
        programme=programme,
        storey_distribution=storey_distribution,
        tuning=tuning,
    )
    timings_ms["plan_grid_ms"] = (time.perf_counter() - t0) * 1000.0
    if grid is None:
        raise ValueError("No viable planning grid fits the requested programme")

    t0 = time.perf_counter()
    massing_profile = _plan_massing_profile(
        utility_type=utility_type,
        grid=grid,
        storey_distribution=storey_distribution,
        tuning=tuning,
        seed=brief.seed,
    )
    timings_ms["plan_massing_ms"] = (time.perf_counter() - t0) * 1000.0
    tower_core_cell = _plan_tower_core_cell(grid)
    return PlanningStageResult(
        programme=programme,
        grid=grid,
        scale_class=scale_class,
        storey_distribution=storey_distribution,
        massing_profile=massing_profile,
        tower_core_cell=tower_core_cell,
        timings_ms=timings_ms,
    )


def resolve_brief_for_request(
    request: HousingRequest,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    tuning: HousingPlanTuning | None = None,
) -> Brief:
    """Infer residential capacity from footprint cap and utility type.

    The site footprint is an upper bound, not a fill target. We search for the
    largest occupant load whose programme can fit a realistic planning grid
    within the cap. If even the smallest viable dwelling does not fit, we fail.
    """
    tuning = tuning or HousingPlanTuning()
    max_storeys = request.max_storeys or DEFAULT_MAX_STOREYS
    cap = design_grid(
        footprint_xz=request.footprint_xz,
        max_storeys=max_storeys,
        cell_voxel_size=cell_voxel_size,
    )
    policy = UTILITY_SIZING_POLICIES[request.utility_type]
    if request.capacity_override is not None:
        occupant_candidates = (request.capacity_override,)
    else:
        upper_occ = min(
            policy.max_occupants,
            max(policy.min_occupants, cap.cells_total // policy.cells_per_occupant_limit),
        )
        if request.utility_type == "residential":
            # Treat compact residential sites as normal dwellings, not maximum-load
            # shared housing. This reduces programme bloat such as duplicate
            # kitchens/bathrooms on small footprints and makes valid solves more
            # probable within modest search budgets.
            footprint_cells = cap.cx * cap.cz
            if footprint_cells <= 9:
                upper_occ = min(upper_occ, footprint_cells)
            if footprint_cells <= 12:
                upper_occ = min(upper_occ, max(3, footprint_cells // 2 + 1))
        occupant_candidates = tuple(range(upper_occ, policy.min_occupants - 1, -1))
    for occupant_count in occupant_candidates:
        household_type = _household_type_for_capacity(request.utility_type, occupant_count)
        brief = Brief(
            occupant_count=occupant_count,
            household_type=household_type,  # type: ignore[arg-type]
            max_storeys=request.max_storeys,
            material_theme=request.material_theme,
            seed=request.seed,
        )
        try:
            _resolve_planning_stages(
                brief,
                footprint_xz=request.footprint_xz,
                utility_type=request.utility_type,
                cell_voxel_size=cell_voxel_size,
                tuning=tuning,
            )
            return brief
        except ValueError:
            continue
    if request.capacity_override is not None:
        raise ValueError(
            "Exact capacity_override does not fit within the requested footprint and storey cap"
        )
    raise ValueError("No viable house fits within the requested footprint and storey cap")


def _plan_mcts_config(
    *,
    iterations: int,
    seed: int,
    tuning: HousingPlanTuning,
    massing_profile: MassingProfile,
) -> MCTSConfig:
    return MCTSConfig(
        iterations=iterations,
        rng_seed=seed,
        expansion_prior_strength=tuning.expansion_prior_strength,
        rollout_prior_strength=tuning.rollout_prior_strength,
        empty_perimeter_strength=_resolved_empty_perimeter_strength(tuning),
        ground_floor_empty_factor=_resolved_ground_floor_empty_factor(tuning),
        terrace_void_strength=massing_profile.terrace_void_strength,
        terrace_start_storey=massing_profile.terrace_start_storey,
        terrace_axis=massing_profile.terrace_axis,
        terrace_direction=massing_profile.terrace_direction,
        terrace_asymmetry_strength=massing_profile.asymmetry_strength,
        occupancy_storey_bias=massing_profile.occupancy_storey_bias,
        void_storey_bias=massing_profile.void_storey_bias,
    )


def _effective_allow_floor_empty(
    tuning: HousingPlanTuning,
    planning: PlanningStageResult,
) -> bool:
    # Tall core-driven towers need an unbroken vertical load path. Allowing
    # FLOOR↔EMPTY at these heights expands search into branches that are either
    # structurally impossible or so low-yield that MCTS exhausts its budget.
    if planning.tower_core_cell is not None:
        return False
    return tuning.allow_floor_empty


def _log_stage_timings(context: str, timings_ms: dict[str, float]) -> None:
    if not logger.isEnabledFor(logging.INFO) or not timings_ms:
        return
    summary = ", ".join(
        f"{name}={value:.2f}"
        for name, value in sorted(timings_ms.items(), key=lambda item: item[1], reverse=True)
    )
    logger.info("%s timings_ms: %s", context, summary)


def _summarise_cells(state: SolverState) -> tuple[HousingPlanCell, ...]:
    tiles = state.tiles
    out: list[HousingPlanCell] = []
    for flat, tid_raw in enumerate(state.assignment.tolist()):
        tid = int(tid_raw)
        if tid < 0:
            continue
        ix, iy, iz = state.grid.from_flat(flat)
        pod_idx = int(tiles.pod_index[tid])
        label = pt.POD_LABELS[pod_idx]
        out.append(
            HousingPlanCell(
                cell_index=(ix, iy, iz),
                label=label,
                role=pt.POD_ROLE[pod_idx],  # type: ignore[arg-type]
                tile_id=tid,
                tile_label=tiles.tile_label[tid],
                rotation_quarters=int(tiles.rotation[tid]),
                occupancy_capacity=int(tiles.occupancy[tid]),
                is_empty=pt.is_void_pod_index(pod_idx),
            )
        )
    return tuple(out)


def _validity_iteration_schedule(search_iterations: int) -> tuple[int, ...]:
    base = max(1, int(search_iterations))
    cap = min(_MAX_VALIDITY_ITERATIONS, max(base, base * _MAX_VALIDITY_ITERATION_MULTIPLIER))
    values: list[int] = []
    current = base
    while current < cap:
        values.append(current)
        current *= 2
    values.append(cap)
    return tuple(dict.fromkeys(values))


def _score_weights_for_site(
    score_weights: ScoreWeights | None,
    *,
    tuning: HousingPlanTuning,
    footprint_xz: tuple[int, int],
    max_storeys: int,
    cell_voxel_size: tuple[int, int, int],
) -> ScoreWeights:
    if score_weights is not None:
        return score_weights
    site_grid = design_grid(
        footprint_xz=footprint_xz,
        max_storeys=max_storeys,
        cell_voxel_size=cell_voxel_size,
    )
    fit_policy = tuning.fit_policy
    if fit_policy.site_footprint_cells is None:
        fit_policy = replace(fit_policy, site_footprint_cells=site_grid.cx * site_grid.cz)
    return ScoreWeights(fit_policy=fit_policy)


def generate_housing_plan(
    brief: Brief,
    *,
    footprint_xz: tuple[int, int],
    utility_type: UtilityType = "residential",
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    search_iterations: int = 256,
    score_weights: ScoreWeights | None = None,
    tuning: HousingPlanTuning | None = None,
) -> HousingPlan:
    """Resolve a request into a solved, utility-marked 3D cell plan."""
    timings: dict[str, float] = {}
    tuning = tuning or HousingPlanTuning()

    planning = _resolve_planning_stages(
        brief,
        footprint_xz=footprint_xz,
        utility_type=utility_type,
        cell_voxel_size=cell_voxel_size,
        tuning=tuning,
    )
    timings.update(planning.timings_ms)

    grid = planning.grid
    t0 = time.perf_counter()
    tiles = build_tile_set(allow_floor_empty=_effective_allow_floor_empty(tuning, planning))
    timings["tiles_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    state = init_state(grid, tiles)
    timings["init_state_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    apply_position_priors(
        state,
        planning.programme,
        utility_type=utility_type,
        public_storey_max=planning.storey_distribution.public_storey_max,
        private_storey_min=planning.storey_distribution.private_storey_min,
        terrace_start_storey=planning.massing_profile.terrace_start_storey,
        tower_core_cell=planning.tower_core_cell,
    )
    timings["position_priors_ms"] = (time.perf_counter() - t0) * 1000.0

    weights = _score_weights_for_site(
        score_weights,
        tuning=tuning,
        footprint_xz=footprint_xz,
        max_storeys=brief.max_storeys or DEFAULT_MAX_STOREYS,
        cell_voxel_size=cell_voxel_size,
    )
    result = None
    total_search_ms = 0.0
    last_report: PlanValidityReport | None = None
    for iterations in _validity_iteration_schedule(search_iterations):
        for attempt_idx in range(_MAX_UNSUPPORTED_RETRIES):
            t0 = time.perf_counter()
            attempt_state = state.copy()
            config = _plan_mcts_config(
                iterations=iterations,
                seed=brief.seed + attempt_idx + iterations,
                tuning=tuning,
                massing_profile=planning.massing_profile,
            )
            result = mcts_search(attempt_state, planning.programme, weights=weights, config=config)
            total_search_ms += (time.perf_counter() - t0) * 1000.0
            timings["search_ms"] = total_search_ms

            if not is_solved(result.best_state):
                logger.warning(
                    "generate_housing_plan attempt %d/%d at %d iterations produced no solved state (%d contradictions)",
                    attempt_idx + 1,
                    _MAX_UNSUPPORTED_RETRIES,
                    iterations,
                    result.contradiction_count,
                )
                continue

            metadata = HousingPlanMetadata(
                seed=brief.seed,
                utility_type=utility_type,
                occupant_count=brief.occupant_count,
                scale_class=planning.scale_class,
                site_footprint_xz=footprint_xz,
                cell_grid_size=(grid.cx, grid.cy, grid.cz),
                cell_voxel_size=cell_voxel_size,
                score_total=result.best_score.total,
                score_breakdown=dict(result.best_score.components),
                stage_timings_ms=dict(timings),
                rollouts=result.iterations_run,
                tuning=tuning,
                storey_distribution=planning.storey_distribution,
                massing_profile=planning.massing_profile,
            )
            cells = _summarise_cells(result.best_state)
            provisional_plan = HousingPlan(
                state=result.best_state,
                programme=planning.programme,
                cells=cells,
                metadata=metadata,
                connection_policy=ConnectionPolicy(),
            )
            candidate_plan = HousingPlan(
                state=provisional_plan.state,
                programme=provisional_plan.programme,
                cells=provisional_plan.cells,
                metadata=provisional_plan.metadata,
                connection_policy=derive_connection_policy(provisional_plan),
            )
            report = validate_housing_plan(candidate_plan)
            if report.is_valid:
                _log_stage_timings("generate_housing_plan", timings)
                return candidate_plan
            last_report = report
            logger.info(
                "generate_housing_plan rejected invalid plan at %d iterations attempt %d/%d: %s",
                iterations,
                attempt_idx + 1,
                _MAX_UNSUPPORTED_RETRIES,
                "; ".join(report.errors),
            )

    raise NoValidPlanError(
        f"MCTS produced no valid solved state after budgets {_validity_iteration_schedule(search_iterations)}",
        last_report,
    )


def render_housing_plan_blocks(
    plan: HousingPlan,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_PLAN_PREVIEW_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
) -> list[SemanticBlockDict]:
    """Render the plan as plain utility-coloured cubes.

    This preview intentionally ignores facade, appendage, roof, and per-pod size
    multipliers. It exists to iterate the planning stage in isolation.
    """
    layout = uniform_layout(plan.state.grid, cell_voxel_size, origin_world)
    blocks: list[SemanticBlockDict] = []
    for cell in plan.cells:
        if cell.is_empty:
            continue
        block_id = _PLAN_PREVIEW_BLOCKS.get(cell.label, "minecraft:white_concrete")
        (x0, y0, z0), (x1, y1, z1) = layout.bbox(*cell.cell_index)
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                for x in range(x0, x1 + 1):
                    blocks.append({"x": x, "y": y, "z": z, "id": block_id})
    return blocks


def generate_housing_plan_for_request(
    request: HousingRequest,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    search_iterations: int = 256,
    score_weights: ScoreWeights | None = None,
    tuning: HousingPlanTuning | None = None,
) -> HousingPlan:
    brief = resolve_brief_for_request(
        request,
        cell_voxel_size=cell_voxel_size,
        tuning=tuning,
    )
    return generate_housing_plan(
        brief,
        footprint_xz=request.footprint_xz,
        utility_type=request.utility_type,
        cell_voxel_size=cell_voxel_size,
        search_iterations=search_iterations,
        score_weights=score_weights,
        tuning=tuning,
    )


__all__ = [
    "DEFAULT_CELL_VOXEL_SIZE",
    "DEFAULT_MAX_STOREYS",
    "DEFAULT_PLAN_PREVIEW_CELL_VOXEL_SIZE",
    "HousingPlan",
    "HousingPlanCell",
    "HousingPlanMetadata",
    "HousingPlanProfile",
    "HousingRequest",
    "HousingPlanTuning",
    "MassingProfile",
    "PLAN_PROFILES",
    "PlanningStageResult",
    "StoreyDistributionPlan",
    "UTILITY_SIZING_POLICIES",
    "generate_housing_plan_for_request",
    "generate_housing_plan",
    "render_housing_plan_blocks",
    "resolve_brief_for_request",
]
