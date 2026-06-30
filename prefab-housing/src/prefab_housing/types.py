"""Public data contracts for the prefab-housing pipeline.

Data-oriented design (DOD) note: these dataclasses are *interface* records used
at the API boundary. Internal hot loops avoid them and operate on packed arrays
indexed by `int` IDs (see `prefab_housing.grid` and `prefab_housing.wfc.tiles`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Renderer-compatible block schema. Re-declared here to avoid a hard import-time
# dependency on voxel-renderer in the production code path; voxel-renderer is a
# dev-only consumer of the *output* of this pipeline.
SemanticBlockDict = dict[str, Any]

UtilityType = Literal["residential", "commercial", "service_building", "storage_utility"]
HouseholdType = Literal["solo", "couple", "single_family", "shared", "multi_family"]
FaceName = Literal["north", "south", "east", "west", "up", "down"]
CellRole = Literal["habitable", "service", "circulation", "exterior"]
OpeningPattern = Literal["sealed", "edge_only", "multi_direction_open"]
BlockStageOperation = Literal["emit", "carve", "clip", "compose"]
BlockStageCategory = Literal["structure", "decor", "interior", "boundary", "composite", "custom"]

SCHEMA_VERSION = "1.4"


@dataclass(frozen=True, slots=True)
class Brief:
    """Explicit input parameter from the orchestrator.

    `occupant_count` and `household_type` jointly drive the required-pod set.
    `outdoor_living_priority`, `material_theme`, and `required_extra_rooms`
    bias the resolver and scorer. `seed` is the sole RNG entropy source.
    """

    occupant_count: int
    household_type: HouseholdType
    outdoor_living_priority: float = 0.3
    max_storeys: int | None = None
    material_theme: str | None = None
    seed: int = 0
    required_extra_rooms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.occupant_count < 1:
            raise ValueError("occupant_count must be >= 1")
        if not 0.0 <= self.outdoor_living_priority <= 1.0:
            raise ValueError("outdoor_living_priority must be in [0, 1]")
        if self.max_storeys is not None and self.max_storeys < 1:
            raise ValueError("max_storeys must be >= 1 when provided")


@dataclass(frozen=True, slots=True)
class SemanticCell:
    """Per-cell record handed to the downstream interior team.

    `voxel_bbox` is `((x0, y0, z0), (x1, y1, z1))` inclusive in world coordinates.
    `interior_volume_voxels` is the scalar budget for furnishing modules; M2 may
    promote this to a polyhedron.
    """

    cell_index: tuple[int, int, int]
    voxel_bbox: tuple[tuple[int, int, int], tuple[int, int, int]]
    label: str
    role: CellRole
    occupancy_capacity: int
    daylight_score: float
    privacy_depth: int
    door_faces: tuple[FaceName, ...]
    window_faces: tuple[FaceName, ...]
    interior_volume_voxels: int
    pod_template_id: str
    properties: dict[str, str] = field(default_factory=dict)
    open_faces: tuple[FaceName, ...] = ()
    opening_pattern: OpeningPattern = "sealed"


@dataclass(frozen=True, slots=True)
class CellConnectionPolicy:
    cell_index: tuple[int, int, int]
    door_faces: tuple[FaceName, ...] = ()
    open_faces: tuple[FaceName, ...] = ()
    window_faces: tuple[FaceName, ...] = ()
    opening_pattern: OpeningPattern = "sealed"


@dataclass(frozen=True, slots=True)
class ConnectionPolicy:
    cells: tuple[CellConnectionPolicy, ...] = ()

    def for_cell(self, cell_index: tuple[int, int, int]) -> CellConnectionPolicy | None:
        for cell in self.cells:
            if cell.cell_index == cell_index:
                return cell
        return None


@dataclass(frozen=True, slots=True)
class RoomSpatialConstraints:
    voxel_size: tuple[int, int, int]
    cell_index: tuple[int, int, int] | None = None
    door_faces: tuple[FaceName, ...] = ()
    window_faces: tuple[FaceName, ...] = ()
    open_faces: tuple[FaceName, ...] = ()
    opening_pattern: OpeningPattern = "sealed"
    privacy_depth: int = -1
    occupancy_capacity: int = 0


@dataclass(frozen=True, slots=True)
class RoomSignature:
    room_type: str
    utility_type: UtilityType
    role: CellRole
    voxel_size: tuple[int, int, int]
    interior_size: tuple[int, int, int]
    floor_area: int
    doorway_count: int
    window_count: int
    privacy_band: str
    exposure: str
    occupancy_band: str
    size_class: str
    lighting_tier: str


@dataclass(frozen=True, slots=True)
class RoomPlan:
    signature: RoomSignature
    core_keywords: tuple[str, ...] = ()
    supplementary_keywords: tuple[str, ...] = ()
    lighting_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoomComponentSpec:
    keyword: str
    block_id: str
    footprint: tuple[int, int]
    anchor: str
    category: str


@dataclass(frozen=True, slots=True)
class RoomComponentPlacement:
    keyword: str
    block_id: str
    category: str
    origin: tuple[int, int, int]
    footprint: tuple[int, int]
    anchor: str


@dataclass(frozen=True, slots=True)
class RoomLayoutPlan:
    plan: RoomPlan
    interior_size: tuple[int, int, int]
    cell_index: tuple[int, int, int] | None = None
    placements: tuple[RoomComponentPlacement, ...] = ()
    door_faces: tuple[FaceName, ...] = ()
    window_faces: tuple[FaceName, ...] = ()
    open_faces: tuple[FaceName, ...] = ()
    opening_pattern: OpeningPattern = "sealed"


@dataclass(frozen=True, slots=True)
class RoomRequest:
    room_type: str
    utility_type: UtilityType
    role: CellRole
    constraints: RoomSpatialConstraints
    signature: RoomSignature | None = None


@dataclass(frozen=True, slots=True)
class RoomInterior:
    cell_index: tuple[int, int, int]
    room_type: str
    variant_id: str
    cache_key: str
    voxel_bbox: tuple[tuple[int, int, int], tuple[int, int, int]]
    blocks: list[SemanticBlockDict]
    signature: RoomSignature | None = None
    plan: RoomPlan | None = None
    layout: RoomLayoutPlan | None = None


@dataclass(frozen=True, slots=True)
class HouseMetadata:
    seed: int
    utility_type: UtilityType
    cell_grid_size: tuple[int, int, int]      # (cx, cy, cz)
    cell_voxel_size: tuple[int, int, int]     # (vx, vy, vz)
    score_total: float
    score_breakdown: dict[str, float]
    stage_timings_ms: dict[str, float]
    rollouts: int
    site_footprint_xz: tuple[int, int] | None = None
    exterior_style: str = "modular_shell"
    wall_face_preset: str | None = None
    wall_face_design_path: str | None = None
    interior_cache_stats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BlockGenerationStage:
    """Block-level generation signal for construction animation hooks.

    ``blocks`` contains blocks emitted by this stage. ``removed_positions`` is
    used by destructive stages such as connection carving and site clipping.
    Consumers can replay stages in ``order`` using ``operation`` to decide
    whether to place or remove blocks.
    """

    name: str
    order: int
    operation: BlockStageOperation = "emit"
    category: BlockStageCategory = "structure"
    include_in_structure_template: bool = False
    blocks: list[SemanticBlockDict] = field(default_factory=list)
    removed_positions: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class HouseResult:
    blocks: list[SemanticBlockDict]
    semantic_cells: list[SemanticCell]
    metadata: HouseMetadata
    exterior_blocks: list[SemanticBlockDict] = field(default_factory=list)
    interior_blocks: list[SemanticBlockDict] = field(default_factory=list)
    room_interiors: list[RoomInterior] = field(default_factory=list)
    block_stages: tuple[BlockGenerationStage, ...] = ()
    schema_version: str = SCHEMA_VERSION


__all__ = [
    "SCHEMA_VERSION",
    "BlockStageCategory",
    "BlockGenerationStage",
    "BlockStageOperation",
    "Brief",
    "CellRole",
    "CellConnectionPolicy",
    "ConnectionPolicy",
    "FaceName",
    "HouseMetadata",
    "HouseResult",
    "HouseholdType",
    "OpeningPattern",
    "RoomInterior",
    "RoomLayoutPlan",
    "RoomPlan",
    "RoomComponentPlacement",
    "RoomComponentSpec",
    "RoomRequest",
    "RoomSignature",
    "RoomSpatialConstraints",
    "SemanticBlockDict",
    "SemanticCell",
    "UtilityType",
]
