"""Minecraft blueprint helpers for residential upgrades and settlement placement.

This module keeps the blueprint data path independent from GDPC.  Live placement
is handled by script-level adapters so tests can verify generation, slot fitting,
diffing, and export without a running Minecraft client.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from prefab_housing.api import assemble_house_from_plan
from prefab_housing.blueprint_package import read_blueprint_package, write_blueprint_package
from prefab_housing.catalogue.shell import choose_wall_face_design_path
from prefab_housing.catalogue import pod_types as pt
from prefab_housing.connection_policy import derive_connection_policy
from prefab_housing.exterior import (
    compose_block_generation_stages,
)
from prefab_housing.grid import FACE_INDEX, HORIZONTAL_FACES
from prefab_housing.housing_plan import (
    DEFAULT_CELL_VOXEL_SIZE,
    HousingPlan,
    HousingPlanCell,
    generate_housing_plan,
)
from prefab_housing.interior import interior_style_profile
from prefab_housing.search.score import ScoreWeights
from prefab_housing.types import ConnectionPolicy, FaceName
from prefab_housing.upgrade import (
    RESIDENTIAL_LEVEL_SPECS,
    brief_for_residential_level,
)
from prefab_housing.validity import validate_housing_plan

BlueprintBlock = dict[str, Any]
AnimationStrategy = Literal["y_up", "y_down", "radial_out"]
ResidentialBlockMode = Literal["core", "full", "structure"]

_COORD_KEYS = frozenset({"x", "y", "z", "dx", "dy", "dz", "id", "properties", "props"})
_AIR_BLOCK_ID = "minecraft:air"
_REPLACEABLE_UPGRADE_STAGE_NAMES = frozenset(
    {"wall_face_textures", "foundation", "trim_bands", "roof"}
)
_WALL_FACE_STAGE_NAMES = frozenset({"wall_face_textures"})
_CORE_REPLACEABLE_UPGRADE_STAGE_NAMES = _REPLACEABLE_UPGRADE_STAGE_NAMES - _WALL_FACE_STAGE_NAMES
_REMOVAL_UPGRADE_STAGE_NAMES = frozenset(
    {"connection_openings", "site_footprint_clip", "interior_site_footprint_clip"}
)
_HORIZONTAL_FACE_NAMES: tuple[FaceName, FaceName, FaceName, FaceName] = (
    "north",
    "east",
    "south",
    "west",
)


def _stringify_property_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


@dataclass(slots=True)
class ResidentialAnimationState:
    level: int
    name: str
    seed: int
    score_total: float
    valid: bool
    wall_face_preset: str | None
    wall_face_design_path: str | None
    entrance_face: str | None
    interior_style_id: str | None
    layout_variant_id: str | None
    source_block_count: int
    source_structure_block_count: int
    blocks: list[BlueprintBlock]
    structure_blocks: list[BlueprintBlock]
    core_blocks: list[BlueprintBlock] = field(default_factory=list)
    wall_face_blocks: list[BlueprintBlock] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class UpgradeDiff:
    from_level: int
    to_level: int
    blocks: list[BlueprintBlock]


@dataclass(frozen=True, slots=True)
class SettlementBuildSlot:
    """Local settlement plot rectangle using the quarantine pipeline contract."""

    x: int
    y: int
    z: int
    width: int
    depth: int
    cell_id: int | None = None
    zone_id: int | None = None
    building_type: str = "residential"

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("settlement build slot width must be positive")
        if self.depth < 1:
            raise ValueError("settlement build slot depth must be positive")
        if not self.building_type:
            raise ValueError("settlement build slot building_type must be non-empty")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_y: int = 0,
    ) -> "SettlementBuildSlot":
        return cls(
            x=int(data["x"]),
            y=int(data.get("y", default_y)),
            z=int(data["z"]),
            width=int(data["width"]),
            depth=int(data["depth"]),
            cell_id=int(data["cell_id"]) if "cell_id" in data else None,
            zone_id=int(data["zone_id"]) if "zone_id" in data else None,
            building_type=str(data.get("building_type", "residential")),
        )


@dataclass(frozen=True, slots=True)
class ResidentialSlotPlacement:
    """Pure block placement result for one residential package in a settlement slot."""

    slot: SettlementBuildSlot
    state: ResidentialAnimationState
    origin: tuple[int, int, int]
    rotation_steps: int
    entrance_face: FaceName | None
    bbox: tuple[int, int, int, int, int, int]
    blocks: list[BlueprintBlock]

    @property
    def level(self) -> int:
        return self.state.level


@dataclass(frozen=True, slots=True)
class ResidentialSlotPlacementRejection:
    slot: SettlementBuildSlot
    reason: str


@dataclass(frozen=True, slots=True)
class ResidentialSettlementPlacementPlan:
    placements: tuple[ResidentialSlotPlacement, ...]
    rejections: tuple[ResidentialSlotPlacementRejection, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.rejections


def _position(block: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(block["dx"]), int(block["dy"]), int(block["dz"])


def _block_signature(block: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    props = block.get("props", {})
    return str(block["id"]), tuple(sorted((str(k), str(v)) for k, v in props.items()))


def _placement_phase(block: Mapping[str, Any]) -> int:
    block_id = str(block["id"])
    if block_id == _AIR_BLOCK_ID:
        return 0
    if block_id.endswith("_bed"):
        props = block.get("props", {})
        if isinstance(props, Mapping) and props.get("part") == "foot":
            return 2
        if isinstance(props, Mapping) and props.get("part") == "head":
            return 3
    return 1


def _placement_sort_key(block: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(block["dy"]),
        _placement_phase(block),
        int(block["dx"]),
        int(block["dz"]),
        str(block["id"]),
    )


def _semantic_position(block: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(block["x"]), int(block["y"]), int(block["z"])


def _summarise_cells_from_state(state: Any) -> tuple[HousingPlanCell, ...]:
    cells: list[HousingPlanCell] = []
    for flat, tid_raw in enumerate(state.assignment.tolist()):
        tid = int(tid_raw)
        ix, iy, iz = state.grid.from_flat(flat)
        pod_idx = int(state.tiles.pod_index[tid])
        label = pt.POD_LABELS[pod_idx]
        cells.append(
            HousingPlanCell(
                cell_index=(ix, iy, iz),
                label=label,
                role=pt.POD_ROLE[pod_idx],  # type: ignore[arg-type]
                tile_id=tid,
                tile_label=state.tiles.tile_label[tid],
                rotation_quarters=int(state.tiles.rotation[tid]),
                occupancy_capacity=int(state.tiles.occupancy[tid]),
                is_empty=pt.is_void_pod_index(pod_idx),
            )
        )
    return tuple(cells)


def _tile_for_label(state: Any, cell_index: tuple[int, int, int], label: str) -> int:
    try:
        pod_index = pt.POD_INDEX[label]
    except KeyError as exc:
        raise ValueError(f"unknown pod label for upgrade lock: {label}") from exc
    flat = state.grid.flat_index(*cell_index)
    for tile_id in state.tiles.pod_to_tiles[pod_index]:
        if bool(state.domain[flat, int(tile_id)]):
            return int(tile_id)
    return int(state.tiles.pod_to_tiles[pod_index][0])


def _set_state_cell_tile(state: Any, cell_index: tuple[int, int, int], tile_id: int) -> None:
    flat = state.grid.flat_index(*cell_index)
    state.assignment[flat] = int(tile_id)
    state.domain[flat] = False
    state.domain[flat, int(tile_id)] = True
    state.entropy_count[flat] = 1


def _void_state_cell(state: Any, cell_index: tuple[int, int, int]) -> None:
    _set_state_cell_tile(state, cell_index, int(state.tiles.structural_void_tile_id))


def _state_label(state: Any, cell_index: tuple[int, int, int]) -> str:
    tid = int(state.assignment[state.grid.flat_index(*cell_index)])
    return pt.POD_LABELS[int(state.tiles.pod_index[tid])]


@dataclass(frozen=True, slots=True)
class _LockedUpgradeCell:
    label: str
    tile_label: str


def _tile_matching_locked_cell(
    state: Any,
    cell_index: tuple[int, int, int],
    locked: _LockedUpgradeCell,
) -> int:
    current_tid = int(state.assignment[state.grid.flat_index(*cell_index)])
    current_label = pt.POD_LABELS[int(state.tiles.pod_index[current_tid])]
    if current_label == locked.label:
        return current_tid

    flat = state.grid.flat_index(*cell_index)
    for tile_id, tile_label in enumerate(state.tiles.tile_label):
        if tile_label == locked.tile_label and bool(state.domain[flat, int(tile_id)]):
            return int(tile_id)
    return _tile_for_label(state, cell_index, locked.label)


def _locked_upgrade_cells_from_plan(
    plan: HousingPlan,
) -> dict[tuple[int, int, int], _LockedUpgradeCell]:
    return {
        cell.cell_index: _LockedUpgradeCell(
            label=cell.label,
            tile_label=cell.tile_label,
        )
        for cell in plan.cells
        if not cell.is_empty
    }


def _cell_counts(cells: Mapping[tuple[int, int, int], str]) -> Counter[str]:
    return Counter(cells.values())


def _can_add_label(label: str, counts: Counter[str], caps: Counter[str]) -> bool:
    cap = caps.get(label)
    return cap is None or counts.get(label, 0) < cap


def _ensure_stairwell_stack(
    generated: HousingPlan,
    selected: dict[tuple[int, int, int], str],
    required_stairwells: int,
) -> None:
    max_y = max(
        max((cell_index[1] for cell_index in selected), default=0),
        required_stairwells - 1,
    )
    if max_y <= 0:
        return

    grid = generated.state.grid
    cells_by_index = {cell.cell_index: cell for cell in generated.cells}
    for ix in range(grid.cx):
        for iz in range(grid.cz):
            if all(selected.get((ix, iy, iz)) == pt.POD_STAIRWELL for iy in range(max_y + 1)):
                return

    best: tuple[int, int, int, int, int, int] | None = None
    for ix in range(grid.cx):
        for iz in range(grid.cz):
            column = tuple((ix, iy, iz) for iy in range(max_y + 1))
            if any(
                cell_index in selected and selected[cell_index] != pt.POD_STAIRWELL
                for cell_index in column
            ):
                continue
            selected_stairs = sum(
                1
                for cell_index in column
                if selected.get(cell_index) == pt.POD_STAIRWELL
            )
            generated_stairs = sum(
                1
                for cell_index in column
                if cells_by_index[cell_index].label == pt.POD_STAIRWELL
            )
            adjacent_selected = 0
            for cell_index in column:
                cx, cy, cz = cell_index
                for face in HORIZONTAL_FACES:
                    neighbour = grid.neighbour(cx, cy, cz, face)
                    if neighbour is not None and neighbour in selected:
                        adjacent_selected += 1
            ground_neighbours = sum(
                1
                for face in HORIZONTAL_FACES
                if (neighbour := grid.neighbour(ix, 0, iz, face)) is not None
                and neighbour in selected
            )
            candidate = (
                selected_stairs,
                generated_stairs,
                adjacent_selected,
                ground_neighbours,
                -iz,
                -ix,
            )
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return

    _, _, _, _, neg_iz, neg_ix = best
    ix = -neg_ix
    iz = -neg_iz
    for iy in range(max_y + 1):
        selected[(ix, iy, iz)] = pt.POD_STAIRWELL


def _horizontal_selected_neighbour_count(
    generated: HousingPlan,
    selected: Mapping[tuple[int, int, int], str],
    cell_index: tuple[int, int, int],
) -> int:
    grid = generated.state.grid
    ix, iy, iz = cell_index
    return sum(
        1
        for face in HORIZONTAL_FACES
        if (neighbour := grid.neighbour(ix, iy, iz, face)) is not None
        and neighbour in selected
    )


def _relocate_isolated_upper_cells(
    generated: HousingPlan,
    selected: dict[tuple[int, int, int], str],
    locked_cells: Mapping[tuple[int, int, int], _LockedUpgradeCell],
) -> None:
    generated_by_index = {cell.cell_index: cell for cell in generated.cells}
    for cell_index, label in tuple(selected.items()):
        if cell_index in locked_cells or label == pt.POD_STAIRWELL or cell_index[1] == 0:
            continue

        selected_without_source = {
            existing_index: existing_label
            for existing_index, existing_label in selected.items()
            if existing_index != cell_index
        }
        below_source = (cell_index[0], cell_index[1] - 1, cell_index[2])
        source_supported = below_source in selected_without_source
        source_has_neighbour = (
            _horizontal_selected_neighbour_count(generated, selected, cell_index) > 0
        )
        if source_supported and source_has_neighbour:
            continue

        best: tuple[int, int, int, int, tuple[int, int, int]] | None = None
        for candidate_index, generated_cell in generated_by_index.items():
            if candidate_index in selected_without_source or candidate_index in locked_cells:
                continue
            if candidate_index[1] != cell_index[1]:
                continue
            below = (candidate_index[0], candidate_index[1] - 1, candidate_index[2])
            if candidate_index[1] > 0 and below not in selected_without_source:
                continue
            neighbour_count = _horizontal_selected_neighbour_count(
                generated,
                selected_without_source,
                candidate_index,
            )
            if neighbour_count <= 0:
                continue
            candidate = (
                1,
                1 if generated_cell.label == label else 0,
                neighbour_count,
                -abs(candidate_index[0] - cell_index[0]) - abs(candidate_index[2] - cell_index[2]),
                candidate_index,
            )
            if best is None or candidate > best:
                best = candidate

        if best is None:
            continue

        _, _, _, _, target_index = best
        selected.pop(cell_index)
        selected[target_index] = label


def _select_upgrade_cells(
    generated: HousingPlan,
    locked_cells: Mapping[tuple[int, int, int], _LockedUpgradeCell],
) -> dict[tuple[int, int, int], str]:
    selected = {cell_index: locked.label for cell_index, locked in locked_cells.items()}
    counts = _cell_counts(selected)
    required = generated.programme.required_counter()
    caps = generated.programme.max_counter()
    generated_cells = [
        cell
        for cell in generated.cells
        if not cell.is_empty and cell.cell_index not in locked_cells
    ]

    for cell in generated_cells:
        if cell.label == pt.POD_STAIRWELL:
            continue
        need = required.get(cell.label, 0)
        if need <= counts.get(cell.label, 0):
            continue
        selected[cell.cell_index] = cell.label
        counts[cell.label] += 1

    missing: list[str] = []
    for label, need in required.items():
        if label == pt.POD_STAIRWELL:
            continue
        missing.extend([label] * max(0, int(need) - counts.get(label, 0)))

    available = [
        cell for cell in generated_cells
        if cell.cell_index not in selected
    ]
    for label in missing:
        if not available:
            raise ValueError(
                f"generated level has no unlocked cells left for required pod {label!r}"
            )
        preferred_index = next(
            (
                index for index, cell in enumerate(available)
                if cell.label == label
            ),
            -1,
        )
        if preferred_index < 0:
            preferred_index = next(
                (
                    index for index, cell in enumerate(available)
                    if cell.label != pt.POD_STAIRWELL and cell.label not in caps
                ),
                0,
            )
        cell = available.pop(preferred_index)
        selected[cell.cell_index] = label
        counts[label] += 1

    for cell in available:
        if cell.label == pt.POD_STAIRWELL:
            continue
        if _can_add_label(cell.label, counts, caps):
            selected[cell.cell_index] = cell.label
            counts[cell.label] += 1

    _ensure_stairwell_stack(
        generated,
        selected,
        required_stairwells=required.get(pt.POD_STAIRWELL, 0),
    )
    _relocate_isolated_upper_cells(generated, selected, locked_cells)
    return selected


def _merge_generated_upgrade_plan(
    generated: HousingPlan,
    locked_cells: Mapping[tuple[int, int, int], _LockedUpgradeCell],
) -> HousingPlan:
    if not locked_cells:
        return generated

    selected = _select_upgrade_cells(generated, locked_cells)
    state = generated.state.copy()
    selected_indices = set(selected)

    for cell in generated.cells:
        if cell.cell_index not in selected_indices:
            _void_state_cell(state, cell.cell_index)

    for cell_index, locked in locked_cells.items():
        if not state.grid.in_bounds(*cell_index):
            raise ValueError(f"locked cell {cell_index} does not fit grid {state.grid}")
        _set_state_cell_tile(
            state,
            cell_index,
            _tile_matching_locked_cell(state, cell_index, locked),
        )

    for cell_index, label in selected.items():
        if cell_index in locked_cells:
            continue
        if _state_label(state, cell_index) != label:
            _set_state_cell_tile(
                state,
                cell_index,
                _tile_for_label(state, cell_index, label),
            )

    timings = dict(generated.metadata.stage_timings_ms)
    timings["upgrade_locked_cells"] = float(len(locked_cells))
    metadata = replace(generated.metadata, stage_timings_ms=timings)
    merged = HousingPlan(
        state=state,
        programme=generated.programme,
        cells=_summarise_cells_from_state(state),
        metadata=metadata,
        connection_policy=ConnectionPolicy(cells=()),
    )
    merged.connection_policy = derive_connection_policy(merged)
    return merged


def _append_only_blocks(
    previous: list[Mapping[str, Any]],
    candidate: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = [dict(block) for block in previous]
    occupied = {_semantic_position(block) for block in previous}
    for block in candidate:
        pos = _semantic_position(block)
        if pos in occupied:
            continue
        out.append(dict(block))
        occupied.add(pos)
    return out


def _remove_positions(
    blocks: list[Mapping[str, Any]],
    positions: set[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    if not positions:
        return [dict(block) for block in blocks]
    return [
        dict(block)
        for block in blocks
        if _semantic_position(block) not in positions
    ]


def _stage_block_positions(
    stages: Iterable[Any],
    stage_names: frozenset[str],
) -> set[tuple[int, int, int]]:
    return {
        _semantic_position(block)
        for stage in stages
        if stage.name in stage_names
        for block in stage.blocks
    }


def _stage_removed_positions(
    stages: Iterable[Any],
    stage_names: frozenset[str],
) -> set[tuple[int, int, int]]:
    return {
        (int(x), int(y), int(z))
        for stage in stages
        if stage.name in stage_names
        for x, y, z in stage.removed_positions
    }


def _occupied_cell_indices(plan: HousingPlan) -> set[tuple[int, int, int]]:
    return {cell.cell_index for cell in plan.cells if not cell.is_empty}


def _entry_exterior_face(plan: HousingPlan) -> str | None:
    grid = plan.state.grid
    for cell in plan.cells:
        if cell.is_empty or cell.label != pt.POD_ENTRY:
            continue
        policy = plan.connection_policy.for_cell(cell.cell_index)
        if policy is None:
            return None
        ix, iy, iz = cell.cell_index
        for face_name in policy.door_faces:
            face_index = FACE_INDEX[str(face_name)]
            if face_index in HORIZONTAL_FACES and grid.neighbour(ix, iy, iz, face_index) is None:
                return str(face_name)
    return None


def _block_in_bbox(
    block: Mapping[str, Any],
    bbox: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> bool:
    x, y, z = _semantic_position(block)
    (x0, y0, z0), (x1, y1, z1) = bbox
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def _blocks_in_cells(
    blocks: list[Mapping[str, Any]],
    semantic_cells: Iterable[Any],
    cell_indices: set[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    bboxes = [
        cell.voxel_bbox
        for cell in semantic_cells
        if cell.cell_index in cell_indices
    ]
    if not bboxes:
        return []
    return [
        dict(block)
        for block in blocks
        if any(_block_in_bbox(block, bbox) for bbox in bboxes)
    ]


def _positions_in_cells(
    positions: set[tuple[int, int, int]],
    semantic_cells: Iterable[Any],
    cell_indices: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    bboxes = [
        cell.voxel_bbox
        for cell in semantic_cells
        if cell.cell_index in cell_indices
    ]
    if not bboxes:
        return set()
    out: set[tuple[int, int, int]] = set()
    for position in positions:
        block = {"x": position[0], "y": position[1], "z": position[2]}
        if any(_block_in_bbox(block, bbox) for bbox in bboxes):
            out.add(position)
    return out


def _compose_non_interior_blocks(house: Any) -> list[dict[str, Any]]:
    stages = tuple(
        stage
        for stage in house.block_stages
        if stage.category != "interior"
        and stage.name != "interior_site_footprint_clip"
    )
    return compose_block_generation_stages(stages)


def _compose_non_interior_blocks_excluding(
    house: Any,
    stage_names: frozenset[str],
) -> list[dict[str, Any]]:
    stages = tuple(
        stage
        for stage in house.block_stages
        if stage.category != "interior"
        and stage.name != "interior_site_footprint_clip"
        and stage.name not in stage_names
    )
    return compose_block_generation_stages(stages)


def _compose_stage_blocks(
    house: Any,
    stage_names: frozenset[str],
) -> list[dict[str, Any]]:
    stages = tuple(stage for stage in house.block_stages if stage.name in stage_names)
    return compose_block_generation_stages(stages)


def _append_only_residential_plan(
    level: int,
    *,
    seed: int,
    material_theme: str,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    locked_cells: Mapping[tuple[int, int, int], _LockedUpgradeCell] | None = None,
) -> HousingPlan:
    try:
        spec = RESIDENTIAL_LEVEL_SPECS[level]
    except KeyError as exc:
        raise ValueError(f"unsupported append-only residential level: {level}") from exc
    brief = brief_for_residential_level(
        level,
        seed=seed,
        material_theme=material_theme,
    )
    generated = generate_housing_plan(
        brief,
        footprint_xz=spec.footprint_xz,
        utility_type="residential",
        cell_voxel_size=cell_voxel_size,
        search_iterations=spec.search_iterations,
        score_weights=ScoreWeights(fit_policy=spec.tuning.fit_policy),
        tuning=spec.tuning,
    )
    generated = _merge_generated_upgrade_plan(generated, locked_cells or {})
    metadata = replace(
        generated.metadata,
        scale_class="append_only_residential_upgrade",
    )
    plan = HousingPlan(
        state=generated.state,
        programme=generated.programme,
        cells=generated.cells,
        metadata=metadata,
        connection_policy=ConnectionPolicy(cells=()),
    )
    plan.connection_policy = derive_connection_policy(plan)
    return plan


def semantic_block_to_blueprint(
    block: Mapping[str, Any],
    *,
    offset: tuple[int, int, int] = (0, 0, 0),
) -> BlueprintBlock:
    """Convert renderer/GDPC-compatible semantic blocks to blueprint blocks."""
    ox, oy, oz = offset
    raw_props: dict[str, Any] = {}
    properties = block.get("properties")
    if isinstance(properties, Mapping):
        raw_props.update(properties)
    props = block.get("props")
    if isinstance(props, Mapping):
        raw_props.update(props)
    for key, value in block.items():
        if key not in _COORD_KEYS and value is not None:
            raw_props[str(key)] = value

    props = {
        str(key): _stringify_property_value(value)
        for key, value in raw_props.items()
        if value is not None
    }
    return {
        "dx": int(block["x"]) + ox,
        "dy": int(block["y"]) + oy,
        "dz": int(block["z"]) + oz,
        "id": str(block["id"]),
        "props": props,
    }


def compute_bounding_box(
    blocks: Iterable[Mapping[str, Any]],
) -> tuple[int, int, int, int, int, int]:
    positions = [_position(block) for block in blocks]
    if not positions:
        return (0, 0, 0, 0, 0, 0)
    xs, ys, zs = zip(*positions, strict=False)
    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)


def residential_state_blocks(
    state: ResidentialAnimationState,
    *,
    block_mode: ResidentialBlockMode = "core",
) -> list[BlueprintBlock]:
    if block_mode == "core":
        return sorted(state.core_blocks or state.blocks, key=_placement_sort_key)
    if block_mode == "full":
        return sorted(state.blocks, key=_placement_sort_key)
    if block_mode == "structure":
        return sorted(state.structure_blocks, key=_placement_sort_key)
    raise ValueError(f"unsupported residential block mode: {block_mode}")


def _face_after_y_rotation(face: str | None, steps: int) -> FaceName | None:
    if face not in _HORIZONTAL_FACE_NAMES:
        return None
    return _HORIZONTAL_FACE_NAMES[(_HORIZONTAL_FACE_NAMES.index(face) + steps) % 4]


def _rotation_steps_between_faces(source_face: str, target_face: FaceName) -> int:
    if source_face not in _HORIZONTAL_FACE_NAMES:
        raise ValueError(f"cannot rotate non-horizontal source face: {source_face}")
    return (
        _HORIZONTAL_FACE_NAMES.index(target_face)
        - _HORIZONTAL_FACE_NAMES.index(source_face)
    ) % 4


def _rotated_block_props(props: Any, steps: int) -> dict[str, str]:
    if not isinstance(props, Mapping):
        return {}
    rotated = {str(key): str(value) for key, value in props.items()}
    facing = rotated.get("facing")
    if facing in _HORIZONTAL_FACE_NAMES:
        rotated["facing"] = str(_face_after_y_rotation(facing, steps))
    axis = rotated.get("axis")
    if steps % 2 == 1 and axis in {"x", "z"}:
        rotated["axis"] = "z" if axis == "x" else "x"
    return rotated


def rotate_blueprint_blocks(
    blocks: list[BlueprintBlock],
    *,
    steps: int,
) -> list[BlueprintBlock]:
    """Rotate blueprint blocks around their own bounding box in 90 degree Y steps."""
    rotation = steps % 4
    if rotation == 0:
        return [dict(block) for block in blocks]

    min_dx, _min_dy, min_dz, max_dx, _max_dy, max_dz = compute_bounding_box(blocks)
    width = max_dx - min_dx
    depth = max_dz - min_dz
    rotated_blocks: list[BlueprintBlock] = []
    for block in blocks:
        x = int(block["dx"]) - min_dx
        z = int(block["dz"]) - min_dz
        if rotation == 1:
            nx, nz = depth - z, x
        elif rotation == 2:
            nx, nz = width - x, depth - z
        else:
            nx, nz = z, width - x

        rotated = dict(block)
        rotated["dx"] = min_dx + nx
        rotated["dz"] = min_dz + nz
        props = _rotated_block_props(block.get("props", {}), rotation)
        if props:
            rotated["props"] = props
        else:
            rotated.pop("props", None)
        rotated_blocks.append(rotated)
    return rotated_blocks


def translate_blueprint_blocks(
    blocks: list[BlueprintBlock],
    *,
    offset: tuple[int, int, int],
) -> list[BlueprintBlock]:
    """Return blocks offset into a parent local coordinate system."""
    ox, oy, oz = offset
    return [
        {
            **block,
            "dx": int(block["dx"]) + ox,
            "dy": int(block["dy"]) + oy,
            "dz": int(block["dz"]) + oz,
        }
        for block in blocks
    ]


def _footprint_size(
    bbox: tuple[int, int, int, int, int, int],
) -> tuple[int, int]:
    min_dx, _min_dy, min_dz, max_dx, _max_dy, max_dz = bbox
    return max_dx - min_dx + 1, max_dz - min_dz + 1


def _candidate_rotation_steps(
    state: ResidentialAnimationState,
    *,
    target_entrance_face: FaceName | None,
    allow_rotate: bool,
) -> tuple[int, ...]:
    if target_entrance_face is None:
        return (0, 1, 2, 3) if allow_rotate else (0,)
    if state.entrance_face is None:
        return (0,)
    return (_rotation_steps_between_faces(state.entrance_face, target_entrance_face),)


def place_residential_state_in_slot(
    state: ResidentialAnimationState,
    slot: SettlementBuildSlot,
    *,
    target_entrance_face: FaceName | None = None,
    allow_rotate: bool = True,
    block_mode: ResidentialBlockMode = "core",
) -> ResidentialSlotPlacement | None:
    """Fit one residential state into a settlement-local slot without GDPC."""
    source_blocks = residential_state_blocks(state, block_mode=block_mode)
    for rotation_steps in _candidate_rotation_steps(
        state,
        target_entrance_face=target_entrance_face,
        allow_rotate=allow_rotate,
    ):
        rotated_blocks = rotate_blueprint_blocks(source_blocks, steps=rotation_steps)
        bbox = compute_bounding_box(rotated_blocks)
        size_x, size_z = _footprint_size(bbox)
        if size_x > slot.width or size_z > slot.depth:
            continue

        min_dx, min_dy, min_dz, _max_dx, _max_dy, _max_dz = bbox
        origin = (
            slot.x + ((slot.width - size_x) // 2) - min_dx,
            slot.y - min_dy,
            slot.z + ((slot.depth - size_z) // 2) - min_dz,
        )
        placed_blocks = translate_blueprint_blocks(rotated_blocks, offset=origin)
        entrance_face = _face_after_y_rotation(state.entrance_face, rotation_steps)
        return ResidentialSlotPlacement(
            slot=slot,
            state=state,
            origin=origin,
            rotation_steps=rotation_steps,
            entrance_face=entrance_face,
            bbox=compute_bounding_box(placed_blocks),
            blocks=placed_blocks,
        )
    return None


def place_residential_upgrade_in_slot(
    states: Sequence[ResidentialAnimationState],
    slot: SettlementBuildSlot,
    *,
    target_entrance_face: FaceName | None = None,
    allow_rotate: bool = True,
    level: int | None = None,
    block_mode: ResidentialBlockMode = "core",
) -> ResidentialSlotPlacement:
    """Place the largest fitting valid residential upgrade level in a slot."""
    candidates = [
        state
        for state in states
        if state.valid and (level is None or state.level == level)
    ]
    for state in sorted(candidates, key=lambda item: item.level, reverse=True):
        placement = place_residential_state_in_slot(
            state,
            slot,
            target_entrance_face=target_entrance_face,
            allow_rotate=allow_rotate,
            block_mode=block_mode,
        )
        if placement is not None:
            return placement

    sizes = {
        state.level: _footprint_size(
            compute_bounding_box(residential_state_blocks(state, block_mode=block_mode))
        )
        for state in candidates
    }
    raise ValueError(
        "no valid residential upgrade level fits settlement slot "
        f"{slot.width}x{slot.depth}; candidate footprints={sizes}"
    )


def plan_residential_settlement_placements(
    states: Sequence[ResidentialAnimationState],
    slots: Sequence[SettlementBuildSlot],
    *,
    target_entrance_face: FaceName | None = None,
    allow_rotate: bool = True,
    level: int | None = None,
    block_mode: ResidentialBlockMode = "core",
    fail_fast: bool = False,
) -> ResidentialSettlementPlacementPlan:
    """Place one residential module per slot using one homogeneous module set."""
    placements: list[ResidentialSlotPlacement] = []
    rejections: list[ResidentialSlotPlacementRejection] = []

    for slot in slots:
        try:
            placements.append(
                place_residential_upgrade_in_slot(
                    states,
                    slot,
                    target_entrance_face=target_entrance_face,
                    allow_rotate=allow_rotate,
                    level=level,
                    block_mode=block_mode,
                )
            )
        except ValueError as exc:
            if fail_fast:
                raise
            rejections.append(
                ResidentialSlotPlacementRejection(slot=slot, reason=str(exc))
            )

    return ResidentialSettlementPlacementPlan(
        placements=tuple(placements),
        rejections=tuple(rejections),
    )


def _residential_state_variant_sets(
    states_or_variants: (
        Sequence[ResidentialAnimationState]
        | Sequence[Sequence[ResidentialAnimationState]]
    ),
) -> tuple[tuple[ResidentialAnimationState, ...], ...]:
    records = tuple(states_or_variants)
    if not records:
        return ()
    if isinstance(records[0], ResidentialAnimationState):
        return (tuple(records),)  # type: ignore[arg-type]
    return tuple(tuple(variant) for variant in records)  # type: ignore[union-attr]


def plan_typed_residential_settlement_placements(
    states_by_building_type: Mapping[
        str,
        Sequence[ResidentialAnimationState] | Sequence[Sequence[ResidentialAnimationState]],
    ],
    slots: Sequence[SettlementBuildSlot],
    *,
    target_entrance_face: FaceName | None = None,
    allow_rotate: bool = True,
    level: int | None = None,
    block_mode: ResidentialBlockMode = "core",
    fail_fast: bool = False,
) -> ResidentialSettlementPlacementPlan:
    """Place residential modules selected by each slot's building type."""
    variant_sets_by_type = {
        building_type: _residential_state_variant_sets(states_or_variants)
        for building_type, states_or_variants in states_by_building_type.items()
    }
    usage_counts: Counter[str] = Counter()
    placements: list[ResidentialSlotPlacement] = []
    rejections: list[ResidentialSlotPlacementRejection] = []

    for slot in slots:
        variant_sets = variant_sets_by_type.get(slot.building_type)
        if not variant_sets:
            reason = (
                "no residential module states registered for building type "
                f"{slot.building_type!r}"
            )
            if fail_fast:
                raise ValueError(reason)
            rejections.append(ResidentialSlotPlacementRejection(slot=slot, reason=reason))
            continue

        start_index = usage_counts[slot.building_type] % len(variant_sets)
        usage_counts[slot.building_type] += 1
        errors: list[str] = []
        try:
            best_placement: ResidentialSlotPlacement | None = None
            for variant_offset in range(len(variant_sets)):
                states = variant_sets[(start_index + variant_offset) % len(variant_sets)]
                try:
                    placement = place_residential_upgrade_in_slot(
                        states,
                        slot,
                        target_entrance_face=target_entrance_face,
                        allow_rotate=allow_rotate,
                        level=level,
                        block_mode=block_mode,
                    )
                    if best_placement is None or placement.level > best_placement.level:
                        best_placement = placement
                    if level is not None:
                        break
                except ValueError as exc:
                    errors.append(str(exc))
            if best_placement is None:
                raise ValueError(
                    "no registered residential variant fits building type "
                    f"{slot.building_type!r}; variant errors={errors}"
                )
            placements.append(best_placement)
        except ValueError as exc:
            if fail_fast:
                raise
            rejections.append(
                ResidentialSlotPlacementRejection(slot=slot, reason=str(exc))
            )

    return ResidentialSettlementPlacementPlan(
        placements=tuple(placements),
        rejections=tuple(rejections),
    )

def _grid(blocks: Iterable[BlueprintBlock]) -> dict[tuple[int, int, int], BlueprintBlock]:
    return {_position(block): block for block in blocks if block["id"] != _AIR_BLOCK_ID}


def diff_block_states(
    before_blocks: list[BlueprintBlock],
    after_blocks: list[BlueprintBlock],
) -> list[BlueprintBlock]:
    """Return placements needed to transform ``before`` into ``after``."""
    before = _grid(before_blocks)
    after = _grid(after_blocks)
    updates: list[BlueprintBlock] = []

    for pos, block in before.items():
        if pos not in after:
            dx, dy, dz = pos
            updates.append(
                {"dx": dx, "dy": dy, "dz": dz, "id": _AIR_BLOCK_ID, "props": {}}
            )
            continue
        if _block_signature(block) != _block_signature(after[pos]):
            updates.append(after[pos])

    for pos, block in after.items():
        if pos not in before:
            updates.append(block)

    return sorted(updates, key=_placement_sort_key)


def build_upgrade_diffs(states: list[ResidentialAnimationState]) -> list[UpgradeDiff]:
    diffs: list[UpgradeDiff] = []
    for before, after in zip(states, states[1:], strict=False):
        diffs.append(
            UpgradeDiff(
                from_level=before.level,
                to_level=after.level,
                blocks=diff_block_states(before.blocks, after.blocks),
            )
        )
    return diffs


def _normalise_state_blocks(
    states: list[ResidentialAnimationState],
) -> list[ResidentialAnimationState]:
    all_positions = [
        (int(block["x"]), int(block["y"]), int(block["z"]))
        for state in states
        for block in state.blocks
    ]
    if not all_positions:
        return states

    xs, ys, zs = zip(*all_positions, strict=False)
    offset = (-min(xs), -min(ys), -min(zs))
    normalised: list[ResidentialAnimationState] = []
    for state in states:
        normalised.append(
            ResidentialAnimationState(
                level=state.level,
                name=state.name,
                seed=state.seed,
                score_total=state.score_total,
                valid=state.valid,
                wall_face_preset=state.wall_face_preset,
                wall_face_design_path=state.wall_face_design_path,
                entrance_face=state.entrance_face,
                interior_style_id=state.interior_style_id,
                layout_variant_id=state.layout_variant_id,
                source_block_count=state.source_block_count,
                source_structure_block_count=state.source_structure_block_count,
                blocks=[
                    semantic_block_to_blueprint(block, offset=offset)
                    for block in state.blocks
                ],
                structure_blocks=[
                    semantic_block_to_blueprint(block, offset=offset)
                    for block in state.structure_blocks
                ],
                core_blocks=[
                    semantic_block_to_blueprint(block, offset=offset)
                    for block in (state.core_blocks or state.blocks)
                ],
                wall_face_blocks=[
                    semantic_block_to_blueprint(block, offset=offset)
                    for block in state.wall_face_blocks
                ],
            )
        )
    return normalised


def build_residential_upgrade_sequence(
    *,
    seed: int = 42,
    material_theme: str = "sci_fi_modular",
    levels: tuple[int, ...] = (1, 2, 3),
    search_iterations: Mapping[int, int] | None = None,
    wall_face_design_path: str | Path | None = None,
) -> list[ResidentialAnimationState]:
    """Generate append-only block states for each residential upgrade level."""
    if search_iterations:
        # Kept for CLI/API compatibility with the earlier MCTS-backed path.
        # The live upgrade sequence is now deterministic so upgrades preserve
        # earlier cells instead of re-solving a replacement house each level.
        _ = search_iterations

    states: list[ResidentialAnimationState] = []
    selected_wall_face_design_path = (
        Path(wall_face_design_path)
        if wall_face_design_path is not None
        else choose_wall_face_design_path(seed, salt="residential_upgrade")
    )
    selected_interior_style_id = f"{interior_style_profile(seed).id}+room_variants"
    selected_layout_variant_id = f"wfc_mcts_append_seed_{seed}"
    cumulative_blocks: list[dict[str, Any]] = []
    cumulative_core_blocks: list[dict[str, Any]] = []
    cumulative_wall_face_blocks: list[dict[str, Any]] = []
    cumulative_structure_blocks: list[dict[str, Any]] = []
    cumulative_cell_indices: set[tuple[int, int, int]] = set()
    cumulative_stairwell_indices: set[tuple[int, int, int]] = set()
    cumulative_interior_positions: set[tuple[int, int, int]] = set()
    locked_cells: dict[tuple[int, int, int], _LockedUpgradeCell] = {}
    replaceable_positions: set[tuple[int, int, int]] = set()
    core_replaceable_positions: set[tuple[int, int, int]] = set()
    wall_face_replaceable_positions: set[tuple[int, int, int]] = set()
    stable_entrance_face: str | None = None
    for level in levels:
        spec = RESIDENTIAL_LEVEL_SPECS[level]
        plan = _append_only_residential_plan(
            level,
            seed=seed,
            material_theme=material_theme,
            locked_cells=locked_cells,
        )
        validation = validate_housing_plan(plan)
        house = assemble_house_from_plan(
            plan,
            material_theme=material_theme,
            wall_face_design_path=selected_wall_face_design_path,
        )
        stable_entrance_face = stable_entrance_face or _entry_exterior_face(plan)
        removed_positions = _stage_removed_positions(
            house.block_stages,
            _REMOVAL_UPGRADE_STAGE_NAMES,
        )
        occupied_cell_indices = _occupied_cell_indices(plan)
        new_cell_indices = occupied_cell_indices - cumulative_cell_indices
        stairwell_cell_indices = {
            cell.cell_index
            for cell in house.semantic_cells
            if cell.label == pt.POD_STAIRWELL
        }
        new_stairwell_indices = stairwell_cell_indices & new_cell_indices
        refresh_interior_indices = (
            new_cell_indices
            | cumulative_stairwell_indices
            | new_stairwell_indices
        )
        stale_interior_positions = _positions_in_cells(
            cumulative_interior_positions,
            house.semantic_cells,
            refresh_interior_indices,
        )
        non_interior_blocks = _compose_non_interior_blocks(house)
        core_non_interior_blocks = _compose_non_interior_blocks_excluding(
            house,
            _WALL_FACE_STAGE_NAMES,
        )
        wall_face_blocks = _compose_stage_blocks(house, _WALL_FACE_STAGE_NAMES)
        new_interior_blocks = _blocks_in_cells(
            house.interior_blocks,
            house.semantic_cells,
            refresh_interior_indices,
        )
        structure_blocks = compose_block_generation_stages(
            tuple(
                stage
                for stage in house.block_stages
                if stage.include_in_structure_template
            )
        )
        cumulative_blocks = _remove_positions(
            cumulative_blocks,
            replaceable_positions
            | stale_interior_positions
            | (removed_positions - cumulative_interior_positions),
        )
        cumulative_core_blocks = _remove_positions(
            cumulative_core_blocks,
            core_replaceable_positions
            | stale_interior_positions
            | (removed_positions - cumulative_interior_positions),
        )
        cumulative_wall_face_blocks = _remove_positions(
            cumulative_wall_face_blocks,
            wall_face_replaceable_positions | removed_positions,
        )
        cumulative_structure_blocks = _remove_positions(
            cumulative_structure_blocks,
            removed_positions,
        )
        cumulative_blocks = _append_only_blocks(
            cumulative_blocks,
            [*non_interior_blocks, *new_interior_blocks],
        )
        cumulative_core_blocks = _append_only_blocks(
            cumulative_core_blocks,
            [*core_non_interior_blocks, *new_interior_blocks],
        )
        cumulative_wall_face_blocks = _append_only_blocks(
            cumulative_wall_face_blocks,
            wall_face_blocks,
        )
        cumulative_interior_positions -= stale_interior_positions
        cumulative_interior_positions.update(
            _semantic_position(block)
            for block in new_interior_blocks
        )
        cumulative_structure_blocks = _append_only_blocks(
            cumulative_structure_blocks,
            structure_blocks,
        )
        replaceable_positions = _stage_block_positions(
            house.block_stages,
            _REPLACEABLE_UPGRADE_STAGE_NAMES,
        )
        core_replaceable_positions = _stage_block_positions(
            house.block_stages,
            _CORE_REPLACEABLE_UPGRADE_STAGE_NAMES,
        )
        wall_face_replaceable_positions = _stage_block_positions(
            house.block_stages,
            _WALL_FACE_STAGE_NAMES,
        )
        cumulative_cell_indices.update(occupied_cell_indices)
        cumulative_stairwell_indices.update(new_stairwell_indices)
        locked_cells = _locked_upgrade_cells_from_plan(plan)
        states.append(
            ResidentialAnimationState(
                level=level,
                name=spec.name,
                seed=seed,
                score_total=plan.metadata.score_total,
                valid=validation.is_valid,
                wall_face_preset=house.metadata.wall_face_preset,
                wall_face_design_path=house.metadata.wall_face_design_path,
                entrance_face=stable_entrance_face,
                interior_style_id=selected_interior_style_id,
                layout_variant_id=selected_layout_variant_id,
                source_block_count=len(cumulative_blocks),
                source_structure_block_count=len(cumulative_structure_blocks),
                blocks=cumulative_blocks,
                structure_blocks=cumulative_structure_blocks,
                core_blocks=cumulative_core_blocks,
                wall_face_blocks=cumulative_wall_face_blocks,
            )
        )
    return _normalise_state_blocks(states)


def iter_batches(
    blocks: list[BlueprintBlock],
    *,
    strategy: AnimationStrategy = "y_up",
) -> Iterable[list[BlueprintBlock]]:
    if strategy in {"y_up", "y_down"}:
        by_y: dict[int, list[BlueprintBlock]] = defaultdict(list)
        for block in blocks:
            by_y[int(block["dy"])].append(block)
        for y in sorted(by_y, reverse=(strategy == "y_down")):
            yield sorted(by_y[y], key=_placement_sort_key)
        return

    if strategy == "radial_out":
        if not blocks:
            return
        cx = sum(int(block["dx"]) for block in blocks) / len(blocks)
        cz = sum(int(block["dz"]) for block in blocks) / len(blocks)
        buckets: dict[int, list[BlueprintBlock]] = defaultdict(list)
        for block in blocks:
            dist = math.hypot(int(block["dx"]) - cx, int(block["dz"]) - cz)
            buckets[int(math.floor(dist))].append(block)
        for bucket in sorted(buckets):
            yield sorted(
                buckets[bucket],
                key=_placement_sort_key,
            )
        return

    raise ValueError(f"unknown animation strategy: {strategy}")


def write_blueprint(path: Path, *, meta: Mapping[str, Any], blocks: list[BlueprintBlock]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": dict(meta), "blocks": blocks}, indent=2),
        encoding="utf-8",
    )


def export_residential_upgrade_sequence(
    states: list[ResidentialAnimationState],
    output_dir: Path,
) -> dict[str, Any]:
    """Write per-level and per-upgrade-diff blueprint JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diffs = build_upgrade_diffs(states)

    level_records: list[dict[str, Any]] = []
    cache_dir = output_dir / "structure_cache"
    for state in states:
        path = output_dir / f"level_{state.level}_{state.name}.json"
        cache_path = cache_dir / f"level_{state.level}_{state.name}_structure.json"
        bbox = compute_bounding_box(state.blocks)
        structure_bbox = compute_bounding_box(state.structure_blocks)
        meta = {
            "kind": "residential_upgrade_level",
            "upgrade_policy": "append_only_residential_cell_schedule",
            "level": state.level,
            "name": state.name,
            "seed": state.seed,
            "score_total": state.score_total,
            "valid": state.valid,
            "wall_face_preset": state.wall_face_preset,
            "wall_face_design_path": state.wall_face_design_path,
            "entrance_face": state.entrance_face,
            "interior_style_id": state.interior_style_id,
            "layout_variant_id": state.layout_variant_id,
            "bbox": bbox,
        }
        write_blueprint(path, meta=meta, blocks=state.blocks)
        write_blueprint(
            cache_path,
            meta={
                "kind": "residential_structure_template_cache",
                "upgrade_policy": "append_only_residential_cell_schedule",
                "level": state.level,
                "name": state.name,
                "seed": state.seed,
                "wall_face_preset": state.wall_face_preset,
                "wall_face_design_path": state.wall_face_design_path,
                "entrance_face": state.entrance_face,
                "interior_style_id": state.interior_style_id,
                "layout_variant_id": state.layout_variant_id,
                "bbox": structure_bbox,
                "source_stage_policy": "include_in_structure_template",
            },
            blocks=state.structure_blocks,
        )
        level_records.append(
            {
                **meta,
                "path": str(path),
                "block_count": len(state.blocks),
                "structure_cache_path": str(cache_path),
                "structure_block_count": len(state.structure_blocks),
                "structure_bbox": structure_bbox,
            }
        )

    diff_records: list[dict[str, Any]] = []
    state_by_level = {state.level: state for state in states}
    for diff in diffs:
        to_state = state_by_level[diff.to_level]
        path = output_dir / f"diff_{diff.from_level}_to_{diff.to_level}.json"
        meta = {
            "kind": "residential_upgrade_diff",
            "upgrade_policy": "append_only_residential_cell_schedule",
            "from_level": diff.from_level,
            "to_level": diff.to_level,
            "wall_face_preset": to_state.wall_face_preset,
            "wall_face_design_path": to_state.wall_face_design_path,
            "entrance_face": to_state.entrance_face,
            "interior_style_id": to_state.interior_style_id,
            "layout_variant_id": to_state.layout_variant_id,
            "bbox": compute_bounding_box(diff.blocks),
        }
        write_blueprint(path, meta=meta, blocks=diff.blocks)
        diff_records.append({**meta, "path": str(path), "block_count": len(diff.blocks)})

    manifest = {
        "kind": "residential_upgrade_animation_manifest",
        "upgrade_policy": "append_only_residential_cell_schedule",
        "levels": level_records,
        "diffs": diff_records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _residential_package_sections(
    states: list[ResidentialAnimationState],
    diffs: list[UpgradeDiff],
) -> tuple[
    dict[str, list[BlueprintBlock]],
    dict[int, str],
    dict[int, str],
    dict[int, str],
    dict[int, str],
    dict[tuple[int, int], str],
]:
    sections: dict[str, list[BlueprintBlock]] = {}
    level_sections: dict[int, str] = {}
    structure_sections: dict[int, str] = {}
    core_sections: dict[int, str] = {}
    wall_face_sections: dict[int, str] = {}
    diff_sections: dict[tuple[int, int], str] = {}

    for state in states:
        level_section = f"level_{state.level}_{state.name}"
        structure_section = f"level_{state.level}_{state.name}_structure"
        core_section = f"level_{state.level}_{state.name}_core"
        wall_face_section = f"level_{state.level}_{state.name}_wall_face"
        sections[level_section] = state.blocks
        sections[structure_section] = state.structure_blocks
        sections[core_section] = state.core_blocks or state.blocks
        sections[wall_face_section] = state.wall_face_blocks
        level_sections[state.level] = level_section
        structure_sections[state.level] = structure_section
        core_sections[state.level] = core_section
        wall_face_sections[state.level] = wall_face_section

    for diff in diffs:
        section = f"diff_{diff.from_level}_to_{diff.to_level}"
        sections[section] = diff.blocks
        diff_sections[(diff.from_level, diff.to_level)] = section

    return (
        sections,
        level_sections,
        structure_sections,
        core_sections,
        wall_face_sections,
        diff_sections,
    )


def export_residential_upgrade_package(
    states: list[ResidentialAnimationState],
    package_path: Path,
) -> dict[str, Any]:
    """Write one compact binary residential upgrade package."""
    diffs = build_upgrade_diffs(states)
    (
        sections,
        level_sections,
        structure_sections,
        core_sections,
        wall_face_sections,
        diff_sections,
    ) = _residential_package_sections(states, diffs)

    level_records: list[dict[str, Any]] = []
    for state in states:
        bbox = compute_bounding_box(state.blocks)
        structure_bbox = compute_bounding_box(state.structure_blocks)
        core_blocks = state.core_blocks or state.blocks
        wall_face_blocks = state.wall_face_blocks
        core_bbox = compute_bounding_box(core_blocks)
        wall_face_bbox = compute_bounding_box(wall_face_blocks)
        level_records.append(
            {
                "kind": "residential_upgrade_level",
                "upgrade_policy": "append_only_residential_cell_schedule",
                "wall_face_policy": "separate_swappable_section",
                "level": state.level,
                "name": state.name,
                "seed": state.seed,
                "score_total": state.score_total,
                "valid": state.valid,
                "wall_face_preset": state.wall_face_preset,
                "wall_face_design_path": state.wall_face_design_path,
                "entrance_face": state.entrance_face,
                "interior_style_id": state.interior_style_id,
                "layout_variant_id": state.layout_variant_id,
                "section": level_sections[state.level],
                "block_count": len(state.blocks),
                "bbox": bbox,
                "structure_section": structure_sections[state.level],
                "structure_block_count": len(state.structure_blocks),
                "structure_bbox": structure_bbox,
                "core_section": core_sections[state.level],
                "core_block_count": len(core_blocks),
                "core_bbox": core_bbox,
                "wall_face_section": wall_face_sections[state.level],
                "wall_face_block_count": len(wall_face_blocks),
                "wall_face_bbox": wall_face_bbox,
            }
        )

    diff_records: list[dict[str, Any]] = []
    state_by_level = {state.level: state for state in states}
    for diff in diffs:
        to_state = state_by_level[diff.to_level]
        diff_records.append(
            {
                "kind": "residential_upgrade_diff",
                "upgrade_policy": "append_only_residential_cell_schedule",
                "wall_face_policy": "legacy_full_diff_includes_wall_face",
                "from_level": diff.from_level,
                "to_level": diff.to_level,
                "wall_face_preset": to_state.wall_face_preset,
                "wall_face_design_path": to_state.wall_face_design_path,
                "entrance_face": to_state.entrance_face,
                "interior_style_id": to_state.interior_style_id,
                "layout_variant_id": to_state.layout_variant_id,
                "section": diff_sections[(diff.from_level, diff.to_level)],
                "block_count": len(diff.blocks),
                "bbox": compute_bounding_box(diff.blocks),
            }
        )

    manifest = {
        "kind": "residential_upgrade_animation_package",
        "upgrade_policy": "append_only_residential_cell_schedule",
        "wall_face_policy": "levels provide core_section plus separate wall_face_section",
        "levels": level_records,
        "diffs": diff_records,
    }
    return write_blueprint_package(package_path, metadata=manifest, sections=sections)


def load_residential_upgrade_package(
    package_path: Path,
) -> tuple[list[ResidentialAnimationState], list[UpgradeDiff], dict[str, Any]]:
    """Load a compact binary residential upgrade package."""
    manifest, sections = read_blueprint_package(package_path)
    states: list[ResidentialAnimationState] = []
    for record in manifest.get("levels", []):
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid level record in {package_path}: {record!r}")
        blocks = sections[str(record["section"])]
        structure_blocks = sections[str(record["structure_section"])]
        core_section = record.get("core_section")
        wall_face_section = record.get("wall_face_section")
        core_blocks = sections[str(core_section)] if core_section is not None else blocks
        wall_face_blocks = (
            sections[str(wall_face_section)] if wall_face_section is not None else []
        )
        states.append(
            ResidentialAnimationState(
                level=int(record["level"]),
                name=str(record["name"]),
                seed=int(record["seed"]),
                score_total=float(record["score_total"]),
                valid=bool(record["valid"]),
                wall_face_preset=record.get("wall_face_preset"),
                wall_face_design_path=record.get("wall_face_design_path"),
                entrance_face=record.get("entrance_face"),
                interior_style_id=record.get("interior_style_id"),
                layout_variant_id=record.get("layout_variant_id"),
                source_block_count=len(blocks),
                source_structure_block_count=len(structure_blocks),
                blocks=blocks,
                structure_blocks=structure_blocks,
                core_blocks=core_blocks,
                wall_face_blocks=wall_face_blocks,
            )
        )

    diffs: list[UpgradeDiff] = []
    for record in manifest.get("diffs", []):
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid diff record in {package_path}: {record!r}")
        diffs.append(
            UpgradeDiff(
                from_level=int(record["from_level"]),
                to_level=int(record["to_level"]),
                blocks=sections[str(record["section"])],
            )
        )

    if not states:
        raise ValueError(f"residential package has no level states: {package_path}")
    if len(states) > 1 and len(diffs) != len(states) - 1:
        raise ValueError(
            f"residential package has {len(states)} levels but {len(diffs)} diffs"
        )
    return states, diffs, manifest


__all__ = [
    "AnimationStrategy",
    "BlueprintBlock",
    "ResidentialBlockMode",
    "ResidentialAnimationState",
    "ResidentialSettlementPlacementPlan",
    "ResidentialSlotPlacement",
    "ResidentialSlotPlacementRejection",
    "SettlementBuildSlot",
    "UpgradeDiff",
    "build_residential_upgrade_sequence",
    "build_upgrade_diffs",
    "compute_bounding_box",
    "diff_block_states",
    "export_residential_upgrade_sequence",
    "export_residential_upgrade_package",
    "iter_batches",
    "load_residential_upgrade_package",
    "plan_residential_settlement_placements",
    "plan_typed_residential_settlement_placements",
    "place_residential_state_in_slot",
    "place_residential_upgrade_in_slot",
    "rotate_blueprint_blocks",
    "residential_state_blocks",
    "semantic_block_to_blueprint",
    "translate_blueprint_blocks",
    "write_blueprint",
]
