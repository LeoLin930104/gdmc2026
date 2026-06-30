"""Public API: ``build_house(brief, footprint, utility_type, ...) → HouseResult``.

Single entry point that wires the current prefab-housing pipeline:

1. Resolve programme from ``brief × utility_type``.
2. Design the cell grid (topology only) from footprint + ``brief.max_storeys``.
3. Build the static tile set.
4. Initialise solver state (boundary EXTERIOR pruning + AC-3).
5. Apply per-cell positional priors.
6. Run MCTS-guided WFC search with utility scoring.
7. Build the spatial layout (per-cell voxel AABBs) post-solve.
8. Emit exterior block stages: structural shell, openings, decor, roof, clipping.
9. Annotate semantic cells and populate room interiors.
10. Bundle flat blocks plus ordered block-stage signals into :class:`HouseResult`.

Determinism: identical ``(brief, footprint, utility_type, search_config)``
yields a bit-exact identical ``HouseResult``.
"""

from __future__ import annotations

import time
from pathlib import Path

from prefab_housing.annotate import annotate
from prefab_housing.boundary import clip_blocks_to_site_footprint_with_removed
from prefab_housing.catalogue.shell import (
    choose_wall_face_design_path,
    get_active_wall_face_design_path,
    set_active_wall_face_design,
)
from prefab_housing.exterior import (
    DEFAULT_EXTERIOR_STYLE,
    compose_block_generation_stages,
    design_plan_exterior_layout,
    render_plan_exterior_stages_with_layout,
)
from prefab_housing.housing_plan import (
    DEFAULT_CELL_VOXEL_SIZE,
    DEFAULT_MAX_STOREYS,
    HousingPlan,
    HousingPlanTuning,
    generate_housing_plan,
)
from prefab_housing.interior import RoomInteriorCache, generate_room_interiors
from prefab_housing.palette import DEFAULT_THEME
from prefab_housing.search.score import ScoreWeights
from prefab_housing.types import (
    Brief,
    BlockGenerationStage,
    HouseMetadata,
    HouseResult,
    UtilityType,
)


def build_house(
    brief: Brief,
    *,
    footprint_xz: tuple[int, int],
    utility_type: UtilityType = "residential",
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
    search_iterations: int = 256,
    score_weights: ScoreWeights | None = None,
    plan_tuning: HousingPlanTuning | None = None,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
    material_theme: str | None = None,
    room_cache: RoomInteriorCache | None = None,
    wall_face_design_path: str | Path | None = None,
) -> HouseResult:
    """Run the full pipeline and return a renderable house."""
    plan = generate_housing_plan(
        brief,
        footprint_xz=footprint_xz,
        utility_type=utility_type,
        cell_voxel_size=cell_voxel_size,
        search_iterations=search_iterations,
        score_weights=score_weights,
        tuning=plan_tuning,
    )

    return assemble_house_from_plan(
        plan,
        cell_voxel_size=cell_voxel_size,
        origin_world=origin_world,
        material_theme=material_theme or brief.material_theme or DEFAULT_THEME,
        exterior_style=exterior_style,
        room_cache=room_cache,
        wall_face_design_path=wall_face_design_path,
    )


def assemble_house_from_plan(
    plan: HousingPlan,
    *,
    cell_voxel_size: tuple[int, int, int] | None = None,
    origin_world: tuple[int, int, int] = (0, 0, 0),
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
    room_cache: RoomInteriorCache | None = None,
    wall_face_design_path: str | Path | None = None,
) -> HouseResult:
    """Assemble exterior and cached interiors for an existing solved plan."""
    resolved_cell_size = cell_voxel_size or plan.metadata.cell_voxel_size
    resolved_wall_face_design_path = (
        Path(wall_face_design_path)
        if wall_face_design_path is not None
        else choose_wall_face_design_path(
            plan.metadata.seed,
            salt=f"{plan.metadata.utility_type}:{plan.metadata.cell_grid_size}",
        )
    )
    wall_face_preset = (
        resolved_wall_face_design_path.name
        if resolved_wall_face_design_path is not None
        else None
    )
    wall_face_path_text = (
        str(resolved_wall_face_design_path)
        if resolved_wall_face_design_path is not None
        else None
    )

    # Layer 2: build voxel-space layout from the solved topology. The banded
    # factory honours per-pod ``POD_SIZE_MULTIPLIER`` along x and z while
    # keeping storey heights uniform; with all v1 multipliers equal to 1.0
    # this reduces to a uniform layout (preserving determinism baseline).
    # The current render path then places boxed modular cells first and only
    # afterwards resolves connection cuts between adjacent cells.
    t0 = time.perf_counter()
    layout = design_plan_exterior_layout(
        plan,
        cell_voxel_size=resolved_cell_size,
        origin_world=origin_world,
    )
    previous_wall_face_design_path = get_active_wall_face_design_path()
    try:
        set_active_wall_face_design(resolved_wall_face_design_path)
        exterior_stages = render_plan_exterior_stages_with_layout(
            plan,
            layout,
            material_theme=material_theme,
            exterior_style=exterior_style,
        )
    finally:
        set_active_wall_face_design(previous_wall_face_design_path)
    exterior_blocks = compose_block_generation_stages(exterior_stages)
    materialise_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    semantic_cells = annotate(plan.state, layout, plan.connection_policy)
    annotate_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    room_interiors, interior_stats = generate_room_interiors(
        semantic_cells,
        utility_type=plan.metadata.utility_type,
        cache=room_cache,
        variant_seed=plan.metadata.seed,
    )
    raw_interior_blocks = [block for room in room_interiors for block in room.blocks]
    interior_blocks, removed_interior_positions = clip_blocks_to_site_footprint_with_removed(
        raw_interior_blocks,
        site_footprint_xz=plan.metadata.site_footprint_xz,
        origin_world=origin_world,
    )
    interior_ms = (time.perf_counter() - t0) * 1000.0

    timings = dict(plan.metadata.stage_timings_ms)
    timings["materialise_ms"] = materialise_ms
    timings["annotate_ms"] = annotate_ms
    timings["interior_ms"] = interior_ms

    metadata = HouseMetadata(
        seed=plan.metadata.seed,
        utility_type=plan.metadata.utility_type,
        site_footprint_xz=plan.metadata.site_footprint_xz,
        cell_grid_size=plan.metadata.cell_grid_size,
        cell_voxel_size=resolved_cell_size,
        score_total=plan.metadata.score_total,
        score_breakdown=dict(plan.metadata.score_breakdown),
        stage_timings_ms=timings,
        rollouts=plan.metadata.rollouts,
        exterior_style=exterior_style,
        wall_face_preset=wall_face_preset,
        wall_face_design_path=wall_face_path_text,
        interior_cache_stats={
            "hits": interior_stats.hits,
            "misses": interior_stats.misses,
            "unique_variants": interior_stats.misses,
        },
    )

    interior_stage_order = len(exterior_stages)
    block_stages = (
        *exterior_stages,
        BlockGenerationStage(
            name="populate_interiors",
            order=interior_stage_order,
            operation="emit",
            category="interior",
            blocks=raw_interior_blocks,
        ),
        BlockGenerationStage(
            name="interior_site_footprint_clip",
            order=interior_stage_order + 1,
            operation="clip",
            category="boundary",
            removed_positions=removed_interior_positions,
        ),
    )

    return HouseResult(
        blocks=exterior_blocks + interior_blocks,
        semantic_cells=semantic_cells,
        metadata=metadata,
        exterior_blocks=exterior_blocks,
        interior_blocks=interior_blocks,
        room_interiors=room_interiors,
        block_stages=block_stages,
    )


__all__ = [
    "DEFAULT_CELL_VOXEL_SIZE",
    "DEFAULT_MAX_STOREYS",
    "assemble_house_from_plan",
    "build_house",
]
