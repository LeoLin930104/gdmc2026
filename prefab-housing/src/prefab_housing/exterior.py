"""Exterior composition for solved housing plans.

This module owns the whole-building exterior pass for a solved topology:

1. place modular cell shells
2. resolve inter-cell passages
3. add wall-face texture panels
4. add swappable decorative bands/foundation
5. add roof geometry
6. clip to the explicit site footprint

Keeping this composition in one module prevents shell and roof design work from
remaining scattered across scripts and API callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from prefab_housing.boundary import clip_blocks_to_site_footprint_with_removed
from prefab_housing.catalogue.shell import (
    get_active_wall_face_design_path,
    set_active_wall_face_design,
)
from prefab_housing.decorate import generate_foundation_blocks, generate_trim_band_blocks
from prefab_housing.housing_plan import DEFAULT_CELL_VOXEL_SIZE, HousingPlan
from prefab_housing.layout import SpatialLayout, banded_layout
from prefab_housing.materialise import (
    carve_connection_openings,
    place_facade_overlays,
    place_structural_shells,
)
from prefab_housing.palette import DEFAULT_THEME
from prefab_housing.palette import resolve_palette
from prefab_housing.roof import generate_roof
from prefab_housing.types import BlockGenerationStage, SemanticBlockDict


DEFAULT_EXTERIOR_STYLE: Final[str] = "modular_shell"
SITE_FOOTPRINT_CLIP_EXEMPT_STAGE_NAMES: Final[frozenset[str]] = frozenset(
    {"wall_face_textures"}
)
BlockStageRenderer = Callable[
    [HousingPlan, SpatialLayout, str],
    tuple[BlockGenerationStage, ...],
]


@dataclass(frozen=True, slots=True)
class ExteriorStyle:
    name: str
    render: Callable[[HousingPlan, SpatialLayout, str], list[SemanticBlockDict]]
    render_stages: BlockStageRenderer | None = None


def _position(block: SemanticBlockDict) -> tuple[int, int, int]:
    return (int(block["x"]), int(block["y"]), int(block["z"]))


def compose_block_generation_stages(
    stages: tuple[BlockGenerationStage, ...],
) -> list[SemanticBlockDict]:
    """Replay block-stage signals into a single deterministic block list."""
    by_pos: dict[tuple[int, int, int], SemanticBlockDict] = {}
    for stage in sorted(stages, key=lambda item: item.order):
        if stage.operation in {"carve", "clip"}:
            for position in stage.removed_positions:
                by_pos.pop(position, None)
        for block in stage.blocks:
            by_pos[_position(block)] = block
    return list(by_pos.values())


def _render_modular_shell(
    plan: HousingPlan,
    layout: SpatialLayout,
    material_theme: str,
) -> list[SemanticBlockDict]:
    return compose_block_generation_stages(
        _render_modular_shell_stages(plan, layout, material_theme)
    )


def _render_modular_shell_stages(
    plan: HousingPlan,
    layout: SpatialLayout,
    material_theme: str,
) -> tuple[BlockGenerationStage, ...]:
    palette = resolve_palette(material_theme)
    structural_shell = place_structural_shells(plan.state, layout, palette)
    _carved_shell, removed_connections = carve_connection_openings(
        plan.state,
        layout,
        plan.connection_policy,
        structural_shell,
    )
    return (
        BlockGenerationStage(
            name="structural_shell",
            order=0,
            operation="emit",
            category="structure",
            include_in_structure_template=True,
            blocks=structural_shell,
        ),
        BlockGenerationStage(
            name="connection_openings",
            order=1,
            operation="carve",
            category="structure",
            include_in_structure_template=True,
            removed_positions=removed_connections,
        ),
        BlockGenerationStage(
            name="wall_face_textures",
            order=2,
            operation="emit",
            category="decor",
            blocks=place_facade_overlays(
                plan.state,
                layout,
                palette,
                plan.connection_policy,
            ),
        ),
        BlockGenerationStage(
            name="foundation",
            order=3,
            operation="emit",
            category="decor",
            blocks=generate_foundation_blocks(plan.state, layout, palette),
        ),
        BlockGenerationStage(
            name="trim_bands",
            order=4,
            operation="emit",
            category="decor",
            blocks=generate_trim_band_blocks(plan.state, layout, palette),
        ),
        BlockGenerationStage(
            name="roof",
            order=5,
            operation="emit",
            category="decor",
            blocks=generate_roof(plan.state, layout, palette),
        ),
    )


def _render_shell_only(
    plan: HousingPlan,
    layout: SpatialLayout,
    material_theme: str,
) -> list[SemanticBlockDict]:
    return compose_block_generation_stages(
        _render_shell_only_stages(plan, layout, material_theme)
    )


def _render_shell_only_stages(
    plan: HousingPlan,
    layout: SpatialLayout,
    material_theme: str,
) -> tuple[BlockGenerationStage, ...]:
    palette = resolve_palette(material_theme)
    structural_shell = place_structural_shells(plan.state, layout, palette)
    _carved_shell, removed_connections = carve_connection_openings(
        plan.state,
        layout,
        plan.connection_policy,
        structural_shell,
    )
    return (
        BlockGenerationStage(
            name="structural_shell",
            order=0,
            operation="emit",
            category="structure",
            include_in_structure_template=True,
            blocks=structural_shell,
        ),
        BlockGenerationStage(
            name="connection_openings",
            order=1,
            operation="carve",
            category="structure",
            include_in_structure_template=True,
            removed_positions=removed_connections,
        ),
    )


def render_plan_exterior_design(
    plan: HousingPlan,
    *,
    wall_face_design_path: str | Path | None,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
) -> list[SemanticBlockDict]:
    previous = get_active_wall_face_design_path()
    try:
        set_active_wall_face_design(wall_face_design_path)
        return render_plan_exterior(
            plan,
            cell_voxel_size=cell_voxel_size,
            origin_world=origin_world,
            material_theme=material_theme,
            exterior_style=exterior_style,
        )
    finally:
        set_active_wall_face_design(previous)


EXTERIOR_STYLE_REGISTRY: dict[str, ExteriorStyle] = {
    DEFAULT_EXTERIOR_STYLE: ExteriorStyle(
        name=DEFAULT_EXTERIOR_STYLE,
        render=_render_modular_shell,
        render_stages=_render_modular_shell_stages,
    ),
    "shell_only": ExteriorStyle(
        name="shell_only",
        render=_render_shell_only,
        render_stages=_render_shell_only_stages,
    ),
}


def register_exterior_style(style: ExteriorStyle) -> None:
    EXTERIOR_STYLE_REGISTRY[style.name] = style


def resolve_exterior_style(name: str | None) -> ExteriorStyle:
    key = name or DEFAULT_EXTERIOR_STYLE
    if key not in EXTERIOR_STYLE_REGISTRY:
        raise KeyError(
            f"unknown exterior_style {key!r}; available: {sorted(EXTERIOR_STYLE_REGISTRY.keys())}"
        )
    return EXTERIOR_STYLE_REGISTRY[key]


def render_plan_exterior(
    plan: HousingPlan,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
) -> list[SemanticBlockDict]:
    """Render the solved plan as a composed exterior shell.

    Uses the same staged shell/decorate/roof path as the public house builder,
    but operates directly on an existing ``HousingPlan`` so topology profiles can
    be re-rendered without repeating search.
    """
    layout = design_plan_exterior_layout(
        plan,
        cell_voxel_size=cell_voxel_size,
        origin_world=origin_world,
    )
    return render_plan_exterior_with_layout(
        plan,
        layout,
        material_theme=material_theme,
        exterior_style=exterior_style,
    )


def render_plan_exterior_stages(
    plan: HousingPlan,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
) -> tuple[BlockGenerationStage, ...]:
    """Render the solved plan as ordered block-generation stage signals."""
    layout = design_plan_exterior_layout(
        plan,
        cell_voxel_size=cell_voxel_size,
        origin_world=origin_world,
    )
    return render_plan_exterior_stages_with_layout(
        plan,
        layout,
        material_theme=material_theme,
        exterior_style=exterior_style,
    )


def design_plan_exterior_layout(
    plan: HousingPlan,
    *,
    cell_voxel_size: tuple[int, int, int] = DEFAULT_CELL_VOXEL_SIZE,
    origin_world: tuple[int, int, int] = (0, 0, 0),
) -> SpatialLayout:
    """Produce the voxel-space layout used by the exterior composition pass."""
    return banded_layout(plan.state, cell_voxel_size, origin_world)


def render_plan_exterior_with_layout(
    plan: HousingPlan,
    layout: SpatialLayout,
    *,
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
) -> list[SemanticBlockDict]:
    """Compose shell, decoration, and roof for a solved plan + layout."""
    return compose_block_generation_stages(
        render_plan_exterior_stages_with_layout(
            plan,
            layout,
            material_theme=material_theme,
            exterior_style=exterior_style,
        )
    )


def render_plan_exterior_stages_with_layout(
    plan: HousingPlan,
    layout: SpatialLayout,
    *,
    material_theme: str = DEFAULT_THEME,
    exterior_style: str = DEFAULT_EXTERIOR_STYLE,
) -> tuple[BlockGenerationStage, ...]:
    """Emit ordered block-generation stages for a solved plan + layout."""
    style = resolve_exterior_style(exterior_style)
    if style.render_stages is None:
        stages = (
            BlockGenerationStage(
                name=f"{style.name}_blocks",
                order=0,
                operation="emit",
                category="custom",
                blocks=style.render(plan, layout, material_theme),
            ),
        )
    else:
        stages = style.render_stages(plan, layout, material_theme)

    # Deliberate contract: the site AABB constrains the modular structure, not
    # swappable wall-face skin. Wall-face texture blocks may protrude beyond
    # the construction footprint so their multi-layer depth is preserved.
    clip_subject_stages = tuple(
        stage
        for stage in stages
        if stage.name not in SITE_FOOTPRINT_CLIP_EXEMPT_STAGE_NAMES
    )
    _clipped, removed_positions = clip_blocks_to_site_footprint_with_removed(
        compose_block_generation_stages(clip_subject_stages),
        site_footprint_xz=plan.metadata.site_footprint_xz,
        origin_world=layout.origin_world,
    )
    return (
        *stages,
        BlockGenerationStage(
            name="site_footprint_clip",
            order=len(stages),
            operation="clip",
            category="boundary",
            removed_positions=removed_positions,
        ),
    )


__all__ = [
    "DEFAULT_EXTERIOR_STYLE",
    "EXTERIOR_STYLE_REGISTRY",
    "ExteriorStyle",
    "compose_block_generation_stages",
    "design_plan_exterior_layout",
    "register_exterior_style",
    "render_plan_exterior_design",
    "render_plan_exterior",
    "render_plan_exterior_stages",
    "render_plan_exterior_stages_with_layout",
    "render_plan_exterior_with_layout",
    "resolve_exterior_style",
]
