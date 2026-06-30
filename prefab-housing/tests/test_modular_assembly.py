from __future__ import annotations

from prefab_housing import (
    Brief,
    ExteriorStyle,
    RESIDENTIAL_LEVEL_SPECS,
    RoomInteriorCache,
    assemble_house_from_plan,
    compose_block_generation_stages,
    generate_housing_plan,
    generate_housing_plan_for_request,
    register_exterior_style,
    render_plan_exterior_stages,
    request_for_residential_level,
)
from prefab_housing.catalogue.shell import list_wall_face_design_paths

WALLFACE_GLASS_BLOCK = "minecraft:light_gray_stained_glass_pane"


def _make_plan(seed: int = 42):
    return generate_housing_plan(
        Brief(
            occupant_count=3,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=seed,
        ),
        footprint_xz=(30, 30),
        search_iterations=96,
    )


def test_assemble_house_from_plan_separates_exterior_and_interior_blocks() -> None:
    plan = _make_plan()
    result = assemble_house_from_plan(plan)
    assert result.exterior_blocks
    assert result.interior_blocks
    assert result.room_interiors
    assert result.blocks == result.exterior_blocks + result.interior_blocks
    assert result.metadata.interior_cache_stats["misses"] >= 1
    assert all(room.signature is not None for room in result.room_interiors)
    assert all(room.plan is not None for room in result.room_interiors)
    assert all(room.plan.lighting_keywords for room in result.room_interiors if room.plan is not None)


def test_assemble_house_from_plan_records_house_wallface_preset() -> None:
    plan = _make_plan(seed=42)
    result = assemble_house_from_plan(plan)
    preset_names = {path.name for path in list_wall_face_design_paths()}
    wall_face_stage = next(stage for stage in result.block_stages if stage.name == "wall_face_textures")

    assert result.metadata.wall_face_preset in preset_names
    assert result.metadata.wall_face_design_path is not None
    assert result.metadata.wall_face_design_path.endswith(result.metadata.wall_face_preset)
    assert any(block["id"] == WALLFACE_GLASS_BLOCK for block in wall_face_stage.blocks)


def test_room_cache_reuses_variants_across_identical_assemblies() -> None:
    plan = _make_plan(seed=42)
    cache = RoomInteriorCache()
    first = assemble_house_from_plan(plan, room_cache=cache)
    second = assemble_house_from_plan(plan, room_cache=cache)
    assert first.room_interiors == second.room_interiors
    assert second.metadata.interior_cache_stats["hits"] > first.metadata.interior_cache_stats["hits"]


def test_bedroom_and_bathroom_receive_required_core_components() -> None:
    plan = _make_plan(seed=42)
    result = assemble_house_from_plan(plan)
    bedrooms = [room for room in result.room_interiors if room.room_type == "bedroom"]
    bathrooms = [room for room in result.room_interiors if room.room_type == "bathroom"]
    assert bedrooms
    assert bathrooms
    assert all(room.plan is not None and "bed_core" in room.plan.core_keywords for room in bedrooms)
    assert all(room.plan is not None and "toilet" in room.plan.core_keywords for room in bathrooms)


def test_custom_exterior_style_is_swappable() -> None:
    def render_marker(plan, layout, material_theme):
        return [{"x": 0, "y": 0, "z": 0, "id": "minecraft:gold_block"}]

    register_exterior_style(ExteriorStyle(name="test_marker", render=render_marker))
    plan = _make_plan()
    result = assemble_house_from_plan(plan, exterior_style="test_marker")
    assert result.exterior_blocks == [{"x": 0, "y": 0, "z": 0, "id": "minecraft:gold_block"}]
    assert result.metadata.exterior_style == "test_marker"


def test_block_generation_stages_expose_construction_order() -> None:
    plan = _make_plan()
    result = assemble_house_from_plan(plan)
    stage_names = [stage.name for stage in result.block_stages]

    assert stage_names.index("wall_face_textures") < stage_names.index("roof")
    assert stage_names.index("roof") < stage_names.index("site_footprint_clip")
    assert stage_names.index("site_footprint_clip") < stage_names.index("populate_interiors")

    exterior_stages = tuple(
        stage
        for stage in result.block_stages
        if stage.name not in {"populate_interiors", "interior_site_footprint_clip"}
    )
    assert result.exterior_blocks == compose_block_generation_stages(exterior_stages)


def test_structure_template_stages_exclude_decor_and_interiors() -> None:
    plan = _make_plan()
    result = assemble_house_from_plan(plan)
    template_stage_names = [
        stage.name
        for stage in result.block_stages
        if stage.include_in_structure_template
    ]

    assert template_stage_names == ["structural_shell", "connection_openings"]
    assert all(
        stage.category == "structure"
        for stage in result.block_stages
        if stage.include_in_structure_template
    )
    assert all(
        not stage.include_in_structure_template
        for stage in result.block_stages
        if stage.name in {"wall_face_textures", "foundation", "trim_bands", "roof", "populate_interiors"}
    )


def test_full_footprint_house_keeps_proud_wall_face_detail_after_site_clip() -> None:
    spec = RESIDENTIAL_LEVEL_SPECS[1]
    request = request_for_residential_level(1, seed=42)
    plan = generate_housing_plan_for_request(
        request,
        search_iterations=spec.search_iterations,
        tuning=spec.tuning,
    )
    stages = render_plan_exterior_stages(
        plan,
        material_theme=request.material_theme or "sci_fi_modular",
    )
    wall_face_stage = next(
        stage for stage in stages if stage.name == "wall_face_textures"
    )
    clip_stage = next(
        stage for stage in stages if stage.name == "site_footprint_clip"
    )
    final_positions = {
        (block["x"], block["y"], block["z"])
        for block in compose_block_generation_stages(stages)
    }
    obsolete_room_colour_ids = {
        "minecraft:orange_concrete",
        "minecraft:yellow_concrete",
        "minecraft:light_blue_concrete",
        "minecraft:purple_concrete",
    }
    sx, sz = spec.footprint_xz
    outside_frame_positions = {
        (block["x"], block["y"], block["z"])
        for block in wall_face_stage.blocks
        if block["id"] == "minecraft:black_concrete"
        and (
            block["x"] < 0
            or block["x"] >= sx
            or block["z"] < 0
            or block["z"] >= sz
        )
    }
    wall_face_ids = {block["id"] for block in wall_face_stage.blocks}

    assert WALLFACE_GLASS_BLOCK in wall_face_ids
    assert wall_face_ids.isdisjoint(obsolete_room_colour_ids)
    assert outside_frame_positions
    assert outside_frame_positions <= final_positions
    assert outside_frame_positions.isdisjoint(clip_stage.removed_positions)


def test_block_stage_composition_prefers_later_roof_blocks() -> None:
    from prefab_housing import BlockGenerationStage

    wall = BlockGenerationStage(
        name="wall_face_textures",
        order=0,
        operation="emit",
        category="decor",
        blocks=[{"x": 1, "y": 2, "z": 3, "id": "minecraft:red_concrete"}],
    )
    roof = BlockGenerationStage(
        name="roof",
        order=1,
        operation="emit",
        category="decor",
        blocks=[{"x": 1, "y": 2, "z": 3, "id": "minecraft:black_concrete"}],
    )

    assert compose_block_generation_stages((wall, roof)) == [
        {"x": 1, "y": 2, "z": 3, "id": "minecraft:black_concrete"}
    ]
