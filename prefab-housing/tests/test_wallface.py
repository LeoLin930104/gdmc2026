from __future__ import annotations

from pathlib import Path

from prefab_housing import Brief, generate_housing_plan, render_plan_exterior_design
from prefab_housing.catalogue.shell import (
    build_face_texture_panel,
    choose_wall_face_design_path,
    get_default_wall_face_design_path,
    list_wall_face_design_paths,
    set_active_wall_face_design,
)
from prefab_housing.palette import resolve_palette
from prefab_housing.wallface import (
    DEFAULT_BASE_WALL_BLOCK,
    EMPTY_SYMBOL,
    emit_wall_face_blocks,
    empty_wall_face_design,
    load_wall_face_design,
    parse_wall_face_design,
    render_wall_face_preview,
    serialise_wall_face_design,
)

WALLFACE_GLASS_BLOCK = "minecraft:light_gray_stained_glass_pane"


def test_wallface_round_trip_sample_design() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "designs" / "modular_default.wallface"
    design = load_wall_face_design(sample_path)
    serialised = serialise_wall_face_design(design)
    reparsed = parse_wall_face_design(serialised)
    block_ids = {
        block_id
        for layer in design.layers.values()
        for row in layer
        for block_id in row
        if block_id is not None
    }
    assert (design.width, design.height) == (10, 6)
    assert WALLFACE_GLASS_BLOCK in block_ids
    assert "minecraft:sea_lantern" in block_ids
    assert "minecraft:red_concrete" not in block_ids
    assert reparsed == design


def test_emit_wallface_blocks_resamples_to_target_size() -> None:
    design = parse_wall_face_design(
        "\n".join(
            (
                "wallface-v1",
                "size 2 2",
                "symbol A minecraft:gold_block",
                "",
                "layer -2",
                "..",
                "..",
                "",
                "layer -1",
                "..",
                "..",
                "",
                "layer 0",
                "AA",
                "AA",
                "",
                "layer 1",
                "..",
                "..",
                "",
                "layer 2",
                "..",
                "..",
                "",
            )
        )
    )
    blocks = emit_wall_face_blocks(
        design,
        axis="x",
        fixed=0,
        outward_sign=1,
        a0=0,
        a1=3,
        y0=0,
        y1=3,
    )
    assert len(blocks) == 16
    assert all(block["id"] == "minecraft:gold_block" for block in blocks)


def test_shell_face_builder_can_load_saved_wallface_design() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "designs" / "modular_default.wallface"
    set_active_wall_face_design(sample_path)
    try:
        blocks = build_face_texture_panel(
            axis="x",
            fixed=0,
            outward_sign=1,
            a0=0,
            a1=7,
            y0=0,
            y1=5,
            palette=resolve_palette("sci_fi_modular"),
            pod_name="living",
        )
    finally:
        set_active_wall_face_design(None)
    assert blocks
    assert any(block["id"] == WALLFACE_GLASS_BLOCK for block in blocks)
    assert any(block["id"] == "minecraft:sea_lantern" for block in blocks)
    assert all(block["id"] != "minecraft:red_concrete" for block in blocks)


def test_shell_face_builder_uses_default_wallface_design_without_override() -> None:
    set_active_wall_face_design(None)
    blocks = build_face_texture_panel(
        axis="x",
        fixed=0,
        outward_sign=1,
        a0=0,
        a1=9,
        y0=0,
        y1=5,
        palette=resolve_palette("sci_fi_modular"),
        pod_name="living",
    )

    assert get_default_wall_face_design_path().exists()
    assert any(block["id"] == "minecraft:sea_lantern" for block in blocks)
    assert any(block["id"] == WALLFACE_GLASS_BLOCK for block in blocks)


def test_wallface_preset_registry_lists_modular_designs() -> None:
    paths = list_wall_face_design_paths()
    names = {path.name for path in paths}

    assert paths[0].name == "modular_default.wallface"
    assert {"modular_default.wallface", "modular_var1.wallface"} <= names
    assert choose_wall_face_design_path(42, salt="residential_upgrade") in paths


def test_render_plan_exterior_design_uses_saved_wallface() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "designs" / "modular_default.wallface"
    plan = generate_housing_plan(
        Brief(
            occupant_count=3,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=42,
        ),
        footprint_xz=(30, 30),
        search_iterations=64,
    )
    blocks = render_plan_exterior_design(plan, wall_face_design_path=sample_path)
    assert blocks
    assert any(block["id"] == WALLFACE_GLASS_BLOCK for block in blocks)
    assert all(block["id"] != "minecraft:red_concrete" for block in blocks)


def test_empty_symbol_is_reserved() -> None:
    assert EMPTY_SYMBOL == "."


def test_empty_design_prefills_base_wall_layer() -> None:
    design = empty_wall_face_design(3, 2)
    assert all(cell == DEFAULT_BASE_WALL_BLOCK for row in design.layers[0] for cell in row)
    assert all(cell is None for layer in (-2, -1, 1, 2) for row in design.layers[layer] for cell in row)


def test_editor_preview_renders_all_standard_views() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "designs" / "modular_default.wallface"
    design = load_wall_face_design(sample_path)
    preview = render_wall_face_preview(design)
    assert set(preview) == {"iso_left", "iso_right", "profile", "top"}
    assert all(preview[name] for name in preview)
