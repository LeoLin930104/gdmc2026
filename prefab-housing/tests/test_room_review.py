from __future__ import annotations

from pathlib import Path

from prefab_housing.interior import (
    RoomInteriorCache,
    _component_blocks,
    layout_variant_index,
    make_room_request,
    plan_room,
    plan_room_layout,
    room_cache_key,
    room_variant_index,
)
from prefab_housing.room_review import analyse_room_layout, save_room_layout_report
from prefab_housing.types import SemanticCell

BED_BLOCK_IDS = {
    "minecraft:red_bed",
    "minecraft:green_bed",
    "minecraft:gray_bed",
    "minecraft:white_bed",
    "minecraft:blue_bed",
    "minecraft:yellow_bed",
}
STORAGE_BLOCK_IDS = {"minecraft:barrel", "minecraft:bookshelf"}
SURFACE_BLOCK_IDS = {
    "minecraft:birch_slab",
    "minecraft:spruce_slab",
    "minecraft:oak_slab",
    "minecraft:bamboo_slab",
    "minecraft:quartz_slab",
    "minecraft:smooth_stone_slab",
    "minecraft:dark_oak_planks",
}
SOFA_BLOCK_IDS = {
    "minecraft:gray_wool",
    "minecraft:spruce_planks",
    "minecraft:bamboo_planks",
    "minecraft:white_concrete",
    "minecraft:gray_concrete",
    "minecraft:birch_planks",
}
KITCHEN_COUNTER_IDS = {
    "minecraft:smooth_stone",
    "minecraft:white_concrete",
    "minecraft:gray_concrete",
    "minecraft:bamboo_planks",
    "minecraft:birch_planks",
    "minecraft:stone_bricks",
}
CARPET_BLOCK_IDS = {
    "minecraft:gray_carpet",
    "minecraft:light_gray_carpet",
    "minecraft:white_carpet",
    "minecraft:green_carpet",
    "minecraft:blue_carpet",
    "minecraft:light_blue_carpet",
    "minecraft:lime_carpet",
    "minecraft:yellow_carpet",
}


def _sample_bedroom_cell(
    *,
    cell_index: tuple[int, int, int] = (0, 0, 0),
) -> SemanticCell:
    return SemanticCell(
        cell_index=cell_index,
        voxel_bbox=((0, 0, 0), (7, 5, 7)),
        label="bedroom",
        role="habitable",
        occupancy_capacity=2,
        daylight_score=1.0,
        privacy_depth=3,
        door_faces=("south",),
        window_faces=("north", "east"),
        interior_volume_voxels=144,
        pod_template_id=f"bedroom@{cell_index[0]}_{cell_index[1]}_{cell_index[2]}",
    )


def _sample_room_cell(label: str, *, role: str = "habitable") -> SemanticCell:
    return SemanticCell(
        cell_index=(0, 0, 0),
        voxel_bbox=((0, 0, 0), (9, 5, 9)),
        label=label,
        role=role,  # type: ignore[arg-type]
        occupancy_capacity=2 if role == "habitable" else 0,
        daylight_score=1.0,
        privacy_depth=2,
        door_faces=("south",),
        window_faces=("north", "east"),
        interior_volume_voxels=256,
        pod_template_id=f"{label}@0",
    )


def test_room_layout_plan_places_core_components() -> None:
    request = make_room_request(_sample_bedroom_cell(), utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    assert layout.placements
    assert any(item.keyword == "bed_core" for item in layout.placements)
    assert any(item.category == "lighting" for item in layout.placements)


def test_room_layout_plan_avoids_solid_furniture_overlap() -> None:
    request = make_room_request(_sample_bedroom_cell(), utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    solid_cells: set[tuple[int, int]] = set()
    for item in layout.placements:
        if item.category == "lighting" or "carpet" in item.block_id or item.keyword == "mirror":
            continue
        x, _, z = item.origin
        fx, fz = item.footprint
        cells = {(ix, iz) for ix in range(x, x + fx) for iz in range(z, z + fz)}
        assert solid_cells.isdisjoint(cells)
        solid_cells.update(cells)


def test_bedroom_layout_keeps_desk_on_window_wall_and_bedside_near_bed() -> None:
    request = make_room_request(_sample_bedroom_cell(), utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    placements = {item.keyword: item for item in layout.placements}
    bed = placements["bed_core"]
    bedside = placements["bedside"]
    desk = placements["desk"]
    bed_cells = {
        (ix, iz)
        for ix in range(bed.origin[0], bed.origin[0] + bed.footprint[0])
        for iz in range(bed.origin[2], bed.origin[2] + bed.footprint[1])
    }
    bedside_cells = {
        (ix, iz)
        for ix in range(bedside.origin[0], bedside.origin[0] + bedside.footprint[0])
        for iz in range(bedside.origin[2], bedside.origin[2] + bedside.footprint[1])
    }
    assert any(
        abs(x - tx) + abs(z - tz) == 1
        for x, z in bedside_cells
        for tx, tz in bed_cells
    )
    assert desk.origin[2] == 1 or desk.origin[0] + desk.footprint[0] - 1 == layout.interior_size[0]


def test_bedroom_component_blocks_emit_readable_furniture_assemblies() -> None:
    request = make_room_request(_sample_bedroom_cell(), utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)

    blocks = _component_blocks(layout)
    ids = {block["id"] for block in blocks}
    occupied_positions = {
        (int(block["x"]), int(block["y"]), int(block["z"]))
        for block in blocks
    }
    ix, iy, iz = layout.interior_size

    assert len(blocks) > len(layout.placements)
    assert len(occupied_positions) == len(blocks)
    assert ids & BED_BLOCK_IDS
    assert ids & STORAGE_BLOCK_IDS
    assert "minecraft:oak_stairs" in ids
    assert ids & CARPET_BLOCK_IDS
    assert ids & {"minecraft:lantern", "minecraft:sea_lantern"}
    assert all(1 <= int(block["x"]) <= ix for block in blocks)
    assert all(1 <= int(block["y"]) <= iy for block in blocks)
    assert all(1 <= int(block["z"]) <= iz for block in blocks)
    assert any(
        block["id"] in BED_BLOCK_IDS
        and block.get("properties", {}).get("part") == "head"
        for block in blocks
    )
    assert any(
        block["id"] == "minecraft:oak_stairs" and "properties" in block
        for block in blocks
    )


def test_repeated_bedroom_cells_emit_stable_visual_variants() -> None:
    cells = tuple(
        _sample_bedroom_cell(cell_index=(index, 0, 0))
        for index in range(3)
    )
    layouts = []
    carpet_palettes = []
    for cell in cells:
        request = make_room_request(cell, utility_type="residential")
        plan = plan_room(request)
        layout = plan_room_layout(plan, request)
        layouts.append(layout)
        blocks = _component_blocks(layout)
        carpet_palettes.append(
            frozenset(
                block["id"]
                for block in blocks
                if str(block["id"]).endswith("_carpet")
            )
        )

    assert {layout_variant_index(layout) for layout in layouts} == {0, 1, 2}
    assert len(set(carpet_palettes)) == 3

    request = make_room_request(cells[1], utility_type="residential")
    assert f"variant={room_variant_index(request)}" in room_cache_key(request)


def test_bedroom_cache_preserves_block_properties_after_translation() -> None:
    cache = RoomInteriorCache()
    room = cache.build_for_cell(_sample_bedroom_cell(), utility_type="residential")

    bed_blocks = [block for block in room.blocks if block["id"] in BED_BLOCK_IDS]
    chair_blocks = [block for block in room.blocks if block["id"] == "minecraft:oak_stairs"]

    assert bed_blocks
    assert chair_blocks
    assert any(block.get("properties", {}).get("part") == "head" for block in bed_blocks)
    assert all("properties" in block for block in chair_blocks)


def test_room_cache_variant_seed_selects_distinct_style_profiles() -> None:
    cell = _sample_bedroom_cell()
    rustic = RoomInteriorCache(variant_seed=43).build_for_cell(
        cell,
        utility_type="residential",
    )
    industrial = RoomInteriorCache(variant_seed=45).build_for_cell(
        cell,
        utility_type="residential",
    )

    rustic_ids = {block["id"] for block in rustic.blocks}
    industrial_ids = {block["id"] for block in industrial.blocks}

    assert "rustic_cabin" in rustic.variant_id
    assert "industrial_loft" in industrial.variant_id
    assert rustic_ids & BED_BLOCK_IDS
    assert industrial_ids & BED_BLOCK_IDS
    assert rustic_ids != industrial_ids


def test_common_room_types_emit_richer_interior_assemblies() -> None:
    expected_groups_by_room = {
        "living": (SOFA_BLOCK_IDS, SURFACE_BLOCK_IDS, STORAGE_BLOCK_IDS),
        "kitchen": (
            KITCHEN_COUNTER_IDS,
            {"minecraft:cauldron"},
            {"minecraft:furnace"},
        ),
        "bathroom": (
            {"minecraft:quartz_stairs"},
            {"minecraft:light_blue_stained_glass", "minecraft:glass_pane"},
        ),
        "entry": (SURFACE_BLOCK_IDS, STORAGE_BLOCK_IDS),
        "corridor": (CARPET_BLOCK_IDS, STORAGE_BLOCK_IDS, {"minecraft:wall_torch"}),
    }
    role_by_room = {
        "entry": "circulation",
        "corridor": "circulation",
    }

    for room_type, expected_groups in expected_groups_by_room.items():
        cell = _sample_room_cell(room_type, role=role_by_room.get(room_type, "habitable"))
        request = make_room_request(cell, utility_type="residential")
        plan = plan_room(request)
        layout = plan_room_layout(plan, request)
        blocks = _component_blocks(layout)
        ids = {block["id"] for block in blocks}
        occupied_positions = {
            (int(block["x"]), int(block["y"]), int(block["z"]))
            for block in blocks
        }
        ix, iy, iz = layout.interior_size

        for expected_ids in expected_groups:
            assert expected_ids & ids, room_type
        assert len(blocks) > len(layout.placements), room_type
        assert len(occupied_positions) == len(blocks), room_type
        assert all(1 <= int(block["x"]) <= ix for block in blocks), room_type
        assert all(1 <= int(block["y"]) <= iy for block in blocks), room_type
        assert all(1 <= int(block["z"]) <= iz for block in blocks), room_type


def test_room_layout_report_is_written(tmp_path: Path) -> None:
    request = make_room_request(_sample_bedroom_cell(), utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    analysis = analyse_room_layout(layout)
    assert analysis.component_count >= 1
    report = save_room_layout_report(layout, tmp_path / "room_layout.png")
    assert report.exists()
    assert report.stat().st_size > 0


def test_stairwell_layout_uses_fixed_stair_geometry() -> None:
    cell = SemanticCell(
        cell_index=(0, 0, 0),
        voxel_bbox=((0, 0, 0), (7, 5, 7)),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=1,
        door_faces=("south",),
        window_faces=(),
        open_faces=("up",),
        opening_pattern="edge_only",
        interior_volume_voxels=144,
        pod_template_id="stairwell@0",
    )
    request = make_room_request(cell, utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    keywords = {item.keyword for item in layout.placements}
    assert "stair_flight" in keywords
    assert "landing_light" in keywords
    stair = next(item for item in layout.placements if item.keyword == "stair_flight")
    assert stair.footprint[0] >= 2
    assert stair.footprint[1] >= 2


def test_top_storey_stairwell_layout_omits_local_platforms() -> None:
    cell = SemanticCell(
        cell_index=(0, 2, 0),
        voxel_bbox=((0, 12, 0), (7, 17, 7)),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=3,
        door_faces=("south",),
        window_faces=(),
        open_faces=("down",),
        opening_pattern="edge_only",
        interior_volume_voxels=144,
        pod_template_id="stairwell@2",
    )
    request = make_room_request(cell, utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)

    assert {item.keyword for item in layout.placements} == {"landing_light"}
