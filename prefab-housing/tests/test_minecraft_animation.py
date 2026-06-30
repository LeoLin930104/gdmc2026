from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prefab_housing.minecraft_animation import (
    SettlementBuildSlot,
    _append_only_residential_plan,
    _locked_upgrade_cells_from_plan,
    build_residential_upgrade_sequence,
    build_upgrade_diffs,
    compute_bounding_box,
    export_residential_upgrade_package,
    export_residential_upgrade_sequence,
    iter_batches,
    load_residential_upgrade_package,
    plan_residential_settlement_placements,
    plan_typed_residential_settlement_placements,
    place_residential_upgrade_in_slot,
    residential_state_blocks,
    semantic_block_to_blueprint,
)
from prefab_housing.interior import ROOM_STYLE_VARIANTS, room_interior_style_profile


_PACKAGE_CACHE = (
    Path(__file__).resolve().parents[1]
    / "production_cache"
    / "residential_upgrade"
    / "seed_043.pbp"
)
_PACKAGE_CACHE_ALT = (
    Path(__file__).resolve().parents[1]
    / "production_cache"
    / "residential_upgrade"
    / "seed_045.pbp"
)
_PACKAGE_CACHE_ROW = (
    Path(__file__).resolve().parents[1]
    / "production_cache"
    / "residential_upgrade"
    / "seed_047.pbp"
)
_PACKAGE_CACHE_ROW_PARTIAL = (
    Path(__file__).resolve().parents[1]
    / "production_cache"
    / "residential_upgrade"
    / "seed_050.pbp"
)
_PACKAGE_CACHE_MODERN = (
    Path(__file__).resolve().parents[1]
    / "production_cache"
    / "residential_upgrade"
    / "seed_044.pbp"
)


def _grid(blocks: list[dict[str, Any]]) -> dict[tuple[int, int, int], tuple[str, tuple]]:
    return {
        (block["dx"], block["dy"], block["dz"]): (
            block["id"],
            tuple(sorted(block.get("props", {}).items())),
        )
        for block in blocks
        if block["id"] != "minecraft:air"
    }


def _block_signatures(
    blocks: list[dict[str, Any]],
) -> list[tuple[int, int, int, str, tuple[tuple[str, str], ...]]]:
    return [
        (
            int(block["dx"]),
            int(block["dy"]),
            int(block["dz"]),
            str(block["id"]),
            tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in block.get("props", {}).items()
                )
            ),
        )
        for block in blocks
    ]


def _apply_diff(
    before: list[dict[str, Any]],
    diff: list[dict[str, Any]],
) -> dict[tuple[int, int, int], tuple[str, tuple]]:
    grid = _grid(before)
    for block in diff:
        pos = (block["dx"], block["dy"], block["dz"])
        if block["id"] == "minecraft:air":
            grid.pop(pos, None)
        else:
            grid[pos] = (block["id"], tuple(sorted(block.get("props", {}).items())))
    return grid


def _assert_inside_slot(placement: Any) -> None:
    min_x, min_y, min_z, max_x, _max_y, max_z = placement.bbox
    slot = placement.slot
    assert min_x >= slot.x
    assert min_y == slot.y
    assert min_z >= slot.z
    assert max_x < slot.x + slot.width
    assert max_z < slot.z + slot.depth


def test_semantic_block_to_blueprint_flattens_nested_block_properties() -> None:
    blueprint = semantic_block_to_blueprint(
        {
            "x": 1,
            "y": 2,
            "z": 3,
            "id": "minecraft:oak_stairs",
            "properties": {"facing": "north", "half": "bottom"},
            "props": {"shape": "straight"},
            "waterlogged": False,
        },
        offset=(10, 20, 30),
    )

    assert blueprint == {
        "dx": 11,
        "dy": 22,
        "dz": 33,
        "id": "minecraft:oak_stairs",
        "props": {
            "facing": "north",
            "half": "bottom",
            "shape": "straight",
            "waterlogged": "false",
        },
    }
    assert "properties" not in blueprint["props"]


def test_residential_package_places_core_building_module_in_exact_slot() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    slot = SettlementBuildSlot(x=65, y=103, z=35, width=32, depth=22, cell_id=11)

    placement = place_residential_upgrade_in_slot(
        states,
        slot,
        target_entrance_face="west",
    )

    assert placement.level == 3
    assert placement.entrance_face == "west"
    assert placement.rotation_steps == 0
    assert compute_bounding_box(placement.blocks) == placement.bbox
    assert len(placement.blocks) == len(residential_state_blocks(states[-1]))
    _assert_inside_slot(placement)


def test_residential_package_rotates_building_module_to_fit_slot() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    slot = SettlementBuildSlot(x=12, y=90, z=40, width=22, depth=32, cell_id=99)

    placement = place_residential_upgrade_in_slot(
        states,
        slot,
        target_entrance_face="north",
    )

    assert placement.level == 3
    assert placement.entrance_face == "north"
    assert placement.rotation_steps == 1
    assert compute_bounding_box(placement.blocks) == placement.bbox
    _assert_inside_slot(placement)


def test_residential_package_uses_largest_level_that_fits_slot() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    slot = SettlementBuildSlot.from_mapping(
        {"x": 32, "y": 94, "z": 66, "width": 28, "depth": 28, "cell_id": 19}
    )

    placement = place_residential_upgrade_in_slot(states, slot)

    assert placement.level == 1
    assert placement.slot.cell_id == 19
    assert len(placement.blocks) == len(residential_state_blocks(states[0]))
    _assert_inside_slot(placement)


def test_residential_package_places_bed_feet_before_heads() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    ordered_blocks = residential_state_blocks(states[-1], block_mode="full")

    bed_parts_by_y: dict[int, list[str]] = {}
    for block in ordered_blocks:
        if not str(block["id"]).endswith("_bed"):
            continue
        bed_parts_by_y.setdefault(int(block["dy"]), []).append(
            str(block.get("props", {}).get("part"))
        )

    assert bed_parts_by_y
    for bed_parts in bed_parts_by_y.values():
        assert bed_parts == sorted(bed_parts, key={"foot": 0, "head": 1}.__getitem__)


def test_residential_package_rejects_too_small_slot() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    slot = SettlementBuildSlot(x=0, y=64, z=0, width=8, depth=8)

    with pytest.raises(ValueError, match="no valid residential upgrade level fits"):
        place_residential_upgrade_in_slot(states, slot)


def test_residential_package_plans_strict_same_type_slots() -> None:
    states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    slots = (
        SettlementBuildSlot(x=0, y=70, z=0, width=32, depth=22, cell_id=1),
        SettlementBuildSlot(x=40, y=70, z=0, width=28, depth=28, cell_id=2),
        SettlementBuildSlot(x=80, y=70, z=0, width=8, depth=8, cell_id=3),
    )

    plan = plan_residential_settlement_placements(
        states,
        slots,
        target_entrance_face="west",
    )

    assert not plan.is_complete
    assert [placement.slot.cell_id for placement in plan.placements] == [1, 2]
    assert [placement.level for placement in plan.placements] == [3, 1]
    assert [placement.slot.building_type for placement in plan.placements] == [
        "residential",
        "residential",
    ]
    assert len(plan.rejections) == 1
    assert plan.rejections[0].slot.cell_id == 3
    assert "no valid residential upgrade level fits" in plan.rejections[0].reason
    for placement in plan.placements:
        _assert_inside_slot(placement)


def test_residential_package_plans_typed_zoning_selection() -> None:
    rustic_states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    loft_states, _diffs_alt, _manifest_alt = load_residential_upgrade_package(
        _PACKAGE_CACHE_ALT
    )
    slots = (
        SettlementBuildSlot(
            x=0,
            y=70,
            z=0,
            width=32,
            depth=22,
            cell_id=1,
            zone_id=10,
            building_type="residential",
        ),
        SettlementBuildSlot(
            x=40,
            y=70,
            z=0,
            width=32,
            depth=22,
            cell_id=2,
            zone_id=11,
            building_type="worker_housing",
        ),
        SettlementBuildSlot(
            x=80,
            y=70,
            z=0,
            width=32,
            depth=22,
            cell_id=3,
            zone_id=12,
            building_type="market",
        ),
    )

    plan = plan_typed_residential_settlement_placements(
        {
            "residential": rustic_states,
            "worker_housing": loft_states,
        },
        slots,
    )

    assert not plan.is_complete
    assert [placement.slot.zone_id for placement in plan.placements] == [10, 11]
    assert [placement.slot.building_type for placement in plan.placements] == [
        "residential",
        "worker_housing",
    ]
    assert [placement.state.interior_style_id for placement in plan.placements] == [
        "rustic_cabin+room_variants",
        "industrial_loft+room_variants",
    ]
    assert len(plan.rejections) == 1
    assert plan.rejections[0].slot.building_type == "market"
    assert "building type 'market'" in plan.rejections[0].reason
    for placement in plan.placements:
        _assert_inside_slot(placement)


def test_residential_package_cycles_variants_within_one_building_type() -> None:
    rustic_states, _diffs, _manifest = load_residential_upgrade_package(_PACKAGE_CACHE)
    modern_states, _diffs_modern, _manifest_modern = load_residential_upgrade_package(
        _PACKAGE_CACHE_MODERN
    )
    loft_states, _diffs_loft, _manifest_loft = load_residential_upgrade_package(
        _PACKAGE_CACHE_ALT
    )
    slots = tuple(
        SettlementBuildSlot(
            x=index * 40,
            y=70,
            z=0,
            width=32,
            depth=32,
            cell_id=index,
            building_type="residential",
        )
        for index in range(3)
    )

    plan = plan_typed_residential_settlement_placements(
        {
            "residential": (rustic_states, modern_states, loft_states),
        },
        slots,
    )

    assert plan.is_complete
    assert [placement.state.seed for placement in plan.placements] == [43, 44, 45]
    assert len({placement.state.layout_variant_id for placement in plan.placements}) == 3
    for placement in plan.placements:
        _assert_inside_slot(placement)


def test_typed_planner_prefers_highest_level_variant_that_fits() -> None:
    full_states, _diffs_full, _manifest_full = load_residential_upgrade_package(
        _PACKAGE_CACHE_ROW
    )
    partial_states, _diffs_partial, _manifest_partial = load_residential_upgrade_package(
        _PACKAGE_CACHE_ROW_PARTIAL
    )
    slots = tuple(
        SettlementBuildSlot(
            x=index * 40,
            y=70,
            z=0,
            width=36,
            depth=36,
            cell_id=index,
            building_type="row_house",
        )
        for index in range(2)
    )

    plan = plan_typed_residential_settlement_placements(
        {
            "row_house": (full_states, partial_states),
        },
        slots,
    )

    assert plan.is_complete
    assert [placement.level for placement in plan.placements] == [3, 3]
    assert [placement.state.seed for placement in plan.placements] == [47, 47]
    for placement in plan.placements:
        _assert_inside_slot(placement)


def test_residential_upgrade_sequence_diffs_reconstruct_next_level() -> None:
    states = build_residential_upgrade_sequence(seed=42)
    diffs = build_upgrade_diffs(states)

    assert [state.level for state in states] == [1, 2, 3]
    assert [diff.from_level for diff in diffs] == [1, 2]
    assert all(state.structure_blocks for state in states)
    assert len({state.wall_face_preset for state in states}) == 1
    assert len({state.wall_face_design_path for state in states}) == 1
    assert [state.entrance_face for state in states] == ["west", "west", "west"]
    assert [state.interior_style_id for state in states] == [
        "classic_modular+room_variants",
        "classic_modular+room_variants",
        "classic_modular+room_variants",
    ]
    assert [state.layout_variant_id for state in states] == [
        "wfc_mcts_append_seed_42",
        "wfc_mcts_append_seed_42",
        "wfc_mcts_append_seed_42",
    ]
    assert all(len(state.structure_blocks) < len(state.blocks) for state in states)
    assert all(state.core_blocks for state in states)
    assert all(state.wall_face_blocks for state in states)
    assert all(len(state.core_blocks) < len(state.blocks) for state in states)
    assert all(
        not any(
            block["id"] == "minecraft:light_gray_stained_glass_pane"
            for block in state.core_blocks
        )
        for state in states
    )
    assert all(state.valid for state in states)
    for state in states:
        block_ids = {block["id"] for block in state.blocks}
        assert any(block_id.endswith("_bed") for block_id in block_ids)
        assert {"minecraft:lantern", "minecraft:sea_lantern"} & block_ids

    for before, after, diff in zip(states, states[1:], diffs, strict=False):
        before_grid = _grid(before.blocks)
        after_grid = _grid(after.blocks)
        before_beds = {
            pos: sig
            for pos, sig in before_grid.items()
            if sig[0].endswith("_bed")
        }
        assert diff.blocks
        assert any(block["id"] == "minecraft:air" for block in diff.blocks)
        assert before_beds.items() <= after_grid.items()
        assert _apply_diff(before.blocks, diff.blocks) == _grid(after.blocks)


def test_residential_upgrade_preserves_locked_room_cells() -> None:
    locked_cells = {}
    entry_cell: tuple[int, int, int] | None = None

    for level in (1, 2, 3):
        plan = _append_only_residential_plan(
            level,
            seed=42,
            material_theme="sci_fi_modular",
            locked_cells=locked_cells,
        )
        current_labels = {
            cell.cell_index: cell.label
            for cell in plan.cells
            if not cell.is_empty
        }

        for cell_index, locked in locked_cells.items():
            assert current_labels[cell_index] == locked.label

        entry_cells = [
            cell_index
            for cell_index, label in current_labels.items()
            if label == "entry"
        ]
        if entry_cell is None:
            assert len(entry_cells) == 1
            entry_cell = entry_cells[0]
        assert entry_cell in entry_cells

        locked_cells = _locked_upgrade_cells_from_plan(plan)


def test_residential_upgrade_diff_places_bed_foot_before_head() -> None:
    diffs = [
        diff
        for seed in (42, 43, 45)
        for diff in build_upgrade_diffs(build_residential_upgrade_sequence(seed=seed))
    ]

    for diff in diffs:
        for batch in iter_batches(diff.blocks, strategy="y_up"):
            beds = [block for block in batch if block["id"].endswith("_bed")]
            if not beds:
                continue
            parts = [block["props"]["part"] for block in beds]
            assert parts == sorted(parts, key={"foot": 0, "head": 1}.__getitem__)


def test_llm_room_style_catalogue_has_three_variants_per_room_type() -> None:
    expected = {"bedroom", "living", "kitchen", "bathroom", "entry", "corridor", "stairwell"}

    assert expected <= set(ROOM_STYLE_VARIANTS)
    assert all(len(ROOM_STYLE_VARIANTS[room_type]) == 3 for room_type in expected)
    assert {
        room_interior_style_profile("bedroom", variant_seed=42, variant_index=index).bed_block
        for index in range(3)
    } == {
        "minecraft:white_bed",
        "minecraft:yellow_bed",
        "minecraft:red_bed",
    }


def test_residential_upgrade_seed_changes_interior_style_payload() -> None:
    rustic = build_residential_upgrade_sequence(seed=43)
    industrial = build_residential_upgrade_sequence(seed=45)

    rustic_ids = {block["id"] for block in rustic[0].blocks}
    industrial_ids = {block["id"] for block in industrial[0].blocks}

    assert rustic[0].interior_style_id == "rustic_cabin+room_variants"
    assert industrial[0].interior_style_id == "industrial_loft+room_variants"
    assert any(block_id.endswith("_bed") for block_id in rustic_ids)
    assert any(block_id.endswith("_bed") for block_id in industrial_ids)
    assert _block_signatures(rustic[-1].blocks) != _block_signatures(industrial[-1].blocks)


def test_residential_upgrade_seed_changes_exterior_layout_payload() -> None:
    states_by_seed = {
        seed: build_residential_upgrade_sequence(seed=seed)
        for seed in (43, 45, 47)
    }

    assert [states[0].layout_variant_id for states in states_by_seed.values()] == [
        "wfc_mcts_append_seed_43",
        "wfc_mcts_append_seed_45",
        "wfc_mcts_append_seed_47",
    ]
    assert all(state.valid for states in states_by_seed.values() for state in states)
    assert all(
        [state.entrance_face for state in states] == ["west", "west", "west"]
        for states in states_by_seed.values()
    )
    assert len(
        {
            tuple(_block_signatures(state[-1].structure_blocks))
            for state in states_by_seed.values()
        }
    ) == 3


def test_residential_upgrade_export_writes_structure_cache(tmp_path: Path) -> None:
    states = build_residential_upgrade_sequence(seed=42)
    manifest = export_residential_upgrade_sequence(states, tmp_path)

    assert len(manifest["levels"]) == 3
    for record in manifest["levels"]:
        level_path = Path(record["path"])
        cache_path = Path(record["structure_cache_path"])
        assert level_path.exists()
        assert cache_path.exists()
        assert record["wall_face_preset"]
        assert record["entrance_face"] == "west"
        assert record["interior_style_id"] == "classic_modular+room_variants"
        assert record["layout_variant_id"] == "wfc_mcts_append_seed_42"
        assert record["structure_block_count"] < record["block_count"]
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload["meta"]["kind"] == "residential_structure_template_cache"
        assert payload["meta"]["wall_face_preset"] == record["wall_face_preset"]
        assert payload["meta"]["entrance_face"] == record["entrance_face"]
        assert payload["meta"]["interior_style_id"] == record["interior_style_id"]
        assert payload["meta"]["layout_variant_id"] == record["layout_variant_id"]
        assert payload["blocks"]


def test_residential_upgrade_package_round_trips_levels_and_diffs(tmp_path: Path) -> None:
    states = build_residential_upgrade_sequence(seed=42)
    package_path = tmp_path / "residential_upgrade_seed_042.pbp"
    manifest = export_residential_upgrade_package(states, package_path)
    loaded_states, loaded_diffs, loaded_manifest = load_residential_upgrade_package(package_path)

    assert package_path.exists()
    assert package_path.stat().st_size < 150_000
    assert loaded_manifest["format"] == "prefab-housing-blueprint-package-v1"
    assert len(loaded_manifest["palette"]) == len(manifest["palette"])
    assert [state.entrance_face for state in loaded_states] == [
        state.entrance_face for state in states
    ]
    assert [state.interior_style_id for state in loaded_states] == [
        state.interior_style_id for state in states
    ]
    assert [state.layout_variant_id for state in loaded_states] == [
        state.layout_variant_id for state in states
    ]
    assert [_block_signatures(state.blocks) for state in loaded_states] == [
        _block_signatures(state.blocks) for state in states
    ]
    assert [_block_signatures(state.structure_blocks) for state in loaded_states] == [
        _block_signatures(state.structure_blocks) for state in states
    ]
    assert [_block_signatures(state.core_blocks) for state in loaded_states] == [
        _block_signatures(state.core_blocks) for state in states
    ]
    assert [_block_signatures(state.wall_face_blocks) for state in loaded_states] == [
        _block_signatures(state.wall_face_blocks) for state in states
    ]
    for record in loaded_manifest["levels"]:
        assert record["wall_face_policy"] == "separate_swappable_section"
        assert record["core_section"]
        assert record["wall_face_section"]
        assert record["core_block_count"] < record["block_count"]
        assert record["wall_face_block_count"] > 0
    assert [_block_signatures(diff.blocks) for diff in loaded_diffs] == [
        _block_signatures(diff.blocks) for diff in build_upgrade_diffs(states)
    ]
