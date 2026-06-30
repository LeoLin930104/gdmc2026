"""End-to-end smoke test for ``prefab_housing.build_house``.

Verifies that the M1 pipeline:

1. Produces a fully-solved house for a representative single-family brief.
2. Satisfies the programme's required-pod floor (functional_adequacy == 1.0).
3. Yields connected habitables (circulation == 1.0).
4. Reaches every required pod from an entry cell via the door/open graph.
5. Is bit-exact deterministic for a fixed seed.

These are *systemic* invariants, not aesthetic targets — failures here imply
a regression in WFC, MCTS, scoring, or the public API contract.
"""

from __future__ import annotations

import pytest

from prefab_housing import Brief, build_house
from prefab_housing.types import HouseResult


# 30x30 voxel footprint = 3x3 cell footprint at the default 10-voxel cell width.
# 2 storeys keeps the search space tractable for CI while still exercising
# vertical stairwell connectivity.
_FOOTPRINT_VOXELS: tuple[int, int] = (30, 30)
_SEARCH_ITERS: int = 128


def _build_default(seed: int = 42) -> HouseResult:
    brief = Brief(
        occupant_count=3,
        household_type="single_family",
        material_theme="sci_fi_modular",
        seed=seed,
    )
    return build_house(
        brief,
        footprint_xz=_FOOTPRINT_VOXELS,
        search_iterations=_SEARCH_ITERS,
    )


def test_build_house_returns_solved_result() -> None:
    """Pipeline runs without raising and emits a non-empty HouseResult."""
    result = _build_default()
    assert isinstance(result, HouseResult)
    assert result.metadata.cell_grid_size[1] >= 2
    assert len(result.blocks) > 0
    assert len(result.exterior_blocks) > 0
    assert len(result.interior_blocks) > 0
    assert len(result.semantic_cells) > 0
    assert len(result.room_interiors) == len(result.semantic_cells)


def test_functional_and_circulation_floors() -> None:
    """Both hard-floored components must be fully satisfied at this seed."""
    result = _build_default()
    breakdown = result.metadata.score_breakdown
    assert breakdown["functional_adequacy"] == pytest.approx(1.0)
    assert breakdown["circulation"] == pytest.approx(1.0)
    # Total must reflect that no hard-floor penalty was applied.
    assert result.metadata.score_total > 0.5


def test_required_pods_present_and_reachable() -> None:
    """Every required pod-label must appear at least N times AND have a
    finite privacy_depth (i.e. be reachable from an entry cell)."""
    result = _build_default()
    by_label: dict[str, list[int]] = {}
    for cell in result.semantic_cells:
        by_label.setdefault(cell.label, []).append(cell.privacy_depth)

    expected_minimums = {
        "entry": 1,
        "living": 1,
        "kitchen": 1,
        "bathroom": 1,
        "bedroom": 2,
    }
    for label, minimum in expected_minimums.items():
        depths = by_label.get(label, [])
        assert len(depths) >= minimum, f"missing {label}: have {len(depths)}, need {minimum}"
        for d in depths:
            assert d >= 0, f"unreachable {label} cell (depth={d})"


def test_generated_house_selects_richer_cached_interiors() -> None:
    """Required residential rooms must carry room-specific block assemblies."""
    result = _build_default()
    expected_groups_by_room = {
        "bedroom": (
            {
                "minecraft:red_bed",
                "minecraft:green_bed",
                "minecraft:gray_bed",
                "minecraft:white_bed",
                "minecraft:blue_bed",
                "minecraft:yellow_bed",
            },
            {"minecraft:barrel", "minecraft:bookshelf"},
            {"minecraft:oak_stairs"},
        ),
        "living": (
            {
                "minecraft:gray_wool",
                "minecraft:spruce_planks",
                "minecraft:bamboo_planks",
                "minecraft:white_concrete",
                "minecraft:gray_concrete",
                "minecraft:birch_planks",
            },
            {
                "minecraft:birch_slab",
                "minecraft:spruce_slab",
                "minecraft:oak_slab",
                "minecraft:bamboo_slab",
                "minecraft:quartz_slab",
                "minecraft:smooth_stone_slab",
                "minecraft:dark_oak_planks",
            },
            {"minecraft:bookshelf", "minecraft:barrel"},
        ),
        "kitchen": (
            {
                "minecraft:smooth_stone",
                "minecraft:white_concrete",
                "minecraft:gray_concrete",
                "minecraft:bamboo_planks",
                "minecraft:birch_planks",
                "minecraft:stone_bricks",
            },
            {"minecraft:cauldron"},
            {"minecraft:furnace"},
        ),
        "bathroom": (
            {"minecraft:quartz_stairs"},
            {"minecraft:light_blue_stained_glass", "minecraft:glass_pane"},
        ),
        "entry": (
            {
                "minecraft:birch_slab",
                "minecraft:spruce_slab",
                "minecraft:oak_slab",
                "minecraft:bamboo_slab",
                "minecraft:quartz_slab",
                "minecraft:smooth_stone_slab",
                "minecraft:dark_oak_planks",
            },
            {"minecraft:barrel", "minecraft:bookshelf"},
        ),
    }

    for room_type, expected_groups in expected_groups_by_room.items():
        rooms = [room for room in result.room_interiors if room.room_type == room_type]
        assert rooms, f"missing generated {room_type} interior"
        ids = {block["id"] for room in rooms for block in room.blocks}
        for expected_ids in expected_groups:
            assert expected_ids & ids, f"{room_type} ids={sorted(ids)}"

    assert any("properties" in block for block in result.interior_blocks)


def test_determinism_bit_exact() -> None:
    """Two runs with identical inputs produce identical outputs."""
    a = _build_default(seed=42)
    b = _build_default(seed=42)
    assert a.blocks == b.blocks
    assert a.semantic_cells == b.semantic_cells
    assert a.metadata.score_total == b.metadata.score_total
    assert a.metadata.score_breakdown == b.metadata.score_breakdown


def test_seed_changes_output() -> None:
    """Different seeds should yield different layouts (statistical sanity)."""
    a = _build_default(seed=42)
    b = _build_default(seed=7)
    # At least one of the two diverges; if both seeds happen to converge on
    # the same global optimum the test would be over-strict, so we tolerate
    # equality on either single dimension as long as some signal differs.
    diverged = (
        a.blocks != b.blocks
        or a.semantic_cells != b.semantic_cells
        or a.metadata.score_total != b.metadata.score_total
    )
    assert diverged, "expected seed change to perturb output"
