from __future__ import annotations

import numpy as np
import pytest

from prefab_housing.town_lighting import (
    ROAD_EMBED_BLOCKS,
    REVERSE_SWEEP_FIXTURE_KIND,
    TownLightingConfig,
    plan_reverse_sweep_lighting,
    plan_town_lighting,
)


def _base_arrays(shape: tuple[int, int] = (32, 32)) -> dict[str, np.ndarray]:
    heightmap = np.full(shape, 70, dtype=np.int32)
    return {
        "heightmap": heightmap,
        "core_mask": np.ones(shape, dtype=bool),
        "path_mask": np.zeros(shape, dtype=bool),
        "path_base_y": heightmap.copy(),
        "path_slab_mask": np.zeros(shape, dtype=bool),
        "farm_mask": np.zeros(shape, dtype=bool),
        "building_mask": np.zeros(shape, dtype=bool),
    }


def _has_adjacent(mask: np.ndarray, local_x: int, local_z: int) -> bool:
    depth, width = mask.shape
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = local_x + dx
        z = local_z + dz
        if 0 <= x < width and 0 <= z < depth and mask[z, x]:
            return True
    return False


def _assert_min_spacing(fixtures: tuple, spacing: int) -> None:
    for index, first in enumerate(fixtures):
        for second in fixtures[index + 1 :]:
            distance_sq = (
                (first.local_x - second.local_x) ** 2
                + (first.local_z - second.local_z) ** 2
            )
            assert distance_sq >= spacing * spacing


def _assert_covered(fixtures: tuple, mask: np.ndarray, radius: int) -> None:
    for local_z, local_x in np.argwhere(mask):
        assert any(
            abs(int(local_x) - fixture.local_x)
            + abs(int(local_z) - fixture.local_z)
            <= radius
            for fixture in fixtures
        )


def _block_lookup(blocks: dict[tuple[int, int, int], str]):
    def block_at(local_x: int, y: int, local_z: int) -> str:
        return blocks.get((local_x, y, local_z), "minecraft:air")

    return block_at


def _flat_floor_blocks(
    shape: tuple[int, int],
    *,
    y: int = 64,
    block_id: str = "minecraft:grass_block",
) -> dict[tuple[int, int, int], str]:
    depth, width = shape
    return {
        (local_x, y, local_z): block_id
        for local_z in range(depth)
        for local_x in range(width)
    }


def test_road_lighting_uses_path_edges_without_replacing_reserved_cells() -> None:
    arrays = _base_arrays()
    arrays["path_mask"][16, 2:30] = True
    arrays["building_mask"][15, 12] = True

    plan = plan_town_lighting(
        **arrays,
        config=TownLightingConfig(
            seed=21,
            road_spacing=6,
            farm_spacing=12,
            max_road_fixtures=8,
            max_farm_fixtures=0,
            max_coverage_fixtures=4,
        ),
    )

    road_posts = tuple(
        fixture for fixture in plan.fixtures if fixture.kind == "road_post"
    )
    assert road_posts
    _assert_min_spacing(road_posts, 6)
    for fixture in road_posts:
        assert _has_adjacent(arrays["path_mask"], fixture.local_x, fixture.local_z)
        assert not arrays["path_mask"][fixture.local_z, fixture.local_x]
        assert not arrays["building_mask"][fixture.local_z, fixture.local_x]
        assert [block.y for block in fixture.blocks] == sorted(
            block.y for block in fixture.blocks
        )
        assert fixture.blocks[-1].block_id == "minecraft:lantern"


def test_farm_lighting_uses_external_farm_edges_and_organic_variants() -> None:
    arrays = _base_arrays()
    arrays["farm_mask"][10:19, 10:19] = True

    plan = plan_town_lighting(
        **arrays,
        config=TownLightingConfig(
            seed=34,
            road_spacing=8,
            farm_spacing=7,
            max_road_fixtures=0,
            max_farm_fixtures=8,
        ),
    )

    assert plan.fixtures
    farm_fixtures = tuple(
        fixture for fixture in plan.fixtures if fixture.kind.startswith("farm_")
    )
    assert farm_fixtures
    _assert_min_spacing(farm_fixtures, 7)
    assert {
        fixture.kind for fixture in farm_fixtures
    }.issubset({"farm_lantern", "farm_shroomlight"})
    for fixture in farm_fixtures:
        assert _has_adjacent(arrays["farm_mask"], fixture.local_x, fixture.local_z)
        assert not arrays["farm_mask"][fixture.local_z, fixture.local_x]
        assert fixture.blocks[0].block_id == "minecraft:oak_fence"
        assert fixture.blocks[-1].block_id in {
            "minecraft:lantern",
            "minecraft:shroomlight",
        }


def test_blocked_mask_prevents_lighting_candidates() -> None:
    arrays = _base_arrays((12, 12))
    arrays["core_mask"][:] = False
    arrays["core_mask"][5:8, :] = True
    arrays["path_mask"][6, :] = True
    blocked = np.zeros((12, 12), dtype=bool)
    blocked[5:8, :] = True

    plan = plan_town_lighting(
        **arrays,
        blocked_mask=blocked,
        config=TownLightingConfig(
            seed=1,
            road_spacing=3,
            farm_spacing=3,
            max_road_fixtures=0,
            max_farm_fixtures=0,
        ),
    )

    assert plan.fixtures == ()


def test_road_embeds_replace_road_surface_at_configured_frequency() -> None:
    arrays = _base_arrays((16, 32))
    arrays["path_mask"][8, 2:30] = True

    plan = plan_town_lighting(
        **arrays,
        config=TownLightingConfig(
            seed=12,
            road_spacing=20,
            road_embed_spacing=5,
            farm_spacing=20,
            coverage_spacing=20,
            max_road_fixtures=2,
            max_road_embeds=6,
            max_farm_fixtures=0,
            max_coverage_fixtures=2,
        ),
    )

    embeds = tuple(
        fixture for fixture in plan.fixtures if fixture.kind == "road_embed"
    )
    assert 1 <= len(embeds) <= 6
    _assert_min_spacing(embeds, 5)
    for fixture in embeds:
        assert arrays["path_mask"][fixture.local_z, fixture.local_x]
        assert len(fixture.blocks) == 1
        assert fixture.blocks[0].y == arrays["path_base_y"][
            fixture.local_z,
            fixture.local_x,
        ]
        assert fixture.blocks[0].block_id in ROAD_EMBED_BLOCKS


def test_coverage_lights_cover_open_core_gaps() -> None:
    arrays = _base_arrays((20, 20))

    plan = plan_town_lighting(
        **arrays,
        config=TownLightingConfig(
            seed=5,
            road_spacing=20,
            road_embed_spacing=20,
            farm_spacing=20,
            coverage_spacing=8,
            coverage_radius=6,
            max_road_fixtures=0,
            max_road_embeds=0,
            max_farm_fixtures=0,
            max_coverage_fixtures=0,
        ),
    )

    coverage_lights = tuple(
        fixture for fixture in plan.fixtures if fixture.kind == "coverage_light"
    )
    assert coverage_lights
    _assert_covered(coverage_lights, arrays["core_mask"], radius=6)
    for fixture in coverage_lights:
        assert fixture.blocks[0].block_id == "minecraft:light"
        assert fixture.blocks[0].props_dict == {"level": "15"}


def test_lighting_planner_fails_fast_on_shape_mismatch() -> None:
    arrays = _base_arrays()
    arrays["path_mask"] = np.zeros((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="path_mask shape"):
        plan_town_lighting(**arrays)


def test_reverse_sweep_consolidates_open_spawn_surfaces() -> None:
    shape = (7, 7)
    floor_y = 64
    blocks = _flat_floor_blocks(shape, y=floor_y)

    plan = plan_reverse_sweep_lighting(
        block_at=_block_lookup(blocks),
        target_mask=np.ones(shape, dtype=bool),
        min_y_by_cell=np.full(shape, floor_y, dtype=np.int32),
        max_y=floor_y,
        config=TownLightingConfig(
            reverse_sweep_min_block_light=1,
            reverse_sweep_light_level=15,
        ),
    )

    assert plan.audit.reverse_sweep_targets == 49
    assert plan.audit.reverse_sweep_uncovered == 0
    assert len(plan.fixtures) == 1
    fixture = plan.fixtures[0]
    assert fixture.kind == REVERSE_SWEEP_FIXTURE_KIND
    assert fixture.blocks[0].block_id == "minecraft:light"
    assert fixture.blocks[0].props_dict == {"level": "15"}


def test_reverse_sweep_places_separate_lights_when_blocks_stop_propagation() -> None:
    shape = (1, 3)
    floor_y = 64
    blocks = _flat_floor_blocks(shape, y=floor_y)
    blocks[(1, floor_y + 1, 0)] = "minecraft:stone"
    blocks[(1, floor_y + 2, 0)] = "minecraft:stone"

    plan = plan_reverse_sweep_lighting(
        block_at=_block_lookup(blocks),
        target_mask=np.ones(shape, dtype=bool),
        min_y_by_cell=np.full(shape, floor_y, dtype=np.int32),
        max_y=floor_y,
        config=TownLightingConfig(
            reverse_sweep_min_block_light=1,
            reverse_sweep_light_level=15,
        ),
    )

    assert plan.audit.reverse_sweep_targets == 2
    assert plan.audit.reverse_sweep_uncovered == 0
    assert len(plan.fixtures) == 2
    assert {(fixture.local_x, fixture.local_z) for fixture in plan.fixtures} == {
        (0, 0),
        (2, 0),
    }


def test_reverse_sweep_fast_path_matches_callback_scanner() -> None:
    shape = (3, 5)
    floor_y = 64
    blocks = _flat_floor_blocks(shape, y=floor_y)
    blocks[(2, floor_y + 1, 1)] = "minecraft:stone"
    blocks[(2, floor_y + 2, 1)] = "minecraft:stone"
    blocks[(4, floor_y + 1, 2)] = "minecraft:light"

    common = dict(
        block_at=_block_lookup(blocks),
        target_mask=np.ones(shape, dtype=bool),
        min_y_by_cell=np.full(shape, floor_y, dtype=np.int32),
        max_y=floor_y,
    )
    fast = plan_reverse_sweep_lighting(
        **common,
        config=TownLightingConfig(
            reverse_sweep_min_block_light=1,
            reverse_sweep_light_level=15,
            reverse_sweep_fast_path=True,
        ),
    )
    slow = plan_reverse_sweep_lighting(
        **common,
        config=TownLightingConfig(
            reverse_sweep_min_block_light=1,
            reverse_sweep_light_level=15,
            reverse_sweep_fast_path=False,
        ),
    )

    assert fast.audit == slow.audit
    assert [(fixture.local_x, fixture.ground_y, fixture.local_z) for fixture in fast.fixtures] == [
        (fixture.local_x, fixture.ground_y, fixture.local_z)
        for fixture in slow.fixtures
    ]


def test_reverse_sweep_counts_existing_light_as_coverage() -> None:
    shape = (5, 5)
    floor_y = 64
    blocks = _flat_floor_blocks(shape, y=floor_y)
    blocks[(2, floor_y + 1, 2)] = "minecraft:light"

    plan = plan_reverse_sweep_lighting(
        block_at=_block_lookup(blocks),
        target_mask=np.ones(shape, dtype=bool),
        min_y_by_cell=np.full(shape, floor_y, dtype=np.int32),
        max_y=floor_y,
        config=TownLightingConfig(
            reverse_sweep_min_block_light=1,
            reverse_sweep_light_level=15,
        ),
    )

    assert plan.audit.reverse_sweep_targets == 25
    assert plan.audit.reverse_sweep_existing_covered == 25
    assert plan.audit.reverse_sweep_uncovered == 0
    assert plan.fixtures == ()
