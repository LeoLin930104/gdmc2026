"""Deterministic settlement-scale lighting plans.

The planner only emits intended fixture blocks. Live-world clearance checks stay
at the GDPC boundary so this module can be tested without a running server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

AIR_BLOCK = "minecraft:air"

SOFT_REPLACEABLE_BLOCKS: frozenset[str] = frozenset(
    {
        AIR_BLOCK,
        "minecraft:short_grass",
        "minecraft:grass",
        "minecraft:tall_grass",
        "minecraft:fern",
        "minecraft:large_fern",
        "minecraft:dead_bush",
        "minecraft:dandelion",
        "minecraft:poppy",
        "minecraft:azure_bluet",
        "minecraft:oxeye_daisy",
        "minecraft:cornflower",
        "minecraft:allium",
        "minecraft:snow",
        "minecraft:light",
    }
)

EMBEDDED_ROAD_REPLACEABLE_BLOCKS: frozenset[str] = frozenset(
    {
        "minecraft:cobblestone",
        "minecraft:stone_bricks",
        "minecraft:polished_andesite",
        "minecraft:chiseled_stone_bricks",
        "minecraft:mossy_cobblestone",
        "minecraft:dirt_path",
        "minecraft:coarse_dirt",
        "minecraft:sea_lantern",
        "minecraft:shroomlight",
        "minecraft:ochre_froglight",
        "minecraft:verdant_froglight",
        "minecraft:pearlescent_froglight",
    }
)

ROAD_EMBED_BLOCKS: tuple[str, ...] = (
    "minecraft:ochre_froglight",
    "minecraft:verdant_froglight",
    "minecraft:pearlescent_froglight",
    "minecraft:sea_lantern",
    "minecraft:shroomlight",
)

UNSAFE_SUPPORT_BLOCKS: frozenset[str] = frozenset(
    {
        AIR_BLOCK,
        "minecraft:water",
        "minecraft:lava",
        "minecraft:farmland",
        "minecraft:wheat",
        "minecraft:carrots",
        "minecraft:potatoes",
        "minecraft:beetroots",
        "minecraft:sugar_cane",
    }
)


@dataclass(frozen=True, slots=True)
class TownLightingConfig:
    seed: int = 1337
    road_spacing: int = 11
    road_embed_spacing: int = 9
    farm_spacing: int = 12
    coverage_spacing: int = 18
    coverage_radius: int = 12
    max_road_fixtures: int = 96
    max_road_embeds: int = 0
    max_farm_fixtures: int = 48
    max_coverage_fixtures: int = 768

    def __post_init__(self) -> None:
        if self.road_spacing < 1:
            raise ValueError("road_spacing must be positive")
        if self.road_embed_spacing < 1:
            raise ValueError("road_embed_spacing must be positive")
        if self.farm_spacing < 1:
            raise ValueError("farm_spacing must be positive")
        if self.coverage_spacing < 1:
            raise ValueError("coverage_spacing must be positive")
        if self.coverage_radius < 1:
            raise ValueError("coverage_radius must be positive")
        if self.max_road_fixtures < 0:
            raise ValueError("max_road_fixtures must be non-negative")
        if self.max_road_embeds < 0:
            raise ValueError("max_road_embeds must be non-negative")
        if self.max_farm_fixtures < 0:
            raise ValueError("max_farm_fixtures must be non-negative")
        if self.max_coverage_fixtures < 0:
            raise ValueError("max_coverage_fixtures must be non-negative")


@dataclass(frozen=True, slots=True)
class LightingBlock:
    local_x: int
    y: int
    local_z: int
    block_id: str
    props: tuple[tuple[str, str], ...] = ()

    @property
    def props_dict(self) -> dict[str, str]:
        return dict(self.props)


@dataclass(frozen=True, slots=True)
class LightingFixture:
    kind: str
    local_x: int
    ground_y: int
    local_z: int
    blocks: tuple[LightingBlock, ...]


@dataclass(frozen=True, slots=True)
class TownLightingPlan:
    fixtures: tuple[LightingFixture, ...]

    @property
    def block_count(self) -> int:
        return sum(len(fixture.blocks) for fixture in self.fixtures)

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fixture in self.fixtures:
            counts[fixture.kind] = counts.get(fixture.kind, 0) + 1
        return counts


@dataclass(frozen=True, slots=True)
class _Candidate:
    local_x: int
    local_z: int
    score: int
    noise: int


def plan_town_lighting(
    *,
    heightmap: np.ndarray,
    core_mask: np.ndarray,
    path_mask: np.ndarray,
    path_base_y: np.ndarray,
    path_slab_mask: np.ndarray,
    farm_mask: np.ndarray,
    building_mask: np.ndarray,
    blocked_mask: np.ndarray | None = None,
    config: TownLightingConfig | None = None,
) -> TownLightingPlan:
    """Plan road guidance and farm-edge lighting without writing blocks."""

    cfg = config or TownLightingConfig()
    heightmap = _int_array(heightmap, "heightmap")
    shape = heightmap.shape
    core_mask = _bool_array(core_mask, "core_mask", shape)
    path_mask = _bool_array(path_mask, "path_mask", shape)
    path_base_y = _int_array(path_base_y, "path_base_y", shape)
    path_slab_mask = _bool_array(path_slab_mask, "path_slab_mask", shape)
    farm_mask = _bool_array(farm_mask, "farm_mask", shape)
    building_mask = _bool_array(building_mask, "building_mask", shape)
    if blocked_mask is None:
        blocked_mask = np.zeros(shape, dtype=bool)
    else:
        blocked_mask = _bool_array(blocked_mask, "blocked_mask", shape)

    reserved_mask = path_mask | farm_mask | building_mask | blocked_mask
    open_mask = core_mask & ~reserved_mask
    road_embed_fixtures = _plan_road_embed_fixtures(
        path_mask=path_mask,
        path_base_y=path_base_y,
        path_slab_mask=path_slab_mask,
        building_mask=building_mask,
        blocked_mask=blocked_mask,
        config=cfg,
    )

    road_fixtures = _plan_road_fixtures(
        heightmap=heightmap,
        path_mask=path_mask,
        path_base_y=path_base_y,
        path_slab_mask=path_slab_mask,
        open_mask=open_mask,
        config=cfg,
    )
    farm_fixtures = _plan_farm_fixtures(
        heightmap=heightmap,
        farm_mask=farm_mask,
        path_mask=path_mask,
        open_mask=open_mask,
        config=cfg,
        existing=road_fixtures,
    )
    coverage_fixtures = _plan_coverage_fixtures(
        heightmap=heightmap,
        core_mask=core_mask,
        open_mask=open_mask,
        building_mask=building_mask,
        blocked_mask=blocked_mask,
        existing=road_embed_fixtures + road_fixtures + farm_fixtures,
        config=cfg,
    )
    return TownLightingPlan(
        fixtures=tuple(
            road_embed_fixtures
            + road_fixtures
            + farm_fixtures
            + coverage_fixtures
        )
    )


def _int_array(
    value: np.ndarray,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    return array


def _bool_array(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    return array


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(mask.shape, dtype=np.uint8)
    counts[1:, :] += mask[:-1, :]
    counts[:-1, :] += mask[1:, :]
    counts[:, 1:] += mask[:, :-1]
    counts[:, :-1] += mask[:, 1:]
    return counts


def _adjacent_cells(
    local_x: int,
    local_z: int,
    *,
    width: int,
    depth: int,
) -> Iterable[tuple[int, int]]:
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = local_x + dx
        z = local_z + dz
        if 0 <= x < width and 0 <= z < depth:
            yield x, z


def _coord_noise(local_x: int, local_z: int, seed: int, salt: int) -> int:
    value = (
        int(local_x) * 73856093
        ^ int(local_z) * 19349663
        ^ int(seed) * 83492791
        ^ int(salt) * 2654435761
    )
    value ^= value >> 13
    value *= 1274126177
    return value & 0xFFFFFFFF


def _selected_with_spacing(
    candidates: Sequence[_Candidate],
    *,
    min_spacing: int,
    max_count: int,
    existing: Sequence[LightingFixture] = (),
) -> list[_Candidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.noise,
            candidate.local_z,
            candidate.local_x,
        ),
    )
    selected: list[_Candidate] = []
    min_spacing_sq = min_spacing * min_spacing
    fixed_points = [(fixture.local_x, fixture.local_z) for fixture in existing]

    for candidate in ranked:
        if max_count and len(selected) >= max_count:
            break
        if any(
            (candidate.local_x - x) ** 2 + (candidate.local_z - z) ** 2
            < min_spacing_sq
            for x, z in fixed_points
        ):
            continue
        if any(
            (candidate.local_x - item.local_x) ** 2
            + (candidate.local_z - item.local_z) ** 2
            < min_spacing_sq
            for item in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _plan_road_embed_fixtures(
    *,
    path_mask: np.ndarray,
    path_base_y: np.ndarray,
    path_slab_mask: np.ndarray,
    building_mask: np.ndarray,
    blocked_mask: np.ndarray,
    config: TownLightingConfig,
) -> list[LightingFixture]:
    path_degree = _neighbour_count(path_mask)
    candidate_mask = path_mask & ~path_slab_mask & ~building_mask & ~blocked_mask
    candidates: list[_Candidate] = []

    for local_z, local_x in np.argwhere(candidate_mask):
        x = int(local_x)
        z = int(local_z)
        degree = int(path_degree[z, x])
        score = 4
        if degree >= 3:
            score += 4
        elif degree == 2:
            score += 1
        if _coord_noise(x, z, config.seed, 71) % 4 == 0:
            score += 1
        candidates.append(
            _Candidate(
                local_x=x,
                local_z=z,
                score=score,
                noise=_coord_noise(x, z, config.seed, 73),
            )
        )

    selected = _selected_with_spacing(
        candidates,
        min_spacing=config.road_embed_spacing,
        max_count=config.max_road_embeds,
    )
    return [
        _road_embed_fixture(
            candidate.local_x,
            int(path_base_y[candidate.local_z, candidate.local_x]),
            candidate.local_z,
            config.seed,
        )
        for candidate in selected
    ]


def _plan_road_fixtures(
    *,
    heightmap: np.ndarray,
    path_mask: np.ndarray,
    path_base_y: np.ndarray,
    path_slab_mask: np.ndarray,
    open_mask: np.ndarray,
    config: TownLightingConfig,
) -> list[LightingFixture]:
    depth, width = heightmap.shape
    path_degree = _neighbour_count(path_mask)
    edge_mask = open_mask & (_neighbour_count(path_mask) > 0)
    candidates: list[_Candidate] = []

    for local_z, local_x in np.argwhere(edge_mask):
        x = int(local_x)
        z = int(local_z)
        adjacent_path_cells = [
            (nx, nz)
            for nx, nz in _adjacent_cells(x, z, width=width, depth=depth)
            if path_mask[nz, nx]
        ]
        if not adjacent_path_cells:
            continue

        ground_y = int(heightmap[z, x])
        path_tops = [
            int(path_base_y[nz, nx]) + (1 if path_slab_mask[nz, nx] else 0)
            for nx, nz in adjacent_path_cells
        ]
        if min(abs(ground_y - path_y) for path_y in path_tops) > 2:
            continue

        max_degree = max(int(path_degree[nz, nx]) for nx, nz in adjacent_path_cells)
        score = 2
        if max_degree >= 3:
            score += 5
        if len(adjacent_path_cells) >= 2:
            score += 2
        if _coord_noise(x, z, config.seed, 11) % 5 == 0:
            score += 1
        candidates.append(
            _Candidate(
                local_x=x,
                local_z=z,
                score=score,
                noise=_coord_noise(x, z, config.seed, 23),
            )
        )

    selected = _selected_with_spacing(
        candidates,
        min_spacing=config.road_spacing,
        max_count=config.max_road_fixtures,
    )
    return [
        _road_fixture(
            candidate.local_x,
            int(heightmap[candidate.local_z, candidate.local_x]),
            candidate.local_z,
            config.seed,
        )
        for candidate in selected
    ]


def _plan_farm_fixtures(
    *,
    heightmap: np.ndarray,
    farm_mask: np.ndarray,
    path_mask: np.ndarray,
    open_mask: np.ndarray,
    config: TownLightingConfig,
    existing: Sequence[LightingFixture],
) -> list[LightingFixture]:
    if not np.any(farm_mask):
        return []

    depth, width = heightmap.shape
    farm_edge_mask = open_mask & (_neighbour_count(farm_mask) > 0)
    path_near_mask = _neighbour_count(path_mask) > 0
    candidates: list[_Candidate] = []

    for local_z, local_x in np.argwhere(farm_edge_mask):
        x = int(local_x)
        z = int(local_z)
        farm_neighbours = sum(
            1
            for nx, nz in _adjacent_cells(x, z, width=width, depth=depth)
            if farm_mask[nz, nx]
        )
        score = 2 + farm_neighbours
        if path_near_mask[z, x]:
            score -= 1
        if _coord_noise(x, z, config.seed, 37) % 4 == 0:
            score += 1
        candidates.append(
            _Candidate(
                local_x=x,
                local_z=z,
                score=score,
                noise=_coord_noise(x, z, config.seed, 41),
            )
        )

    selected = _selected_with_spacing(
        candidates,
        min_spacing=config.farm_spacing,
        max_count=config.max_farm_fixtures,
        existing=existing,
    )
    return [
        _farm_fixture(
            candidate.local_x,
            int(heightmap[candidate.local_z, candidate.local_x]),
            candidate.local_z,
            config.seed,
        )
        for candidate in selected
    ]


def _plan_coverage_fixtures(
    *,
    heightmap: np.ndarray,
    core_mask: np.ndarray,
    open_mask: np.ndarray,
    building_mask: np.ndarray,
    blocked_mask: np.ndarray,
    existing: Sequence[LightingFixture],
    config: TownLightingConfig,
) -> list[LightingFixture]:
    target_mask = core_mask & ~building_mask & ~blocked_mask
    if not np.any(target_mask) or not np.any(open_mask):
        return []

    covered = _coverage_mask(
        target_mask.shape,
        existing,
        radius=config.coverage_radius,
    )
    ignored = np.zeros(target_mask.shape, dtype=bool)
    selected_points = {(fixture.local_x, fixture.local_z) for fixture in existing}
    fixtures: list[LightingFixture] = []

    while True:
        uncovered = target_mask & ~covered & ~ignored
        if not np.any(uncovered):
            break

        progressed = False
        for target_x, target_z in _tile_targets(
            uncovered,
            spacing=config.coverage_spacing,
            seed=config.seed,
        ):
            if not uncovered[target_z, target_x]:
                continue
            candidate = _nearest_open_candidate(
                open_mask,
                target_x,
                target_z,
                radius=config.coverage_radius,
                seed=config.seed,
                excluded=selected_points,
            )
            if candidate is None:
                ignored[target_z, target_x] = True
                progressed = True
                continue

            local_x, local_z = candidate
            fixtures.append(
                _coverage_fixture(
                    local_x,
                    int(heightmap[local_z, local_x]),
                    local_z,
                )
            )
            selected_points.add(candidate)
            _mark_coverage(
                covered,
                local_x,
                local_z,
                radius=config.coverage_radius,
            )
            progressed = True
            if (
                config.max_coverage_fixtures
                and len(fixtures) >= config.max_coverage_fixtures
            ):
                return fixtures

        if not progressed:
            break

    return fixtures


def _coverage_mask(
    shape: tuple[int, int],
    fixtures: Sequence[LightingFixture],
    *,
    radius: int,
) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for fixture in fixtures:
        _mark_coverage(covered, fixture.local_x, fixture.local_z, radius=radius)
    return covered


def _mark_coverage(
    covered: np.ndarray,
    local_x: int,
    local_z: int,
    *,
    radius: int,
) -> None:
    depth, width = covered.shape
    for dz in range(-radius, radius + 1):
        z = local_z + dz
        if not (0 <= z < depth):
            continue
        span = radius - abs(dz)
        x0 = max(0, local_x - span)
        x1 = min(width, local_x + span + 1)
        covered[z, x0:x1] = True


def _tile_targets(
    uncovered: np.ndarray,
    *,
    spacing: int,
    seed: int,
) -> Iterable[tuple[int, int]]:
    depth, width = uncovered.shape
    for tile_z in range(0, depth, spacing):
        for tile_x in range(0, width, spacing):
            tile = uncovered[
                tile_z : min(depth, tile_z + spacing),
                tile_x : min(width, tile_x + spacing),
            ]
            if not np.any(tile):
                continue
            centre_x = tile_x + (tile.shape[1] - 1) / 2.0
            centre_z = tile_z + (tile.shape[0] - 1) / 2.0
            choices = []
            for offset_z, offset_x in np.argwhere(tile):
                x = tile_x + int(offset_x)
                z = tile_z + int(offset_z)
                dist = (x - centre_x) ** 2 + (z - centre_z) ** 2
                choices.append((dist, _coord_noise(x, z, seed, 79), x, z))
            _dist, _noise, x, z = min(choices)
            yield x, z


def _nearest_open_candidate(
    open_mask: np.ndarray,
    target_x: int,
    target_z: int,
    *,
    radius: int,
    seed: int,
    excluded: set[tuple[int, int]],
) -> tuple[int, int] | None:
    depth, width = open_mask.shape
    best: tuple[int, int, int, int] | None = None
    for dz in range(-radius, radius + 1):
        z = target_z + dz
        if not (0 <= z < depth):
            continue
        span = radius - abs(dz)
        for dx in range(-span, span + 1):
            x = target_x + dx
            if not (0 <= x < width):
                continue
            if not open_mask[z, x] or (x, z) in excluded:
                continue
            dist = abs(dx) + abs(dz)
            key = (dist, _coord_noise(x, z, seed, 83), x, z)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    _dist, _noise, x, z = best
    return x, z


def _props(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in mapping.items()))


def _road_embed_fixture(
    local_x: int,
    ground_y: int,
    local_z: int,
    seed: int,
) -> LightingFixture:
    block_id = ROAD_EMBED_BLOCKS[
        _coord_noise(local_x, local_z, seed, 89) % len(ROAD_EMBED_BLOCKS)
    ]
    return LightingFixture(
        kind="road_embed",
        local_x=local_x,
        ground_y=ground_y,
        local_z=local_z,
        blocks=(LightingBlock(local_x, ground_y, local_z, block_id),),
    )


def _road_fixture(
    local_x: int,
    ground_y: int,
    local_z: int,
    seed: int,
) -> LightingFixture:
    post_blocks = (
        "minecraft:dark_oak_fence",
        "minecraft:spruce_fence",
        "minecraft:oak_fence",
    )
    post_block = post_blocks[
        _coord_noise(local_x, local_z, seed, 53) % len(post_blocks)
    ]
    blocks = (
        LightingBlock(local_x, ground_y + 1, local_z, post_block),
        LightingBlock(local_x, ground_y + 2, local_z, post_block),
        LightingBlock(
            local_x,
            ground_y + 3,
            local_z,
            "minecraft:lantern",
            _props({"hanging": "false"}),
        ),
    )
    return LightingFixture(
        kind="road_post",
        local_x=local_x,
        ground_y=ground_y,
        local_z=local_z,
        blocks=blocks,
    )


def _farm_fixture(
    local_x: int,
    ground_y: int,
    local_z: int,
    seed: int,
) -> LightingFixture:
    if _coord_noise(local_x, local_z, seed, 67) % 3 == 0:
        blocks = (
            LightingBlock(local_x, ground_y + 1, local_z, "minecraft:oak_fence"),
            LightingBlock(local_x, ground_y + 2, local_z, "minecraft:oak_fence"),
            LightingBlock(
                local_x,
                ground_y + 3,
                local_z,
                "minecraft:lantern",
                _props({"hanging": "false"}),
            ),
        )
        kind = "farm_lantern"
    else:
        blocks = (
            LightingBlock(local_x, ground_y + 1, local_z, "minecraft:oak_fence"),
            LightingBlock(local_x, ground_y + 2, local_z, "minecraft:shroomlight"),
        )
        kind = "farm_shroomlight"

    return LightingFixture(
        kind=kind,
        local_x=local_x,
        ground_y=ground_y,
        local_z=local_z,
        blocks=blocks,
    )


def _coverage_fixture(
    local_x: int,
    ground_y: int,
    local_z: int,
) -> LightingFixture:
    return LightingFixture(
        kind="coverage_light",
        local_x=local_x,
        ground_y=ground_y,
        local_z=local_z,
        blocks=(
            LightingBlock(
                local_x,
                ground_y + 1,
                local_z,
                "minecraft:light",
                _props({"level": "15"}),
            ),
        ),
    )


__all__ = [
    "AIR_BLOCK",
    "EMBEDDED_ROAD_REPLACEABLE_BLOCKS",
    "LightingBlock",
    "LightingFixture",
    "ROAD_EMBED_BLOCKS",
    "SOFT_REPLACEABLE_BLOCKS",
    "TownLightingConfig",
    "TownLightingPlan",
    "UNSAFE_SUPPORT_BLOCKS",
    "plan_town_lighting",
]
