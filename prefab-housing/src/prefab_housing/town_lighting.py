"""Deterministic settlement-scale lighting plans.

The planner only emits intended fixture blocks. Live-world clearance checks stay
at the GDPC boundary so this module can be tested without a running server.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependency sync
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

AIR_BLOCK = "minecraft:air"
MOB_PROOF_LIGHT_BLOCK = "minecraft:light"
REVERSE_SWEEP_FIXTURE_KIND = "reverse_sweep_light"

_LIGHT_FLAG_CLEAR = np.uint8(1)
_LIGHT_FLAG_TRANSPARENT = np.uint8(2)
_LIGHT_FLAG_SUPPORT = np.uint8(4)

SOFT_REPLACEABLE_BLOCKS: frozenset[str] = frozenset(
    {
        AIR_BLOCK,
        "minecraft:cave_air",
        "minecraft:void_air",
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
        MOB_PROOF_LIGHT_BLOCK,
    }
)

BODY_CLEAR_BLOCKS: frozenset[str] = SOFT_REPLACEABLE_BLOCKS

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
        "minecraft:cave_air",
        "minecraft:void_air",
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

LIGHT_EMISSION_LEVELS: Mapping[str, int] = {
    MOB_PROOF_LIGHT_BLOCK: 15,
    "minecraft:beacon": 15,
    "minecraft:campfire": 15,
    "minecraft:conduit": 15,
    "minecraft:crying_obsidian": 10,
    "minecraft:end_rod": 14,
    "minecraft:fire": 15,
    "minecraft:glowstone": 15,
    "minecraft:jack_o_lantern": 15,
    "minecraft:lantern": 15,
    "minecraft:lava": 15,
    "minecraft:magma_block": 3,
    "minecraft:ochre_froglight": 15,
    "minecraft:pearlescent_froglight": 15,
    "minecraft:portal": 11,
    "minecraft:redstone_torch": 7,
    "minecraft:sea_lantern": 15,
    "minecraft:shroomlight": 15,
    "minecraft:soul_campfire": 10,
    "minecraft:soul_fire": 10,
    "minecraft:soul_lantern": 10,
    "minecraft:soul_torch": 10,
    "minecraft:torch": 14,
    "minecraft:verdant_froglight": 15,
    "minecraft:wall_torch": 14,
}

LIGHT_TRANSPARENT_SUFFIXES: tuple[str, ...] = (
    "_button",
    "_carpet",
    "_chain",
    "_door",
    "_fence",
    "_fence_gate",
    "_glass",
    "_glass_pane",
    "_leaves",
    "_pressure_plate",
    "_rail",
    "_sapling",
    "_sign",
    "_slab",
    "_stairs",
    "_torch",
    "_trapdoor",
    "_wall",
)
LIGHT_TRANSPARENT_TOKENS: tuple[str, ...] = (
    "flower",
    "lantern",
    "mushroom",
    "roots",
    "torch",
    "vine",
)
NON_SPAWNABLE_SUPPORT_SUFFIXES: tuple[str, ...] = (
    "_button",
    "_carpet",
    "_chain",
    "_door",
    "_fence",
    "_fence_gate",
    "_glass",
    "_glass_pane",
    "_leaves",
    "_pressure_plate",
    "_rail",
    "_sapling",
    "_sign",
    "_torch",
    "_trapdoor",
    "_wall",
)
NON_SPAWNABLE_SUPPORT_TOKENS: tuple[str, ...] = (
    "flower",
    "lantern",
    "mushroom",
    "roots",
    "torch",
    "vine",
)

BlockLookup = Callable[[int, int, int], str]
ProgressCallback = Callable[[str], None]


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
    reverse_sweep_min_block_light: int = 1
    reverse_sweep_light_level: int = 15
    max_reverse_sweep_fixtures: int = 0
    reverse_sweep_fast_path: bool = True

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
        if not (1 <= self.reverse_sweep_min_block_light <= 15):
            raise ValueError("reverse_sweep_min_block_light must be between 1 and 15")
        if not (
            self.reverse_sweep_min_block_light
            <= self.reverse_sweep_light_level
            <= 15
        ):
            raise ValueError(
                "reverse_sweep_light_level must be between "
                "reverse_sweep_min_block_light and 15"
            )
        if self.max_reverse_sweep_fixtures < 0:
            raise ValueError("max_reverse_sweep_fixtures must be non-negative")


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
class LightingAudit:
    reverse_sweep_targets: int = 0
    reverse_sweep_existing_covered: int = 0
    reverse_sweep_added: int = 0
    reverse_sweep_uncovered: int = 0


@dataclass(frozen=True, slots=True)
class TownLightingPlan:
    fixtures: tuple[LightingFixture, ...]
    audit: LightingAudit = field(default_factory=LightingAudit)

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


@dataclass(frozen=True, slots=True)
class _LightSource:
    local_x: int
    y: int
    local_z: int
    level: int


@dataclass(frozen=True, slots=True)
class _SpawnTarget:
    local_x: int
    support_y: int
    local_z: int

    @property
    def spawn_pos(self) -> tuple[int, int, int]:
        return self.local_x, self.support_y + 1, self.local_z


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


def plan_reverse_sweep_lighting(
    *,
    block_at: BlockLookup,
    target_mask: np.ndarray,
    min_y_by_cell: np.ndarray,
    max_y: int,
    existing_fixtures: Sequence[LightingFixture] = (),
    config: TownLightingConfig | None = None,
    progress: ProgressCallback | None = None,
) -> TownLightingPlan:
    """Patch final exposed spawn surfaces that still have zero block light.

    The sweep works from the completed block state rather than from intended
    road/building masks. It treats each target column independently for surface
    discovery, then uses a conservative transparent-block propagation model to
    avoid placing one invisible light per surface when an existing or newly
    planned source already covers the spawn space.
    """

    cfg = config or TownLightingConfig()
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.ndim != 2:
        raise ValueError("target_mask must be a 2D array")
    min_y_by_cell = _int_array(min_y_by_cell, "min_y_by_cell", target_mask.shape)
    if max_y < 0:
        raise ValueError("max_y must be non-negative")

    scan_max_y = min(317, int(max_y))
    if scan_max_y < 0 or not np.any(target_mask):
        return TownLightingPlan(fixtures=())

    if cfg.reverse_sweep_fast_path:
        return _plan_reverse_sweep_lighting_array(
            block_at=block_at,
            target_mask=target_mask,
            min_y_by_cell=min_y_by_cell,
            max_y=scan_max_y,
            existing_fixtures=existing_fixtures,
            config=cfg,
            progress=progress,
        )

    if progress is not None:
        progress(
            "scanning exposed settlement surfaces "
            f"({int(np.count_nonzero(target_mask))} columns, y<= {scan_max_y})"
        )
    targets, world_sources = _scan_reverse_sweep_targets(
        block_at=block_at,
        target_mask=target_mask,
        min_y_by_cell=min_y_by_cell,
        max_y=scan_max_y,
    )
    if not targets:
        if progress is not None:
            progress("no exposed spawnable targets found")
        return TownLightingPlan(fixtures=())

    min_light = cfg.reverse_sweep_min_block_light
    sources = world_sources + _sources_from_fixtures(existing_fixtures)
    if progress is not None:
        progress(
            f"propagating {len(sources)} existing/planned light source(s) "
            f"across {len(targets)} target(s)"
        )
    light_by_pos = _propagate_sources(
        block_at=block_at,
        shape=target_mask.shape,
        min_y_by_cell=min_y_by_cell,
        max_y=scan_max_y + 2,
        sources=sources,
        min_light=min_light,
    )

    target_by_pos = {target.spawn_pos: target for target in targets}
    uncovered = {
        pos for pos in target_by_pos if light_by_pos.get(pos, 0) < min_light
    }
    initially_uncovered = len(uncovered)
    fixtures: list[LightingFixture] = []
    if progress is not None:
        progress(f"patching {initially_uncovered} uncovered target(s)")

    while uncovered:
        if (
            cfg.max_reverse_sweep_fixtures
            and len(fixtures) >= cfg.max_reverse_sweep_fixtures
        ):
            break

        source_x, source_y, source_z = min(
            uncovered,
            key=lambda pos: (pos[2], pos[0], pos[1]),
        )
        source = _LightSource(
            local_x=source_x,
            y=source_y,
            local_z=source_z,
            level=cfg.reverse_sweep_light_level,
        )
        fixtures.append(
            _reverse_sweep_fixture(
                source.local_x,
                source.y - 1,
                source.local_z,
                level=source.level,
            )
        )
        touched = _propagate_source(
            block_at=block_at,
            shape=target_mask.shape,
            min_y_by_cell=min_y_by_cell,
            max_y=scan_max_y + 2,
            source=source,
            min_light=min_light,
            light_by_pos=light_by_pos,
        )
        covered_now = {
            pos
            for pos in touched
            if pos in uncovered and light_by_pos.get(pos, 0) >= min_light
        }
        uncovered.difference_update(covered_now)
        uncovered.discard((source_x, source_y, source_z))
        if progress is not None and len(fixtures) % 128 == 0:
            progress(
                f"planned {len(fixtures)} patch light(s); "
                f"{len(uncovered)} target(s) still uncovered"
            )

    if progress is not None:
        progress(
            f"planned {len(fixtures)} patch light(s); "
            f"{len(uncovered)} target(s) still uncovered"
        )
    return TownLightingPlan(
        fixtures=tuple(fixtures),
        audit=LightingAudit(
            reverse_sweep_targets=len(targets),
            reverse_sweep_existing_covered=len(targets) - initially_uncovered,
            reverse_sweep_added=len(fixtures),
            reverse_sweep_uncovered=len(uncovered),
        ),
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


def _block_id(value: str) -> str:
    return str(value).split("[", 1)[0]


def _block_props(props: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {str(key): str(value) for key, value in props}


def _emission_level(
    block_id: str,
    props: tuple[tuple[str, str], ...] = (),
) -> int:
    block_id = _block_id(block_id)
    if block_id == MOB_PROOF_LIGHT_BLOCK:
        level = _block_props(props).get("level")
        if level is not None:
            try:
                return max(0, min(15, int(level)))
            except ValueError:
                return 15
    return int(LIGHT_EMISSION_LEVELS.get(block_id, 0))


def _is_light_transparent(block_id: str) -> bool:
    block_id = _block_id(block_id)
    if block_id in BODY_CLEAR_BLOCKS:
        return True
    if block_id in LIGHT_EMISSION_LEVELS:
        return True
    if block_id.endswith(LIGHT_TRANSPARENT_SUFFIXES):
        return True
    return any(token in block_id for token in LIGHT_TRANSPARENT_TOKENS)


def _is_spawn_body_clear(block_id: str) -> bool:
    return _block_id(block_id) in BODY_CLEAR_BLOCKS


def _is_spawnable_support(block_id: str) -> bool:
    block_id = _block_id(block_id)
    if block_id in UNSAFE_SUPPORT_BLOCKS:
        return False
    if block_id in BODY_CLEAR_BLOCKS:
        return False
    if block_id in LIGHT_EMISSION_LEVELS:
        return False
    if block_id.endswith(NON_SPAWNABLE_SUPPORT_SUFFIXES):
        return False
    if any(token in block_id for token in NON_SPAWNABLE_SUPPORT_TOKENS):
        return False
    return True


def _block_light_flags(block_id: str) -> np.uint8:
    flags = np.uint8(0)
    if _is_spawn_body_clear(block_id):
        flags |= _LIGHT_FLAG_CLEAR
    if _is_light_transparent(block_id):
        flags |= _LIGHT_FLAG_TRANSPARENT
    if _is_spawnable_support(block_id):
        flags |= _LIGHT_FLAG_SUPPORT
    return flags


def _reverse_sweep_volume_bounds(
    *,
    min_y_by_cell: np.ndarray,
    max_y: int,
) -> tuple[int, int]:
    return (
        max(0, int(np.min(min_y_by_cell))),
        min(319, int(max_y) + 2),
    )


def _build_reverse_sweep_volume(
    *,
    block_at: BlockLookup,
    shape: tuple[int, int],
    min_y_by_cell: np.ndarray,
    y_min: int,
    y_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth, width = shape
    volume_height = y_max - y_min + 1
    flags = np.zeros((volume_height, depth, width), dtype=np.uint8)
    emissions = np.zeros((volume_height, depth, width), dtype=np.uint8)

    for z in range(depth):
        for x in range(width):
            column_y_min = max(y_min, max(0, int(min_y_by_cell[z, x])))
            for y in range(column_y_min, y_max + 1):
                block_id = _block_id(block_at(x, y, z))
                yi = y - y_min
                flags[yi, z, x] = _block_light_flags(block_id)
                emissions[yi, z, x] = np.uint8(_emission_level(block_id))
    return flags, emissions


def _valid_volume_mask(
    *,
    min_y_by_cell: np.ndarray,
    y_min: int,
    count: int,
) -> np.ndarray:
    y_values = np.arange(y_min, y_min + count, dtype=np.int32)
    return y_values[:, None, None] >= min_y_by_cell[None, :, :]


def _scan_reverse_sweep_targets_from_volume(
    *,
    flags: np.ndarray,
    emissions: np.ndarray,
    target_mask: np.ndarray,
    min_y_by_cell: np.ndarray,
    y_min: int,
    max_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[_LightSource]]:
    target_count = max(0, min(max_y - y_min + 1, flags.shape[0] - 2))
    if target_count <= 0:
        target_coords = np.empty((0, 3), dtype=np.int32)
    else:
        support = (flags[:target_count] & _LIGHT_FLAG_SUPPORT) != 0
        body_1_clear = (flags[1 : target_count + 1] & _LIGHT_FLAG_CLEAR) != 0
        body_2_clear = (flags[2 : target_count + 2] & _LIGHT_FLAG_CLEAR) != 0
        valid = _valid_volume_mask(
            min_y_by_cell=min_y_by_cell,
            y_min=y_min,
            count=target_count,
        )
        target_coords = np.argwhere(
            support
            & body_1_clear
            & body_2_clear
            & valid
            & target_mask[None, :, :]
        ).astype(np.int32)

    if len(target_coords):
        order = np.lexsort(
            (
                target_coords[:, 0],
                target_coords[:, 2],
                target_coords[:, 1],
            )
        )
        target_coords = target_coords[order]
    target_y_indices = target_coords[:, 0] + 1
    target_z = target_coords[:, 1]
    target_x = target_coords[:, 2]

    source_count = min(flags.shape[0], min(319, max_y + 2) - y_min + 1)
    if source_count <= 0:
        source_coords = np.empty((0, 3), dtype=np.int32)
    else:
        valid_sources = _valid_volume_mask(
            min_y_by_cell=min_y_by_cell,
            y_min=y_min,
            count=source_count,
        )
        source_coords = np.argwhere(
            (emissions[:source_count] > 0)
            & valid_sources
            & target_mask[None, :, :]
        ).astype(np.int32)

    sources = [
        _LightSource(
            local_x=int(x),
            y=int(y_min + yi),
            local_z=int(z),
            level=int(emissions[int(yi), int(z), int(x)]),
        )
        for yi, z, x in source_coords
    ]
    return target_x, target_y_indices, target_z, sources


@njit(cache=True)
def _propagate_source_volume_numba(
    flags: np.ndarray,
    min_y_by_cell: np.ndarray,
    y_min: int,
    max_y: int,
    source_x: int,
    source_y: int,
    source_z: int,
    source_level: int,
    min_light: int,
    light_levels: np.ndarray,
) -> None:
    if source_level < min_light:
        return

    volume_height, depth, width = flags.shape
    if source_x < 0 or source_x >= width or source_z < 0 or source_z >= depth:
        return
    if source_y < 0 or source_y > min(319, max_y):
        return
    if source_y < max(0, int(min_y_by_cell[source_z, source_x])):
        return
    source_yi = source_y - y_min
    if source_yi < 0 or source_yi >= volume_height:
        return
    if light_levels[source_yi, source_z, source_x] >= source_level:
        return

    # Light level is capped at 15, so a 65k queue is comfortably above the
    # unique Manhattan-ball volume plus duplicate frontier entries.
    max_queue = 65536
    qx = np.empty(max_queue, dtype=np.int32)
    qy = np.empty(max_queue, dtype=np.int32)
    qz = np.empty(max_queue, dtype=np.int32)
    ql = np.empty(max_queue, dtype=np.uint8)
    head = 0
    tail = 1
    qx[0] = source_x
    qy[0] = source_y
    qz[0] = source_z
    ql[0] = np.uint8(source_level)
    light_levels[source_yi, source_z, source_x] = np.uint8(source_level)

    while head < tail:
        x = qx[head]
        y = qy[head]
        z = qz[head]
        level = int(ql[head])
        head += 1
        next_level = level - 1
        if next_level < min_light:
            continue

        for direction in range(6):
            nx = x
            ny = y
            nz = z
            if direction == 0:
                nx += 1
            elif direction == 1:
                nx -= 1
            elif direction == 2:
                ny += 1
            elif direction == 3:
                ny -= 1
            elif direction == 4:
                nz += 1
            else:
                nz -= 1

            if nx < 0 or nx >= width or nz < 0 or nz >= depth:
                continue
            if ny < 0 or ny > min(319, max_y):
                continue
            if ny < max(0, int(min_y_by_cell[nz, nx])):
                continue
            nyi = ny - y_min
            if nyi < 0 or nyi >= volume_height:
                continue
            if (flags[nyi, nz, nx] & _LIGHT_FLAG_TRANSPARENT) == 0:
                continue
            if light_levels[nyi, nz, nx] >= next_level:
                continue

            light_levels[nyi, nz, nx] = np.uint8(next_level)
            if tail >= max_queue:
                continue
            qx[tail] = nx
            qy[tail] = ny
            qz[tail] = nz
            ql[tail] = np.uint8(next_level)
            tail += 1


def _propagate_volume_sources(
    *,
    flags: np.ndarray,
    min_y_by_cell: np.ndarray,
    y_min: int,
    max_y: int,
    sources: Sequence[_LightSource],
    min_light: int,
    light_levels: np.ndarray,
) -> None:
    for source in sources:
        _propagate_source_volume_numba(
            flags,
            min_y_by_cell,
            y_min,
            max_y,
            int(source.local_x),
            int(source.y),
            int(source.local_z),
            int(source.level),
            int(min_light),
            light_levels,
        )


def _plan_reverse_sweep_lighting_array(
    *,
    block_at: BlockLookup,
    target_mask: np.ndarray,
    min_y_by_cell: np.ndarray,
    max_y: int,
    existing_fixtures: Sequence[LightingFixture],
    config: TownLightingConfig,
    progress: ProgressCallback | None,
) -> TownLightingPlan:
    y_min, y_max = _reverse_sweep_volume_bounds(
        min_y_by_cell=min_y_by_cell,
        max_y=max_y,
    )
    if y_max < y_min:
        return TownLightingPlan(fixtures=())

    if progress is not None:
        engine = "numba" if _NUMBA_AVAILABLE else "python"
        progress(
            "pre-sampling reverse-sweep volume "
            f"({int(np.count_nonzero(target_mask))} columns, "
            f"y={y_min}..{y_max}, propagation={engine})"
        )
    flags, emissions = _build_reverse_sweep_volume(
        block_at=block_at,
        shape=target_mask.shape,
        min_y_by_cell=min_y_by_cell,
        y_min=y_min,
        y_max=y_max,
    )
    target_x, target_y_indices, target_z, world_sources = (
        _scan_reverse_sweep_targets_from_volume(
            flags=flags,
            emissions=emissions,
            target_mask=target_mask,
            min_y_by_cell=min_y_by_cell,
            y_min=y_min,
            max_y=max_y,
        )
    )
    target_count = int(len(target_x))
    if target_count == 0:
        if progress is not None:
            progress("no exposed spawnable targets found")
        return TownLightingPlan(fixtures=())

    min_light = config.reverse_sweep_min_block_light
    sources = world_sources + _sources_from_fixtures(existing_fixtures)
    light_levels = np.zeros(flags.shape, dtype=np.uint8)
    if progress is not None:
        progress(
            f"propagating {len(sources)} existing/planned light source(s) "
            f"across {target_count} target(s)"
        )
    _propagate_volume_sources(
        flags=flags,
        min_y_by_cell=min_y_by_cell,
        y_min=y_min,
        max_y=y_max,
        sources=sources,
        min_light=min_light,
        light_levels=light_levels,
    )

    target_light = light_levels[target_y_indices, target_z, target_x]
    uncovered = target_light < min_light
    initially_uncovered = int(np.count_nonzero(uncovered))
    fixtures: list[LightingFixture] = []
    if progress is not None:
        progress(f"patching {initially_uncovered} uncovered target(s)")

    while bool(np.any(uncovered)):
        if (
            config.max_reverse_sweep_fixtures
            and len(fixtures) >= config.max_reverse_sweep_fixtures
        ):
            break

        index = int(np.flatnonzero(uncovered)[0])
        source_x = int(target_x[index])
        source_y = int(y_min + target_y_indices[index])
        source_z = int(target_z[index])
        level = int(config.reverse_sweep_light_level)
        fixtures.append(
            _reverse_sweep_fixture(
                source_x,
                source_y - 1,
                source_z,
                level=level,
            )
        )
        _propagate_source_volume_numba(
            flags,
            min_y_by_cell,
            y_min,
            y_max,
            source_x,
            source_y,
            source_z,
            level,
            int(min_light),
            light_levels,
        )
        target_light = light_levels[target_y_indices, target_z, target_x]
        uncovered = target_light < min_light
        if progress is not None and len(fixtures) % 128 == 0:
            progress(
                f"planned {len(fixtures)} patch light(s); "
                f"{int(np.count_nonzero(uncovered))} target(s) still uncovered"
            )

    uncovered_count = int(np.count_nonzero(uncovered))
    if progress is not None:
        progress(
            f"planned {len(fixtures)} patch light(s); "
            f"{uncovered_count} target(s) still uncovered"
        )
    return TownLightingPlan(
        fixtures=tuple(fixtures),
        audit=LightingAudit(
            reverse_sweep_targets=target_count,
            reverse_sweep_existing_covered=target_count - initially_uncovered,
            reverse_sweep_added=len(fixtures),
            reverse_sweep_uncovered=uncovered_count,
        ),
    )


def _scan_reverse_sweep_targets(
    *,
    block_at: BlockLookup,
    target_mask: np.ndarray,
    min_y_by_cell: np.ndarray,
    max_y: int,
) -> tuple[list[_SpawnTarget], list[_LightSource]]:
    targets: list[_SpawnTarget] = []
    sources: list[_LightSource] = []
    source_scan_max_y = min(319, max_y + 2)

    for local_z, local_x in np.argwhere(target_mask):
        x = int(local_x)
        z = int(local_z)
        min_y = max(0, int(min_y_by_cell[z, x]))
        for y in range(source_scan_max_y, min_y - 1, -1):
            block_id = _block_id(block_at(x, y, z))
            level = _emission_level(block_id)
            if level > 0:
                sources.append(_LightSource(x, y, z, level))

            if y > max_y or y > 317:
                continue
            if not _is_spawnable_support(block_id):
                continue
            if not _is_spawn_body_clear(block_at(x, y + 1, z)):
                continue
            if not _is_spawn_body_clear(block_at(x, y + 2, z)):
                continue
            targets.append(_SpawnTarget(x, y, z))

    return targets, sources


def _sources_from_fixtures(
    fixtures: Sequence[LightingFixture],
) -> list[_LightSource]:
    sources: list[_LightSource] = []
    for fixture in fixtures:
        for block in fixture.blocks:
            level = _emission_level(block.block_id, block.props)
            if level > 0:
                sources.append(
                    _LightSource(
                        int(block.local_x),
                        int(block.y),
                        int(block.local_z),
                        level,
                    )
                )
    return sources


def _volume_contains(
    *,
    shape: tuple[int, int],
    min_y_by_cell: np.ndarray,
    max_y: int,
    local_x: int,
    y: int,
    local_z: int,
) -> bool:
    depth, width = shape
    if not (0 <= local_x < width and 0 <= local_z < depth):
        return False
    if y < max(0, int(min_y_by_cell[local_z, local_x])):
        return False
    return 0 <= y <= min(319, max_y)


def _propagate_sources(
    *,
    block_at: BlockLookup,
    shape: tuple[int, int],
    min_y_by_cell: np.ndarray,
    max_y: int,
    sources: Sequence[_LightSource],
    min_light: int,
) -> dict[tuple[int, int, int], int]:
    light_by_pos: dict[tuple[int, int, int], int] = {}
    for source in sources:
        _propagate_source(
            block_at=block_at,
            shape=shape,
            min_y_by_cell=min_y_by_cell,
            max_y=max_y,
            source=source,
            min_light=min_light,
            light_by_pos=light_by_pos,
        )
    return light_by_pos


def _propagate_source(
    *,
    block_at: BlockLookup,
    shape: tuple[int, int],
    min_y_by_cell: np.ndarray,
    max_y: int,
    source: _LightSource,
    min_light: int,
    light_by_pos: dict[tuple[int, int, int], int],
) -> set[tuple[int, int, int]]:
    if source.level < min_light:
        return set()
    if not _volume_contains(
        shape=shape,
        min_y_by_cell=min_y_by_cell,
        max_y=max_y,
        local_x=source.local_x,
        y=source.y,
        local_z=source.local_z,
    ):
        return set()

    queue: deque[tuple[int, int, int, int]] = deque(
        [(source.local_x, source.y, source.local_z, source.level)]
    )
    touched: set[tuple[int, int, int]] = set()

    while queue:
        x, y, z, level = queue.popleft()
        pos = (x, y, z)
        if light_by_pos.get(pos, -1) >= level:
            continue
        light_by_pos[pos] = level
        touched.add(pos)
        next_level = level - 1
        if next_level < min_light:
            continue

        for nx, ny, nz in (
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ):
            if not _volume_contains(
                shape=shape,
                min_y_by_cell=min_y_by_cell,
                max_y=max_y,
                local_x=nx,
                y=ny,
                local_z=nz,
            ):
                continue
            if not _is_light_transparent(block_at(nx, ny, nz)):
                continue
            if light_by_pos.get((nx, ny, nz), -1) >= next_level:
                continue
            queue.append((nx, ny, nz, next_level))

    return touched


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


def _reverse_sweep_fixture(
    local_x: int,
    ground_y: int,
    local_z: int,
    *,
    level: int,
) -> LightingFixture:
    return LightingFixture(
        kind=REVERSE_SWEEP_FIXTURE_KIND,
        local_x=local_x,
        ground_y=ground_y,
        local_z=local_z,
        blocks=(
            LightingBlock(
                local_x,
                ground_y + 1,
                local_z,
                MOB_PROOF_LIGHT_BLOCK,
                _props({"level": str(level)}),
            ),
        ),
    )


__all__ = [
    "AIR_BLOCK",
    "BODY_CLEAR_BLOCKS",
    "EMBEDDED_ROAD_REPLACEABLE_BLOCKS",
    "LightingAudit",
    "LightingBlock",
    "LightingFixture",
    "MOB_PROOF_LIGHT_BLOCK",
    "REVERSE_SWEEP_FIXTURE_KIND",
    "ROAD_EMBED_BLOCKS",
    "SOFT_REPLACEABLE_BLOCKS",
    "TownLightingConfig",
    "TownLightingPlan",
    "UNSAFE_SUPPORT_BLOCKS",
    "plan_reverse_sweep_lighting",
    "plan_town_lighting",
]
