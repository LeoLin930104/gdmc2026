"""Run the quarantined settlement pipeline and place cached residential prefabs."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from prefab_housing.town_lighting import (
    EMBEDDED_ROAD_REPLACEABLE_BLOCKS,
    LightingFixture,
    SOFT_REPLACEABLE_BLOCKS,
    UNSAFE_SUPPORT_BLOCKS,
    TownLightingConfig,
    TownLightingPlan,
    plan_reverse_sweep_lighting,
    plan_town_lighting,
)
from prefab_housing.minecraft_animation import (
    BlueprintBlock,
    ResidentialBlockMode,
    ResidentialSettlementPlacementPlan,
    SettlementBuildSlot,
    compute_bounding_box,
    load_residential_upgrade_package,
    plan_residential_settlement_placements,
    plan_typed_residential_settlement_placements,
)
from prefab_housing.types import FaceName

REPO_ROOT = Path(__file__).resolve().parent.parent
# The repo root IS the upstream base generator now (builder.py, plotter.py,
# voronoi.py, map_manager.py, ... live here), so default to it directly.
DEFAULT_UPSTREAM_DIR = REPO_ROOT
DEFAULT_CACHE_DIR = REPO_ROOT / "prefab-housing" / "production_cache" / "residential_upgrade"
DEFAULT_TYPED_PACKAGES = (
    "residential=seed_043.pbp",
    "residential=seed_044.pbp",
    "worker_housing=seed_045.pbp",
    "worker_housing=seed_046.pbp",
    "row_house=seed_047.pbp",
    "row_house=seed_050.pbp",
)
HORIZONTAL_FACES: tuple[FaceName, FaceName, FaceName, FaceName] = (
    "north",
    "east",
    "south",
    "west",
)
PATH_BLOCKS = (
    "minecraft:cobblestone",
    "minecraft:stone_bricks",
    "minecraft:polished_andesite",
    "minecraft:chiseled_stone_bricks",
    "minecraft:mossy_cobblestone",
)
PATH_SLAB_BLOCKS = {
    "minecraft:cobblestone": "minecraft:cobblestone_slab",
    "minecraft:stone_bricks": "minecraft:stone_brick_slab",
    "minecraft:polished_andesite": "minecraft:polished_andesite_slab",
    "minecraft:chiseled_stone_bricks": "minecraft:stone_brick_slab",
    "minecraft:mossy_cobblestone": "minecraft:mossy_cobblestone_slab",
}
CELL_SURFACE_BLOCK = "minecraft:grass_block"
FOUNDATION_BLOCK = "minecraft:dirt"
PREFAB_PAD_BLOCK = "minecraft:coarse_dirt"
AIR_BLOCK = "minecraft:air"
LEVEL_THREE_FOOTPRINT = 30
UPSTREAM_HOUSE_MARKER_BLOCK = "minecraft:terracotta"
PAD_REPLACEABLE_BLOCKS = {
    AIR_BLOCK,
    CELL_SURFACE_BLOCK,
    FOUNDATION_BLOCK,
    PREFAB_PAD_BLOCK,
    UPSTREAM_HOUSE_MARKER_BLOCK,
    "minecraft:podzol",
    "minecraft:rooted_dirt",
    "minecraft:mycelium",
    "minecraft:moss_block",
    "minecraft:sand",
    "minecraft:red_sand",
    "minecraft:gravel",
    "minecraft:clay",
    "minecraft:snow",
    "minecraft:snow_block",
    "minecraft:short_grass",
    "minecraft:grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:large_fern",
}


@contextlib.contextmanager
def _pushd(path: Path) -> Iterable[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the quarantined GDMC settlement pipeline, deploy terrain/paths/farms, "
            "then place cached residential prefabs into generated building rectangles."
        )
    )
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--host", default="http://localhost:9000")
    parser.add_argument(
        "--max-water-ratio",
        type=float,
        default=0.35,
        help=(
            "Reject freshly captured upstream map slices when more than this "
            "fraction is water. This prevents silently planning towns in oceans."
        ),
    )
    parser.add_argument(
        "--allow-water-settlement",
        action="store_true",
        help="Allow high-water upstream map captures instead of failing fast.",
    )
    parser.add_argument(
        "--map-sample-radius",
        type=int,
        default=1,
        help=(
            "When the player/override-centred slice is too watery, sample this "
            "many neighbouring rings before failing."
        ),
    )
    parser.add_argument(
        "--map-sample-step",
        type=int,
        default=192,
        help="World-block distance between neighbouring map-sample centres.",
    )
    parser.add_argument(
        "--town-area-size",
        type=int,
        default=384,
        help="Width/depth of the upstream live world slice captured for town planning.",
    )
    parser.add_argument(
        "--region-center",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "Override the live capture centre. If omitted, the script uses the "
            "current player position and samples neighbouring centres from there."
        ),
    )
    parser.add_argument(
        "--region-origin",
        nargs=2,
        type=int,
        metavar=("X", "Z"),
        default=None,
        help=(
            "Override the exact live capture Rect origin. Neighbour sampling "
            "offsets this origin directly."
        ),
    )
    parser.add_argument(
        "--seed-spacing",
        type=int,
        default=38,
        help="Spacing between upstream Voronoi drift seeds. Larger values create larger house cells.",
    )
    parser.add_argument(
        "--seed-jitter",
        type=float,
        default=0.18,
        help="Deterministic jitter ratio applied to upstream Voronoi seed positions.",
    )
    parser.add_argument(
        "--seed-random",
        type=int,
        default=42,
        help="Seed used for deterministic upstream Voronoi jitter.",
    )
    parser.add_argument(
        "--drift-steps",
        type=int,
        default=10,
        help="Number of upstream slope-drift iterations for Voronoi seeds.",
    )
    parser.add_argument(
        "--drift-speed",
        type=float,
        default=1.0,
        help="Upstream slope-drift speed for Voronoi seeds.",
    )
    parser.add_argument(
        "--buffer-seed-stride",
        type=int,
        default=8,
        help="Stride for obstacle-edge buffer seeds in upstream Voronoi generation.",
    )
    parser.add_argument("--module-size", type=int, default=10)
    parser.add_argument("--setback", type=float, default=2.0)
    parser.add_argument("--min-slot-width", type=int, default=22)
    parser.add_argument("--min-slot-depth", type=int, default=22)
    parser.add_argument(
        "--max-prefabs",
        type=int,
        default=12,
        help="Maximum residential lots to keep. Use 0 for no cap.",
    )
    parser.add_argument("--lot-width", type=int, default=36)
    parser.add_argument("--lot-depth", type=int, default=36)
    parser.add_argument("--lot-gap", type=int, default=8)
    parser.add_argument("--lot-margin", type=int, default=8)
    parser.add_argument("--lot-buildable-threshold", type=float, default=0.70)
    parser.add_argument("--lot-inner-buildable-threshold", type=float, default=0.70)
    parser.add_argument("--lot-max-height-delta", type=int, default=8)
    parser.add_argument("--floor-y-offset", type=int, default=1)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument(
        "--plot-source",
        choices=("lots", "dense", "upstream"),
        default="upstream",
        help=(
            "upstream preserves the quarantined module/farm classifier; dense "
            "uses largest rectangles from all core cells; lots creates synthetic "
            "large stress-test lots."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "typed"),
        default="typed",
        help="Use one residential package everywhere, or cycle packages by building type.",
    )
    parser.add_argument(
        "--strict-package",
        default="seed_043.pbp",
        help="Package used when --mode strict is selected.",
    )
    parser.add_argument(
        "--typed-package",
        action="append",
        default=None,
        metavar="TYPE=PACKAGE",
        help=(
            "Building type to package mapping. May be repeated. Defaults to six "
            "checked-in residential package variants."
        ),
    )
    parser.add_argument(
        "--block-mode",
        choices=("core", "full", "structure"),
        default="full",
        help="Which cached package block section to place into the live town.",
    )
    parser.add_argument(
        "--target-entrance-face",
        choices=HORIZONTAL_FACES,
        default=None,
        help="Optional common entrance face for all residential prefabs.",
    )
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument(
        "--reuse-upstream-data",
        action="store_true",
        help="Skip map capture/planning and reuse existing upstream data/*.npz files.",
    )
    parser.add_argument(
        "--use-upstream-deploy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the quarantined deployer for terrain, paths, farms, and "
            "landscaping. Placeholder buildings are suppressed and replaced by "
            "cached residential prefabs."
        ),
    )
    parser.add_argument(
        "--skip-upstream-deploy",
        action="store_true",
        help="Deprecated alias for --no-use-upstream-deploy.",
    )
    parser.add_argument(
        "--skip-town-surfaces",
        action="store_true",
        help="Skip tracked terrain/path cleanup before prefab placement.",
    )
    parser.add_argument(
        "--town-lighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Place deterministic road and farm lighting after live prefab "
            "placement. The pass only writes into verified soft-replaceable cells."
        ),
    )
    parser.add_argument(
        "--lighting-only",
        action="store_true",
        help=(
            "Reuse existing upstream data and placement packages, skip all town "
            "and prefab placement, and run only the final lighting pass. Use "
            "this after narrative block placement."
        ),
    )
    parser.add_argument("--lighting-seed", type=int, default=1337)
    parser.add_argument(
        "--road-light-spacing",
        type=int,
        default=11,
        help="Minimum local-block spacing between planned road lamp posts.",
    )
    parser.add_argument(
        "--road-embed-light-spacing",
        type=int,
        default=9,
        help="Minimum local-block spacing between embedded luminous road blocks.",
    )
    parser.add_argument(
        "--farm-light-spacing",
        type=int,
        default=12,
        help="Minimum local-block spacing between planned farm lighting fixtures.",
    )
    parser.add_argument(
        "--coverage-light-spacing",
        type=int,
        default=18,
        help=(
            "Coarse spacing for invisible fallback coverage lights across the "
            "settlement core."
        ),
    )
    parser.add_argument(
        "--coverage-light-radius",
        type=int,
        default=12,
        help="2D planning radius assumed for each light source coverage pass.",
    )
    parser.add_argument(
        "--max-road-lights",
        type=int,
        default=96,
        help="Maximum planned road lighting fixtures. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-road-embed-lights",
        type=int,
        default=0,
        help="Maximum embedded road lighting fixtures. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-farm-lights",
        type=int,
        default=48,
        help="Maximum planned farm lighting fixtures. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-coverage-lights",
        type=int,
        default=768,
        help="Maximum invisible fallback coverage lights. Use 0 for no cap.",
    )
    parser.add_argument(
        "--reverse-sweep-lighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After decorative lighting, scan the final world slice for exposed "
            "spawnable surfaces and patch remaining zero-block-light targets."
        ),
    )
    parser.add_argument(
        "--reverse-sweep-min-block-light",
        type=int,
        default=1,
        help="Minimum block light required at each swept spawn position.",
    )
    parser.add_argument(
        "--reverse-sweep-light-level",
        type=int,
        default=15,
        help="Light level used for invisible reverse-sweep patch blocks.",
    )
    parser.add_argument(
        "--max-reverse-sweep-lights",
        type=int,
        default=0,
        help="Maximum reverse-sweep patch fixtures. Use 0 for no cap.",
    )
    parser.add_argument(
        "--reverse-sweep-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the array-backed cached-numba reverse-sweep planner. Disable "
            "only when debugging against the slower callback scanner."
        ),
    )
    parser.add_argument(
        "--town-clear-height",
        type=int,
        default=36,
        help="Clear this many blocks above town terrain before placing clean surfaces.",
    )
    parser.add_argument(
        "--clear-debug-y",
        type=int,
        default=120,
        help="Clear the upstream diagnostic sky-frame layer. Use -1 to disable.",
    )
    parser.add_argument(
        "--clear-prefab-volume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear each placed prefab bounding box before writing prefab blocks.",
    )
    parser.add_argument("--prefab-clear-extra-y", type=int, default=2)
    parser.add_argument(
        "--prefab-pad-buffer",
        type=int,
        default=2,
        help=(
            "Blocks of visible ground apron around the actual prefab footprint. "
            "Use 0 to support only the exact house shape."
        ),
    )
    parser.add_argument(
        "--prefab-pad-block",
        default=PREFAB_PAD_BLOCK,
        help="Ground block used for visible prefab aprons around placed houses.",
    )
    parser.add_argument("--flush-every", type=int, default=512)
    parser.add_argument(
        "--teleport",
        "--teleport-player",
        dest="teleport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Teleport the nearest player to an elevated overview point after live "
            "placement. The command is always printed, but only run when enabled."
        ),
    )
    parser.add_argument(
        "--teleport-y-offset",
        type=int,
        default=48,
        help="Blocks above the tallest placed prefab for the overview teleport.",
    )
    parser.add_argument(
        "--teleport-min-y",
        type=int,
        default=96,
        help="Minimum Y coordinate for the overview teleport.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "out" / "minecraft" / "town_prefab_session.json",
    )
    args = parser.parse_args()
    if args.module_size < 1:
        parser.error("--module-size must be positive")
    if args.min_slot_width < 1 or args.min_slot_depth < 1:
        parser.error("--min-slot-width and --min-slot-depth must be positive")
    if args.prefab_clear_extra_y < 0:
        parser.error("--prefab-clear-extra-y must be non-negative")
    if args.prefab_pad_buffer < 0:
        parser.error("--prefab-pad-buffer must be non-negative")
    if args.max_prefabs < 0:
        parser.error("--max-prefabs must be non-negative")
    if not (0.0 <= args.max_water_ratio <= 1.0):
        parser.error("--max-water-ratio must be between 0 and 1")
    if args.map_sample_radius < 0:
        parser.error("--map-sample-radius must be non-negative")
    if args.map_sample_step < 1:
        parser.error("--map-sample-step must be positive")
    if args.town_area_size < 64:
        parser.error("--town-area-size must be at least 64")
    if args.region_center is not None and args.region_origin is not None:
        parser.error("--region-center and --region-origin are mutually exclusive")
    if args.seed_spacing < 8:
        parser.error("--seed-spacing must be at least 8")
    if args.seed_jitter < 0:
        parser.error("--seed-jitter must be non-negative")
    if args.drift_steps < 0:
        parser.error("--drift-steps must be non-negative")
    if args.drift_speed < 0:
        parser.error("--drift-speed must be non-negative")
    if args.buffer_seed_stride < 1:
        parser.error("--buffer-seed-stride must be positive")
    if args.lot_width < 1 or args.lot_depth < 1:
        parser.error("--lot-width and --lot-depth must be positive")
    if args.lot_gap < 0 or args.lot_margin < 0:
        parser.error("--lot-gap and --lot-margin must be non-negative")
    if not (0.0 <= args.lot_buildable_threshold <= 1.0):
        parser.error("--lot-buildable-threshold must be between 0 and 1")
    if not (0.0 <= args.lot_inner_buildable_threshold <= 1.0):
        parser.error("--lot-inner-buildable-threshold must be between 0 and 1")
    if args.lot_max_height_delta < 0:
        parser.error("--lot-max-height-delta must be non-negative")
    if args.town_clear_height < 0:
        parser.error("--town-clear-height must be non-negative")
    if args.road_light_spacing < 1:
        parser.error("--road-light-spacing must be positive")
    if args.road_embed_light_spacing < 1:
        parser.error("--road-embed-light-spacing must be positive")
    if args.farm_light_spacing < 1:
        parser.error("--farm-light-spacing must be positive")
    if args.coverage_light_spacing < 1:
        parser.error("--coverage-light-spacing must be positive")
    if args.coverage_light_radius < 1:
        parser.error("--coverage-light-radius must be positive")
    if args.max_road_lights < 0:
        parser.error("--max-road-lights must be non-negative")
    if args.max_road_embed_lights < 0:
        parser.error("--max-road-embed-lights must be non-negative")
    if args.max_farm_lights < 0:
        parser.error("--max-farm-lights must be non-negative")
    if args.max_coverage_lights < 0:
        parser.error("--max-coverage-lights must be non-negative")
    if not (1 <= args.reverse_sweep_min_block_light <= 15):
        parser.error("--reverse-sweep-min-block-light must be between 1 and 15")
    if not (
        args.reverse_sweep_min_block_light
        <= args.reverse_sweep_light_level
        <= 15
    ):
        parser.error(
            "--reverse-sweep-light-level must be between "
            "--reverse-sweep-min-block-light and 15"
        )
    if args.max_reverse_sweep_lights < 0:
        parser.error("--max-reverse-sweep-lights must be non-negative")
    if args.teleport_y_offset < 0:
        parser.error("--teleport-y-offset must be non-negative")
    if args.skip_upstream_deploy:
        args.use_upstream_deploy = False
    return args


def _normalise_host(host: str) -> str:
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}"


def _resolve_package_path(cache_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.suffix:
        path = path.with_suffix(".pbp")
    if not path.is_absolute():
        path = cache_dir / path
    return path


def _parse_typed_package(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--typed-package must use TYPE=PACKAGE, for example residential=seed_043.pbp"
        )
    building_type, package_name = value.split("=", 1)
    building_type = building_type.strip()
    package_name = package_name.strip()
    if not building_type or not package_name:
        raise argparse.ArgumentTypeError("--typed-package values cannot be empty")
    return building_type, package_name


def _resolve_package_entries(args: argparse.Namespace) -> list[str]:
    # Explicit --typed-package always wins. Otherwise, if the narrative wrote
    # data/settlement_identity.json and the matching mood/biome prefab variants
    # were baked, place those; else fall back to the default packages.
    if args.typed_package:
        return list(args.typed_package)
    identity_path = args.upstream_dir.resolve() / "data" / "settlement_identity.json"
    mood = family = None
    if identity_path.exists():
        try:
            data = json.loads(identity_path.read_text(encoding="utf-8"))
            mood = data.get("mood_tier")
            family = data.get("biome_family")
        except Exception as exc:  # noqa: BLE001 - a bad identity file must not abort placement
            print(f"[wallface] could not read {identity_path} ({exc}); using default packages.")
    if not mood:
        return list(DEFAULT_TYPED_PACKAGES)
    suffix = f"__{mood}__{family}" if family else f"__{mood}"
    variants: list[str] = []
    for entry in DEFAULT_TYPED_PACKAGES:
        building_type, package_name = _parse_typed_package(entry)
        variants.append(f"{building_type}={Path(package_name).stem}{suffix}.pbp")
    missing = [
        v for v in variants
        if not _resolve_package_path(args.cache_dir, _parse_typed_package(v)[1]).exists()
    ]
    if missing:
        print(f"[wallface] {len(missing)} mood/biome variant package(s) missing; using default packages.")
        return list(DEFAULT_TYPED_PACKAGES)
    print(f"[wallface] using mood={mood!r} biome_family={family!r} prefab variants.")
    return variants


def _extract_rect(record: object) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if isinstance(record, np.ndarray):
        for item in reversed(record.tolist()):
            if isinstance(item, Mapping):
                return dict(item)
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes)):
        for item in reversed(record):
            if isinstance(item, Mapping):
                return dict(item)
    raise ValueError(f"building_rects record does not contain a mapping: {record!r}")


def _load_plot_rects(upstream_dir: Path) -> list[dict[str, Any]]:
    path = upstream_dir / "data" / "settlement_plots.npz"
    data = np.load(path, allow_pickle=True)
    if "building_rects" not in data:
        raise ValueError(f"{path} does not contain a building_rects array")
    return [_extract_rect(record) for record in data["building_rects"]]


def _plot_rect_count(upstream_dir: Path) -> int:
    path = upstream_dir / "data" / "settlement_plots.npz"
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    if "building_rects" not in data:
        return 0
    return int(len(data["building_rects"]))


def _load_settlement_context(upstream_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = upstream_dir / "data" / "settlement_data.npz"
    data = np.load(path, allow_pickle=True)
    origin = np.asarray(data["origin"], dtype=int)
    zone_map = np.asarray(data["zone_map"]) if "zone_map" in data else None
    return origin, zone_map


def _load_settlement_arrays(upstream_dir: Path) -> dict[str, Any]:
    path = upstream_dir / "data" / "settlement_data.npz"
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _nearest_seed_membership_mask(
    *,
    seeds: np.ndarray,
    selected_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    if seeds.ndim != 2 or seeds.shape[1] < 2:
        raise ValueError("settlement seeds must have shape (n, 2)")

    selected_indices = np.asarray(selected_indices, dtype=np.int64).ravel()
    selected_indices = selected_indices[
        (selected_indices >= 0) & (selected_indices < seeds.shape[0])
    ]
    if selected_indices.size == 0:
        return np.zeros(shape, dtype=bool)

    selected = np.zeros(seeds.shape[0], dtype=bool)
    selected[selected_indices] = True

    depth, width = shape
    seed_x = np.asarray(seeds[:, 0], dtype=np.float64)
    seed_z = np.asarray(seeds[:, 1], dtype=np.float64)
    x_values = np.arange(width, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    chunk_depth = 32

    for z0 in range(0, depth, chunk_depth):
        z1 = min(depth, z0 + chunk_depth)
        z_values = np.arange(z0, z1, dtype=np.float64)
        dx = x_values[None, :, None] - seed_x[None, None, :]
        dz = z_values[:, None, None] - seed_z[None, None, :]
        nearest = np.argmin((dx * dx) + (dz * dz), axis=2)
        mask[z0:z1, :] = selected[nearest]

    return mask


def _settlement_footprint_mask(
    upstream_dir: Path,
    *,
    shape: tuple[int, int],
    fallback_core_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(fallback_core_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("fallback_core_mask shape must match shape")
    mask = mask.copy()

    core_path = upstream_dir.resolve() / "data" / "settlement_core.npz"
    if not core_path.exists():
        return mask

    payload = _load_npz_payload(core_path)
    seeds = np.asarray(payload.get("seeds", np.empty((0, 2))), dtype=np.float64)
    core_indices = np.asarray(payload.get("core_indices", np.array([], dtype=np.int64)))
    if seeds.size == 0 or core_indices.size == 0:
        return mask

    try:
        mask |= _nearest_seed_membership_mask(
            seeds=seeds,
            selected_indices=core_indices,
            shape=shape,
        )
    except ValueError:
        return mask
    return mask


@dataclass(frozen=True, slots=True)
class CapturedMapQuality:
    water_ratio: float
    flat_ratio: float
    origin: tuple[int, int, int]


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _write_npz_payload(path: Path, payload: Mapping[str, Any]) -> None:
    np.savez_compressed(path, **dict(payload))


def _captured_map_quality(payload: Mapping[str, Any]) -> CapturedMapQuality | None:
    water_map = np.asarray(payload.get("water_map"))
    flat_mask = np.asarray(payload.get("flat_mask"))
    if water_map.size == 0 or flat_mask.size == 0:
        return None

    water_ratio = float(np.count_nonzero(water_map)) / float(water_map.size)
    flat_ratio = float(np.count_nonzero(flat_mask)) / float(flat_mask.size)
    origin = tuple(
        int(value)
        for value in np.asarray(payload.get("origin", [0, 0, 0]), dtype=int).tolist()
    )
    return CapturedMapQuality(
        water_ratio=water_ratio,
        flat_ratio=flat_ratio,
        origin=origin,
    )


def _captured_map_quality_from_file(path: Path) -> CapturedMapQuality | None:
    if not path.exists():
        return None
    return _captured_map_quality(_load_npz_payload(path))


def _map_sample_offsets(radius: int, step: int) -> list[tuple[int, int]]:
    offsets = {
        (x_ring * step, z_ring * step)
        for z_ring in range(-radius, radius + 1)
        for x_ring in range(-radius, radius + 1)
    }
    return sorted(
        offsets,
        key=lambda item: (
            abs(item[0]) + abs(item[1]),
            item[0] * item[0] + item[1] * item[1],
            item[1],
            item[0],
        ),
    )


def _high_water_message(quality: CapturedMapQuality, *, prefix: str) -> str:
    return (
        f"{prefix} "
        f"(water_ratio={quality.water_ratio:.3f}, flat_ratio={quality.flat_ratio:.3f}, "
        f"origin={list(quality.origin)})."
    )


def _resolve_map_manager_center(manager: Any) -> tuple[int, int, int]:
    resolver = getattr(manager, "resolve_center", None)
    if callable(resolver):
        return tuple(int(value) for value in resolver())
    try:
        get_player_pos = getattr(manager.editor, "getPlayerPos", None)
        if callable(get_player_pos):
            center = get_player_pos()
            if center is not None:
                return tuple(int(value) for value in center)
        return _get_player_position_from_http(manager.editor.host)
    except Exception as exc:
        player_error = exc
    else:
        player_error = None
    default_center = getattr(manager, "default_center", None)
    if default_center is not None:
        return tuple(int(value) for value in default_center)
    raise RuntimeError(
        "No capture centre is available. Join the Minecraft world so the player "
        "position can be read, or pass --region-center / --region-origin."
    ) from player_error


def _parse_player_position(data: str) -> tuple[int, int, int] | None:
    pos_match = re.search(r"Pos:\[([^\]]+)\]", data)
    if not pos_match:
        return None
    pos_values = [part.strip().rstrip("dD") for part in pos_match.group(1).split(",")]
    if len(pos_values) < 3:
        return None
    return (
        int(math.floor(float(pos_values[0]))),
        int(math.floor(float(pos_values[1]))),
        int(math.floor(float(pos_values[2]))),
    )


def _get_player_position_from_http(host: str) -> tuple[int, int, int]:
    import requests

    response = requests.get(
        f"{_normalise_host(host).rstrip('/')}/players",
        params={"includeData": "true"},
        timeout=1.0,
    )
    response.raise_for_status()
    players = response.json()
    if not players:
        raise RuntimeError("GDMC server returned no players")
    position = _parse_player_position(players[0].get("data", ""))
    if position is None:
        raise RuntimeError("could not parse player position from GDMC player data")
    return position


def _args_region_center(args: argparse.Namespace) -> tuple[int, int, int] | None:
    value = getattr(args, "region_center", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _args_region_origin(args: argparse.Namespace) -> tuple[int, int] | None:
    value = getattr(args, "region_origin", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _base_capture_center(args: argparse.Namespace, manager: Any) -> tuple[int, int, int]:
    region_center = _args_region_center(args)
    if region_center is not None:
        return region_center

    region_origin = _args_region_origin(args)
    if region_origin is not None:
        half = int(getattr(args, "town_area_size", 384)) // 2
        return (region_origin[0] + half, 0, region_origin[1] + half)

    return _resolve_map_manager_center(manager)


def _uses_player_origin_capture(args: argparse.Namespace) -> bool:
    return (
        not bool(getattr(args, "reuse_upstream_data", False))
        and _args_region_center(args) is None
        and _args_region_origin(args) is None
    )


def _offset_region_origin(
    args: argparse.Namespace,
    offset_x: int,
    offset_z: int,
) -> tuple[int, int] | None:
    region_origin = _args_region_origin(args)
    if region_origin is None:
        return None
    return (region_origin[0] + offset_x, region_origin[1] + offset_z)


def _capture_environment_dataset(
    manager: Any,
    center: tuple[int, int, int],
    *,
    region_origin: tuple[int, int] | None = None,
) -> dict[str, Any]:
    try:
        return manager.load_environment_dataset(center=center, origin=region_origin)
    except TypeError:
        if region_origin is None:
            try:
                return manager.load_environment_dataset(center=center)
            except TypeError:
                pass

    from gdpc import Rect

    if not manager.is_minecraft_available():
        return manager.load_environment_dataset()

    print("✅ Minecraft detected. Parsing live environment context...")
    if region_origin is not None:
        x1, z1 = region_origin
    else:
        cx, _cy, cz = center
        half = int(manager.area_size) // 2
        x1 = (int(cx - half) // 16) * 16
        z1 = (int(cz - half) // 16) * 16
    rect = Rect((x1, z1), (int(manager.area_size), int(manager.area_size)))
    print(f"🔌 Loading World Slice from bounds: {rect}")
    world_slice = manager.editor.loadWorldSlice(rect)
    heightmap, tree_map, water_map = manager.extract_and_orient_maps(world_slice)
    slope, flat_mask = manager.compute_slopes(heightmap)
    origin = np.array([x1, 0, z1])
    np.savez_compressed(
        "data/data.npz",
        seeds=np.array([]),
        heightmap=heightmap,
        slope=slope,
        flat_mask=flat_mask,
        tree_map=tree_map,
        water_map=water_map,
        origin=origin,
        num_drift=0,
    )
    print("💾 Fresh map context cached to: data/data.npz")
    dataset = {
        "heightmap": heightmap,
        "slope": slope,
        "flat_mask": flat_mask,
        "tree_map": tree_map,
        "water_map": water_map,
        "origin": origin,
    }
    print("Dataset confirmation keys available:", list(dataset.keys()))
    return dataset


def _validate_captured_map_data(args: argparse.Namespace) -> CapturedMapQuality | None:
    """Validate upstream map capture before Voronoi seed generation."""
    path = args.upstream_dir.resolve() / "data" / "data.npz"
    quality = _captured_map_quality_from_file(path)
    if quality is None:
        return None
    if quality.water_ratio <= float(args.max_water_ratio):
        return quality
    message = _high_water_message(
        quality,
        prefix=(
            "upstream cached map slice is high-water; rerun without "
            "--reuse-upstream-data to let the script resample"
        ),
    )
    if args.allow_water_settlement:
        print(f"[upstream:warning] {message}")
        return quality
    raise RuntimeError(message)


def _capture_valid_upstream_map_data(
    args: argparse.Namespace,
    *,
    manager_cls: Any | None = None,
) -> CapturedMapQuality | None:
    if manager_cls is None:
        from map_manager import MapManager

        manager_cls = MapManager

    town_area_size = int(getattr(args, "town_area_size", 384))
    manager_kwargs: dict[str, Any] = {
        "area_size": town_area_size,
        "host": _normalise_host(getattr(args, "host", "http://localhost:9000")),
    }
    region_center = _args_region_center(args)
    region_origin = _args_region_origin(args)
    if region_center is not None:
        manager_kwargs["region_center"] = region_center
    if region_origin is not None:
        manager_kwargs["region_origin"] = region_origin
    try:
        manager = manager_cls(**manager_kwargs)
    except TypeError:
        manager = manager_cls()
        if hasattr(manager, "area_size"):
            manager.area_size = town_area_size
    base_center = _base_capture_center(args, manager)
    offsets = _map_sample_offsets(
        int(args.map_sample_radius),
        int(args.map_sample_step),
    )
    data_path = args.upstream_dir.resolve() / "data" / "data.npz"
    best: tuple[CapturedMapQuality, dict[str, Any]] | None = None

    for sample_index, (offset_x, offset_z) in enumerate(offsets, start=1):
        center = (
            base_center[0] + offset_x,
            base_center[1],
            base_center[2] + offset_z,
        )
        sample_origin = _offset_region_origin(args, offset_x, offset_z)
        print(
            "[upstream:capture] "
            f"sample {sample_index}/{len(offsets)} center={center}"
            + (f" origin={sample_origin}" if sample_origin is not None else "")
        )
        _capture_environment_dataset(manager, center, region_origin=sample_origin)
        payload = _load_npz_payload(data_path)
        quality = _captured_map_quality(payload)
        if quality is None:
            return None
        print(
            "[upstream:capture] "
            f"origin={list(quality.origin)} water_ratio={quality.water_ratio:.3f} "
            f"flat_ratio={quality.flat_ratio:.3f}"
        )
        if best is None or quality.water_ratio < best[0].water_ratio:
            best = (quality, payload)
        if args.allow_water_settlement or quality.water_ratio <= float(args.max_water_ratio):
            return quality

    if best is not None:
        best_quality, best_payload = best
        _write_npz_payload(data_path, best_payload)
        message = _high_water_message(
            best_quality,
            prefix=(
                f"sampled {len(offsets)} map slices and all were high-water; "
                "best candidate kept in data/data.npz"
            ),
        )
    else:
        message = "sampled no map slices; check --map-sample-radius and --map-sample-step."
    raise RuntimeError(
        f"{message} Increase --map-sample-radius or reduce --max-water-ratio only if intentional."
    )


def _write_dense_building_plots(args: argparse.Namespace) -> None:
    upstream_dir = args.upstream_dir.resolve()
    from plotter import largest_buildable_rectangle_from_data

    data = np.load(upstream_dir / "data" / "settlement_data.npz", allow_pickle=True)
    core_data = np.load(upstream_dir / "data" / "settlement_core.npz", allow_pickle=True)
    seeds = data["seeds"]
    heightmap = data["heightmap"]
    origin = data["origin"]
    zone_map = np.asarray(data["zone_map"]) if "zone_map" in data else None
    core_indices = sorted(set(int(item) for item in core_data["core_indices"].tolist()))

    rects: list[dict[str, Any]] = []
    for cell_id in core_indices:
        rect = largest_buildable_rectangle_from_data(
            seeds,
            heightmap,
            cell_id,
            setback=args.setback,
            origin=origin,
        )
        if rect is None:
            continue
        if rect["width"] < args.min_slot_width or rect["depth"] < args.min_slot_depth:
            continue
        if zone_map is not None:
            zone_id = _rect_zone_id(rect, zone_map)
            if zone_id is not None:
                rect["zone_id"] = zone_id
        rects.append(rect)

    rects.sort(key=lambda item: (-int(item["area"]), int(item["cell_id"])))
    if args.max_prefabs:
        rects = rects[: args.max_prefabs]

    plot_path = upstream_dir / "data" / "settlement_plots.npz"
    existing = _load_npz_payload(plot_path) if plot_path.exists() else {}
    np.savez(
        plot_path,
        plots=existing.get("plots", np.array([], dtype=object)),
        farms=existing.get("farms", np.array([], dtype=object)),
        building_rects=np.array(
            [(int(rect["cell_id"]), rect) for rect in rects],
            dtype=object,
        ),
        module_size=np.array(args.module_size),
        setback=np.array(args.setback),
        farm_setback=np.array(0.0),
        min_build_width=np.array(args.min_slot_width),
        min_build_depth=np.array(args.min_slot_depth),
        plot_source=np.array("dense_rectangles"),
        max_prefabs=np.array(args.max_prefabs),
    )
    print(
        "[plots] dense residential rectangles: "
        f"{len(rects)} -> {plot_path}"
    )


def _bool_array(
    arrays: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, int],
    default: bool,
) -> np.ndarray:
    if key not in arrays:
        return np.full(shape, default, dtype=bool)
    array = np.asarray(arrays[key], dtype=bool)
    if array.shape != shape:
        return np.full(shape, default, dtype=bool)
    return array


def _write_lot_building_plots(args: argparse.Namespace) -> None:
    upstream_dir = args.upstream_dir.resolve()
    arrays = _load_settlement_arrays(upstream_dir)
    heightmap = np.asarray(arrays["heightmap"])
    origin = np.asarray(arrays["origin"], dtype=int)
    zone_map = np.asarray(arrays["zone_map"]) if "zone_map" in arrays else None
    shape = heightmap.shape
    core_mask = _bool_array(arrays, "core_cell_mask", shape=shape, default=True)
    path_mask = _bool_array(arrays, "path_mask", shape=shape, default=False)
    water_mask = _bool_array(arrays, "water_map", shape=shape, default=False)
    chasm_mask = _bool_array(arrays, "chasm_mask", shape=shape, default=False)
    buildable_mask = core_mask & ~path_mask & ~water_mask & ~chasm_mask

    lot_width = int(args.lot_width)
    lot_depth = int(args.lot_depth)
    inner_width = min(LEVEL_THREE_FOOTPRINT, lot_width)
    inner_depth = min(LEVEL_THREE_FOOTPRINT, lot_depth)
    step_x = lot_width + int(args.lot_gap)
    step_z = lot_depth + int(args.lot_gap)
    depth, width = shape
    rects: list[dict[str, Any]] = []
    candidate_index = 0

    for z in range(int(args.lot_margin), max(int(args.lot_margin), depth - lot_depth + 1), step_z):
        for x in range(int(args.lot_margin), max(int(args.lot_margin), width - lot_width + 1), step_x):
            region = buildable_mask[z : z + lot_depth, x : x + lot_width]
            if region.shape != (lot_depth, lot_width):
                continue

            inner_x0 = x + ((lot_width - inner_width) // 2)
            inner_z0 = z + ((lot_depth - inner_depth) // 2)
            inner = buildable_mask[
                inner_z0 : inner_z0 + inner_depth,
                inner_x0 : inner_x0 + inner_width,
            ]
            if inner.shape != (inner_depth, inner_width):
                continue

            buildable_ratio = float(np.count_nonzero(region)) / float(region.size)
            inner_ratio = float(np.count_nonzero(inner)) / float(inner.size)
            if buildable_ratio < float(args.lot_buildable_threshold):
                continue
            if inner_ratio < float(args.lot_inner_buildable_threshold):
                continue

            height_region = heightmap[z : z + lot_depth, x : x + lot_width]
            buildable_heights = height_region[region]
            if buildable_heights.size == 0:
                continue
            min_height = int(buildable_heights.min())
            max_height = int(buildable_heights.max())
            if max_height - min_height > int(args.lot_max_height_delta):
                continue

            rect: dict[str, Any] = {
                "cell_id": 10000 + candidate_index,
                "x": int(x),
                "z": int(z),
                "world_x": int(origin[0]) + int(x),
                "world_z": int(origin[2]) + int(z),
                "y": max_height,
                "width": lot_width,
                "depth": lot_depth,
                "area": lot_width * lot_depth,
                "buildable_ratio": buildable_ratio,
                "inner_buildable_ratio": inner_ratio,
                "height_min": min_height,
                "height_max": max_height,
            }
            zone_id = _rect_zone_id(rect, zone_map)
            if zone_id is not None:
                rect["zone_id"] = zone_id
            rects.append(rect)
            candidate_index += 1

    centre_x = width / 2.0
    centre_z = depth / 2.0
    rects.sort(
        key=lambda rect: (
            (int(rect["x"]) + int(rect["width"]) / 2.0 - centre_x) ** 2
            + (int(rect["z"]) + int(rect["depth"]) / 2.0 - centre_z) ** 2,
            int(rect.get("zone_id", 0)),
            int(rect["z"]),
            int(rect["x"]),
        )
    )
    if args.max_prefabs:
        rects = rects[: args.max_prefabs]

    plot_path = upstream_dir / "data" / "settlement_plots.npz"
    existing = _load_npz_payload(plot_path) if plot_path.exists() else {}
    np.savez(
        plot_path,
        plots=existing.get("plots", np.array([], dtype=object)),
        farms=existing.get("farms", np.array([], dtype=object)),
        building_rects=np.array(
            [(int(rect["cell_id"]), rect) for rect in rects],
            dtype=object,
        ),
        module_size=np.array(args.module_size),
        setback=np.array(args.setback),
        farm_setback=np.array(0.0),
        min_build_width=np.array(args.min_slot_width),
        min_build_depth=np.array(args.min_slot_depth),
        plot_source=np.array("level3_lots"),
        max_prefabs=np.array(args.max_prefabs),
        lot_width=np.array(args.lot_width),
        lot_depth=np.array(args.lot_depth),
        lot_gap=np.array(args.lot_gap),
        lot_margin=np.array(args.lot_margin),
        lot_buildable_threshold=np.array(args.lot_buildable_threshold),
        lot_inner_buildable_threshold=np.array(args.lot_inner_buildable_threshold),
        lot_max_height_delta=np.array(args.lot_max_height_delta),
    )
    print(
        "[plots] level-3 residential lots: "
        f"{len(rects)} -> {plot_path}"
    )


def _filter_rects(
    rects: Sequence[Mapping[str, Any]],
    *,
    min_width: int,
    min_depth: int,
) -> list[Mapping[str, Any]]:
    return [
        rect
        for rect in rects
        if int(rect["width"]) >= min_width and int(rect["depth"]) >= min_depth
    ]


def _rect_fit_tier(rect: Mapping[str, Any]) -> int:
    width = int(rect["width"])
    depth = int(rect["depth"])
    short_side = min(width, depth)
    long_side = max(width, depth)
    if short_side >= 32:
        return 3
    if short_side >= 22 and long_side >= 32:
        return 2
    if short_side >= 22:
        return 1
    return 0


def _select_varied_rects(
    rects: Sequence[Mapping[str, Any]],
    *,
    max_count: int,
) -> list[Mapping[str, Any]]:
    ranked = sorted(
        rects,
        key=lambda rect: (
            -_rect_fit_tier(rect),
            -int(rect["area"]),
            int(rect.get("cell_id", 0)),
        ),
    )
    if max_count <= 0 or len(ranked) <= max_count:
        return ranked

    by_tier: dict[int, list[Mapping[str, Any]]] = {1: [], 2: [], 3: []}
    for rect in ranked:
        tier = _rect_fit_tier(rect)
        if tier in by_tier:
            by_tier[tier].append(rect)

    selected: list[Mapping[str, Any]] = []
    tier_cycle = (3, 2, 1)
    while len(selected) < max_count and any(by_tier.values()):
        progressed = False
        for tier in tier_cycle:
            if by_tier[tier]:
                selected.append(by_tier[tier].pop(0))
                progressed = True
                if len(selected) >= max_count:
                    break
        if not progressed:
            break
    return selected


def _rect_zone_id(rect: Mapping[str, Any], zone_map: np.ndarray | None) -> int | None:
    if "zone_id" in rect:
        return int(rect["zone_id"])
    if zone_map is None:
        return None
    depth, width = zone_map.shape
    x = min(width - 1, max(0, int(rect["x"]) + int(rect["width"]) // 2))
    z = min(depth - 1, max(0, int(rect["z"]) + int(rect["depth"]) // 2))
    zone_id = int(zone_map[z, x])
    return zone_id if zone_id >= 0 else None


def _slots_from_rects(
    rects: Sequence[Mapping[str, Any]],
    *,
    zone_map: np.ndarray | None,
    building_types: Sequence[str],
    default_building_type: str,
    floor_y_offset: int,
) -> list[SettlementBuildSlot]:
    slots: list[SettlementBuildSlot] = []
    for index, rect in enumerate(rects):
        zone_id = _rect_zone_id(rect, zone_map)
        if "building_type" in rect:
            building_type = str(rect["building_type"])
        elif zone_id is not None and building_types:
            building_type = building_types[zone_id % len(building_types)]
        elif building_types:
            building_type = building_types[index % len(building_types)]
        else:
            building_type = default_building_type

        slots.append(
            SettlementBuildSlot(
                x=int(rect["x"]),
                y=int(rect.get("y", 0)) + floor_y_offset,
                z=int(rect["z"]),
                width=int(rect["width"]),
                depth=int(rect["depth"]),
                cell_id=int(rect["cell_id"]) if "cell_id" in rect else None,
                zone_id=zone_id,
                building_type=building_type,
            )
        )
    return slots


def _load_strict_plan(args: argparse.Namespace, slots: Sequence[SettlementBuildSlot]) -> ResidentialSettlementPlacementPlan:
    package_path = _resolve_package_path(args.cache_dir, args.strict_package)
    states, _diffs, _manifest = load_residential_upgrade_package(package_path)
    return plan_residential_settlement_placements(
        states,
        slots,
        target_entrance_face=args.target_entrance_face,
        allow_rotate=not args.no_rotate,
        level=args.level,
        block_mode=args.block_mode,
        fail_fast=False,
    )


def _load_typed_plan(args: argparse.Namespace, slots: Sequence[SettlementBuildSlot]) -> ResidentialSettlementPlacementPlan:
    package_entries = _resolve_package_entries(args)
    typed_packages: dict[str, list[str]] = defaultdict(list)
    for entry in package_entries:
        building_type, package_name = _parse_typed_package(entry)
        typed_packages[building_type].append(package_name)

    states_by_type = {
        building_type: tuple(
            load_residential_upgrade_package(
                _resolve_package_path(args.cache_dir, package_name)
            )[0]
            for package_name in package_names
        )
        for building_type, package_names in typed_packages.items()
    }
    return plan_typed_residential_settlement_placements(
        states_by_type,
        slots,
        target_entrance_face=args.target_entrance_face,
        allow_rotate=not args.no_rotate,
        level=args.level,
        block_mode=args.block_mode,
        fail_fast=False,
    )


def _run_upstream_pipeline(args: argparse.Namespace) -> None:
    upstream_dir = args.upstream_dir.resolve()
    sys.path.insert(0, str(upstream_dir))
    with _pushd(upstream_dir):
        from marker import generate_zones, isolate_buildable_plot, mark_path_and_perimeter
        from plotter import find_modular_plots
        from terraformer import apply_terraforming
        from voronoi import generate_voronoi_diagram

        if not args.reuse_upstream_data:
            print("[upstream] capturing map and generating settlement plan")
            _capture_valid_upstream_map_data(args)
            generate_voronoi_diagram(
                grid_spacing=args.seed_spacing,
                drift_steps=args.drift_steps,
                drift_speed=args.drift_speed,
                jitter_ratio=args.seed_jitter,
                random_seed=args.seed_random,
                buffer_stride=args.buffer_seed_stride,
            )
            if not isolate_buildable_plot():
                raise RuntimeError("upstream found no buildable settlement core")
            generate_zones(num_zones=4)
            mark_path_and_perimeter()
            apply_terraforming(plot_setback=args.setback)
            if args.plot_source in {"lots", "dense"}:
                find_modular_plots(
                    module_size=args.module_size,
                    setback=args.setback,
                    min_build_width=args.min_slot_width,
                    min_build_depth=args.min_slot_depth,
                )
            if args.plot_source == "lots":
                _write_lot_building_plots(args)
            elif args.plot_source == "dense":
                _write_dense_building_plots(args)
            else:
                find_modular_plots(
                    module_size=args.module_size,
                    setback=args.setback,
                    min_build_width=args.min_slot_width,
                    min_build_depth=args.min_slot_depth,
                )
            if args.plot_source == "upstream" and _plot_rect_count(upstream_dir) == 0:
                print("[plots:repair] upstream classifier produced no house rectangles; falling back to dense core rectangles")
                _write_dense_building_plots(args)
        else:
            print("[upstream] reusing existing settlement data files")
            _validate_captured_map_data(args)
            if args.plot_source in {"lots", "dense", "upstream"}:
                find_modular_plots(
                    module_size=args.module_size,
                    setback=args.setback,
                    min_build_width=args.min_slot_width,
                    min_build_depth=args.min_slot_depth,
                )
            if args.plot_source == "lots":
                _write_lot_building_plots(args)
            elif args.plot_source == "dense":
                _write_dense_building_plots(args)
            elif _plot_rect_count(upstream_dir) == 0:
                print("[plots:repair] upstream classifier produced no house rectangles; falling back to dense core rectangles")
                _write_dense_building_plots(args)


def _deploy_upstream_settlement(args: argparse.Namespace) -> None:
    upstream_dir = args.upstream_dir.resolve()
    sys.path.insert(0, str(upstream_dir))
    with _pushd(upstream_dir):
        from builder import deploy_settlement

        print(
            "[upstream] deploying terrain, paths, farms, and landscaping "
            "with placeholder buildings suppressed"
        )
        deploy_settlement(place_debug_frame=False, place_placeholders=False)


def _filter_plot_buildings_for_plan(
    upstream_dir: Path,
    plan: ResidentialSettlementPlacementPlan,
) -> dict[str, int]:
    """Keep only building plot records that will receive prefab placements."""
    plot_path = upstream_dir.resolve() / "data" / "settlement_plots.npz"
    payload = _load_npz_payload(plot_path)
    selected_cell_ids = {
        int(placement.slot.cell_id)
        for placement in plan.placements
        if placement.slot.cell_id is not None
    }

    plots = dict(payload.get("plots", np.array([], dtype=object)))
    building_rects = dict(payload.get("building_rects", np.array([], dtype=object)))

    filtered_plots = [
        (cell_id, modules)
        for cell_id, modules in plots.items()
        if int(cell_id) in selected_cell_ids
    ]
    filtered_rects = []
    for cell_id, rect in building_rects.items():
        rect_cell_id = int(rect.get("cell_id", cell_id))
        if rect_cell_id in selected_cell_ids:
            filtered_rects.append((cell_id, rect))

    payload["plots"] = np.array(filtered_plots, dtype=object)
    payload["building_rects"] = np.array(filtered_rects, dtype=object)
    payload["prefab_selected_cell_ids"] = np.array(
        sorted(selected_cell_ids),
        dtype=np.int32,
    )
    payload["prefab_unfilled_plot_count"] = np.array(
        max(0, len(plots) - len(filtered_plots)),
        dtype=np.int32,
    )
    payload["prefab_unfilled_rect_count"] = np.array(
        max(0, len(building_rects) - len(filtered_rects)),
        dtype=np.int32,
    )
    np.savez(plot_path, **payload)

    stats = {
        "selected_cell_ids": len(selected_cell_ids),
        "plots_before": len(plots),
        "plots_after": len(filtered_plots),
        "rects_before": len(building_rects),
        "rects_after": len(filtered_rects),
    }
    print(
        "[plots] reserved prefab lots only: "
        f"rects {stats['rects_before']}->{stats['rects_after']} "
        f"plots {stats['plots_before']}->{stats['plots_after']}"
    )
    return stats


def _import_gdpc() -> tuple[Any, Any]:
    try:
        from gdpc import Editor
        from gdpc.block import Block
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live prefab placement requires the optional 'gdpc' package and a running GDMC-HTTP server."
        ) from exc
    return Editor, Block


def _path_block_for(local_x: int, local_z: int) -> str:
    return PATH_BLOCKS[(local_x * 31 + local_z * 17) % len(PATH_BLOCKS)]


def _path_slab_for(local_x: int, local_z: int) -> str:
    return PATH_SLAB_BLOCKS[_path_block_for(local_x, local_z)]


def _iter_mask(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    for local_z, local_x in np.argwhere(mask):
        yield int(local_x), int(local_z)


def _building_mask_from_plan(
    plan: ResidentialSettlementPlacementPlan,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    depth, width = shape
    mask = np.zeros((depth, width), dtype=bool)
    for placement in plan.placements:
        min_x, _min_y, min_z, max_x, _max_y, max_z = placement.bbox
        x0 = max(0, min_x)
        z0 = max(0, min_z)
        x1 = min(width, max_x + 1)
        z1 = min(depth, max_z + 1)
        if x1 > x0 and z1 > z0:
            mask[z0:z1, x0:x1] = True
    return mask


def _building_floor_y_from_plan(
    plan: ResidentialSettlementPlacementPlan,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    depth, width = shape
    floor_y = np.full((depth, width), -1, dtype=np.int32)
    for placement in plan.placements:
        min_x, min_y, min_z, max_x, _max_y, max_z = placement.bbox
        x0 = max(0, min_x)
        z0 = max(0, min_z)
        x1 = min(width, max_x + 1)
        z1 = min(depth, max_z + 1)
        if x1 > x0 and z1 > z0:
            floor_y[z0:z1, x0:x1] = np.maximum(
                floor_y[z0:z1, x0:x1],
                int(min_y) - 1,
            )
    return floor_y


def _prefab_support_maps_from_plan(
    plan: ResidentialSettlementPlacementPlan,
    *,
    shape: tuple[int, int],
    buffer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, width = shape
    footprint = np.zeros((depth, width), dtype=bool)
    support_mask = np.zeros((depth, width), dtype=bool)
    floor_y = np.full((depth, width), -1, dtype=np.int32)

    for placement in plan.placements:
        target_ground_y = int(placement.bbox[1]) - 1
        cells = {
            (int(block["dx"]), int(block["dz"]))
            for block in placement.blocks
            if str(block["id"]).split("[")[0] != AIR_BLOCK
        }
        for local_x, local_z in cells:
            if 0 <= local_x < width and 0 <= local_z < depth:
                footprint[local_z, local_x] = True

        for local_x, local_z in cells:
            for offset_z in range(-buffer, buffer + 1):
                for offset_x in range(-buffer, buffer + 1):
                    x = local_x + offset_x
                    z = local_z + offset_z
                    if 0 <= x < width and 0 <= z < depth:
                        support_mask[z, x] = True
                        floor_y[z, x] = max(floor_y[z, x], target_ground_y)

    return footprint, support_mask, floor_y


def _farm_mask_from_plots(upstream_dir: Path, *, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    plot_path = upstream_dir.resolve() / "data" / "settlement_plots.npz"
    if not plot_path.exists():
        return mask
    payload = _load_npz_payload(plot_path)
    farms = dict(payload.get("farms", np.array([], dtype=object)))
    depth, width = shape
    for cells in farms.values():
        for local_x, local_z in cells:
            x = int(local_x)
            z = int(local_z)
            if 0 <= x < width and 0 <= z < depth:
                mask[z, x] = True
    return mask


def _place_live_block(
    *,
    editor: Any,
    block_cls: Any,
    origin: np.ndarray,
    local_x: int,
    y: int,
    local_z: int,
    block_id: str,
    props: Mapping[str, str] | None = None,
) -> None:
    editor.placeBlock(
        (int(origin[0]) + local_x, int(y), int(origin[2]) + local_z),
        block_cls(block_id, dict(props or {})),
    )


def _deploy_town_surfaces_live(
    args: argparse.Namespace,
    *,
    origin: np.ndarray,
    plan: ResidentialSettlementPlacementPlan,
) -> dict[str, int]:
    arrays = _load_settlement_arrays(args.upstream_dir)
    heightmap = np.asarray(arrays["heightmap"])
    core_mask = np.asarray(arrays.get("core_cell_mask"), dtype=bool)
    path_mask = np.asarray(arrays.get("path_mask"), dtype=bool)
    path_base_y = np.asarray(arrays.get("path_base_y", heightmap))
    path_slab_mask = np.asarray(
        arrays.get("path_slab_mask", np.zeros_like(path_mask, dtype=bool)),
        dtype=bool,
    )
    if core_mask.shape != heightmap.shape:
        core_mask = np.ones_like(heightmap, dtype=bool)
    if path_mask.shape != heightmap.shape:
        path_mask = np.zeros_like(heightmap, dtype=bool)
    if path_base_y.shape != heightmap.shape:
        path_base_y = heightmap
    if path_slab_mask.shape != heightmap.shape:
        path_slab_mask = np.zeros_like(path_mask, dtype=bool)

    building_mask = _building_mask_from_plan(plan, shape=heightmap.shape)
    building_floor_y = _building_floor_y_from_plan(plan, shape=heightmap.shape)
    target_mask = core_mask | path_mask | building_mask
    Editor, Block = _import_gdpc()
    editor = Editor(buffering=True, host=_normalise_host(args.host))
    stats = {"cleared": 0, "terrain": 0, "paths": 0, "debug_cleared": 0}
    try:
        for local_x, local_z in _iter_mask(target_mask):
            surface_y = int(heightmap[local_z, local_x])
            clear_base_y = surface_y
            if building_mask[local_z, local_x]:
                clear_base_y = max(surface_y, int(building_floor_y[local_z, local_x]))
            clear_top_y = max(surface_y + args.town_clear_height, clear_base_y + args.town_clear_height)
            for y in range(surface_y + 1, min(320, clear_top_y + 1)):
                _place_live_block(
                    editor=editor,
                    block_cls=Block,
                    origin=origin,
                    local_x=local_x,
                    y=y,
                    local_z=local_z,
                    block_id=AIR_BLOCK,
                )
                stats["cleared"] += 1

            if building_mask[local_z, local_x]:
                target_ground_y = max(surface_y, int(building_floor_y[local_z, local_x]))
                for y in range(surface_y, target_ground_y + 1):
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=y,
                        local_z=local_z,
                        block_id=FOUNDATION_BLOCK,
                    )
                    stats["terrain"] += 1
            elif path_mask[local_z, local_x]:
                path_y = int(path_base_y[local_z, local_x])
                _place_live_block(
                    editor=editor,
                    block_cls=Block,
                    origin=origin,
                    local_x=local_x,
                    y=path_y,
                    local_z=local_z,
                    block_id=_path_block_for(local_x, local_z),
                )
                if path_slab_mask[local_z, local_x]:
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=path_y + 1,
                        local_z=local_z,
                        block_id=_path_slab_for(local_x, local_z),
                        props={"type": "bottom"},
                    )
                stats["paths"] += 1
            elif core_mask[local_z, local_x]:
                _place_live_block(
                    editor=editor,
                    block_cls=Block,
                    origin=origin,
                    local_x=local_x,
                    y=surface_y,
                    local_z=local_z,
                    block_id=CELL_SURFACE_BLOCK,
                )
                stats["terrain"] += 1

            total = stats["cleared"] + stats["terrain"] + stats["paths"]
            if args.flush_every > 0 and total % args.flush_every == 0:
                editor.flushBuffer()

        if args.clear_debug_y >= 0:
            depth, width = heightmap.shape
            for local_z in range(depth):
                for local_x in range(width):
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=args.clear_debug_y,
                        local_z=local_z,
                        block_id=AIR_BLOCK,
                    )
                    stats["debug_cleared"] += 1
                if args.flush_every > 0 and local_z % 8 == 0:
                    editor.flushBuffer()
    finally:
        editor.flushBuffer()
        editor.buffering = False
    print(
        "[town-surfaces] "
        f"terrain={stats['terrain']} paths={stats['paths']} "
        f"cleared={stats['cleared']} debug_cleared={stats['debug_cleared']}"
    )
    return stats


def _prepare_prefab_foundations_live(
    args: argparse.Namespace,
    *,
    origin: np.ndarray,
    plan: ResidentialSettlementPlacementPlan,
) -> dict[str, int]:
    arrays = _load_settlement_arrays(args.upstream_dir)
    heightmap = np.asarray(arrays["heightmap"])
    path_mask = _bool_array(arrays, "path_mask", shape=heightmap.shape, default=False)
    farm_mask = _farm_mask_from_plots(args.upstream_dir, shape=heightmap.shape)
    footprint_mask, support_mask, support_floor_y = _prefab_support_maps_from_plan(
        plan,
        shape=heightmap.shape,
        buffer=int(args.prefab_pad_buffer),
    )
    Editor, Block = _import_gdpc()
    from gdpc import Rect

    editor = Editor(buffering=True, host=_normalise_host(args.host))
    depth, width = heightmap.shape
    world_slice = editor.loadWorldSlice(
        Rect((int(origin[0]), int(origin[2])), (width, depth))
    )
    stats = {"foundations": 0, "pads": 0, "cleared": 0, "debug_cleared": 0}

    def existing_block(local_x: int, y: int, local_z: int) -> str:
        return world_slice.getBlock((local_x, y, local_z)).id.split("[")[0]

    try:
        for local_x, local_z in _iter_mask(support_mask):
            surface_y = int(heightmap[local_z, local_x])
            target_ground_y = int(support_floor_y[local_z, local_x])
            if target_ground_y < 0:
                continue

            is_structure = bool(footprint_mask[local_z, local_x])
            is_reserved = bool(path_mask[local_z, local_x] or farm_mask[local_z, local_x])
            if not is_structure and is_reserved:
                continue

            if not is_structure:
                clear_to_y = max(surface_y, target_ground_y)
                if any(
                    existing_block(local_x, y, local_z) not in PAD_REPLACEABLE_BLOCKS
                    for y in range(target_ground_y, clear_to_y + 1)
                ):
                    continue

            fill_start_y = min(surface_y, target_ground_y)
            for y in range(fill_start_y, target_ground_y + 1):
                if y == target_ground_y:
                    block_id = args.prefab_pad_block
                    stats["pads"] += 1
                else:
                    block_id = FOUNDATION_BLOCK
                    stats["foundations"] += 1
                _place_live_block(
                    editor=editor,
                    block_cls=Block,
                    origin=origin,
                    local_x=local_x,
                    y=y,
                    local_z=local_z,
                    block_id=block_id,
                )

            if is_structure:
                clear_top_y = min(319, target_ground_y + int(args.town_clear_height))
                for y in range(target_ground_y + 1, clear_top_y + 1):
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=y,
                        local_z=local_z,
                        block_id=AIR_BLOCK,
                    )
                    stats["cleared"] += 1
            elif surface_y > target_ground_y:
                for y in range(target_ground_y + 1, surface_y + 1):
                    if existing_block(local_x, y, local_z) == AIR_BLOCK:
                        continue
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=y,
                        local_z=local_z,
                        block_id=AIR_BLOCK,
                    )
                    stats["cleared"] += 1

            total = stats["foundations"] + stats["pads"] + stats["cleared"]
            if args.flush_every > 0 and total % args.flush_every == 0:
                editor.flushBuffer()

        if args.clear_debug_y >= 0:
            for local_z in range(depth):
                for local_x in range(width):
                    _place_live_block(
                        editor=editor,
                        block_cls=Block,
                        origin=origin,
                        local_x=local_x,
                        y=args.clear_debug_y,
                        local_z=local_z,
                        block_id=AIR_BLOCK,
                    )
                    stats["debug_cleared"] += 1
                if args.flush_every > 0 and local_z % 8 == 0:
                    editor.flushBuffer()
    finally:
        editor.flushBuffer()
        editor.buffering = False
    print(
        "[prefab-foundations] "
        f"foundations={stats['foundations']} pads={stats['pads']} "
        f"cleared={stats['cleared']} "
        f"debug_cleared={stats['debug_cleared']}"
    )
    return stats


def _lighting_config_from_args(args: argparse.Namespace) -> TownLightingConfig:
    return TownLightingConfig(
        seed=int(args.lighting_seed),
        road_spacing=int(args.road_light_spacing),
        road_embed_spacing=int(args.road_embed_light_spacing),
        farm_spacing=int(args.farm_light_spacing),
        coverage_spacing=int(args.coverage_light_spacing),
        coverage_radius=int(args.coverage_light_radius),
        max_road_fixtures=int(args.max_road_lights),
        max_road_embeds=int(args.max_road_embed_lights),
        max_farm_fixtures=int(args.max_farm_lights),
        max_coverage_fixtures=int(args.max_coverage_lights),
        reverse_sweep_min_block_light=int(args.reverse_sweep_min_block_light),
        reverse_sweep_light_level=int(args.reverse_sweep_light_level),
        max_reverse_sweep_fixtures=int(args.max_reverse_sweep_lights),
        reverse_sweep_fast_path=bool(args.reverse_sweep_fast_path),
    )


def _load_lighting_plan(
    args: argparse.Namespace,
    *,
    plan: ResidentialSettlementPlacementPlan,
) -> TownLightingPlan:
    arrays = _load_settlement_arrays(args.upstream_dir)
    heightmap = np.asarray(arrays["heightmap"], dtype=np.int32)
    shape = heightmap.shape
    core_mask = _bool_array(arrays, "core_cell_mask", shape=shape, default=True)
    path_mask = _bool_array(arrays, "path_mask", shape=shape, default=False)
    path_base_y = np.asarray(arrays.get("path_base_y", heightmap), dtype=np.int32)
    path_slab_mask = _bool_array(
        arrays,
        "path_slab_mask",
        shape=shape,
        default=False,
    )
    if path_base_y.shape != shape:
        path_base_y = heightmap

    farm_mask = _farm_mask_from_plots(args.upstream_dir, shape=shape)
    water_mask = _bool_array(arrays, "water_map", shape=shape, default=False)
    chasm_mask = _bool_array(arrays, "chasm_mask", shape=shape, default=False)
    _footprint_mask, support_mask, _support_floor_y = _prefab_support_maps_from_plan(
        plan,
        shape=shape,
        buffer=int(args.prefab_pad_buffer),
    )
    building_mask = _building_mask_from_plan(plan, shape=shape) | support_mask
    return plan_town_lighting(
        heightmap=heightmap,
        core_mask=core_mask,
        path_mask=path_mask,
        path_base_y=path_base_y,
        path_slab_mask=path_slab_mask,
        farm_mask=farm_mask,
        building_mask=building_mask,
        blocked_mask=water_mask | chasm_mask,
        config=_lighting_config_from_args(args),
    )


def _reverse_sweep_max_y(
    args: argparse.Namespace,
    *,
    heightmap: np.ndarray,
    plan: ResidentialSettlementPlacementPlan,
) -> int:
    max_y = int(np.max(heightmap)) + int(args.town_clear_height) + 2
    for placement in plan.placements:
        max_y = max(max_y, int(placement.bbox[4]) + 2)
    return min(317, max_y)


def _load_reverse_sweep_lighting_plan(
    args: argparse.Namespace,
    *,
    plan: ResidentialSettlementPlacementPlan,
    block_at: Any,
    existing_fixtures: Sequence[LightingFixture],
    progress: Callable[[str], None] | None = None,
) -> TownLightingPlan:
    arrays = _load_settlement_arrays(args.upstream_dir)
    heightmap = np.asarray(arrays["heightmap"], dtype=np.int32)
    shape = heightmap.shape
    core_mask = _bool_array(arrays, "core_cell_mask", shape=shape, default=True)
    path_mask = _bool_array(arrays, "path_mask", shape=shape, default=False)
    water_mask = _bool_array(arrays, "water_map", shape=shape, default=False)
    chasm_mask = _bool_array(arrays, "chasm_mask", shape=shape, default=False)
    settlement_mask = _settlement_footprint_mask(
        args.upstream_dir,
        shape=shape,
        fallback_core_mask=core_mask,
    )
    building_mask = _building_mask_from_plan(plan, shape=shape)
    target_mask = (settlement_mask | path_mask | building_mask) & ~(
        water_mask | chasm_mask
    )

    min_y_by_cell = np.maximum(0, heightmap - 2).astype(np.int32)
    building_floor_y = _building_floor_y_from_plan(plan, shape=shape)
    has_building_floor = building_floor_y >= 0
    min_y_by_cell[has_building_floor] = np.minimum(
        min_y_by_cell[has_building_floor],
        np.maximum(0, building_floor_y[has_building_floor] - 1),
    )

    return plan_reverse_sweep_lighting(
        block_at=block_at,
        target_mask=target_mask,
        min_y_by_cell=min_y_by_cell,
        max_y=_reverse_sweep_max_y(args, heightmap=heightmap, plan=plan),
        existing_fixtures=existing_fixtures,
        config=_lighting_config_from_args(args),
        progress=progress,
    )


def _deploy_town_lighting_live(
    args: argparse.Namespace,
    *,
    origin: np.ndarray,
    plan: ResidentialSettlementPlacementPlan,
) -> dict[str, Any]:
    lighting_plan = _load_lighting_plan(args, plan=plan)
    arrays = _load_settlement_arrays(args.upstream_dir)
    heightmap = np.asarray(arrays["heightmap"])
    depth, width = heightmap.shape

    Editor, Block = _import_gdpc()
    from gdpc import Rect

    editor = Editor(buffering=True, host=_normalise_host(args.host))
    world_slice = editor.loadWorldSlice(
        Rect((int(origin[0]), int(origin[2])), (width, depth))
    )
    placed_blocks: dict[tuple[int, int, int], str] = {}
    stats: dict[str, Any] = {
        "planned_fixtures": len(lighting_plan.fixtures),
        "planned_blocks": lighting_plan.block_count,
        "planned_reverse_sweep_fixtures": 0,
        "placed_fixtures": 0,
        "skipped_fixtures": 0,
        "placed_blocks": 0,
        "reverse_sweep_targets": 0,
        "reverse_sweep_existing_covered": 0,
        "reverse_sweep_added": 0,
        "reverse_sweep_uncovered": 0,
        "by_kind": lighting_plan.counts_by_kind(),
        "placed_by_kind": {},
    }
    placed_fixtures: list[LightingFixture] = []

    def existing_block(local_x: int, y: int, local_z: int) -> str:
        cached = placed_blocks.get((local_x, y, local_z))
        if cached is not None:
            return cached
        return world_slice.getBlock((local_x, y, local_z)).id.split("[")[0]

    def fixture_has_clearance(fixture: LightingFixture) -> bool:
        if fixture.kind != "road_embed":
            support_block = existing_block(
                fixture.local_x,
                fixture.ground_y,
                fixture.local_z,
            )
            if support_block in UNSAFE_SUPPORT_BLOCKS:
                return False
        for block in fixture.blocks:
            if not (
                0 <= int(block.local_x) < width
                and 0 <= int(block.local_z) < depth
            ):
                return False
            if not (0 <= int(block.y) < 320):
                return False
            target_block = existing_block(block.local_x, block.y, block.local_z)
            replaceable_blocks = (
                EMBEDDED_ROAD_REPLACEABLE_BLOCKS
                if fixture.kind == "road_embed"
                else SOFT_REPLACEABLE_BLOCKS
            )
            if target_block not in replaceable_blocks:
                return False
        return True

    def place_fixture(fixture: LightingFixture) -> bool:
        if not fixture_has_clearance(fixture):
            stats["skipped_fixtures"] += 1
            return False
        for block in fixture.blocks:
            _place_live_block(
                editor=editor,
                block_cls=Block,
                origin=origin,
                local_x=int(block.local_x),
                y=int(block.y),
                local_z=int(block.local_z),
                block_id=str(block.block_id),
                props=block.props_dict,
            )
            placed_blocks[
                (int(block.local_x), int(block.y), int(block.local_z))
            ] = str(block.block_id)
            stats["placed_blocks"] += 1
        stats["placed_fixtures"] += 1
        stats["placed_by_kind"][fixture.kind] = (
            stats["placed_by_kind"].get(fixture.kind, 0) + 1
        )
        placed_fixtures.append(fixture)
        if (
            args.flush_every > 0
            and stats["placed_blocks"] % args.flush_every == 0
        ):
            editor.flushBuffer()
        return True

    try:
        for fixture in lighting_plan.fixtures:
            place_fixture(fixture)

        if args.reverse_sweep_lighting:
            print("[lighting] reverse sweep: starting final mob-proofing audit...")
            reverse_plan = _load_reverse_sweep_lighting_plan(
                args,
                plan=plan,
                block_at=existing_block,
                existing_fixtures=placed_fixtures,
                progress=lambda message: print(
                    f"[lighting] reverse sweep: {message}",
                    flush=True,
                ),
            )
            stats["planned_reverse_sweep_fixtures"] = len(reverse_plan.fixtures)
            stats["planned_fixtures"] += len(reverse_plan.fixtures)
            stats["planned_blocks"] += reverse_plan.block_count
            for kind, count in reverse_plan.counts_by_kind().items():
                stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + count
            stats["reverse_sweep_targets"] = (
                reverse_plan.audit.reverse_sweep_targets
            )
            stats["reverse_sweep_existing_covered"] = (
                reverse_plan.audit.reverse_sweep_existing_covered
            )
            stats["reverse_sweep_added"] = reverse_plan.audit.reverse_sweep_added
            stats["reverse_sweep_uncovered"] = (
                reverse_plan.audit.reverse_sweep_uncovered
            )
            for fixture in reverse_plan.fixtures:
                place_fixture(fixture)
            print(
                "[lighting] reverse sweep: placement complete "
                f"({len(reverse_plan.fixtures)} patch fixture(s), "
                f"{reverse_plan.audit.reverse_sweep_uncovered} target(s) "
                "still uncovered)."
            )
    finally:
        editor.flushBuffer()
        editor.buffering = False

    print(
        "[lighting] "
        f"fixtures={stats['placed_fixtures']}/{stats['planned_fixtures']} "
        f"blocks={stats['placed_blocks']}/{stats['planned_blocks']} "
        f"skipped={stats['skipped_fixtures']} "
        f"reverse_targets={stats['reverse_sweep_targets']} "
        f"reverse_uncovered={stats['reverse_sweep_uncovered']} "
        f"by_kind={stats['by_kind']} placed_by_kind={stats['placed_by_kind']}"
    )
    return stats


def _world_pos(origin: np.ndarray, block: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(origin[0]) + int(block["dx"]),
        int(block["dy"]),
        int(origin[2]) + int(block["dz"]),
    )


def _world_bbox(
    origin: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return (
        int(origin[0]) + min_x,
        min_y,
        int(origin[2]) + min_z,
        int(origin[0]) + max_x,
        max_y,
        int(origin[2]) + max_z,
    )


def _teleport_target(summary: Mapping[str, Any]) -> tuple[int, int, int]:
    placements = summary.get("placements")
    if isinstance(placements, list) and placements:
        bboxes = [
            tuple(int(value) for value in record["world_bbox"])
            for record in placements
            if isinstance(record, Mapping) and "world_bbox" in record
        ]
        if bboxes:
            min_x = min(bbox[0] for bbox in bboxes)
            max_x = max(bbox[3] for bbox in bboxes)
            min_z = min(bbox[2] for bbox in bboxes)
            max_z = max(bbox[5] for bbox in bboxes)
            max_y = max(bbox[4] for bbox in bboxes)
            return (
                (min_x + max_x) // 2,
                max(max_y + int(summary["teleport_y_offset"]), int(summary["teleport_min_y"])),
                (min_z + max_z) // 2,
            )

    footprint = summary.get("footprint")
    if isinstance(footprint, Mapping):
        return (
            (int(footprint["min_x"]) + int(footprint["max_x"])) // 2,
            int(summary["teleport_min_y"]),
            (int(footprint["min_z"]) + int(footprint["max_z"])) // 2,
        )
    return (0, int(summary["teleport_min_y"]), 0)


def _teleport_command(target: tuple[int, int, int]) -> str:
    x, y, z = target
    return f"tp @p {x} {y} {z}"


def _teleport_player_live(args: argparse.Namespace, command: str) -> None:
    Editor, _Block = _import_gdpc()
    editor = Editor(buffering=False, host=_normalise_host(args.host))
    editor.runCommandGlobal(command, syncWithBuffer=True)
    print(f"[tp] ran: /{command}")


def _placement_record(origin: np.ndarray, placement: Any) -> dict[str, Any]:
    return {
        "cell_id": placement.slot.cell_id,
        "zone_id": placement.slot.zone_id,
        "building_type": placement.slot.building_type,
        "seed": placement.state.seed,
        "level": placement.level,
        "entrance_face": placement.entrance_face,
        "rotation_steps": placement.rotation_steps,
        "local_slot": {
            "x": placement.slot.x,
            "y": placement.slot.y,
            "z": placement.slot.z,
            "width": placement.slot.width,
            "depth": placement.slot.depth,
        },
        "local_bbox": list(placement.bbox),
        "world_bbox": list(_world_bbox(origin, placement.bbox)),
        "block_count": len(placement.blocks),
    }


def _clear_bbox(
    *,
    editor: Any,
    block_cls: Any,
    origin: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
    extra_y: int,
) -> int:
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    air = block_cls("minecraft:air")
    count = 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + extra_y + 1):
            for z in range(min_z, max_z + 1):
                editor.placeBlock(
                    (int(origin[0]) + x, y, int(origin[2]) + z),
                    air,
                )
                count += 1
    return count


def _place_blocks(
    *,
    editor: Any,
    block_cls: Any,
    origin: np.ndarray,
    blocks: Sequence[BlueprintBlock],
    flush_every: int,
) -> int:
    count = 0
    for block in blocks:
        editor.placeBlock(
            _world_pos(origin, block),
            block_cls(str(block["id"]), dict(block.get("props", {}))),
        )
        count += 1
        if flush_every > 0 and count % flush_every == 0:
            editor.flushBuffer()
    return count


def _place_prefabs_live(
    args: argparse.Namespace,
    *,
    origin: np.ndarray,
    plan: ResidentialSettlementPlacementPlan,
) -> dict[str, int]:
    Editor, Block = _import_gdpc()
    editor = Editor(buffering=True, host=_normalise_host(args.host))
    placed = 0
    cleared = 0
    try:
        for index, placement in enumerate(plan.placements, start=1):
            if args.clear_prefab_volume:
                cleared += _clear_bbox(
                    editor=editor,
                    block_cls=Block,
                    origin=origin,
                    bbox=placement.bbox,
                    extra_y=args.prefab_clear_extra_y,
                )
                editor.flushBuffer()
            placed += _place_blocks(
                editor=editor,
                block_cls=Block,
                origin=origin,
                blocks=placement.blocks,
                flush_every=args.flush_every,
            )
            editor.flushBuffer()
            print(
                "[prefab] "
                f"{index}/{len(plan.placements)} cell={placement.slot.cell_id} "
                f"type={placement.slot.building_type} seed={placement.state.seed} "
                f"level={placement.level} world_bbox={_world_bbox(origin, placement.bbox)} "
                f"blocks={len(placement.blocks)}"
            )
    finally:
        editor.flushBuffer()
        editor.buffering = False
    return {"placed": placed, "cleared": cleared}


def _summarise_plan(plan: ResidentialSettlementPlacementPlan) -> dict[str, Any]:
    return {
        "complete": plan.is_complete,
        "placements": len(plan.placements),
        "rejections": len(plan.rejections),
        "levels": dict(Counter(str(placement.level) for placement in plan.placements)),
        "seeds": dict(Counter(str(placement.state.seed) for placement in plan.placements)),
        "building_types": dict(Counter(placement.slot.building_type for placement in plan.placements)),
        "rejections_sample": [
            {
                "cell_id": rejection.slot.cell_id,
                "building_type": rejection.slot.building_type,
                "width": rejection.slot.width,
                "depth": rejection.slot.depth,
                "reason": rejection.reason,
            }
            for rejection in plan.rejections[:8]
        ],
    }


def main() -> int:
    args = _parse_args()
    if args.lighting_only:
        print("[lighting] lighting-only mode: reusing existing settlement data.")
    else:
        _run_upstream_pipeline(args)

    origin, zone_map = _load_settlement_context(args.upstream_dir)
    footprint = None
    if zone_map is not None:
        depth, width = zone_map.shape
        footprint = {
            "min_x": int(origin[0]),
            "min_z": int(origin[2]),
            "max_x": int(origin[0]) + width - 1,
            "max_z": int(origin[2]) + depth - 1,
            "width": int(width),
            "depth": int(depth),
        }
    raw_rects = _load_plot_rects(args.upstream_dir)
    compatible_rects = _filter_rects(
        raw_rects,
        min_width=args.min_slot_width,
        min_depth=args.min_slot_depth,
    )
    rects = _select_varied_rects(compatible_rects, max_count=args.max_prefabs)
    package_entries = _resolve_package_entries(args)
    building_types = tuple(dict(_parse_typed_package(entry) for entry in package_entries))
    slots = _slots_from_rects(
        rects,
        zone_map=zone_map,
        building_types=building_types,
        default_building_type="residential",
        floor_y_offset=args.floor_y_offset,
    )

    plan = (
        _load_strict_plan(args, slots)
        if args.mode == "strict"
        else _load_typed_plan(args, slots)
    )
    summary = {
        "origin": [int(value) for value in origin.tolist()],
        "source_rectangles": len(raw_rects),
        "compatible_rectangles": len(compatible_rects),
        "selected_rectangles": len(rects),
        "filtered_rectangles": len(raw_rects) - len(compatible_rects),
        "block_mode": args.block_mode,
        "mode": args.mode,
        "teleport_y_offset": args.teleport_y_offset,
        "teleport_min_y": args.teleport_min_y,
        "plan": _summarise_plan(plan),
        "footprint": footprint,
        "placements": [
            _placement_record(origin, placement)
            for placement in plan.placements
        ],
    }
    if footprint is not None:
        print(
            "[town] "
            f"origin=({int(origin[0])}, {int(origin[1])}, {int(origin[2])}) "
            f"footprint=x:{footprint['min_x']}..{footprint['max_x']} "
            f"z:{footprint['min_z']}..{footprint['max_z']}"
        )
    print(
        "[prefab-plan] "
        f"source={len(raw_rects)} compatible={len(compatible_rects)} "
        f"selected={len(rects)} placements={len(plan.placements)} "
        f"rejections={len(plan.rejections)} "
        f"levels={summary['plan']['levels']} seeds={summary['plan']['seeds']}"
    )
    for record in summary["placements"]:
        print(
            "[prefab-plan] "
            f"cell={record['cell_id']} type={record['building_type']} "
            f"seed={record['seed']} level={record['level']} "
            f"world_bbox={tuple(record['world_bbox'])}"
        )
    for rejection in summary["plan"]["rejections_sample"]:
        print(
            "[prefab-reject] "
            f"cell={rejection['cell_id']} type={rejection['building_type']} "
            f"slot={rejection['width']}x{rejection['depth']} reason={rejection['reason']}"
        )

    if args.dry_run:
        print("[prefab] dry run; live prefab placement skipped")
    elif args.lighting_only:
        print("[lighting] lighting-only mode: town and prefab placement skipped.")
        summary["lighting"] = _deploy_town_lighting_live(
            args,
            origin=origin,
            plan=plan,
        )
    else:
        if args.use_upstream_deploy:
            summary["reserved_prefab_lots"] = _filter_plot_buildings_for_plan(
                args.upstream_dir,
                plan,
            )
            _deploy_upstream_settlement(args)
            if args.skip_town_surfaces:
                print("[prefab-foundations] skipped by --skip-town-surfaces")
            else:
                summary["prefab_foundations"] = _prepare_prefab_foundations_live(
                    args,
                    origin=origin,
                    plan=plan,
                )
        elif args.skip_town_surfaces:
            print("[town-surfaces] skipped by --skip-town-surfaces")
        else:
            summary["town_surfaces"] = _deploy_town_surfaces_live(
                args,
                origin=origin,
                plan=plan,
            )
        live_stats = _place_prefabs_live(args, origin=origin, plan=plan)
        summary["live"] = live_stats
        print(
            "[prefab] live placement complete: "
            f"placed={live_stats['placed']} cleared={live_stats['cleared']}"
        )
        if args.town_lighting:
            summary["lighting"] = _deploy_town_lighting_live(
                args,
                origin=origin,
                plan=plan,
            )
        else:
            print("[lighting] skipped by --no-town-lighting")

    teleport_target = _teleport_target(summary)
    teleport_command = _teleport_command(teleport_target)
    summary["teleport_target"] = list(teleport_target)
    summary["teleport_command"] = f"/{teleport_command}"
    if not args.dry_run and args.teleport:
        if _uses_player_origin_capture(args):
            summary["teleport_skipped_reason"] = "player_origin_capture"
            print(
                "[tp] skipped: generation used player-centred capture; "
                "overview command was not run"
            )
        else:
            _teleport_player_live(args, teleport_command)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] wrote {args.summary_out}")
    print(f"[next] overview teleport: /{teleport_command}")
    return 0 if plan.is_complete else 1


if __name__ == "__main__":
    sys.exit(main())
