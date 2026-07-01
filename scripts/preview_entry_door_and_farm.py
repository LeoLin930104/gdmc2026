"""Offline voxel previews for the entry-door module and hydrated farm layout."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from prefab_housing.entry_door import generate_entry_door_blocks
from prefab_housing.grid import CellGrid
from prefab_housing.types import SemanticBlockDict, SemanticCell
from voxel_renderer.api import compose_gallery_grid, compose_view_grid, render_orthographic_views


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_entry_door_and_farm"
PREMADE_DIR = REPO_ROOT / "narrative" / "Premade Builds"

CELL_SIZE = (8, 6, 8)


def _block(
    x: int,
    y: int,
    z: int,
    block_id: str,
    properties: dict[str, str] | None = None,
) -> SemanticBlockDict:
    block: SemanticBlockDict = {"x": x, "y": y, "z": z, "id": block_id}
    if properties:
        block["properties"] = dict(properties)
    return block


def _load_farm_field() -> ModuleType:
    if str(PREMADE_DIR) not in sys.path:
        sys.path.insert(0, str(PREMADE_DIR))
    spec = importlib.util.spec_from_file_location("farm_field", PREMADE_DIR / "farm_field.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load farm_field.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry_cell() -> SemanticCell:
    vx, vy, vz = CELL_SIZE
    return SemanticCell(
        cell_index=(0, 0, 0),
        voxel_bbox=((0, 0, 0), (vx - 1, vy - 1, vz - 1)),
        label="entry",
        role="circulation",
        occupancy_capacity=1,
        daylight_score=0.0,
        privacy_depth=0,
        door_faces=("north",),
        window_faces=(),
        interior_volume_voxels=(vx - 2) * (vy - 2) * (vz - 2),
        pod_template_id="entry@0",
        opening_pattern="edge_only",
    )


def _entry_context_blocks() -> list[SemanticBlockDict]:
    vx, vy, vz = CELL_SIZE
    blocks: list[SemanticBlockDict] = []

    for x in range(vx):
        for z in range(vz):
            blocks.append(_block(x, 0, z, "minecraft:spruce_planks"))

    # Front wall around the carved aperture; the door module fills the opening.
    for y in range(1, vy):
        for x in range(vx):
            if 2 <= x <= 5 and 1 <= y <= 3:
                continue
            blocks.append(_block(x, y, 0, "minecraft:white_concrete"))

    for z in range(1, vz):
        for y in range(1, vy):
            blocks.append(_block(0, y, z, "minecraft:white_concrete"))
            blocks.append(_block(vx - 1, y, z, "minecraft:white_concrete"))
    return blocks


def build_entry_door_scene() -> tuple[list[SemanticBlockDict], dict[str, Any]]:
    cell = _entry_cell()
    blocks = _entry_context_blocks() + generate_entry_door_blocks(
        [cell],
        CellGrid(cx=1, cy=1, cz=1),
        material_theme="sci_fi_modular",
    )
    stats = {
        "scene": "entry_door_module",
        "door_blocks": sum(1 for block in blocks if str(block["id"]).endswith("_door")),
        "total_blocks": len(blocks),
    }
    return blocks, stats


def build_entry_door_focus_scene() -> tuple[list[SemanticBlockDict], dict[str, Any]]:
    cell = _entry_cell()
    blocks = generate_entry_door_blocks(
        [cell],
        CellGrid(cx=1, cy=1, cz=1),
        material_theme="sci_fi_modular",
    )
    stats = {
        "scene": "entry_door_module_only",
        "door_blocks": sum(1 for block in blocks if str(block["id"]).endswith("_door")),
        "total_blocks": len(blocks),
    }
    return blocks, stats


def _farm_cells(width: int, depth: int) -> list[tuple[int, int]]:
    return [(x, z) for x in range(width) for z in range(depth)]


def _is_hydrated(cell: tuple[int, int], water: set[tuple[int, int]], radius: int) -> bool:
    x, z = cell
    return any(max(abs(x - wx), abs(z - wz)) <= radius for wx, wz in water)


def build_farm_scene(width: int, depth: int) -> tuple[list[SemanticBlockDict], dict[str, Any]]:
    farm_field = _load_farm_field()
    cell_set, border, water, crop_land = farm_field.farm_layout(_farm_cells(width, depth))

    blocks: list[SemanticBlockDict] = []
    for x, z in sorted(cell_set):
        if (x, z) in border:
            blocks.append(_block(x, 0, z, "minecraft:dirt"))
            blocks.append(_block(x, 1, z, "minecraft:oak_log"))
        elif (x, z) in water:
            blocks.append(_block(x, 0, z, "minecraft:water"))
        else:
            blocks.append(_block(x, 0, z, "minecraft:coarse_dirt"))
            marker = "minecraft:yellow_carpet" if (x + z) % 3 else "minecraft:lime_carpet"
            blocks.append(_block(x, 1, z, marker))

    radius = int(farm_field.HYDRATION_RADIUS)
    uncovered = sorted(cell for cell in crop_land if not _is_hydrated(cell, water, radius))
    stats = {
        "scene": "hydrated_farm_layout",
        "width": width,
        "depth": depth,
        "border_cells": len(border),
        "water_cells": len(water),
        "crop_land_cells": len(crop_land),
        "uncovered_crop_land_cells": len(uncovered),
        "total_blocks": len(blocks),
    }
    return blocks, stats


def _render_scene(
    name: str,
    blocks: list[SemanticBlockDict],
    stats: dict[str, Any],
    *,
    width: int,
    height: int,
    backend: str,
) -> dict[str, str]:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    views = render_orthographic_views(blocks, width=width, height=height, backend=backend)
    for view_name, b64 in views.items():
        (out_dir / f"{view_name}.png").write_bytes(base64.b64decode(b64))
    (out_dir / "composite.png").write_bytes(compose_view_grid(views))
    return views


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render offline voxel previews for the entry door and hydrated farm layout."
    )
    parser.add_argument("--farm-size", default="24x18", help="Farm preview size as WIDTHxDEPTH.")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--backend", choices=("auto", "pyrender", "trimesh"), default="auto")
    return parser.parse_args()


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("farm size must use WIDTHxDEPTH, for example 24x18")
    try:
        width, depth = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("farm size dimensions must be integers") from exc
    if width < 3 or depth < 3:
        raise argparse.ArgumentTypeError("farm size dimensions must be at least 3")
    return width, depth


def main() -> int:
    args = _parse_args()
    farm_width, farm_depth = _parse_size(args.farm_size)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    entry_blocks, entry_stats = build_entry_door_scene()
    entry_views = _render_scene(
        "entry_door_module",
        entry_blocks,
        entry_stats,
        width=args.width,
        height=args.height,
        backend=args.backend,
    )

    focus_blocks, focus_stats = build_entry_door_focus_scene()
    focus_views = _render_scene(
        "entry_door_module_only",
        focus_blocks,
        focus_stats,
        width=args.width,
        height=args.height,
        backend=args.backend,
    )

    farm_blocks, farm_stats = build_farm_scene(farm_width, farm_depth)
    farm_views = _render_scene(
        "hydrated_farm_layout",
        farm_blocks,
        farm_stats,
        width=args.width,
        height=args.height,
        backend=args.backend,
    )

    overview = compose_gallery_grid(
        [
            ("entry_door_module", entry_views["profile"]),
            ("entry_door_only", focus_views["profile"]),
            ("hydrated_farm_layout", farm_views["top"]),
        ],
        columns=3,
    )
    (OUT_ROOT / "overview.png").write_bytes(overview)

    print(
        f"entry_door_module: blocks={entry_stats['total_blocks']} "
        f"door_blocks={entry_stats['door_blocks']}"
    )
    print(
        f"entry_door_module_only: blocks={focus_stats['total_blocks']} "
        f"door_blocks={focus_stats['door_blocks']}"
    )
    print(
        f"hydrated_farm_layout: size={farm_width}x{farm_depth} "
        f"water={farm_stats['water_cells']} crop_land={farm_stats['crop_land_cells']} "
        f"uncovered={farm_stats['uncovered_crop_land_cells']}"
    )
    print(f"out_dir={OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
