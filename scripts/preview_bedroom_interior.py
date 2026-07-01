"""Render an isolated bedroom interior with the camera-facing shell removed."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from prefab_housing import make_room_request, plan_room, plan_room_layout
from prefab_housing.interior import _component_blocks  # type: ignore[attr-defined]
from prefab_housing.types import SemanticBlockDict, SemanticCell
from voxel_renderer.api import compose_view_grid, render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_bedroom_interior"


def _parse_cell_size(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "cell size must use WIDTHxHEIGHTxDEPTH, for example 10x6x10"
        )
    try:
        vx, vy, vz = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cell size dimensions must be integers") from exc
    if vx < 4 or vy < 4 or vz < 4:
        raise argparse.ArgumentTypeError("cell size must leave at least a 2x2x2 usable interior")
    return (vx, vy, vz)


def _sample_bedroom_cell(cell_size: tuple[int, int, int]) -> SemanticCell:
    vx, vy, vz = cell_size
    return SemanticCell(
        cell_index=(0, 0, 0),
        voxel_bbox=((0, 0, 0), (vx - 1, vy - 1, vz - 1)),
        label="bedroom",
        role="habitable",
        occupancy_capacity=2,
        daylight_score=1.0,
        privacy_depth=3,
        door_faces=("south",),
        window_faces=("north", "east"),
        interior_volume_voxels=max(1, (vx - 2) * (vy - 2) * (vz - 2)),
        pod_template_id="bedroom@preview",
    )


def _cutaway_shell_blocks(cell_size: tuple[int, int, int]) -> list[SemanticBlockDict]:
    vx, vy, vz = cell_size
    blocks: list[SemanticBlockDict] = []
    for x in range(vx):
        for z in range(vz):
            blocks.append({"x": x, "y": 0, "z": z, "id": "minecraft:spruce_planks"})
    for x in range(vx):
        for y in range(1, vy):
            blocks.append({"x": x, "y": y, "z": 0, "id": "minecraft:white_concrete"})
    for z in range(1, vz):
        for y in range(1, vy):
            blocks.append({"x": 0, "y": y, "z": z, "id": "minecraft:white_concrete"})
    return blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one generated bedroom interior.")
    parser.add_argument(
        "--cell-size",
        type=_parse_cell_size,
        default=(10, 6, 10),
        help="Cell voxel size as WIDTHxHEIGHTxDEPTH. Default: 10x6x10.",
    )
    parser.add_argument(
        "--shell",
        choices=("cutaway", "none"),
        default="cutaway",
        help="Render a floor plus north/west walls, or furniture only. Default: cutaway.",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--backend", choices=("auto", "pyrender", "trimesh"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cell = _sample_bedroom_cell(args.cell_size)
    request = make_room_request(cell, utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    furniture_blocks = _component_blocks(layout)
    shell_blocks = _cutaway_shell_blocks(args.cell_size) if args.shell == "cutaway" else []
    blocks = shell_blocks + furniture_blocks

    size_label = f"{args.cell_size[0]}x{args.cell_size[1]}x{args.cell_size[2]}"
    out_dir = OUT_ROOT / f"{size_label}_{args.shell}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")

    views = render_orthographic_views(
        blocks,
        width=args.width,
        height=args.height,
        backend=args.backend,
    )
    for name, b64 in views.items():
        (out_dir / f"{name}.png").write_bytes(base64.b64decode(b64))
    (out_dir / "composite.png").write_bytes(compose_view_grid(views))

    print(
        f"cell_size={size_label} shell={args.shell} placements={len(layout.placements)} "
        f"furniture_blocks={len(furniture_blocks)} total_blocks={len(blocks)} out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
