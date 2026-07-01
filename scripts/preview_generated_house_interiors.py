"""Render interiors selected by an actual generated prefab house."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from prefab_housing import Brief, build_house
from prefab_housing.types import HouseResult, SemanticBlockDict
from voxel_renderer.api import compose_view_grid, render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_generated_house_interiors"


def _parse_footprint(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("footprint must use WIDTHxDEPTH, for example 24x24")
    try:
        vx, vz = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("footprint dimensions must be integers") from exc
    if vx < 8 or vz < 8:
        raise argparse.ArgumentTypeError("footprint dimensions must be at least one default cell")
    return (vx, vz)


def _floor_plate_blocks(result: HouseResult) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    for cell in result.semantic_cells:
        (x0, y0, z0), (x1, _, z1) = cell.voxel_bbox
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                blocks.append({"x": x, "y": y0, "z": z, "id": "minecraft:spruce_planks"})
    return blocks


def _room_summary(result: HouseResult) -> dict[str, object]:
    rooms: list[dict[str, object]] = []
    for room in result.room_interiors:
        ids = sorted({str(block["id"]) for block in room.blocks})
        property_count = sum(1 for block in room.blocks if "properties" in block)
        rooms.append(
            {
                "cell_index": room.cell_index,
                "room_type": room.room_type,
                "variant_id": room.variant_id,
                "block_count": len(room.blocks),
                "property_block_count": property_count,
                "block_ids": ids,
            }
        )
    return {
        "metadata": {
            "seed": result.metadata.seed,
            "cell_grid_size": result.metadata.cell_grid_size,
            "cell_voxel_size": result.metadata.cell_voxel_size,
            "score_total": result.metadata.score_total,
            "interior_cache_stats": result.metadata.interior_cache_stats,
        },
        "rooms": rooms,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview interiors from build_house().")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occupants", type=int, default=3)
    parser.add_argument("--footprint", type=_parse_footprint, default=(24, 24))
    parser.add_argument("--search-iterations", type=int, default=128)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--backend", choices=("auto", "pyrender", "trimesh"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_house(
        Brief(
            occupant_count=args.occupants,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=args.seed,
        ),
        footprint_xz=args.footprint,
        search_iterations=args.search_iterations,
    )
    blocks = _floor_plate_blocks(result) + result.interior_blocks

    footprint_label = f"{args.footprint[0]}x{args.footprint[1]}"
    out_dir = OUT_ROOT / f"seed_{args.seed}_{footprint_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(_room_summary(result), indent=2), encoding="utf-8")

    views = render_orthographic_views(
        blocks,
        width=args.width,
        height=args.height,
        backend=args.backend,
    )
    for name, b64 in views.items():
        (out_dir / f"{name}.png").write_bytes(base64.b64decode(b64))
    (out_dir / "composite.png").write_bytes(compose_view_grid(views))

    room_counts: dict[str, int] = {}
    for room in result.room_interiors:
        room_counts[room.room_type] = room_counts.get(room.room_type, 0) + 1
    print(
        f"seed={args.seed} footprint={footprint_label} score={result.metadata.score_total:.3f} "
        f"rooms={room_counts} interior_blocks={len(result.interior_blocks)} out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
