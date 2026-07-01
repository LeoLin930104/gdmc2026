"""Render a simplified room-layout review for one generated room."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prefab_housing import Brief, build_house, save_room_layout_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_room_layout"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one room layout plan.")
    parser.add_argument("--room-type", default="bedroom")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = build_house(
        Brief(
            occupant_count=3,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=42,
        ),
        footprint_xz=(24, 24),
        search_iterations=96,
    )
    room = next((item for item in result.room_interiors if item.room_type == args.room_type), None)
    if room is None or room.layout is None:
        raise RuntimeError(f"No room layout found for room_type={args.room_type!r}")
    out_path = OUT_ROOT / f"{args.room_type}.png"
    save_room_layout_report(room.layout, out_path, title=f"{args.room_type} layout")
    print(f"room={args.room_type} out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
