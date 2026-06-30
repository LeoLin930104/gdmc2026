"""Render a saved .wallface design through the live shell face builder."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from prefab_housing.catalogue.shell import build_face_texture_panel, set_active_wall_face_design
from prefab_housing.palette import resolve_palette
from voxel_renderer.api import render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_wallface_design"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a saved wall-face design.")
    parser.add_argument("design", help="Path to a .wallface file")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--axis", choices=("x", "z"), default="x")
    parser.add_argument("--pod", default="living")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_active_wall_face_design(args.design)
    try:
        blocks = build_face_texture_panel(
            axis=args.axis,
            fixed=0,
            outward_sign=1,
            a0=0,
            a1=args.width - 1,
            y0=0,
            y1=args.height - 1,
            palette=resolve_palette("sci_fi_modular"),
            pod_name=args.pod,
        )
    finally:
        set_active_wall_face_design(None)

    views = render_orthographic_views(blocks, width=720, height=540, backend="auto")
    for name, b64 in views.items():
        (OUT_ROOT / f"{Path(args.design).stem}_{name}.png").write_bytes(base64.b64decode(b64))
    print(f"blocks={len(blocks)} out_dir={OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
