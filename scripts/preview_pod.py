"""Standalone exterior face-study render.

Renders the current shared exterior-face builder in isolation, without WFC,
planning, roof, or house-scale assembly.

The study tracks the live shell rule rather than bespoke preview geometry:

- base wall plane on the cell face
- one proud outer frame rectangle
- one neutral glass inset on the base wall plane
"""

from __future__ import annotations

import argparse
import base64
import sys
from dataclasses import dataclass
from pathlib import Path

from prefab_housing.catalogue.shell import build_face_texture_panel, set_active_wall_face_design
from prefab_housing.palette import resolve_palette
from voxel_renderer.api import compose_gallery_grid, render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_pod"

RENDER_W = 720
RENDER_H = 540


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the current wall face treatment.")
    parser.add_argument(
        "--design",
        default=None,
        help="Optional path to a .wallface design file to preview through the live shell builder.",
    )
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class FaceStudy:
    name: str
    axis: str
    face_span: tuple[int, int]
    pod_name: str


STUDIES: tuple[FaceStudy, ...] = (
    FaceStudy("default_living_x", axis="x", face_span=(8, 6), pod_name="living"),
    FaceStudy("default_bedroom_z", axis="z", face_span=(8, 6), pod_name="bedroom"),
    FaceStudy("wide_entry_x", axis="x", face_span=(12, 6), pod_name="entry"),
    FaceStudy("tall_kitchen_z", axis="z", face_span=(8, 8), pod_name="kitchen"),
)


def _translate_positive_origin(blocks: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    min_x = min(int(block["x"]) for block in blocks)
    min_y = min(int(block["y"]) for block in blocks)
    min_z = min(int(block["z"]) for block in blocks)
    out: list[dict[str, int | str]] = []
    for block in blocks:
        out.append(
            {
                "x": int(block["x"]) - min_x,
                "y": int(block["y"]) - min_y,
                "z": int(block["z"]) - min_z,
                "id": str(block["id"]),
            }
        )
    return out


def _build_study_blocks(study: FaceStudy) -> list[dict[str, int | str]]:
    span_axis, span_y = study.face_span
    palette = resolve_palette("sci_fi_modular")
    blocks = build_face_texture_panel(
        axis=study.axis,
        fixed=0,
        outward_sign=1,
        a0=0,
        a1=span_axis - 1,
        y0=0,
        y1=span_y - 1,
        palette=palette,
        pod_name=study.pod_name,
    )
    return _translate_positive_origin(blocks)


def main() -> int:
    args = _parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gallery_tiles: list[tuple[str, str]] = []
    set_active_wall_face_design(args.design)

    for study in STUDIES:
        blocks = _build_study_blocks(study)
        views = render_orthographic_views(
            blocks,
            width=RENDER_W,
            height=RENDER_H,
            backend="auto",
        )
        for name, b64 in views.items():
            (OUT_ROOT / f"{study.name}_{name}.png").write_bytes(base64.b64decode(b64))
        gallery_tiles.append((study.name, views["profile"]))
        print(f"{study.name}: blocks={len(blocks)}")

    gallery = compose_gallery_grid(gallery_tiles, columns=2)
    (OUT_ROOT / "profile_gallery.png").write_bytes(gallery)
    print(f"out_dir={OUT_ROOT}")
    set_active_wall_face_design(None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
