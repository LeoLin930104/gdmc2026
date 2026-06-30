"""Topology-only housing-plan preview.

Renders the extracted planning stage as plain utility-coloured cubes. This is
the first isolated iteration loop: house request -> 3D utility-marked cell
plan, with no facade, appendage, roof, or interior noise.
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
from pathlib import Path

from prefab_housing import Brief, HousingPlanTuning, generate_housing_plan, render_housing_plan_blocks
from voxel_renderer.api import render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_housing_plan"

FOOTPRINT = (24, 24)
ITERS = 64
RENDER_W = 720
RENDER_H = 540


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a topology-only housing plan.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level for pipeline timing output.",
    )
    return parser.parse_args()


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    brief = Brief(
        occupant_count=4,
        household_type="single_family",
        material_theme="sci_fi_modular",
        seed=42,
    )
    plan = generate_housing_plan(
        brief,
        footprint_xz=FOOTPRINT,
        search_iterations=ITERS,
        tuning=HousingPlanTuning(quirkiness=0.5),
    )
    blocks = render_housing_plan_blocks(plan)
    print(
        f"grid={plan.metadata.cell_grid_size} blocks={len(blocks)} "
        f"score={plan.metadata.score_total:.3f}"
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    views = render_orthographic_views(
        blocks, width=RENDER_W, height=RENDER_H, backend="auto"
    )
    for name, b64 in views.items():
        (OUT_ROOT / f"{name}.png").write_bytes(base64.b64decode(b64))
    print(f"out_dir={OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
