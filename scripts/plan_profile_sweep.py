"""Render whole-exterior reviews across standard housing-plan presets."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prefab_housing import (
    HousingRequest,
    PLAN_PROFILES,
    generate_housing_plan_for_request,
    render_plan_exterior,
)
from prefab_housing.plan_review import save_housing_plan_report
from voxel_renderer.api import render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "plan_profile_sweep"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render whole-exterior reviews across standard housing-plan presets.")
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


def _save_iso_overview(entries: list[tuple[str, Path]]) -> Path:
    cols = 3
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes_list:
        ax.axis("off")
    for ax, (name, path) in zip(axes_list, entries, strict=False):
        ax.imshow(plt.imread(path))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Housing Plan Iso Overview")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_ROOT / "overview_iso.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _write_failure_summary(out_dir: Path, *, profile_name: str, error: Exception) -> None:
    payload = {
        "profile": profile_name,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    iso_entries: list[tuple[str, Path]] = []
    summary: list[dict[str, object]] = []
    for profile in PLAN_PROFILES.values():
        out_dir = OUT_ROOT / profile.name
        out_dir.mkdir(parents=True, exist_ok=True)
        request = HousingRequest(
            footprint_xz=profile.footprint_xz,
            utility_type=profile.utility_type,
            capacity_override=profile.capacity_override,
            max_storeys=profile.max_storeys,
            material_theme="sci_fi_modular",
            seed=42,
        )
        try:
            plan = generate_housing_plan_for_request(
                request,
                search_iterations=profile.search_iterations,
                tuning=profile.tuning,
            )
            blocks = render_plan_exterior(plan, material_theme=request.material_theme or "sci_fi_modular")
            views = render_orthographic_views(blocks, width=720, height=540, backend="auto")
            for name, b64 in views.items():
                (out_dir / f"{name}.png").write_bytes(base64.b64decode(b64))
            iso_entries.append((profile.name, out_dir / "iso_left.png"))
            save_housing_plan_report(plan, out_dir / "report.png", title=profile.name)
            record = {
                "profile": profile.name,
                "status": "ok",
                "grid": plan.metadata.cell_grid_size,
                "score": plan.metadata.score_total,
                "occupants": plan.metadata.occupant_count,
                "out_dir": str(out_dir),
            }
            summary.append(record)
            (out_dir / "summary.json").write_text(json.dumps(record, indent=2))
            print(
                f"{profile.name}: grid={plan.metadata.cell_grid_size} "
                f"score={plan.metadata.score_total:.3f} out={out_dir}"
            )
        except Exception as error:
            logging.exception("Profile sweep failed for %s", profile.name)
            _write_failure_summary(out_dir, profile_name=profile.name, error=error)
            record = {
                "profile": profile.name,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "out_dir": str(out_dir),
            }
            summary.append(record)
            print(f"{profile.name}: failed error={type(error).__name__}: {error} out={out_dir}")
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    if iso_entries:
        overview = _save_iso_overview(iso_entries)
        print(f"overview={overview}")
    else:
        print("overview=none")
    print(f"summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
