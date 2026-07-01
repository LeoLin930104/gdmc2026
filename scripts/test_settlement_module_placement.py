"""Dry-test cached residential modules against settlement plot rectangles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prefab_housing.minecraft_animation import (
    ResidentialSettlementPlacementPlan,
    SettlementBuildSlot,
    load_residential_upgrade_package,
    plan_residential_settlement_placements,
    plan_typed_residential_settlement_placements,
)
from prefab_housing.types import FaceName

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLOTS_PATH = REPO_ROOT / "data" / "settlement_plots.npz"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run cached residential package placement against settlement plots. "
            "This does not call GDPC or write to Minecraft."
        )
    )
    parser.add_argument("--plots", type=Path, default=DEFAULT_PLOTS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--strict-package", default="seed_043.pbp")
    parser.add_argument(
        "--typed-package",
        action="append",
        default=None,
        metavar="TYPE=PACKAGE",
        help=(
            "Building type to cached package mapping. May be repeated. "
            "Defaults to the checked-in residential package variants."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "typed", "both"),
        default="both",
    )
    parser.add_argument(
        "--target-entrance-face",
        choices=HORIZONTAL_FACES,
        default=None,
        help="Optional common entrance face constraint for all placed modules.",
    )
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument(
        "--block-mode",
        choices=("core", "full", "structure"),
        default="core",
        help="Which cached residential block section to fit/place.",
    )
    parser.add_argument(
        "--min-slot-width",
        type=int,
        default=20,
        help="Drop raw settlement rectangles narrower than this before placement.",
    )
    parser.add_argument(
        "--min-slot-depth",
        type=int,
        default=20,
        help="Drop raw settlement rectangles shallower than this before placement.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable summary.",
    )
    return parser.parse_args()


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


def _load_plot_rects(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    data = np.load(path, allow_pickle=True)
    if "building_rects" not in data:
        raise ValueError(f"{path} does not contain a building_rects array")
    module_size = int(data["module_size"]) if "module_size" in data else None
    return [_extract_rect(record) for record in data["building_rects"]], module_size


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


def _strict_slots(rects: Sequence[Mapping[str, Any]]) -> list[SettlementBuildSlot]:
    return [
        SettlementBuildSlot(
            x=int(rect["x"]),
            y=int(rect.get("y", 0)),
            z=int(rect["z"]),
            width=int(rect["width"]),
            depth=int(rect["depth"]),
            cell_id=int(rect["cell_id"]) if "cell_id" in rect else None,
            zone_id=int(rect["zone_id"]) if "zone_id" in rect else None,
            building_type="residential",
        )
        for rect in rects
    ]


def _typed_slots(
    rects: Sequence[Mapping[str, Any]],
    building_types: Sequence[str],
) -> list[SettlementBuildSlot]:
    if not building_types:
        raise ValueError("at least one building type is required for typed placement")
    slots: list[SettlementBuildSlot] = []
    for index, rect in enumerate(rects):
        synthetic_zone = int(rect.get("cell_id", index)) % len(building_types)
        slots.append(
            SettlementBuildSlot(
                x=int(rect["x"]),
                y=int(rect.get("y", 0)),
                z=int(rect["z"]),
                width=int(rect["width"]),
                depth=int(rect["depth"]),
                cell_id=int(rect["cell_id"]) if "cell_id" in rect else None,
                zone_id=int(rect.get("zone_id", synthetic_zone)),
                building_type=str(rect.get("building_type", building_types[index % len(building_types)])),
            )
        )
    return slots


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


def _load_states(package_path: Path) -> tuple[Any, dict[str, Any]]:
    states, _diffs, manifest = load_residential_upgrade_package(package_path)
    return states, manifest


def _summarise_plan(plan: ResidentialSettlementPlacementPlan) -> dict[str, Any]:
    return {
        "complete": plan.is_complete,
        "placements": len(plan.placements),
        "rejections": len(plan.rejections),
        "levels": dict(Counter(str(placement.level) for placement in plan.placements)),
        "seeds": dict(Counter(str(placement.state.seed) for placement in plan.placements)),
        "interior_styles": dict(
            Counter(str(placement.state.interior_style_id) for placement in plan.placements)
        ),
        "layout_variants": dict(
            Counter(str(placement.state.layout_variant_id) for placement in plan.placements)
        ),
        "building_types": dict(
            Counter(placement.slot.building_type for placement in plan.placements)
        ),
        "sample_rejections": [
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


def _print_summary(label: str, summary: Mapping[str, Any]) -> None:
    print(
        f"[{label}] placements={summary['placements']} "
        f"rejections={summary['rejections']} complete={summary['complete']} "
        f"levels={summary['levels']} seeds={summary['seeds']} "
        f"types={summary['building_types']}"
    )
    for rejection in summary["sample_rejections"]:
        print(
            f"[{label}:reject] cell={rejection['cell_id']} "
            f"type={rejection['building_type']} "
            f"slot={rejection['width']}x{rejection['depth']} "
            f"reason={rejection['reason']}"
        )


def main() -> int:
    args = _parse_args()
    raw_rects, module_size = _load_plot_rects(args.plots)
    rects = _filter_rects(
        raw_rects,
        min_width=args.min_slot_width,
        min_depth=args.min_slot_depth,
    )
    target_face: FaceName | None = args.target_entrance_face
    summary: dict[str, Any] = {
        "plots": str(args.plots),
        "source_plot_count": len(raw_rects),
        "plot_count": len(rects),
        "filtered_plot_count": len(raw_rects) - len(rects),
        "min_slot_width": args.min_slot_width,
        "min_slot_depth": args.min_slot_depth,
        "module_size": module_size,
        "block_mode": args.block_mode,
        "modes": {},
    }

    print(
        f"[plots] path={args.plots} source_count={len(raw_rects)} "
        f"compatible_count={len(rects)} filtered={len(raw_rects) - len(rects)} "
        f"module_size={module_size} block_mode={args.block_mode}"
    )

    if args.mode in {"strict", "both"}:
        strict_path = _resolve_package_path(args.cache_dir, args.strict_package)
        strict_states, strict_manifest = _load_states(strict_path)
        strict_plan = plan_residential_settlement_placements(
            strict_states,
            _strict_slots(rects),
            target_entrance_face=target_face,
            allow_rotate=not args.no_rotate,
            block_mode=args.block_mode,
            fail_fast=args.fail_fast,
        )
        strict_summary = _summarise_plan(strict_plan)
        strict_summary["package"] = str(strict_path)
        strict_summary["package_manifest"] = strict_manifest
        summary["modes"]["strict"] = strict_summary
        _print_summary("strict", strict_summary)

    if args.mode in {"typed", "both"}:
        package_entries = args.typed_package or list(DEFAULT_TYPED_PACKAGES)
        typed_packages: dict[str, list[str]] = defaultdict(list)
        for entry in package_entries:
            building_type, package_name = _parse_typed_package(entry)
            typed_packages[building_type].append(package_name)
        states_by_type = {
            building_type: tuple(
                _load_states(_resolve_package_path(args.cache_dir, package_name))[0]
                for package_name in package_names
            )
            for building_type, package_names in typed_packages.items()
        }
        typed_plan = plan_typed_residential_settlement_placements(
            states_by_type,
            _typed_slots(rects, tuple(typed_packages)),
            target_entrance_face=target_face,
            allow_rotate=not args.no_rotate,
            block_mode=args.block_mode,
            fail_fast=args.fail_fast,
        )
        typed_summary = _summarise_plan(typed_plan)
        typed_summary["packages"] = {
            building_type: [
                str(_resolve_package_path(args.cache_dir, package_name))
                for package_name in package_names
            ]
            for building_type, package_names in typed_packages.items()
        }
        summary["modes"]["typed"] = typed_summary
        _print_summary("typed", typed_summary)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[summary] wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
