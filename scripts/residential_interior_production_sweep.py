"""Run residential generated-house interior production checks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from prefab_housing import (
    RESIDENTIAL_LEVEL_SPECS,
    analyse_interior_production,
    brief_for_residential_level,
    build_house,
    expected_room_counts_from_programme,
)
from prefab_housing.programme import resolve_programme

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "residential_interior_production_sweep"
MIN_ACCEPTED_SCORE = 0.7


@dataclass(frozen=True, slots=True)
class ResidentialSweepCase:
    name: str
    level: int
    footprint_xz: tuple[int, int]
    max_storeys: int
    search_iterations: int


RESIDENTIAL_SWEEP_CASES: tuple[ResidentialSweepCase, ...] = tuple(
    ResidentialSweepCase(
        name=spec.name,
        level=spec.level,
        footprint_xz=spec.footprint_xz,
        max_storeys=spec.max_storeys,
        search_iterations=spec.search_iterations,
    )
    for spec in RESIDENTIAL_LEVEL_SPECS.values()
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residential interior production checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed generated-house interior check.",
    )
    return parser.parse_args()


def _run_case(case: ResidentialSweepCase, *, seed: int) -> dict[str, object]:
    spec = RESIDENTIAL_LEVEL_SPECS[case.level]
    brief = brief_for_residential_level(case.level, seed=seed)
    result = build_house(
        brief,
        footprint_xz=case.footprint_xz,
        search_iterations=case.search_iterations,
        plan_tuning=spec.tuning,
    )
    programme = resolve_programme(brief, "residential")
    report = analyse_interior_production(
        result,
        expected_room_counts=expected_room_counts_from_programme(programme),
    )
    functional = result.metadata.score_breakdown.get("functional_adequacy", 0.0)
    status = (
        "ok"
        if report.is_valid
        and functional == 1.0
        and result.metadata.score_total >= MIN_ACCEPTED_SCORE
        else "failed"
    )
    return {
        "case": asdict(case),
        "status": status,
        "min_accepted_score": MIN_ACCEPTED_SCORE,
        "score_total": result.metadata.score_total,
        "score_breakdown": dict(result.metadata.score_breakdown),
        "grid": result.metadata.cell_grid_size,
        "interior_cache_stats": result.metadata.interior_cache_stats,
        "interior_report": asdict(report),
    }


def main() -> int:
    args = _parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    exit_code = 0
    for case in RESIDENTIAL_SWEEP_CASES:
        record = _run_case(case, seed=args.seed)
        records.append(record)
        status = str(record["status"])
        score = float(record["score_total"])
        report = record["interior_report"]
        assert isinstance(report, dict)
        print(
            f"{case.name}: status={status} score={score:.3f} "
            f"rooms={report['room_counts']} blocks={report['interior_block_count']}"
        )
        if status != "ok":
            exit_code = 1
            if args.fail_fast:
                break
    out_path = OUT_ROOT / f"seed_{args.seed}_summary.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"summary={out_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
