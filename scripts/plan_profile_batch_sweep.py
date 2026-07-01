"""Planning-only batch sweep across standard housing-plan presets.

For each preset, rerun planning with progressively larger search budgets until
either the requested score threshold is reached or the configured max-iteration
cap is exhausted. This intentionally avoids exterior rendering so poor options
do not pay render cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from prefab_housing import HousingRequest, PLAN_PROFILES, generate_housing_plan_for_request, save_housing_plan_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "plan_profile_batch_sweep"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planning-only batch sweeps across standard housing-plan presets.")
    parser.add_argument("--target-score", type=float, default=0.8, help="Stop early once a profile reaches this score.")
    parser.add_argument("--max-iterations", type=int, default=512, help="Maximum search iterations per profile.")
    parser.add_argument("--iteration-step", type=int, default=64, help="Iteration increase applied between attempts.")
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


def _attempt_profile(
    request: HousingRequest,
    *,
    base_iterations: int,
    max_iterations: int,
    iteration_step: int,
    tuning,
    target_score: float,
):
    attempts: list[dict[str, object]] = []
    best_plan = None
    best_iterations = base_iterations
    best_score = float("-inf")
    iterations = base_iterations
    while iterations <= max_iterations:
        try:
            plan = generate_housing_plan_for_request(
                request,
                search_iterations=iterations,
                tuning=tuning,
            )
        except Exception as error:
            attempts.append(
                {
                    "iterations": iterations,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            logging.warning("Profile attempt failed at %s iterations: %s", iterations, error)
            iterations += iteration_step
            continue
        attempt = {
            "iterations": iterations,
            "status": "ok",
            "score": plan.metadata.score_total,
            "grid": plan.metadata.cell_grid_size,
            "occupants": plan.metadata.occupant_count,
            "stage_timings_ms": dict(plan.metadata.stage_timings_ms),
        }
        attempts.append(attempt)
        if plan.metadata.score_total > best_score:
            best_plan = plan
            best_iterations = iterations
            best_score = plan.metadata.score_total
        if plan.metadata.score_total >= target_score:
            break
        iterations += iteration_step
    if best_plan is None:
        raise RuntimeError(f"No valid plan generated across iterations {base_iterations}..{max_iterations}")
    return best_plan, best_iterations, attempts


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
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
            best_plan, best_iterations, attempts = _attempt_profile(
                request,
                base_iterations=profile.search_iterations,
                max_iterations=max(args.max_iterations, profile.search_iterations),
                iteration_step=max(1, args.iteration_step),
                tuning=profile.tuning,
                target_score=args.target_score,
            )
            save_housing_plan_report(best_plan, out_dir / "report.png", title=profile.name)
            record = {
                "profile": profile.name,
                "status": "ok",
                "footprint_xz": profile.footprint_xz,
                "capacity_override": profile.capacity_override,
                "best_iterations": best_iterations,
                "best_score": best_plan.metadata.score_total,
                "best_grid": best_plan.metadata.cell_grid_size,
                "attempts": attempts,
            }
            summary.append(record)
            (out_dir / "summary.json").write_text(json.dumps(record, indent=2))
            print(
                f"{profile.name}: best_score={best_plan.metadata.score_total:.3f} "
                f"best_iterations={best_iterations} grid={best_plan.metadata.cell_grid_size} out={out_dir}"
            )
        except Exception as error:
            logging.exception("Batch sweep failed for %s", profile.name)
            record = {
                "profile": profile.name,
                "status": "failed",
                "footprint_xz": profile.footprint_xz,
                "capacity_override": profile.capacity_override,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            summary.append(record)
            (out_dir / "summary.json").write_text(json.dumps(record, indent=2))
            print(f"{profile.name}: failed error={type(error).__name__}: {error} out={out_dir}")
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
