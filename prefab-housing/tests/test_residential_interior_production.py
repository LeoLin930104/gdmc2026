from __future__ import annotations

import pytest

from prefab_housing import (
    RESIDENTIAL_LEVEL_SPECS,
    analyse_interior_production,
    brief_for_residential_level,
    build_house,
    expected_room_counts_from_programme,
)
from prefab_housing.programme import resolve_programme


@pytest.mark.parametrize("level", tuple(RESIDENTIAL_LEVEL_SPECS))
def test_residential_levels_produce_required_interiors(level: int) -> None:
    spec = RESIDENTIAL_LEVEL_SPECS[level]
    brief = brief_for_residential_level(level, seed=42)
    result = build_house(
        brief,
        footprint_xz=spec.footprint_xz,
        search_iterations=spec.search_iterations,
        plan_tuning=spec.tuning,
    )
    programme = resolve_programme(brief, "residential")
    report = analyse_interior_production(
        result,
        expected_room_counts=expected_room_counts_from_programme(programme),
    )

    assert result.metadata.score_breakdown["functional_adequacy"] == pytest.approx(1.0)
    assert result.metadata.score_total >= 0.7
    assert report.is_valid, report
    assert report.interior_block_count > 0
    assert report.property_block_count > 0
