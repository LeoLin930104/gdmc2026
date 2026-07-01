"""Stage-level tests for the extracted housing-plan module."""

from __future__ import annotations

from pathlib import Path

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.structure import analyse_structure
from prefab_housing import (
    DEFAULT_EXTERIOR_STYLE,
    PLAN_PROFILES,
    Brief,
    HousingRequest,
    HousingPlanTuning,
    analyse_housing_plan,
    render_plan_exterior,
    generate_housing_plan,
    generate_housing_plan_for_request,
    render_housing_plan_blocks,
    resolve_brief_for_request,
    save_housing_plan_report,
)


def _make_plan(seed: int = 42):
    return generate_housing_plan(
        Brief(
            occupant_count=3,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=seed,
        ),
        footprint_xz=(30, 30),
        search_iterations=128,
        tuning=HousingPlanTuning(quirkiness=0.5),
    )


def test_generate_housing_plan_returns_solved_plan() -> None:
    plan = _make_plan()
    assert plan.metadata.cell_grid_size[1] >= 2
    assert plan.metadata.occupant_count == 3
    assert plan.metadata.scale_class in {"compact", "family", "stacked", "vertical"}
    assert plan.metadata.storey_distribution.min_storeys >= 2
    assert plan.metadata.score_total > 0.0
    assert len(plan.cells) == (
        plan.metadata.cell_grid_size[0]
        * plan.metadata.cell_grid_size[1]
        * plan.metadata.cell_grid_size[2]
    )
    assert any(not cell.is_empty for cell in plan.cells)


def test_render_housing_plan_blocks_emits_topology_preview() -> None:
    plan = _make_plan()
    blocks = render_housing_plan_blocks(plan)
    assert blocks
    assert all(block["id"].endswith("_concrete") for block in blocks)


def test_render_plan_exterior_emits_shell_blocks() -> None:
    plan = _make_plan()
    blocks = render_plan_exterior(plan)
    assert blocks
    assert any(block["id"] == "minecraft:black_concrete" for block in blocks)
    assert DEFAULT_EXTERIOR_STYLE == "modular_shell"


def test_housing_plan_is_deterministic() -> None:
    a = _make_plan(seed=42)
    b = _make_plan(seed=42)
    assert a.cells == b.cells
    assert a.metadata.score_total == b.metadata.score_total
    assert a.metadata.score_breakdown == b.metadata.score_breakdown


def test_plan_profiles_cover_small_to_tall() -> None:
    assert "small_house" in PLAN_PROFILES
    assert "grand_mansion" in PLAN_PROFILES
    assert "sky_scraper" in PLAN_PROFILES
    assert PLAN_PROFILES["sky_scraper"].max_storeys > PLAN_PROFILES["small_house"].max_storeys
    assert PLAN_PROFILES["small_house"].capacity_override == 2
    assert PLAN_PROFILES["courtyard_family"].capacity_override == 4


def test_plan_review_emits_report_file(tmp_path: Path) -> None:
    plan = _make_plan()
    analysis = analyse_housing_plan(plan)
    assert analysis.occupied_cells > 0
    report = save_housing_plan_report(plan, tmp_path / "report.png")
    assert report.exists()
    assert report.stat().st_size > 0


def test_request_resolves_capacity_from_footprint() -> None:
    request = HousingRequest(
        footprint_xz=(30, 30),
        utility_type="residential",
        seed=42,
    )
    brief = resolve_brief_for_request(request, tuning=HousingPlanTuning())
    assert brief.occupant_count >= 1
    assert brief.max_storeys is None


def test_request_capacity_override_is_exact_when_viable() -> None:
    request = HousingRequest(
        footprint_xz=(30, 30),
        utility_type="residential",
        capacity_override=2,
        seed=42,
    )
    brief = resolve_brief_for_request(request, tuning=HousingPlanTuning())
    assert brief.occupant_count == 2


def test_request_capacity_override_raises_when_not_viable() -> None:
    request = HousingRequest(
        footprint_xz=(8, 8),
        utility_type="service_building",
        capacity_override=24,
        max_storeys=1,
        seed=42,
    )
    try:
        resolve_brief_for_request(request, tuning=HousingPlanTuning())
    except ValueError as exc:
        assert "capacity_override" in str(exc)
    else:
        raise AssertionError("expected infeasible capacity_override to raise")


def test_request_driven_plan_generation() -> None:
    request = HousingRequest(
        footprint_xz=(30, 30),
        utility_type="residential",
        seed=42,
    )
    plan = generate_housing_plan_for_request(
        request,
        search_iterations=64,
        tuning=HousingPlanTuning(),
    )
    assert plan.metadata.cell_grid_size[1] >= 2
    assert plan.metadata.storey_distribution.min_storeys >= 2
    assert plan.metadata.score_total > 0.0


def test_request_driven_plan_keeps_essential_residential_rooms() -> None:
    plan = generate_housing_plan_for_request(
        HousingRequest(
            footprint_xz=(30, 30),
            utility_type="residential",
            seed=42,
        ),
        search_iterations=96,
        tuning=HousingPlanTuning(),
    )
    counts: dict[str, int] = {}
    for cell in plan.cells:
        if cell.is_empty:
            continue
        counts[cell.label] = counts.get(cell.label, 0) + 1
    assert counts.get(pt.POD_ENTRY, 0) >= 1
    assert counts.get(pt.POD_LIVING, 0) >= 1
    assert counts.get(pt.POD_KITCHEN, 0) == 1
    assert counts.get(pt.POD_BATHROOM, 0) == 1


def test_residential_profiles_respect_realistic_compact_scale() -> None:
    courtyard = generate_housing_plan_for_request(
        HousingRequest(
            footprint_xz=PLAN_PROFILES["courtyard_family"].footprint_xz,
            utility_type=PLAN_PROFILES["courtyard_family"].utility_type,
            capacity_override=PLAN_PROFILES["courtyard_family"].capacity_override,
            seed=42,
        ),
        search_iterations=64,
        tuning=PLAN_PROFILES["courtyard_family"].tuning,
    )
    mansion = generate_housing_plan_for_request(
        HousingRequest(
            footprint_xz=PLAN_PROFILES["grand_mansion"].footprint_xz,
            utility_type=PLAN_PROFILES["grand_mansion"].utility_type,
            capacity_override=PLAN_PROFILES["grand_mansion"].capacity_override,
            seed=42,
        ),
        search_iterations=64,
        tuning=PLAN_PROFILES["grand_mansion"].tuning,
    )
    assert courtyard.metadata.occupant_count == 4
    assert mansion.metadata.occupant_count == 6
    assert courtyard.metadata.storey_distribution.target_storeys == 2
    assert mansion.metadata.storey_distribution.target_storeys >= 2
    assert courtyard.metadata.cell_grid_size[1] == 2
    assert mansion.metadata.cell_grid_size[1] >= 3


def test_tall_service_plan_reserves_supported_tower_core() -> None:
    profile = PLAN_PROFILES["sky_scraper"]
    plan = generate_housing_plan_for_request(
        HousingRequest(
            footprint_xz=profile.footprint_xz,
            utility_type=profile.utility_type,
            capacity_override=profile.capacity_override,
            max_storeys=profile.max_storeys,
            seed=42,
        ),
        search_iterations=profile.search_iterations,
        tuning=profile.tuning,
    )
    counts: dict[tuple[int, int], int] = {}
    for cell in plan.cells:
        if cell.label != pt.POD_STAIRWELL:
            continue
        key = (cell.cell_index[0], cell.cell_index[2])
        counts[key] = counts.get(key, 0) + 1
    assert counts
    assert max(counts.values()) >= 4
    structure = analyse_structure(plan.state)
    assert structure.unsupported_cells == 0
