from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from prefab_housing import (
    MAX_RESIDENTIAL_LEVEL,
    RESIDENTIAL_LEVEL_SPECS,
    assemble_house_from_plan,
    compose_block_generation_stages,
    generate_housing_plan_for_request,
    request_for_residential_level,
    residential_level_spec,
    validate_housing_plan,
)
from prefab_housing.catalogue import pod_types as pt
from prefab_housing.programme import Programme


def _counts(plan) -> Counter[str]:
    return Counter(cell.label for cell in plan.cells if not cell.is_empty)


def _occupied_columns(plan) -> int:
    return len({(cell.cell_index[0], cell.cell_index[2]) for cell in plan.cells if not cell.is_empty})


def _used_storeys(plan) -> int:
    return len({cell.cell_index[1] for cell in plan.cells if not cell.is_empty})


def _assert_non_wall_face_blocks_inside_footprint(
    result,
    footprint_xz: tuple[int, int],
) -> None:
    fx, fz = footprint_xz
    bounded_stages = tuple(
        stage for stage in result.block_stages if stage.name != "wall_face_textures"
    )
    for block in compose_block_generation_stages(bounded_stages):
        assert 0 <= block["x"] < fx
        assert 0 <= block["z"] < fz


def test_residential_level_specs_are_bounded_and_monotonic() -> None:
    assert tuple(RESIDENTIAL_LEVEL_SPECS) == (1, 2, 3)
    assert MAX_RESIDENTIAL_LEVEL == 3
    assert [spec.occupant_count for spec in RESIDENTIAL_LEVEL_SPECS.values()] == [1, 2, 3]
    assert [spec.level for spec in RESIDENTIAL_LEVEL_SPECS.values()] == [1, 2, 3]


def test_residential_level_specs_define_fit_policy() -> None:
    for spec in RESIDENTIAL_LEVEL_SPECS.values():
        policy = spec.tuning.fit_policy
        assert policy.ground_fill_min is not None
        assert policy.ground_fill_target is not None
        assert policy.ground_fill_max is not None
        assert policy.storeys_min is not None
        assert policy.storeys_target is not None
        assert policy.storeys_max is not None
        assert policy.occupied_cells_min is not None
        assert policy.occupied_cells_target is not None
        assert policy.occupied_cells_max is not None


def test_unknown_residential_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        residential_level_spec(4)


def test_residential_levels_generate_valid_plans() -> None:
    for level, spec in RESIDENTIAL_LEVEL_SPECS.items():
        plan = generate_housing_plan_for_request(
            request_for_residential_level(level, seed=42),
            search_iterations=spec.search_iterations,
            tuning=spec.tuning,
        )
        report = validate_housing_plan(plan)
        result = assemble_house_from_plan(plan)
        counts = _counts(plan)
        policy = spec.tuning.fit_policy
        site_cx = max(1, spec.footprint_xz[0] // plan.metadata.cell_voxel_size[0])
        site_cz = max(1, spec.footprint_xz[1] // plan.metadata.cell_voxel_size[2])
        ground_fill = _occupied_columns(plan) / (site_cx * site_cz)
        used_storeys = _used_storeys(plan)
        occupied_cells = sum(counts.values())

        assert report.is_valid, report.errors
        assert plan.metadata.cell_voxel_size == (10, 6, 10)
        assert plan.metadata.site_footprint_xz == spec.footprint_xz
        assert result.metadata.site_footprint_xz == spec.footprint_xz
        _assert_non_wall_face_blocks_inside_footprint(result, spec.footprint_xz)
        assert policy.ground_fill_min is not None
        assert policy.ground_fill_max is not None
        assert policy.storeys_min is not None
        assert policy.storeys_max is not None
        assert policy.occupied_cells_min is not None
        assert policy.occupied_cells_max is not None
        assert policy.ground_fill_min <= ground_fill <= policy.ground_fill_max
        assert policy.storeys_min <= used_storeys <= policy.storeys_max
        assert policy.occupied_cells_min <= occupied_cells <= policy.occupied_cells_max
        assert counts[pt.POD_ENTRY] == 1
        assert counts[pt.POD_KITCHEN] == 1
        assert counts[pt.POD_BATHROOM] == 1
        assert counts[pt.POD_BEDROOM] >= (2 if level == 3 else 1)
        if level == 1:
            assert counts.get(pt.POD_LIVING, 0) == 0
            assert plan.metadata.cell_grid_size[1] == 1
        if level == 2:
            assert plan.metadata.cell_grid_size[1] >= 2
            assert plan.metadata.massing_profile.preferred_storeys == 1
        if level == 3:
            assert counts[pt.POD_LIVING] == 1
            assert counts[pt.POD_STAIRWELL] >= 3
            assert plan.metadata.cell_grid_size[1] >= 3
            assert plan.metadata.massing_profile.preferred_storeys == 2


def test_validator_flags_missing_required_rooms() -> None:
    spec = RESIDENTIAL_LEVEL_SPECS[1]
    plan = generate_housing_plan_for_request(
        request_for_residential_level(1, seed=42),
        search_iterations=spec.search_iterations,
        tuning=spec.tuning,
    )
    impossible_programme = Programme(
        required_pods=((pt.POD_BEDROOM, 4),),
        max_pods=((pt.POD_BEDROOM, 4),),
        optional_pods=(),
        target_min_cells=4,
    )
    invalid_plan = replace(plan, programme=impossible_programme)

    report = validate_housing_plan(invalid_plan)

    assert not report.is_valid
    assert dict(report.missing_required) == {pt.POD_BEDROOM: 3}
