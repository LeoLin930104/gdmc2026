from __future__ import annotations

from collections import Counter

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.programme import resolve_programme, validate_pod_counts
from prefab_housing.types import Brief


def test_residential_programme_caps_single_kitchen_and_bathroom() -> None:
    programme = resolve_programme(
        Brief(
            occupant_count=4,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=42,
        ),
        "residential",
    )
    caps = programme.max_counter()
    assert caps[pt.POD_ENTRY] == 1
    assert caps[pt.POD_LIVING] == 1
    assert caps[pt.POD_KITCHEN] == 1
    assert caps[pt.POD_BATHROOM] == 1


def test_validate_pod_counts_flags_missing_and_excess() -> None:
    programme = resolve_programme(
        Brief(
            occupant_count=3,
            household_type="single_family",
            material_theme="sci_fi_modular",
            seed=42,
        ),
        "residential",
    )
    counts = Counter(
        {
            pt.POD_ENTRY: 1,
            pt.POD_LIVING: 1,
            pt.POD_KITCHEN: 2,
            pt.POD_BEDROOM: 2,
        }
    )
    validation = validate_pod_counts(counts, programme)
    assert not validation.is_valid
    assert dict(validation.missing_pods) == {pt.POD_BATHROOM: 1}
    assert dict(validation.excess_pods) == {pt.POD_KITCHEN: 1}
