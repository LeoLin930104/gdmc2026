from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import design_grid
from prefab_housing.programme import Programme
from prefab_housing.search.score import boxiness_penalty, massing_articulation
from prefab_housing.wfc.solver import init_state
from prefab_housing.wfc.tiles import build_tile_set


def _state_with_pods(cx: int, cy: int, cz: int, pods: dict[tuple[int, int, int], str]):
    grid = design_grid(
        footprint_xz=(cx * 8, cz * 8),
        max_storeys=cy,
        cell_voxel_size=(8, 6, 8),
    )
    tiles = build_tile_set()
    state = init_state(grid, tiles)
    first_tid: dict[str, int] = {}
    for tid in range(len(tiles.pod_index)):
        label = pt.POD_LABELS[int(tiles.pod_index[tid])]
        first_tid.setdefault(label, tid)
    for (ix, iy, iz), pod_label in pods.items():
        state.assignment[grid.flat_index(ix, iy, iz)] = first_tid[pod_label]
    return state


def _empty_programme() -> Programme:
    return Programme(required_pods=(), max_pods=(), optional_pods=(), target_min_cells=0)


def test_supported_stepback_scores_above_full_box() -> None:
    full_box = _state_with_pods(
        2,
        2,
        2,
        {(ix, iy, iz): pt.POD_LIVING for iy in range(2) for iz in range(2) for ix in range(2)},
    )
    stepped = _state_with_pods(
        2,
        2,
        2,
        {
            (0, 0, 0): pt.POD_LIVING,
            (1, 0, 0): pt.POD_LIVING,
            (0, 0, 1): pt.POD_LIVING,
            (1, 0, 1): pt.POD_LIVING,
            (0, 1, 0): pt.POD_LIVING,
            (0, 1, 1): pt.POD_LIVING,
        },
    )
    assert boxiness_penalty(stepped, _empty_programme()) > boxiness_penalty(full_box, _empty_programme())


def test_massing_articulation_scores_supported_stepback_above_full_box() -> None:
    full_box = _state_with_pods(
        3,
        2,
        3,
        {(ix, iy, iz): pt.POD_LIVING for iy in range(2) for iz in range(3) for ix in range(3)},
    )
    stepped = _state_with_pods(
        3,
        2,
        3,
        {
            **{(ix, 0, iz): pt.POD_LIVING for iz in range(3) for ix in range(3)},
            (0, 1, 0): pt.POD_LIVING,
            (1, 1, 0): pt.POD_LIVING,
            (1, 1, 1): pt.POD_LIVING,
        },
    )
    assert massing_articulation(stepped, _empty_programme()) > massing_articulation(full_box, _empty_programme())
