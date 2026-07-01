from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import design_grid
from prefab_housing.structure import analyse_structure
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


def test_anchored_and_overhang_ratios_split_supported_cells() -> None:
    state = _state_with_pods(
        2,
        2,
        2,
        {
            (0, 0, 0): pt.POD_LIVING,
            (1, 0, 0): pt.POD_LIVING,
            (0, 0, 1): pt.POD_LIVING,
            (0, 1, 0): pt.POD_LIVING,
            (1, 1, 0): pt.POD_LIVING,
            (0, 1, 1): pt.POD_LIVING,
            (1, 1, 1): pt.POD_LIVING,
        },
    )
    report = analyse_structure(state)
    assert report.supported_cells == 7
    assert report.anchored_cells == 6
    assert report.anchored_ratio == 6 / 7
    assert report.overhang_ratio == 1 / 7
