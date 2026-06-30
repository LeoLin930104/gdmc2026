"""Unit tests for the cantilever-aware structural_plausibility component.

Constructs minimal SolverState fixtures with hand-placed pod assignments
and asserts the score reflects the support topology, not just per-cell
"is below me non-EMPTY" — the v1 rule it replaced.
"""

from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import design_grid
from prefab_housing.programme import Programme
from prefab_housing.search.score import structural_plausibility
from prefab_housing.structure import assigned_cells_have_support_potential
from prefab_housing.wfc.solver import init_state
from prefab_housing.wfc.tiles import build_tile_set


def _state_with_pods(cx: int, cy: int, cz: int, pods: dict[tuple[int, int, int], str]):
    """Build a SolverState then assign each (ix, iy, iz) → first tile-id whose
    pod_index matches the requested pod label.

    Cells not present in ``pods`` are left at -1 (unassigned). Test fixtures
    therefore mix EMPTY-mapped pods (POD_EMPTY) with non-empty pods to
    exercise the support rule.
    """
    grid = design_grid(
        footprint_xz=(cx * 8, cz * 8),
        max_storeys=cy,
        cell_voxel_size=(8, 6, 8),
    )
    assert (grid.cx, grid.cy, grid.cz) == (cx, cy, cz)

    tiles = build_tile_set()
    state = init_state(grid, tiles)

    # Map pod_label → first matching tile_id.
    first_tid: dict[str, int] = {}
    for tid in range(len(tiles.pod_index)):
        label = pt.POD_LABELS[int(tiles.pod_index[tid])]
        first_tid.setdefault(label, tid)

    for (ix, iy, iz), pod_label in pods.items():
        tid = first_tid[pod_label]
        state.assignment[grid.flat_index(ix, iy, iz)] = tid

    return state


def _empty_programme() -> Programme:
    # structural_plausibility ignores the programme — pass an empty one.
    return Programme(required_pods=(), max_pods=(), optional_pods=(), target_min_cells=0)


def test_all_storey0_cells_supported():
    state = _state_with_pods(2, 1, 2, {
        (0, 0, 0): pt.POD_LIVING,
        (1, 0, 0): pt.POD_LIVING,
        (0, 0, 1): pt.POD_LIVING,
        (1, 0, 1): pt.POD_LIVING,
    })
    assert structural_plausibility(state, _empty_programme()) == 1.0


def test_floating_island_penalised():
    # 2x2x2: storey 0 has only (0,0,0); storey 1 has (1,1,1) — floating, no
    # storey-0 column under it and no same-storey supported neighbour.
    state = _state_with_pods(2, 2, 2, {
        (0, 0, 0): pt.POD_LIVING,
        (1, 1, 1): pt.POD_LIVING,
    })
    score = structural_plausibility(state, _empty_programme())
    # Storey-weighted: storey 0 weight = 1, storey 1 weight = 2.
    # Total weight = 1 (storey 0 cell) + 2 (storey 1 cell) = 3.
    # Supported weight = 1 (only the storey-0 cell). Expect 1/3.
    assert score == 1.0 / 3.0


def test_single_neighbour_cantilever_rejected():
    # Storey 1 cell with only one supported horizontal neighbour should no
    # longer count as supported.
    state = _state_with_pods(2, 2, 1, {
        (0, 0, 0): pt.POD_LIVING,
        (0, 1, 0): pt.POD_LIVING,
        (1, 1, 0): pt.POD_LIVING,
    })
    assert structural_plausibility(state, _empty_programme()) == 3.0 / 5.0


def test_disconnected_cantilever_penalised():
    # Storey 0: (0,0,0). Storey 1: (0,1,0) supported by stack; (1,1,1) is a
    # diagonal-only neighbour — NOT 4-connected to (0,1,0). Should be unsupported.
    state = _state_with_pods(2, 2, 2, {
        (0, 0, 0): pt.POD_LIVING,
        (0, 1, 0): pt.POD_LIVING,
        (1, 1, 1): pt.POD_LIVING,
    })
    score = structural_plausibility(state, _empty_programme())
    # Total weight = 1 (storey 0) + 2 (stacked storey 1) + 2 (floating storey 1) = 5.
    # Supported weight = 1 + 2 = 3.
    assert score == 3.0 / 5.0


def test_ground_empty_cascades_through_stack():
    # Storey 0 is EMPTY at column (0,0,0); storeys 1 and 2 have cells stacked
    # there with no lateral supported neighbour. Both upper cells must be
    # flagged unsupported, and storey 2 should weigh more than storey 1.
    state = _state_with_pods(1, 3, 1, {
        (0, 1, 0): pt.POD_LIVING,
        (0, 2, 0): pt.POD_LIVING,
    })
    score = structural_plausibility(state, _empty_programme())
    # Total weight = 2 (storey 1) + 3 (storey 2) = 5; supported = 0.
    assert score == 0.0


def test_quorum_supported_overhang_recovers():
    # Three upper cells are directly supported by stack. The fourth upper cell
    # has two supported same-storey neighbours, so it passes the quorum rule.
    state = _state_with_pods(2, 2, 2, {
        (0, 0, 0): pt.POD_LIVING,
        (1, 0, 0): pt.POD_LIVING,
        (0, 0, 1): pt.POD_LIVING,
        (0, 1, 0): pt.POD_LIVING,
        (1, 1, 0): pt.POD_LIVING,
        (0, 1, 1): pt.POD_LIVING,
        (1, 1, 1): pt.POD_LIVING,
    })
    assert structural_plausibility(state, _empty_programme()) == 1.0


def test_partial_state_without_future_support_is_rejected() -> None:
    state = _state_with_pods(1, 2, 1, {
        (0, 0, 0): pt.POD_STRUCTURAL_VOID,
        (0, 1, 0): pt.POD_LIVING,
    })
    # The ground cell is already committed to void and there are no same-storey
    # neighbours, so support is impossible even if every undecided cell later
    # fills.
    assert not assigned_cells_have_support_potential(state)


def test_partial_state_with_future_support_remains_viable() -> None:
    state = _state_with_pods(2, 2, 2, {
        (1, 1, 1): pt.POD_LIVING,
    })
    # Two same-storey neighbours and a column below can still be filled later,
    # so search should retain this branch.
    assert assigned_cells_have_support_potential(state)
