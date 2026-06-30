"""Tests for the spatial-layout factories.

Covers the two invariants that justify the topology/geometry split:

1. ``banded_layout`` with all multipliers equal to 1.0 reproduces
   ``uniform_layout`` byte-for-byte. This is the v1 parity guarantee.
2. ``banded_layout`` aggregates multipliers per ix-column / iz-row so
   shared horizontal faces between neighbouring cells keep matching
   extents (no ragged seams).
"""

from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import design_grid
from prefab_housing.layout import banded_layout, uniform_layout
from prefab_housing.wfc.solver import init_state
from prefab_housing.wfc.tiles import build_tile_set


def _solved_state_uniform_pod(grid_shape: tuple[int, int, int], pod: str):
    """Build a solver state with every cell pinned to the rotation-0 tile of
    ``pod``. Bypasses MCTS — we only need a deterministic fully-assigned
    state for layout testing."""
    cx, cy, cz = grid_shape
    grid = design_grid(footprint_xz=(cx * 8, cz * 8), max_storeys=cy, cell_voxel_size=(8, 6, 8))
    tiles = build_tile_set()
    state = init_state(grid, tiles)
    pod_idx = pt.POD_INDEX[pod]
    # Pick the first tile whose pod matches and rotation is 0.
    target_tid = -1
    for tid in range(tiles.num_tiles):
        if int(tiles.pod_index[tid]) == pod_idx and int(tiles.rotation[tid]) == 0:
            target_tid = tid
            break
    assert target_tid >= 0, f"no rotation-0 tile for pod {pod}"
    state.assignment[:] = target_tid
    return state


def test_banded_equals_uniform_when_multipliers_all_one() -> None:
    state = _solved_state_uniform_pod((3, 2, 3), pt.POD_LIVING)
    base = (8, 6, 8)
    origin = (10, 0, -5)
    uni = uniform_layout(state.grid, base, origin)
    band = banded_layout(state, base, origin)
    assert uni.cell_bbox == band.cell_bbox
    assert uni.origin_world == band.origin_world


def test_banded_inflates_column_to_max_multiplier(monkeypatch) -> None:
    state = _solved_state_uniform_pod((3, 1, 3), pt.POD_LIVING)
    # Inflate LIVING to 1.5× horizontally; all other pods stay at 1.0.
    living_idx = pt.POD_INDEX[pt.POD_LIVING]
    new = list(pt.POD_SIZE_MULTIPLIER)
    new[living_idx] = 1.5
    monkeypatch.setattr(pt, "POD_SIZE_MULTIPLIER", tuple(new))
    layout = banded_layout(state, (8, 6, 8), (0, 0, 0))
    # Every cell carries LIVING → every column inflated. ceil(8 * 1.5) = 12.
    sizes = {layout.cell_size(ix, 0, iz) for ix in range(3) for iz in range(3)}
    assert sizes == {(12, 6, 12)}


def test_banded_shared_face_extents_match() -> None:
    """Shared horizontal faces between vertically-stacked or horizontally
    adjacent cells must share their face area — i.e. neighbours along a
    given axis must have identical extents on the perpendicular axes."""
    state = _solved_state_uniform_pod((3, 2, 3), pt.POD_BEDROOM)
    layout = banded_layout(state, (8, 6, 8), (0, 0, 0))
    grid = state.grid
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx - 1):
                a = layout.cell_size(ix, iy, iz)
                b = layout.cell_size(ix + 1, iy, iz)
                assert a[1] == b[1] and a[2] == b[2], (
                    f"x-neighbour face mismatch at ({ix},{iy},{iz})"
                )
            for ix in range(grid.cx):
                if iz < grid.cz - 1:
                    a = layout.cell_size(ix, iy, iz)
                    c = layout.cell_size(ix, iy, iz + 1)
                    assert a[0] == c[0] and a[1] == c[1], (
                        f"z-neighbour face mismatch at ({ix},{iy},{iz})"
                    )
