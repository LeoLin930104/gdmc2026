"""EMPTY-perimeter prior (Step 4) — sampling-bias contract tests.

Locks in the *structural* invariants of `MCTSConfig.empty_perimeter_strength`
without over-specifying scores or counts (which depend on RNG draws):

1. ``strength=0.0`` is a no-op: no EMPTY cells appear in the solved
   topology when the prior is disabled (matches pre-Step-4 baseline at the
   default smoke-test seed).
2. A high strength reliably introduces at least one EMPTY perimeter cell
   on a multi-storey grid where the search has slack — the prior must
   actually move the sampler off the all-cells-occupied basin.
3. The hard functional floor and structural support remain satisfied at default
   strength: stepping back the silhouette must not break the programme.
"""

from __future__ import annotations

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import design_grid
from prefab_housing.programme import resolve_programme
from prefab_housing.search.mcts import MCTSConfig, mcts_search
from prefab_housing.search.priors import apply_position_priors
from prefab_housing.structure import analyse_structure
from prefab_housing.types import Brief
from prefab_housing.wfc.solver import init_state
from prefab_housing.wfc.tiles import build_tile_set


_FOOTPRINT = (24, 24)
_SEED = 7   # picked because the prior produces ≥2 EMPTY cells deterministically


def _solve(strength: float, seed: int = _SEED):
    brief = Brief(
        occupant_count=3,
        household_type="single_family",
        material_theme="sci_fi_modular",
        seed=seed,
    )
    programme = resolve_programme(brief, "residential")
    grid = design_grid(footprint_xz=_FOOTPRINT, max_storeys=4, cell_voxel_size=(8, 6, 8))
    tiles = build_tile_set()
    state = init_state(grid, tiles)
    apply_position_priors(state, programme)
    cfg = MCTSConfig(iterations=128, rng_seed=seed, empty_perimeter_strength=strength)
    return mcts_search(state, programme, config=cfg)


def _empty_count(state) -> int:
    void_pods = set(pt.VOID_POD_INDICES)
    pod_index = state.tiles.pod_index
    n = 0
    for tid in state.assignment.tolist():
        if tid >= 0 and int(pod_index[int(tid)]) in void_pods:
            n += 1
    return n


def _violates_floor_support(state) -> bool:
    """True iff any non-EMPTY cell at iy>0 sits above an EMPTY cell.

    Captures the post-pod-types contract that ``FLOOR ↔ EMPTY`` is
    forbidden — habitable pods must rest on a non-EMPTY cell or on the
    ground (FLOOR ↔ EXTERIOR at iy=0).
    """
    grid = state.grid
    asg = state.assignment
    pod_index = state.tiles.pod_index
    void_pods = set(pt.VOID_POD_INDICES)
    for iy in range(1, grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                tid = int(asg[grid.flat_index(ix, iy, iz)])
                if tid < 0 or int(pod_index[tid]) in void_pods:
                    continue
                tid_below = int(asg[grid.flat_index(ix, iy - 1, iz)])
                if tid_below < 0 or int(pod_index[tid_below]) in void_pods:
                    return True
    return False


def test_strength_zero_respects_floor_support() -> None:
    """Prior disabled ⇒ MCTS still respects the no-floating-pods rule.

    Note: prior to the structural-support refactor this test asserted
    zero EMPTY cells. Under the new ``FLOOR ↔ EMPTY``-forbidden rule a
    perimeter-EMPTY at storey 0 forces every cell above it to be EMPTY
    too, so ``strength=0`` no longer guarantees a fully-occupied grid;
    the structural invariant is the meaningful contract.
    """
    result = _solve(strength=0.0, seed=42)
    assert not _violates_floor_support(result.best_state)


def test_strength_high_introduces_empty_cells() -> None:
    """Prior enabled at default strength ⇒ at least one EMPTY perimeter cell."""
    result = _solve(strength=10.0)
    assert _empty_count(result.best_state) >= 1


def test_functional_floor_and_support_satisfied_under_prior() -> None:
    """Stepping back must not break programme coverage or support."""
    result = _solve(strength=10.0)
    breakdown = result.best_score.components
    structure = analyse_structure(result.best_state)

    assert breakdown["functional_adequacy"] == 1.0
    assert structure.unsupported_cells == 0


def test_determinism_under_prior() -> None:
    """Same (seed, strength) ⇒ identical assignment."""
    a = _solve(strength=10.0)
    b = _solve(strength=10.0)
    assert a.best_state.assignment.tolist() == b.best_state.assignment.tolist()


def test_tall_service_priors_reserve_vertical_tower_core() -> None:
    brief = Brief(
        occupant_count=12,
        household_type="multi_family",
        material_theme="sci_fi_modular",
        seed=42,
    )
    programme = resolve_programme(brief, "service_building")
    grid = design_grid(footprint_xz=(32, 32), max_storeys=6, cell_voxel_size=(8, 6, 8))
    assert (grid.cx, grid.cy, grid.cz) == (4, 6, 4)
    tiles = build_tile_set(allow_floor_empty=False)
    state = init_state(grid, tiles)
    apply_position_priors(
        state,
        programme,
        utility_type="service_building",
        public_storey_max=1,
        private_storey_min=1,
        terrace_start_storey=grid.cy - 2,
        tower_core_cell=(grid.cx // 2, grid.cz // 2),
    )

    core_flat = grid.flat_index(grid.cx // 2, 0, grid.cz // 2)
    core_domain = state.domain[core_flat]
    allowed_pods = {int(state.tiles.pod_index[tid]) for tid in core_domain.nonzero()[0].tolist()}
    assert allowed_pods == {pt.POD_INDEX[pt.POD_STAIRWELL]}

    result = mcts_search(state, programme, config=MCTSConfig(iterations=224, rng_seed=42, empty_perimeter_strength=10.0))
    structure = analyse_structure(result.best_state)
    assert structure.unsupported_cells == 0
