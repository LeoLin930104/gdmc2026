"""MCTS-guided WFC collapse.

Tree
----
Each node owns a :class:`SolverState` (copied lazily on expansion) and the
``flat`` cell index it is responsible for collapsing — the lowest-entropy
undecided cell at the moment the node is first visited. Children correspond
to the legal tile_ids at that cell.

Action selection
----------------
- **Selection**: UCB1 over visited children:
  ``Q(c) + c_puct * P(c) * sqrt(ln N_parent / (1 + N_c))``.
  ``P(c)`` is a programme prior (uniform if no required-pod bias applies).
- **Expansion**: instantiate one untried child by copying the parent state and
  invoking ``collapse_to``. Contradictions are discarded immediately
  (the action is removed and the child not created).
- **Rollout**: weighted-random WFC playthrough using the same programme prior
  on top of the WFC entropy heuristic. Terminates on solved-or-contradicted.
- **Backup**: the value is the utility :func:`score` of the terminal state
  (0.0 if contradicted), aggregated as a running sum.

Determinism
-----------
A single :class:`numpy.random.Generator` keyed on ``config.rng_seed`` drives
all stochastic decisions: UCB1 is deterministic, but tie-breaks during action
shuffling and rollout sampling consult the RNG.

The function records the highest-scoring *fully solved* state encountered
during search and returns it; the search tree itself does not memoise
assignments along its path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Final
import math

import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import CellGrid
from prefab_housing.programme import Programme, validate_pod_counts
from prefab_housing.search.score import ScoreReport, ScoreWeights, score
from prefab_housing.structure import assigned_cells_have_support_potential
from prefab_housing.wfc.solver import (
    SolverState,
    candidate_tiles,
    collapse_to,
    is_contradicted,
    is_solved,
    lowest_entropy_cell,
)
from prefab_housing.wfc.tiles import TileSet


# Sentinel indices for topology-native void tiles.
_STRUCTURAL_VOID_POD_INDEX: Final[int] = pt.POD_INDEX[pt.POD_STRUCTURAL_VOID]
_TERRACE_VOID_POD_INDEX: Final[int] = pt.POD_INDEX[pt.POD_TERRACE_VOID]


# --- Configuration ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    iterations: int = 256
    c_puct: float = 1.41421356        # sqrt(2)
    rollout_prior_strength: float = 4.0   # multiplier applied to required-pod weights
    expansion_prior_strength: float = 2.0
    rng_seed: int = 0
    # EMPTY-perimeter prior: encourages perimeter cells (especially corners
    # at upper storeys) to collapse to ``_empty``, breaking the bounding-box
    # silhouette into set-back / chamfered-corner forms.
    #
    # bias_at(cell) = 1.0 + strength * (horiz_boundary_count / 2) * depth(iy)
    #
    # ``depth(iy)`` interpolates between ``ground_floor_empty_factor`` (at
    # iy=0) and 1.0 (at iy=cy-1). Ground floor receives a non-zero share of
    # the EMPTY pressure so irregular ground footprints (recessed entries,
    # carport notches) can emerge, while upper storeys still carry the bulk
    # of the carve-out probability. Setting the factor to 0.0 reproduces the
    # pre-refinement behaviour (no ground-floor EMPTY pressure).
    empty_perimeter_strength: float = 10.0
    ground_floor_empty_factor: float = 0.25
    terrace_void_strength: float = 0.5
    terrace_start_storey: int = 1
    terrace_axis: str = "x"
    terrace_direction: int = 1
    terrace_asymmetry_strength: float = 0.5
    occupancy_storey_bias: tuple[float, ...] = ()
    void_storey_bias: tuple[float, ...] = ()


@dataclass
class MCTSResult:
    best_state: SolverState
    best_score: ScoreReport
    iterations_run: int
    solved_count: int
    contradiction_count: int


# --- Programme prior --------------------------------------------------------


def _compute_empty_perimeter_bias(
    grid: CellGrid, strength: float, ground_floor_empty_factor: float = 0.0
) -> np.ndarray:
    """Per-cell multiplier applied to the EMPTY tile's sampling weight.

    Result is ``float64[C]``. Returns all-ones array when strength is 0 —
    parity with pre-Step-4 behaviour. Computed once per search; the grid
    topology never changes during MCTS.

    Bias formula
    ------------
    For cell ``(ix, iy, iz)``:
        horiz_b = count of horizontal faces (N/E/S/W) that hit the grid edge
        t       = iy / max(1, cy - 1)                              # 0 → 1
        depth   = lerp(ground_floor_empty_factor, 1.0, t)
        bias    = 1.0 + strength * (horiz_b / 2) * depth

    Ground floor (``t=0``) receives ``ground_floor_empty_factor`` of the
    full EMPTY pressure — non-zero enables recessed entries/carport notches
    while keeping upper storeys carrying the bulk of the silhouette
    carve-out. Setting the factor to 0.0 reproduces pre-refinement
    behaviour (ground bias=1.0).
    """
    C = grid.cells_total
    bias = np.ones(C, dtype=np.float64)
    if strength <= 0.0:
        return bias
    if grid.cy <= 1:
        # Single-storey grid: depth collapses to ground_floor_empty_factor.
        depth_single = float(ground_floor_empty_factor)
        if depth_single <= 0.0:
            return bias
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                horiz_b = 0
                for f in range(4):
                    if grid.neighbour(ix, 0, iz, f) is None:
                        horiz_b += 1
                if horiz_b == 0:
                    continue
                flat = grid.flat_index(ix, 0, iz)
                bias[flat] = 1.0 + strength * (horiz_b / 2.0) * depth_single
        return bias
    cy_norm = float(grid.cy - 1)
    factor = float(ground_floor_empty_factor)
    for iy in range(grid.cy):
        t = iy / cy_norm
        depth = factor + (1.0 - factor) * t
        if depth <= 0.0:
            continue
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                horiz_b = 0
                for f in range(4):  # N=0, E=1, S=2, W=3
                    if grid.neighbour(ix, iy, iz, f) is None:
                        horiz_b += 1
                if horiz_b == 0:
                    continue
                flat = grid.flat_index(ix, iy, iz)
                bias[flat] = 1.0 + strength * (horiz_b / 2.0) * depth
    return bias


def _storey_multiplier(values: tuple[float, ...], iy: int) -> float:
    if not values:
        return 1.0
    if iy < len(values):
        return max(0.01, float(values[iy]))
    return max(0.01, float(values[-1]))


def _required_unmet(state: SolverState, programme: Programme) -> Counter[str]:
    """Return required-pod multiset minus pods already assigned anywhere.

    Pure helper. Reused for both expansion and rollout priors.
    """
    required = programme.required_counter()
    if not required:
        return required
    placed: Counter[str] = Counter()
    assigned = state.assignment
    pod_index = state.tiles.pod_index
    for flat in range(assigned.shape[0]):
        tid = int(assigned[flat])
        if tid < 0:
            continue
        pod_idx = int(pod_index[tid])
        label = pt.POD_LABELS[pod_idx]
        placed[label] += 1
    unmet = Counter()
    for label, count in required.items():
        deficit = count - placed.get(label, 0)
        if deficit > 0:
            unmet[label] = deficit
    return unmet


def _placed_pod_counts(state: SolverState) -> Counter[str]:
    counts: Counter[str] = Counter()
    pod_index = state.tiles.pod_index
    for tid_raw in state.assignment:
        tid = int(tid_raw)
        if tid < 0:
            continue
        counts[pt.POD_LABELS[int(pod_index[tid])]] += 1
    return counts


def _tile_weights(
    state: SolverState,
    programme: Programme,
    tile_ids: np.ndarray,
    *,
    config: MCTSConfig,
    prior_strength: float,
    flat: int,
    empty_bias: np.ndarray,
) -> np.ndarray:
    """Return positive weight per candidate tile for sampling.

    Weight is ``1.0`` baseline, multiplied by ``prior_strength`` for tiles
    whose pod is currently required-and-unmet. Void tiles receive specialised
    pressure: structural void gets ``empty_bias[flat]``; terrace void gets a
    softer fraction so upper setbacks remain reachable without erasing entire
    columns as aggressively. The terrace bias is stage-aware: it only activates
    at or above the planned setback storey and prefers one side/axis to avoid
    reverting to a symmetric carved box.
    """
    if tile_ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    unmet = _required_unmet(state, programme)
    placed = _placed_pod_counts(state)
    capped = programme.max_counter()
    weights = np.ones(tile_ids.size, dtype=np.float64)
    pod_index = state.tiles.pod_index
    cell_empty_bias = float(empty_bias[flat])
    ix, iy, iz = state.grid.from_flat(flat)
    occupancy_storey_bias = _storey_multiplier(config.occupancy_storey_bias, iy)
    void_storey_bias = _storey_multiplier(config.void_storey_bias, iy)
    terrace_axis = config.terrace_axis if config.terrace_axis in ("x", "z") else "x"
    terrace_coord = ix if terrace_axis == "x" else iz
    terrace_extent = state.grid.cx if terrace_axis == "x" else state.grid.cz
    asymmetry_factor = 1.0
    if iy >= config.terrace_start_storey and terrace_extent > 1:
        normalised = terrace_coord / float(terrace_extent - 1)
        if config.terrace_direction < 0:
            normalised = 1.0 - normalised
        asymmetry_factor += config.terrace_asymmetry_strength * normalised
    for i, tid in enumerate(tile_ids):
        pod_idx = int(pod_index[int(tid)])
        if pod_idx == _STRUCTURAL_VOID_POD_INDEX:
            weights[i] *= cell_empty_bias * void_storey_bias
            continue
        if pod_idx == _TERRACE_VOID_POD_INDEX:
            if iy < config.terrace_start_storey:
                weights[i] *= 0.25 * void_storey_bias
            else:
                terrace_bias = 1.0 + config.terrace_void_strength * max(0.0, cell_empty_bias - 1.0)
                weights[i] *= terrace_bias * asymmetry_factor * void_storey_bias
            continue
        weights[i] *= occupancy_storey_bias
        label = pt.POD_LABELS[pod_idx]
        if unmet:
            if unmet.get(label, 0) > 0:
                weights[i] *= prior_strength
        if label in capped and placed.get(label, 0) >= capped[label]:
            weights[i] *= 0.05
    return weights


# --- Tree node --------------------------------------------------------------


@dataclass
class _Node:
    """An MCTS tree node.

    ``state`` is owned (mutated by :func:`collapse_to` at construction). Once
    the node is created its assignment never changes; rollouts copy the state.
    """

    state: SolverState
    parent: "_Node | None"
    cell_flat: int                 # cell this node will branch on; -1 if terminal
    untried: list[int] = field(default_factory=list)   # tile_ids
    untried_priors: list[float] = field(default_factory=list)
    children: dict[int, "_Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.cell_flat < 0

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _make_node(
    state: SolverState,
    parent: "_Node | None",
    programme: Programme,
    config: MCTSConfig,
    empty_bias: np.ndarray,
) -> _Node:
    if is_contradicted(state) or is_solved(state):
        return _Node(state=state, parent=parent, cell_flat=-1)
    cell = lowest_entropy_cell(state)
    if cell is None:
        return _Node(state=state, parent=parent, cell_flat=-1)
    candidates = candidate_tiles(state, cell)
    weights = _tile_weights(
        state, programme, candidates,
        config=config,
        prior_strength=config.expansion_prior_strength,
        flat=cell,
        empty_bias=empty_bias,
    )
    # Sort candidates by descending prior so popping from the back picks the
    # most promising first; weights co-sorted to feed the priors list.
    order = np.argsort(-weights, kind="stable")
    untried = [int(candidates[i]) for i in order]
    priors = [float(weights[i]) for i in order]
    total = sum(priors) or 1.0
    priors = [p / total for p in priors]
    return _Node(
        state=state,
        parent=parent,
        cell_flat=cell,
        untried=untried,
        untried_priors=priors,
    )


# --- Search loop ------------------------------------------------------------


def _ucb_select(node: _Node, c_puct: float) -> _Node:
    """Pick the child maximising UCB1; ties broken by ascending tile_id."""
    log_n = math.log(node.visits) if node.visits > 0 else 0.0
    best: _Node | None = None
    best_score = -math.inf
    for tile_id in sorted(node.children):
        child = node.children[tile_id]
        if child.visits == 0:
            return child
        ucb = child.q() + c_puct * math.sqrt(log_n / child.visits)
        if ucb > best_score:
            best_score = ucb
            best = child
    assert best is not None
    return best


def _expand(
    node: _Node,
    programme: Programme,
    config: MCTSConfig,
    empty_bias: np.ndarray,
) -> _Node | None:
    """Pop one untried action, attempt to collapse, return resulting child.

    Returns ``None`` if every untried action led to a contradiction.
    """
    while node.untried:
        tile_id = node.untried.pop(0)
        node.untried_priors.pop(0)
        child_state = node.state.copy()
        collapse_to(child_state, node.cell_flat, tile_id)
        if is_contradicted(child_state):
            continue
        if not assigned_cells_have_support_potential(child_state):
            continue
        child = _make_node(child_state, node, programme, config, empty_bias)
        node.children[tile_id] = child
        return child
    return None


def _rollout(
    state: SolverState,
    programme: Programme,
    weights: ScoreWeights,
    rng: np.random.Generator,
    config: MCTSConfig,
    empty_bias: np.ndarray,
) -> tuple[SolverState, ScoreReport]:
    """Random WFC playthrough on a *copied* state. Returns final state + score."""
    work = state.copy()
    while not is_solved(work) and not is_contradicted(work):
        flat = lowest_entropy_cell(work)
        if flat is None:
            break
        cands = candidate_tiles(work, flat)
        if cands.size == 0:
            break
        w = _tile_weights(
            work, programme, cands,
            config=config,
            prior_strength=config.rollout_prior_strength,
            flat=flat,
            empty_bias=empty_bias,
        )
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            tile_id = int(cands[rng.integers(0, cands.size)])
        else:
            probs = w / w_sum
            tile_id = int(rng.choice(cands, p=probs))
        collapse_to(work, flat, tile_id)
        if not is_contradicted(work) and not assigned_cells_have_support_potential(work):
            work.contradicted = True
            break
    if is_contradicted(work):
        return work, ScoreReport(total=0.0, components={})
    return work, score(work, programme, weights)


def _backup(node: _Node, value: float) -> None:
    cur: _Node | None = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += value
        cur = cur.parent


def mcts_search(
    initial_state: SolverState,
    programme: Programme,
    *,
    weights: ScoreWeights | None = None,
    config: MCTSConfig | None = None,
) -> MCTSResult:
    """Run MCTS for ``config.iterations`` iterations and return the best
    fully-solved state encountered (by utility score).

    If no rollout ever produces a solved state, the returned ``best_state`` is
    the initial state and ``best_score.total`` is ``0.0``.
    """
    weights = weights or ScoreWeights()
    config = config or MCTSConfig()
    rng = np.random.default_rng(config.rng_seed)
    empty_bias = _compute_empty_perimeter_bias(
        initial_state.grid,
        config.empty_perimeter_strength,
        config.ground_floor_empty_factor,
    )

    root = _make_node(initial_state.copy(), None, programme, config, empty_bias)

    best_state: SolverState = initial_state
    best_report = ScoreReport(total=0.0, components={})
    best_valid_state: SolverState = initial_state
    best_valid_report = ScoreReport(total=0.0, components={})
    has_solved = False
    has_valid_solved = False
    solved_count = 0
    contradiction_count = 0

    for _ in range(config.iterations):
        # --- Selection ---
        node = root
        while not node.terminal and not node.untried and node.children:
            node = _ucb_select(node, config.c_puct)

        # --- Expansion ---
        if not node.terminal and node.untried:
            child = _expand(node, programme, config, empty_bias)
            if child is None:
                # All expansions contradicted: this node is now an effective
                # dead-end. Back up a zero and continue.
                _backup(node, 0.0)
                contradiction_count += 1
                continue
            node = child

        # --- Rollout ---
        if node.terminal:
            terminal_state = node.state
            if is_solved(terminal_state):
                report = score(terminal_state, programme, weights)
            else:
                report = ScoreReport(total=0.0, components={})
        else:
            terminal_state, report = _rollout(
                node.state, programme, weights, rng, config, empty_bias
            )

        # --- Bookkeeping ---
        if is_contradicted(terminal_state):
            contradiction_count += 1
        elif is_solved(terminal_state):
            solved_count += 1
            if not has_solved or report.total > best_report.total:
                best_report = report
                best_state = terminal_state
                has_solved = True
            if validate_pod_counts(_placed_pod_counts(terminal_state), programme).is_valid:
                if not has_valid_solved or report.total > best_valid_report.total:
                    best_valid_report = report
                    best_valid_state = terminal_state
                    has_valid_solved = True

        # --- Backup ---
        _backup(node, report.total)

    return MCTSResult(
        best_state=best_valid_state if has_valid_solved else best_state,
        best_score=best_valid_report if has_valid_solved else best_report,
        iterations_run=config.iterations,
        solved_count=solved_count,
        contradiction_count=contradiction_count,
    )


__all__ = [
    "MCTSConfig",
    "MCTSResult",
    "mcts_search",
]
