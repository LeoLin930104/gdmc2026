"""Utility scoring components.

Each component is a pure function over an *assigned* solver state plus the
resolved programme. All components return a scalar in ``[0, 1]``; the
:func:`score` aggregator combines them via :class:`ScoreWeights`.

Hard floor
----------
``functional_adequacy`` may return ``0.0`` when required pods are missing;
:func:`score` propagates this as a total of ``0.0`` when ``hard_floor=True``
in the weights.

Determinism
-----------
No RNG is consulted in scoring. All outputs depend only on
``(state.assignment, state.tiles, state.grid, programme)``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import (
    DOWN,
    HORIZONTAL_FACES,
    NUM_FACES,
    OPPOSITE_FACE,
    UP,
)
from prefab_housing.programme import Programme, validate_pod_counts
from prefab_housing.structure import analyse_structure
from prefab_housing.wfc.solver import SolverState


@dataclass(frozen=True, slots=True)
class PlanFitPolicy:
    """Level/site fit targets used as soft scoring bands.

    ``None`` means "do not score this axis". Ranges are inclusive target bands:
    plans inside the band score well, while plans below or above it lose score.
    This avoids turning land use, height, or room count into one-way maximisers.
    """

    ground_fill_min: float | None = None
    ground_fill_target: float | None = None
    ground_fill_max: float | None = None
    storeys_min: int | None = None
    storeys_target: int | None = None
    storeys_max: int | None = None
    occupied_cells_min: int | None = None
    occupied_cells_target: int | None = None
    occupied_cells_max: int | None = None
    site_footprint_cells: int | None = None

    def __post_init__(self) -> None:
        for name in ("ground_fill_min", "ground_fill_target", "ground_fill_max"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] when provided")
        if (
            self.ground_fill_min is not None
            and self.ground_fill_max is not None
            and self.ground_fill_min > self.ground_fill_max
        ):
            raise ValueError("ground_fill_min must be <= ground_fill_max")
        for name in (
            "storeys_min",
            "storeys_target",
            "storeys_max",
            "occupied_cells_min",
            "occupied_cells_target",
            "occupied_cells_max",
            "site_footprint_cells",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when provided")
        if self.storeys_min is not None and self.storeys_max is not None and self.storeys_min > self.storeys_max:
            raise ValueError("storeys_min must be <= storeys_max")
        if (
            self.occupied_cells_min is not None
            and self.occupied_cells_max is not None
            and self.occupied_cells_min > self.occupied_cells_max
        ):
            raise ValueError("occupied_cells_min must be <= occupied_cells_max")


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    functional_adequacy: float = 0.25
    circulation: float = 0.15
    privacy_gradient: float = 0.10
    daylight: float = 0.10
    vertical_service_stack: float = 0.10
    programme_efficiency: float = 0.06
    room_budget_fit: float = 0.12
    land_utilisation: float = 0.12
    storey_profile: float = 0.12
    storey_diversity: float = 0.06
    structural_physics: float = 0.08
    boxiness_penalty: float = 0.12
    massing_articulation: float = 0.12
    aesthetic_facade: float = 0.02
    structural_plausibility: float = 0.13
    hard_floor_on_functional: bool = True
    hard_floor_on_circulation: bool = False
    fit_policy: PlanFitPolicy = field(default_factory=PlanFitPolicy)


@dataclass
class ScoreReport:
    total: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _ScoreContext:
    structure: object | None = None


# --- Component implementations ------------------------------------------------


def _pod_counts(state: SolverState) -> Counter[str]:
    counts: Counter[str] = Counter()
    pod_idx = state.tiles.pod_index
    for tid in state.assignment:
        if tid < 0:
            continue
        counts[pt.POD_LABELS[int(pod_idx[int(tid)])]] += 1
    return counts


def _band_score_float(
    value: float,
    *,
    minimum: float | None,
    target: float | None,
    maximum: float | None,
    fallback_span: float,
) -> float:
    if minimum is None and target is None and maximum is None:
        return 1.0
    if minimum is None:
        minimum = target if target is not None else 0.0
    if maximum is None:
        maximum = target if target is not None else 1.0
    if target is None:
        target = (minimum + maximum) * 0.5
    if minimum <= value <= maximum:
        width = max(abs(target - minimum), abs(maximum - target), fallback_span)
        return float(np.clip(1.0 - 0.10 * abs(value - target) / width, 0.90, 1.0))
    if value < minimum:
        return float(np.clip(1.0 - (minimum - value) / fallback_span, 0.0, 1.0))
    return float(np.clip(1.0 - (value - maximum) / fallback_span, 0.0, 1.0))


def _band_score_int(
    value: int,
    *,
    minimum: int | None,
    target: int | None,
    maximum: int | None,
    fallback_span: int,
) -> float:
    return _band_score_float(
        float(value),
        minimum=float(minimum) if minimum is not None else None,
        target=float(target) if target is not None else None,
        maximum=float(maximum) if maximum is not None else None,
        fallback_span=float(max(1, fallback_span)),
    )


def _used_storeys(state: SolverState) -> int:
    grid = state.grid
    pod_idx = state.tiles.pod_index
    used_storeys = 0
    for iy in range(grid.cy):
        any_used = False
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                tid = int(state.assignment[grid.flat_index(ix, iy, iz)])
                if tid < 0:
                    continue
                if pt.is_void_pod_index(int(pod_idx[tid])):
                    continue
                any_used = True
                break
            if any_used:
                break
        if any_used:
            used_storeys += 1
    return used_storeys


def _occupied_columns(state: SolverState) -> int:
    grid = state.grid
    pod_idx = state.tiles.pod_index
    occupied = 0
    for ix in range(grid.cx):
        for iz in range(grid.cz):
            for iy in range(grid.cy):
                tid = int(state.assignment[grid.flat_index(ix, iy, iz)])
                if tid < 0:
                    continue
                if pt.is_void_pod_index(int(pod_idx[tid])):
                    continue
                occupied += 1
                break
    return occupied


def _is_door_or_open(category: int) -> bool:
    return category == pt.DOOR or category == pt.OPEN


def _door_graph_edges(state: SolverState) -> list[tuple[int, int]]:
    """Cell-pair indices connected by a DOOR or OPEN face on at least one side
    AND a category-compatible face on the other.

    Two adjacent cells are connected iff each side's face categories permit
    passage: DOOR↔DOOR, DOOR↔OPEN, OPEN↔OPEN.
    """
    edges: list[tuple[int, int]] = []
    grid = state.grid
    faces = state.tiles.faces
    asg = state.assignment

    for flat in range(grid.cells_total):
        tid = int(asg[flat])
        if tid < 0:
            continue
        ix, iy, iz = grid.from_flat(flat)
        for f in HORIZONTAL_FACES:
            n = grid.neighbour(ix, iy, iz, f)
            if n is None:
                continue
            n_flat = grid.flat_index(*n)
            if n_flat <= flat:
                continue  # avoid duplicate edges (only emit forward direction)
            n_tid = int(asg[n_flat])
            if n_tid < 0:
                continue
            cat_a = int(faces[tid, f])
            cat_b = int(faces[n_tid, OPPOSITE_FACE[f]])
            if _is_door_or_open(cat_a) and _is_door_or_open(cat_b):
                edges.append((flat, n_flat))
        # Vertical connectivity through stairwell pods.
        # A stairwell has OPEN on UP and DOWN; vertical neighbour also needs OPEN/DOOR there.
        for f in (UP, DOWN):
            n = grid.neighbour(ix, iy, iz, f)
            if n is None:
                continue
            n_flat = grid.flat_index(*n)
            if n_flat <= flat:
                continue
            n_tid = int(asg[n_flat])
            if n_tid < 0:
                continue
            cat_a = int(faces[tid, f])
            cat_b = int(faces[n_tid, OPPOSITE_FACE[f]])
            if _is_door_or_open(cat_a) and _is_door_or_open(cat_b):
                edges.append((flat, n_flat))
    return edges


def _entry_cells(state: SolverState) -> list[int]:
    pod_idx = state.tiles.pod_index
    entry_pod = pt.POD_INDEX[pt.POD_ENTRY]
    return [
        flat
        for flat, tid in enumerate(state.assignment)
        if tid >= 0 and int(pod_idx[int(tid)]) == entry_pod
    ]


def _bfs_depths(num_cells: int, edges: list[tuple[int, int]], sources: list[int]) -> np.ndarray:
    depth = np.full(num_cells, -1, dtype=np.int32)
    if not sources:
        return depth
    adj: list[list[int]] = [[] for _ in range(num_cells)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    q: deque[int] = deque()
    for s in sources:
        depth[s] = 0
        q.append(s)
    while q:
        v = q.popleft()
        for u in adj[v]:
            if depth[u] < 0:
                depth[u] = depth[v] + 1
                q.append(u)
    return depth


def functional_adequacy(state: SolverState, programme: Programme) -> float:
    counts = _pod_counts(state)
    validation = validate_pod_counts(counts, programme)
    total = programme.target_min_cells
    if total == 0:
        return 1.0
    penalty = validation.total_missing + validation.total_excess
    if penalty == 0:
        return 1.0
    # Continuous score below 1 so MCTS can climb the gradient even when not
    # yet satisfying the hard floor.
    return max(0.0, 1.0 - penalty / total)


def circulation(state: SolverState, programme: Programme) -> float:
    """Fraction of habitable cells reachable from any entry cell."""
    pod_idx = state.tiles.pod_index
    role = state.tiles.role
    entries = _entry_cells(state)

    habitable_role = 1  # ROLE_HABITABLE in tiles.ROLE_INDEX
    habitable_cells = [
        flat
        for flat, tid in enumerate(state.assignment)
        if tid >= 0 and int(role[int(tid)]) == habitable_role
    ]
    if not habitable_cells:
        return 0.0
    if not entries:
        return 0.0

    edges = _door_graph_edges(state)
    depth = _bfs_depths(state.grid.cells_total, edges, entries)
    reachable = sum(1 for c in habitable_cells if depth[c] >= 0)
    return reachable / len(habitable_cells)


def privacy_gradient(state: SolverState, programme: Programme) -> float:
    """Score how well private rooms (bedroom, bathroom) sit at greater BFS
    depth from the entry than public rooms (living, kitchen).

    Output: 1.0 if every private cell's depth >= every public cell's depth.
    """
    pod_idx = state.tiles.pod_index
    entries = _entry_cells(state)
    if not entries:
        return 0.0
    edges = _door_graph_edges(state)
    depth = _bfs_depths(state.grid.cells_total, edges, entries)

    public = {pt.POD_INDEX[pt.POD_LIVING], pt.POD_INDEX[pt.POD_KITCHEN]}
    private = {pt.POD_INDEX[pt.POD_BEDROOM], pt.POD_INDEX[pt.POD_BATHROOM]}

    public_depths: list[int] = []
    private_depths: list[int] = []
    for flat, tid in enumerate(state.assignment):
        if tid < 0 or depth[flat] < 0:
            continue
        p = int(pod_idx[int(tid)])
        if p in public:
            public_depths.append(int(depth[flat]))
        elif p in private:
            private_depths.append(int(depth[flat]))

    if not public_depths or not private_depths:
        return 0.5  # nothing to compare; neutral

    max_pub = max(public_depths)
    # Reward each private cell that sits deeper than the deepest public cell.
    deeper = sum(1 for d in private_depths if d > max_pub)
    return deeper / len(private_depths)


def daylight(state: SolverState, programme: Programme) -> float:
    """Fraction of habitable cells with at least one WINDOW face that meets
    the grid boundary (i.e. opens onto exterior)."""
    grid = state.grid
    faces = state.tiles.faces
    role = state.tiles.role
    asg = state.assignment

    habitable_role = 1
    habitable_cells: list[int] = []
    lit = 0
    for flat in range(grid.cells_total):
        tid = int(asg[flat])
        if tid < 0:
            continue
        if int(role[tid]) != habitable_role:
            continue
        habitable_cells.append(flat)
        ix, iy, iz = grid.from_flat(flat)
        for f in HORIZONTAL_FACES:
            if grid.neighbour(ix, iy, iz, f) is not None:
                continue
            if int(faces[tid, f]) == pt.WINDOW:
                lit += 1
                break
    if not habitable_cells:
        return 1.0
    return lit / len(habitable_cells)


def vertical_service_stack(state: SolverState, programme: Programme) -> float:
    """Reward bathrooms/kitchens that align vertically with same-type or
    same-service pods directly below."""
    grid = state.grid
    pod_idx = state.tiles.pod_index
    asg = state.assignment

    service_pods = {pt.POD_INDEX[pt.POD_BATHROOM], pt.POD_INDEX[pt.POD_KITCHEN]}
    candidates = 0
    aligned = 0

    for flat in range(grid.cells_total):
        tid = int(asg[flat])
        if tid < 0:
            continue
        p = int(pod_idx[tid])
        if p not in service_pods:
            continue
        ix, iy, iz = grid.from_flat(flat)
        if iy == 0:
            continue  # storey 0 has no cell below
        candidates += 1
        below = grid.neighbour(ix, iy, iz, DOWN)
        if below is None:
            continue
        b_tid = int(asg[grid.flat_index(*below)])
        if b_tid < 0:
            continue
        if int(pod_idx[b_tid]) == p:
            aligned += 1
    if candidates == 0:
        return 1.0
    return aligned / candidates


def programme_efficiency(state: SolverState, programme: Programme) -> float:
    """Reward occupied-cell count staying close to the required utility load."""
    pod_idx = state.tiles.pod_index
    occupied = 0
    for tid in state.assignment:
        if tid < 0:
            continue
        if not pt.is_void_pod_index(int(pod_idx[int(tid)])):
            occupied += 1
    if occupied <= 0 or programme.target_min_cells <= 0:
        return 1.0
    if occupied <= programme.target_min_cells:
        return 1.0
    return max(0.0, programme.target_min_cells / occupied)


def room_budget_fit(
    state: SolverState,
    programme: Programme,
    policy: PlanFitPolicy,
    *,
    context: _ScoreContext | None = None,
) -> float:
    if (
        policy.occupied_cells_min is None
        and policy.occupied_cells_target is None
        and policy.occupied_cells_max is None
    ):
        return 1.0
    report = _structure_report(context or _ScoreContext(), state)
    return _band_score_int(
        report.occupied_cells,
        minimum=policy.occupied_cells_min,
        target=policy.occupied_cells_target,
        maximum=policy.occupied_cells_max,
        fallback_span=max(1, programme.target_min_cells),
    )


def land_utilisation(state: SolverState, programme: Programme, policy: PlanFitPolicy) -> float:
    if (
        policy.ground_fill_min is None
        and policy.ground_fill_target is None
        and policy.ground_fill_max is None
    ):
        return 1.0
    site_area = policy.site_footprint_cells or (state.grid.cx * state.grid.cz)
    fill_ratio = _occupied_columns(state) / max(1, site_area)
    return _band_score_float(
        fill_ratio,
        minimum=policy.ground_fill_min,
        target=policy.ground_fill_target,
        maximum=policy.ground_fill_max,
        fallback_span=0.35,
    )


def storey_profile(state: SolverState, programme: Programme, policy: PlanFitPolicy) -> float:
    if policy.storeys_min is None and policy.storeys_target is None and policy.storeys_max is None:
        return 1.0
    return _band_score_int(
        _used_storeys(state),
        minimum=policy.storeys_min,
        target=policy.storeys_target,
        maximum=policy.storeys_max,
        fallback_span=max(1, state.grid.cy),
    )


def storey_diversity(state: SolverState, programme: Programme) -> float:
    """Penalise single-storey plans when the programme merits vertical spread."""
    used_storeys = _used_storeys(state)
    if used_storeys == 0:
        return 1.0
    desired = 2 if programme.target_min_cells <= 6 else 2 if programme.target_min_cells <= 12 else 3 if programme.target_min_cells <= 24 else 4
    desired = min(desired, state.grid.cy)
    return min(1.0, used_storeys / max(1, desired))


def _structure_report(context: _ScoreContext, state: SolverState):
    if context.structure is None:
        context.structure = analyse_structure(state)
    return context.structure


def boxiness_penalty(
    state: SolverState,
    programme: Programme,
    *,
    context: _ScoreContext | None = None,
) -> float:
    """Reward supported, articulated massing over plain filled prisms."""
    report = _structure_report(context or _ScoreContext(), state)
    occ = report.occupied_mask & report.supported_mask
    if report.occupied_cells <= 1:
        return 0.0
    xs, ys, zs = np.nonzero(occ)
    if xs.size == 0:
        return 0.0
    bbox_vol = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) * (zs.max() - zs.min() + 1)
    fill_ratio = report.occupied_cells / max(1, bbox_vol)
    per_storey = [int(occ[:, iy, :].sum()) for iy in range(state.grid.cy)]
    active = [n for n in per_storey if n > 0]
    if len(active) <= 1:
        storey_variation = 0.0
        stepback_bonus = 0.0
    else:
        storey_variation = (max(active) - min(active)) / max(active)
        stepbacks = 0
        transitions = 0
        prev = active[0]
        for cur in active[1:]:
            transitions += 1
            if cur < prev:
                stepbacks += 1
            prev = cur
        stepback_bonus = stepbacks / transitions if transitions else 0.0
    notch_bonus = 1.0 - min(1.0, fill_ratio)
    return float(
        np.clip(
            (0.45 * notch_bonus + 0.25 * storey_variation + 0.30 * stepback_bonus)
            * report.support_ratio,
            0.0,
            1.0,
        )
    )


def _occupied_storey_mask(state: SolverState, iy: int, occ: np.ndarray) -> np.ndarray:
    return occ[:, iy, :]


def _storey_perimeter(mask: np.ndarray) -> int:
    perimeter = 0
    width, depth = mask.shape
    for ix in range(width):
        for iz in range(depth):
            if not mask[ix, iz]:
                continue
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = ix + dx, iz + dz
                if nx < 0 or nx >= width or nz < 0 or nz >= depth or not mask[nx, nz]:
                    perimeter += 1
    return perimeter


def _column_top_regions(occ: np.ndarray) -> int:
    top = np.full((occ.shape[0], occ.shape[2]), -1, dtype=np.int32)
    for ix in range(occ.shape[0]):
        for iz in range(occ.shape[2]):
            for iy in range(occ.shape[1] - 1, -1, -1):
                if occ[ix, iy, iz]:
                    top[ix, iz] = iy
                    break
    seen = np.zeros_like(top, dtype=bool)
    regions = 0
    for sx in range(top.shape[0]):
        for sz in range(top.shape[1]):
            if seen[sx, sz] or top[sx, sz] < 0:
                continue
            regions += 1
            target = int(top[sx, sz])
            q: deque[tuple[int, int]] = deque([(sx, sz)])
            seen[sx, sz] = True
            while q:
                ix, iz = q.popleft()
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, nz = ix + dx, iz + dz
                    if nx < 0 or nx >= top.shape[0] or nz < 0 or nz >= top.shape[1]:
                        continue
                    if seen[nx, nz] or int(top[nx, nz]) != target:
                        continue
                    seen[nx, nz] = True
                    q.append((nx, nz))
    return regions


def massing_articulation(
    state: SolverState,
    programme: Programme,
    *,
    context: _ScoreContext | None = None,
) -> float:
    """Reward supported notches, tapering, and modest overhangs.

    This is deliberately a style preference, not a validity rule. Unsupported
    cells still score poorly through the structural components and are rejected
    by the hard validator.
    """
    report = _structure_report(context or _ScoreContext(), state)
    if report.occupied_cells <= 1:
        return 0.0
    occ = report.occupied_mask & report.supported_mask
    per_storey = [int(occ[:, iy, :].sum()) for iy in range(state.grid.cy)]
    active = [count for count in per_storey if count > 0]
    if not active:
        return 0.0

    complexity_scores: list[float] = []
    for iy, area in enumerate(per_storey):
        if area <= 1:
            continue
        mask = _occupied_storey_mask(state, iy, occ)
        perimeter = _storey_perimeter(mask)
        ideal_perimeter = 4.0 * float(np.sqrt(area))
        complexity_scores.append(float(np.clip((perimeter / max(1.0, ideal_perimeter) - 1.0) / 0.75, 0.0, 1.0)))
    perimeter_complexity = float(np.mean(complexity_scores)) if complexity_scores else 0.0

    if len(active) <= 1:
        taper_score = 0.0
    else:
        top_area = active[-1]
        base_area = active[0]
        if top_area <= base_area:
            taper_score = min(1.0, (base_area - top_area) / max(1, base_area))
        else:
            taper_score = 0.0

    regions = _column_top_regions(occ)
    region_score = 0.0 if regions <= 1 else min(1.0, (regions - 1) / 3.0)
    overhang_score = 1.0 - min(1.0, abs(report.overhang_ratio - 0.18) / 0.18)
    if report.overhang_ratio <= 0.0:
        overhang_score = 0.0

    return float(
        np.clip(
            (
                0.35 * perimeter_complexity
                + 0.30 * taper_score
                + 0.20 * region_score
                + 0.15 * overhang_score
            )
            * report.support_ratio,
            0.0,
            1.0,
        )
    )


def aesthetic_facade(state: SolverState, programme: Programme) -> float:
    """Light continuity preference for boundary rows, not a boxiness incentive."""
    grid = state.grid
    faces = state.tiles.faces
    asg = state.assignment

    transitions = 0
    boundary_segments = 0

    for iy in range(grid.cy):
        # North row (z=0): cells (ix, iy, 0).face=NORTH
        # South row (z=cz-1): face=SOUTH
        # West column (x=0): face=WEST
        # East column (x=cx-1): face=EAST
        for face_idx, walk in (
            (0, [(ix, iy, 0) for ix in range(grid.cx)]),
            (2, [(ix, iy, grid.cz - 1) for ix in range(grid.cx)]),
            (3, [(0, iy, iz) for iz in range(grid.cz)]),
            (1, [(grid.cx - 1, iy, iz) for iz in range(grid.cz)]),
        ):
            prev = -1
            seg_started = False
            for ix, _iy, iz in walk:
                tid = int(asg[grid.flat_index(ix, _iy, iz)])
                if tid < 0:
                    continue
                cat = int(faces[tid, face_idx])
                if cat == pt.EMPTY:
                    continue
                if not seg_started:
                    seg_started = True
                    prev = cat
                    boundary_segments += 1
                    continue
                if cat != prev:
                    transitions += 1
                    prev = cat

    if boundary_segments == 0:
        return 1.0
    # Normalise: average transitions per segment row. 0 transitions = perfect.
    avg = transitions / boundary_segments
    return float(np.clip(1.0 - avg / 6.0, 0.0, 1.0))


def structural_plausibility(
    state: SolverState,
    programme: Programme,
    *,
    context: _ScoreContext | None = None,
) -> float:
    """Score the fraction of non-EMPTY cells that are *structurally supported*.

    Definition (cantilever-aware)
    -----------------------------
    A non-EMPTY cell ``(ix, iy, iz)`` is **supported** iff:

    1. ``iy == 0`` (storey-0 cells are ground-anchored, always supported), OR
    2. the cell directly below ``(ix, iy-1, iz)`` is non-EMPTY *and* itself
       supported (transitive through stacks), OR
    3. the cell shares a same-storey 4-connected face with at least one
       non-EMPTY neighbour that is supported (cantilever — the cell hangs
       off a structurally-anchored neighbour).

    Rule (3) permits 1-deep overhangs (bay windows, eaves) which read as
    plausible; floating islands without any anchored neighbour remain
    penalised.

    Algorithm
    ---------
    Sweep storeys bottom-up. For each storey:
      - Mark cells whose below is supported (vertical propagation).
      - Run a BFS over 4-connected non-EMPTY cells at that storey, seeded
        by the already-marked supported cells at that storey, propagating
        the "supported" flag laterally (cantilever propagation).

    Vulnerability
    -------------
    Cantilever propagation runs storey-by-storey, so a chain of cantilevers
    that hops between storeys (a stair-step diagonal of overhangs) will
    still resolve correctly because each storey reads the support state of
    the storey beneath it. However, *only same-storey* cantilever support
    is recognised — a cell that overhangs purely from a diagonal cell on
    the storey below is flagged (correct: such a configuration is
    physically marginal in voxel housing).
    """
    report = _structure_report(context or _ScoreContext(), state)
    if report.occupied_cells == 0:
        return 1.0
    total_weight = 0.0
    supported_weight = 0.0
    occ = report.occupied_mask
    sup = report.supported_mask
    for iy in range(state.grid.cy):
        weight = float(iy + 1)
        for iz in range(state.grid.cz):
            for ix in range(state.grid.cx):
                if not occ[ix, iy, iz]:
                    continue
                total_weight += weight
                if sup[ix, iy, iz]:
                    supported_weight += weight
    if total_weight == 0.0:
        return 1.0
    return supported_weight / total_weight


def structural_physics(
    state: SolverState,
    programme: Programme,
    *,
    context: _ScoreContext | None = None,
) -> float:
    """Secondary physics quality score beyond binary support reachability."""
    report = _structure_report(context or _ScoreContext(), state)
    if report.occupied_cells == 0:
        return 1.0
    cantilever_vals = report.cantilever_distance[report.cantilever_distance >= 0]
    max_cantilever = int(cantilever_vals.max()) if cantilever_vals.size else 0
    cantilever_score = 1.0 / (1.0 + max(0, max_cantilever - 1))
    cluster_score = report.largest_cluster / report.occupied_cells
    altitude_penalty = 1.0 / (1.0 + max(0, report.max_altitude - 3) * 0.15)
    anchored_core_score = report.anchored_ratio
    overhang_penalty = 1.0 - min(1.0, report.overhang_ratio)
    return float(
        np.clip(
            0.30 * report.support_ratio
            + 0.20 * anchored_core_score
            + 0.20 * cluster_score
            + 0.15 * cantilever_score
            + 0.10 * overhang_penalty
            + 0.05 * altitude_penalty,
            0.0,
            1.0,
        )
    )


COMPONENTS: Final[dict[str, "callable"]] = {
    "functional_adequacy": functional_adequacy,
    "circulation": circulation,
    "privacy_gradient": privacy_gradient,
    "daylight": daylight,
    "vertical_service_stack": vertical_service_stack,
    "programme_efficiency": programme_efficiency,
    "room_budget_fit": room_budget_fit,
    "land_utilisation": land_utilisation,
    "storey_profile": storey_profile,
    "storey_diversity": storey_diversity,
    "structural_physics": structural_physics,
    "boxiness_penalty": boxiness_penalty,
    "massing_articulation": massing_articulation,
    "aesthetic_facade": aesthetic_facade,
    "structural_plausibility": structural_plausibility,
}


def score(state: SolverState, programme: Programme, weights: ScoreWeights | None = None) -> ScoreReport:
    w = weights or ScoreWeights()
    components: dict[str, float] = {}
    context = _ScoreContext()
    for name, fn in COMPONENTS.items():
        if name in {"boxiness_penalty", "massing_articulation", "room_budget_fit", "structural_physics", "structural_plausibility"}:
            if name == "room_budget_fit":
                components[name] = float(fn(state, programme, w.fit_policy, context=context))
                continue
            components[name] = float(fn(state, programme, context=context))
        elif name in {"land_utilisation", "storey_profile"}:
            components[name] = float(fn(state, programme, w.fit_policy))
        else:
            components[name] = float(fn(state, programme))

    weighted_total = (
        w.functional_adequacy * components["functional_adequacy"]
        + w.circulation * components["circulation"]
        + w.privacy_gradient * components["privacy_gradient"]
        + w.daylight * components["daylight"]
        + w.vertical_service_stack * components["vertical_service_stack"]
        + w.programme_efficiency * components["programme_efficiency"]
        + w.room_budget_fit * components["room_budget_fit"]
        + w.land_utilisation * components["land_utilisation"]
        + w.storey_profile * components["storey_profile"]
        + w.storey_diversity * components["storey_diversity"]
        + w.structural_physics * components["structural_physics"]
        + w.boxiness_penalty * components["boxiness_penalty"]
        + w.massing_articulation * components["massing_articulation"]
        + w.aesthetic_facade * components["aesthetic_facade"]
        + w.structural_plausibility * components["structural_plausibility"]
    )
    weight_sum = (
        w.functional_adequacy
        + w.circulation
        + w.privacy_gradient
        + w.daylight
        + w.vertical_service_stack
        + w.programme_efficiency
        + w.room_budget_fit
        + w.land_utilisation
        + w.storey_profile
        + w.storey_diversity
        + w.structural_physics
        + w.boxiness_penalty
        + w.massing_articulation
        + w.aesthetic_facade
        + w.structural_plausibility
    )
    total = weighted_total / weight_sum if weight_sum > 0.0 else 0.0

    if w.hard_floor_on_functional and components["functional_adequacy"] < 1.0:
        # Soft penalty: scale the total so the hard requirement still dominates.
        # Applied first so MCTS climbs the functional gradient before circulation.
        total *= 0.5 * components["functional_adequacy"]
    elif w.hard_floor_on_circulation and components["circulation"] < 1.0:
        # Only after functional is satisfied: disconnected habitables are a
        # programme-defeating failure, on par with missing required pods.
        total *= 0.5 * components["circulation"]

    return ScoreReport(total=total, components=components)


__all__ = [
    "COMPONENTS",
    "ScoreReport",
    "ScoreWeights",
    "aesthetic_facade",
    "circulation",
    "daylight",
    "functional_adequacy",
    "massing_articulation",
    "land_utilisation",
    "privacy_gradient",
    "boxiness_penalty",
    "programme_efficiency",
    "PlanFitPolicy",
    "room_budget_fit",
    "score",
    "structural_physics",
    "storey_profile",
    "storey_diversity",
    "structural_plausibility",
    "vertical_service_stack",
]
