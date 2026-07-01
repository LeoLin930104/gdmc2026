"""Topology-only analysis and report rendering for housing plans."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import OPPOSITE_FACE
from prefab_housing.housing_plan import HousingPlan
from prefab_housing.programme import validate_pod_counts
from prefab_housing.structure import analyse_structure


UTILITY_COLOURS: dict[str, str] = {
    pt.POD_ENTRY: "#f28c28",
    pt.POD_LIVING: "#b22222",
    pt.POD_KITCHEN: "#d4a017",
    pt.POD_BATHROOM: "#4f94cd",
    pt.POD_BEDROOM: "#6a3dad",
    pt.POD_CORRIDOR: "#7f8c8d",
    pt.POD_STAIRWELL: "#4caf50",
    pt.POD_STRUCTURAL_VOID: "#111111",
    pt.POD_TERRACE_VOID: "#f2f2f2",
}


@dataclass(frozen=True, slots=True)
class PlanAnalysis:
    occupied_cells: int
    empty_cells: int
    occupancy_ratio: float
    occupied_per_storey: tuple[int, ...]
    utility_counts: tuple[tuple[str, int], ...]
    supported_cells: int
    unsupported_cells: int
    support_ratio: float
    unsupported_indices: tuple[tuple[int, int, int], ...]
    circulation_reachable_ratio: float
    missing_required: tuple[tuple[str, int], ...]
    excess_capped: tuple[tuple[str, int], ...]


def _occupied_mask(plan: HousingPlan) -> np.ndarray:
    grid = plan.state.grid
    mask = np.zeros((grid.cx, grid.cy, grid.cz), dtype=bool)
    tiles = plan.state.tiles
    for flat, tid_raw in enumerate(plan.state.assignment.tolist()):
        tid = int(tid_raw)
        if tid < 0:
            continue
        ix, iy, iz = grid.from_flat(flat)
        mask[ix, iy, iz] = not pt.is_void_pod_index(int(tiles.pod_index[tid]))
    return mask


def _support_mask(plan: HousingPlan) -> np.ndarray:
    grid = plan.state.grid
    non_empty = _occupied_mask(plan)
    supported = np.zeros_like(non_empty)
    if grid.cy == 0:
        return supported
    supported[:, 0, :] = non_empty[:, 0, :]
    for iy in range(1, grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if non_empty[ix, iy, iz] and supported[ix, iy - 1, iz]:
                    supported[ix, iy, iz] = True
        q: deque[tuple[int, int]] = deque()
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if supported[ix, iy, iz]:
                    q.append((ix, iz))
        while q:
            ix, iz = q.popleft()
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = ix + dx, iz + dz
                if not (0 <= nx < grid.cx and 0 <= nz < grid.cz):
                    continue
                if not non_empty[nx, iy, nz] or supported[nx, iy, nz]:
                    continue
                supported[nx, iy, nz] = True
                q.append((nx, nz))
    return supported


def _circulation_reachable_ratio(plan: HousingPlan) -> float:
    state = plan.state
    grid = state.grid
    faces = state.tiles.faces
    pod_index = state.tiles.pod_index
    asg = state.assignment

    def is_openish(cat: int) -> bool:
        return cat == pt.DOOR or cat == pt.OPEN

    adj: list[list[int]] = [[] for _ in range(grid.cells_total)]
    sources: list[int] = []
    habitable: list[int] = []

    for flat in range(grid.cells_total):
        tid = int(asg[flat])
        if tid < 0:
            continue
        pod = int(pod_index[tid])
        label = pt.POD_LABELS[pod]
        if pt.is_void_pod_index(pod):
            continue
        if label == pt.POD_ENTRY:
            sources.append(flat)
        if pt.POD_ROLE[pod] == pt.ROLE_HABITABLE:
            habitable.append(flat)
        ix, iy, iz = grid.from_flat(flat)
        for f in range(6):
            nb = grid.neighbour(ix, iy, iz, f)
            if nb is None:
                continue
            nflat = grid.flat_index(*nb)
            ntid = int(asg[nflat])
            if ntid < 0:
                continue
            npod = int(pod_index[ntid])
            if pt.is_void_pod_index(npod):
                continue
            if is_openish(int(faces[tid, f])) and is_openish(int(faces[ntid, OPPOSITE_FACE[f]])):
                adj[flat].append(nflat)

    if not habitable or not sources:
        return 0.0 if habitable else 1.0
    seen = set(sources)
    q: deque[int] = deque(sources)
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return sum(1 for flat in habitable if flat in seen) / len(habitable)


def analyse_housing_plan(plan: HousingPlan) -> PlanAnalysis:
    grid = plan.state.grid
    structure = analyse_structure(plan.state)
    occupied = structure.occupied_mask
    supported = structure.supported_mask
    occupied_cells = int(occupied.sum())
    total_cells = int(grid.cells_total)
    empty_cells = total_cells - occupied_cells
    occupied_per_storey = tuple(int(occupied[:, iy, :].sum()) for iy in range(grid.cy))
    counts: Counter[str] = Counter(cell.label for cell in plan.cells if not cell.is_empty)
    validation = validate_pod_counts(counts, plan.programme)
    unsupported_indices: list[tuple[int, int, int]] = []
    for iy in range(grid.cy):
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if occupied[ix, iy, iz] and not supported[ix, iy, iz]:
                    unsupported_indices.append((ix, iy, iz))
    supported_cells = structure.supported_cells
    return PlanAnalysis(
        occupied_cells=occupied_cells,
        empty_cells=empty_cells,
        occupancy_ratio=occupied_cells / total_cells if total_cells else 0.0,
        occupied_per_storey=occupied_per_storey,
        utility_counts=tuple(sorted(counts.items())),
        supported_cells=supported_cells,
        unsupported_cells=structure.unsupported_cells,
        support_ratio=structure.support_ratio,
        unsupported_indices=structure.unsupported_indices,
        circulation_reachable_ratio=_circulation_reachable_ratio(plan),
        missing_required=validation.missing_pods,
        excess_capped=validation.excess_pods,
    )


def save_housing_plan_report(
    plan: HousingPlan,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    analysis = analyse_housing_plan(plan)
    grid = plan.state.grid
    occupancy = _occupied_mask(plan)
    structure = analyse_structure(plan.state)
    support = structure.supported_mask

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])

    ax_stack = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_storeys = [fig.add_subplot(gs[1, i]) for i in range(3)]

    storey_idx = np.arange(grid.cy)
    ax_stack.bar(storey_idx, analysis.occupied_per_storey, color="#4e79a7")
    ax_stack.set_title("Occupied Cells Per Storey")
    ax_stack.set_xlabel("Storey")
    ax_stack.set_ylabel("Occupied cells")

    utility_labels = [k for k, _ in analysis.utility_counts]
    utility_vals = [v for _, v in analysis.utility_counts]
    utility_cols = [UTILITY_COLOURS.get(k, "#cccccc") for k in utility_labels]
    ax_bar.barh(utility_labels, utility_vals, color=utility_cols)
    ax_bar.set_title("Utility Mix")
    ax_bar.set_xlabel("Cells")

    ax_metrics.axis("off")
    metrics_text = "\n".join(
        [
            f"score_total: {plan.metadata.score_total:.3f}",
            f"occupants: {plan.metadata.occupant_count}",
            f"scale_class: {plan.metadata.scale_class}",
            f"occupancy_ratio: {analysis.occupancy_ratio:.3f}",
            f"support_ratio: {analysis.support_ratio:.3f}",
            f"reachable_habitable: {analysis.circulation_reachable_ratio:.3f}",
            f"empty_cells: {analysis.empty_cells}",
            f"unsupported_cells: {analysis.unsupported_cells}",
            f"max_altitude: {structure.max_altitude}",
            f"largest_cluster: {structure.largest_cluster}",
            f"max_cantilever: {int(structure.cantilever_distance.max())}",
            f"quirkiness: {plan.metadata.tuning.quirkiness:.2f}",
            f"allow_floor_empty: {plan.metadata.tuning.allow_floor_empty}",
            f"target_storeys: {plan.metadata.storey_distribution.target_storeys}",
            f"terrace_from: {plan.metadata.massing_profile.terrace_start_storey}",
            f"missing_required: {dict(analysis.missing_required)}",
            f"excess_capped: {dict(analysis.excess_capped)}",
        ]
    )
    ax_metrics.text(0.0, 1.0, metrics_text, va="top", family="monospace")
    ax_metrics.set_title("Plan Metrics")

    max_panels = min(3, grid.cy)
    for panel_idx in range(3):
        ax = ax_storeys[panel_idx]
        ax.axis("off")
        if panel_idx >= max_panels:
            continue
        iy = panel_idx
        img = np.zeros((grid.cz, grid.cx, 3), dtype=float)
        for iz in range(grid.cz):
            for ix in range(grid.cx):
                if not occupancy[ix, iy, iz]:
                    img[grid.cz - 1 - iz, ix, :] = np.array(matplotlib.colors.to_rgb(UTILITY_COLOURS[pt.POD_STRUCTURAL_VOID]))
                    continue
                flat = grid.flat_index(ix, iy, iz)
                tid = int(plan.state.assignment[flat])
                label = pt.POD_LABELS[int(plan.state.tiles.pod_index[tid])]
                rgb = matplotlib.colors.to_rgb(UTILITY_COLOURS.get(label, "#ffffff"))
                if not support[ix, iy, iz]:
                    rgb = (1.0, 0.0, 0.0)
                img[grid.cz - 1 - iz, ix, :] = np.array(rgb)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"Storey {iy}")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colour, markersize=10, label=label)
        for label, colour in UTILITY_COLOURS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title or "Housing Plan Review")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


__all__ = [
    "PlanAnalysis",
    "analyse_housing_plan",
    "save_housing_plan_report",
]
