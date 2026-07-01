"""Simplified room-layout review rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from prefab_housing.types import RoomLayoutPlan, RoomSignature


_CATEGORY_COLOURS: dict[str, str] = {
    "core": "#b22222",
    "supplementary": "#4f94cd",
    "lighting": "#f1c40f",
}


@dataclass(frozen=True, slots=True)
class RoomLayoutAnalysis:
    component_count: int
    core_count: int
    supplementary_count: int
    lighting_count: int


def analyse_room_layout(layout: RoomLayoutPlan) -> RoomLayoutAnalysis:
    core = sum(1 for item in layout.placements if item.category == "core")
    supplementary = sum(1 for item in layout.placements if item.category == "supplementary")
    lighting = sum(1 for item in layout.placements if item.category == "lighting")
    return RoomLayoutAnalysis(
        component_count=len(layout.placements),
        core_count=core,
        supplementary_count=supplementary,
        lighting_count=lighting,
    )


def _is_floor_cover(block_id: str) -> bool:
    return "carpet" in block_id


def _placement_lines(layout: RoomLayoutPlan) -> list[str]:
    lines = ["placements:"]
    for placement in layout.placements:
        x, y, z = placement.origin
        fx, fz = placement.footprint
        lines.append(
            f"- {placement.category:<13} {placement.keyword:<14} y={y} xz=({x},{z}) fp={fx}x{fz}"
        )
    return lines


def _face_annotation_lines(layout: RoomLayoutPlan) -> list[str]:
    lines = [f"opening_pattern: {layout.opening_pattern}"]
    lines.append(f"door_faces: {layout.door_faces or ('-',)}")
    lines.append(f"open_faces: {layout.open_faces or ('-',)}")
    lines.append(f"window_faces: {layout.window_faces or ('-',)}")
    return lines


def _draw_face_annotations(ax_room: plt.Axes, layout: RoomLayoutPlan) -> None:
    ix, _, iz = layout.interior_size
    edge_style = {"fontsize": 9, "weight": "bold", "color": "#222222", "clip_on": False}
    face_labels: dict[str, str] = {}
    for face in layout.door_faces:
        face_labels[face] = "D"
    for face in layout.open_faces:
        face_labels[face] = "O" if face not in face_labels else face_labels[face] + "/O"
    for face in layout.window_faces:
        face_labels[face] = "W" if face not in face_labels else face_labels[face] + "/W"

    if "north" in face_labels:
        ax_room.text(ix / 2, -0.25, face_labels["north"], ha="center", va="center", **edge_style)
    if "south" in face_labels:
        ax_room.text(ix / 2, iz + 0.25, face_labels["south"], ha="center", va="center", **edge_style)
    if "west" in face_labels:
        ax_room.text(-0.25, iz / 2, face_labels["west"], ha="center", va="center", rotation=90, **edge_style)
    if "east" in face_labels:
        ax_room.text(ix + 0.25, iz / 2, face_labels["east"], ha="center", va="center", rotation=270, **edge_style)


def _legend_handles() -> list[object]:
    return [
        Patch(
            facecolor=_CATEGORY_COLOURS["core"],
            edgecolor="#111111",
            alpha=0.72,
            label="Core furniture",
        ),
        Patch(
            facecolor=_CATEGORY_COLOURS["supplementary"],
            edgecolor="#111111",
            alpha=0.72,
            label="Supplementary furniture",
        ),
        Patch(
            facecolor="white",
            edgecolor=_CATEGORY_COLOURS["supplementary"],
            hatch="//",
            linewidth=2.0,
            label="Floor cover",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_CATEGORY_COLOURS["lighting"],
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            markersize=10,
            label="Lighting fixture",
        ),
    ]


def save_room_layout_report(
    layout: RoomLayoutPlan,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    ix, _, iz = layout.interior_size
    analysis = analyse_room_layout(layout)
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    room_gs = gs[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.20], hspace=0.04)
    ax_room = fig.add_subplot(room_gs[0, 0])
    ax_legend = fig.add_subplot(room_gs[1, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax_room.set_xlim(0, ix)
    ax_room.set_ylim(0, iz)
    ax_room.set_aspect("equal")
    ax_room.invert_yaxis()
    ax_room.set_title("Room Layout")
    ax_room.set_xlabel("x")
    ax_room.set_ylabel("z")
    ax_room.set_xticks(range(ix + 1))
    ax_room.set_yticks(range(iz + 1))
    ax_room.grid(True, color="#e5e5e5", linewidth=0.8)
    ax_room.set_facecolor("#fafafa")
    ax_room.add_patch(Rectangle((0, 0), ix, iz, fill=False, edgecolor="#dddddd", linewidth=2))

    footprint_placements = [item for item in layout.placements if item.category != "lighting"]
    lighting_placements = [item for item in layout.placements if item.category == "lighting"]

    for placement in sorted(
        footprint_placements,
        key=lambda item: item.footprint[0] * item.footprint[1],
        reverse=True,
    ):
        x, _, z = placement.origin
        colour = _CATEGORY_COLOURS.get(placement.category, "#95a5a6")
        is_floor_cover = _is_floor_cover(placement.block_id)
        rect = Rectangle(
            (x - 1, z - 1),
            placement.footprint[0],
            placement.footprint[1],
            facecolor="none" if is_floor_cover else colour,
            alpha=1.0 if is_floor_cover else 0.72,
            edgecolor=colour if is_floor_cover else "#111111",
            hatch="//" if is_floor_cover else None,
            linewidth=2.0 if is_floor_cover else 1.5,
            zorder=2 if is_floor_cover else 3,
        )
        ax_room.add_patch(rect)

    light_offsets: dict[tuple[int, int], int] = defaultdict(int)
    marker_offsets = ((0.0, 0.0), (-0.18, -0.18), (0.18, -0.18), (-0.18, 0.18), (0.18, 0.18))
    for placement in lighting_placements:
        x, _, z = placement.origin
        key = (x, z)
        marker_index = light_offsets[key]
        dx, dz = marker_offsets[min(marker_index, len(marker_offsets) - 1)]
        light_offsets[key] += 1
        ax_room.scatter(
            [x - 0.5 + dx],
            [z - 0.5 + dz],
            s=180,
            marker="o",
            facecolor="#f1c40f",
            edgecolor="#111111",
            linewidth=1.2,
            zorder=5,
        )

    _draw_face_annotations(ax_room, layout)

    ax_legend.axis("off")
    ax_legend.legend(
        handles=_legend_handles(),
        loc="center",
        ncol=2,
        frameon=False,
        title="Legend",
    )

    ax_text.axis("off")
    signature: RoomSignature = layout.plan.signature
    text = "\n".join(
        (
            f"room_type: {signature.room_type}",
            f"role: {signature.role}",
            f"size_class: {signature.size_class}",
            f"privacy: {signature.privacy_band}",
            f"exposure: {signature.exposure}",
            f"lighting: {signature.lighting_tier}",
            f"components: {analysis.component_count}",
            f"core: {analysis.core_count}",
            f"supplementary: {analysis.supplementary_count}",
            f"lighting fixtures: {analysis.lighting_count}",
            *_face_annotation_lines(layout),
            f"core_keywords: {layout.plan.core_keywords}",
            f"supp_keywords: {layout.plan.supplementary_keywords}",
            f"light_keywords: {layout.plan.lighting_keywords}",
            "",
            *_placement_lines(layout),
        )
    )
    ax_text.text(0.0, 1.0, text, va="top", family="monospace", fontsize=9)
    ax_text.set_title("Room Metrics")

    fig.suptitle(title or f"Room Review: {signature.room_type}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


__all__ = [
    "RoomLayoutAnalysis",
    "analyse_room_layout",
    "save_room_layout_report",
]
