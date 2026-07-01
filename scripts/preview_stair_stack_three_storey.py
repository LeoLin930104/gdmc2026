"""Render a three-storey cutaway stair stack for vertical review."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from prefab_housing.catalogue.shell import build_placeholder_cell
from prefab_housing.palette import resolve_palette
from prefab_housing.stairwell import StairStackPlan, build_stair_stack_plan, stairwell_opening_rect
from prefab_housing.types import SemanticBlockDict, SemanticCell
from voxel_renderer.api import render_orthographic_views

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "galleries" / "preview_stair_stack_three_storey"
CELL_VOXEL_SIZE = (8, 6, 8)


def _stack_cell(storey: int, *, has_up: bool, has_down: bool) -> SemanticCell:
    height = CELL_VOXEL_SIZE[1]
    y0 = storey * height
    y1 = y0 + height - 1
    open_faces = tuple(
        face
        for face, enabled in (("up", has_up), ("down", has_down))
        if enabled
    )
    return SemanticCell(
        cell_index=(0, storey, 0),
        voxel_bbox=((0, y0, 0), (CELL_VOXEL_SIZE[0] - 1, y1, CELL_VOXEL_SIZE[2] - 1)),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=storey + 1,
        door_faces=("south",),
        window_faces=(),
        open_faces=open_faces,
        opening_pattern="edge_only",
        interior_volume_voxels=(CELL_VOXEL_SIZE[0] - 2) * (CELL_VOXEL_SIZE[1] - 2) * (CELL_VOXEL_SIZE[2] - 2),
        pod_template_id=f"stairwell@{storey}",
    )


def _translate(blocks: list[SemanticBlockDict], offset: tuple[int, int, int]) -> list[SemanticBlockDict]:
    ox, oy, oz = offset
    return [
        {
            **block,
            "x": int(block["x"]) + ox,
            "y": int(block["y"]) + oy,
            "z": int(block["z"]) + oz,
        }
        for block in blocks
    ]


def _vertical_aperture(cell: SemanticCell) -> set[tuple[int, int, int]]:
    vx, vy, vz = CELL_VOXEL_SIZE
    blocked: set[tuple[int, int, int]] = set()
    if "up" in cell.open_faces:
        rx0, rx1, rz0, rz1 = stairwell_opening_rect((vx, vy, vz), cell.cell_index, direction="up")
        blocked.update((x, vy - 1, z) for x in range(rx0, rx1 + 1) for z in range(rz0, rz1 + 1))
    if "down" in cell.open_faces:
        rx0, rx1, rz0, rz1 = stairwell_opening_rect((vx, vy, vz), cell.cell_index, direction="down")
        blocked.update((x, 0, z) for x in range(rx0, rx1 + 1) for z in range(rz0, rz1 + 1))
    return blocked


def _cutaway_shell(blocks: list[SemanticBlockDict], *, cell_index: tuple[int, int, int]) -> list[SemanticBlockDict]:
    return _cutaway_shell_mode(blocks, cell_index=cell_index, mode="standard")


def _cutaway_shell_mode(
    blocks: list[SemanticBlockDict],
    *,
    cell_index: tuple[int, int, int],
    mode: str,
) -> list[SemanticBlockDict]:
    east_entry = cell_index[1] % 2 == 0
    storey = cell_index[1]
    rx0, rx1, rz0, rz1 = stairwell_opening_rect(CELL_VOXEL_SIZE, cell_index, direction="up")

    def in_cutaway_corner(x: int, z: int) -> bool:
        if east_entry:
            return x >= rx1 and z >= rz1
        return x <= rx0 and z >= rz1

    def keep(block: SemanticBlockDict) -> bool:
        x = int(block["x"])
        y = int(block["y"])
        z = int(block["z"])
        if mode == "landing_focus" and y <= 1:
            if east_entry and x >= rx0:
                return False
            if not east_entry and x <= rx1:
                return False
        if mode == "deep" and in_cutaway_corner(x, z) and (y > 0 or storey > 0):
            return False
        if y == 0:
            return True
        if east_entry and (x == CELL_VOXEL_SIZE[0] - 1 or z == CELL_VOXEL_SIZE[2] - 1):
            return False
        if not east_entry and (x == 0 or z == CELL_VOXEL_SIZE[2] - 1):
            return False
        return True

    return [block for block in blocks if keep(block)]


def _render_cell_shell(
    cell: SemanticCell,
    *,
    palette: dict[str, str],
    cutaway_mode: str,
    remove_roof: bool,
) -> list[SemanticBlockDict]:
    shell = build_placeholder_cell(
        cell_voxel_size=CELL_VOXEL_SIZE,
        palette=palette,
        pod_name="stairwell",
    )
    aperture = _vertical_aperture(cell)
    shell = [block for block in shell if (int(block["x"]), int(block["y"]), int(block["z"])) not in aperture]
    shell = _cutaway_shell_mode(shell, cell_index=cell.cell_index, mode=cutaway_mode)
    if remove_roof:
        shell = [block for block in shell if int(block["y"]) != CELL_VOXEL_SIZE[1] - 1]
    return shell


def _render_stack_interiors(cells: tuple[SemanticCell, ...]) -> dict[tuple[int, int, int], list[SemanticBlockDict]]:
    stack_plan = build_stair_stack_plan(cells)
    out: dict[tuple[int, int, int], list[SemanticBlockDict]] = {}
    for cell in cells:
        cell_plan = stack_plan.for_cell(cell.cell_index)
        out[cell.cell_index] = [] if cell_plan is None else list(cell_plan.local_blocks)
    return out


def _render_stack_interiors_from_plan(
    cells: tuple[SemanticCell, ...],
    stack_plan: StairStackPlan,
) -> dict[tuple[int, int, int], list[SemanticBlockDict]]:
    out: dict[tuple[int, int, int], list[SemanticBlockDict]] = {}
    for cell in cells:
        cell_plan = stack_plan.for_cell(cell.cell_index)
        out[cell.cell_index] = [] if cell_plan is None else list(cell_plan.local_blocks)
    return out



def _render_scene(
    cells: tuple[SemanticCell, ...],
    *,
    palette: dict[str, str],
    cutaway_mode: str,
    roofless_storeys: frozenset[int],
    include_shell: bool = True,
    stack_plan: StairStackPlan | None = None,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    interiors = (
        _render_stack_interiors(cells)
        if stack_plan is None
        else _render_stack_interiors_from_plan(cells, stack_plan)
    )
    for cell in cells:
        shell = []
        if include_shell:
            shell = _render_cell_shell(
                cell,
                palette=palette,
                cutaway_mode=cutaway_mode,
                remove_roof=cell.cell_index[1] in roofless_storeys,
            )
        origin, _ = cell.voxel_bbox
        blocks.extend(
            _translate(shell + interiors.get(cell.cell_index, []), origin)
        )
    return blocks


def build_preview_scenes() -> dict[str, list[SemanticBlockDict]]:
    palette = resolve_palette("sci_fi_modular")
    cells = (
        _stack_cell(0, has_up=True, has_down=False),
        _stack_cell(1, has_up=True, has_down=True),
        _stack_cell(2, has_up=False, has_down=True),
    )
    stack_plan = build_stair_stack_plan(cells)
    return {
        "full_stack": _render_scene(
            cells,
            palette=palette,
            cutaway_mode="deep",
            roofless_storeys=frozenset({2}),
            stack_plan=stack_plan,
        ),
        "storeys_1_2": _render_scene(
            cells[:2],
            palette=palette,
            cutaway_mode="standard",
            roofless_storeys=frozenset({1}),
            stack_plan=stack_plan,
        ),
        "stairs_only_1_2": _render_scene(
            cells[:2],
            palette=palette,
            cutaway_mode="standard",
            roofless_storeys=frozenset(),
            include_shell=False,
            stack_plan=stack_plan,
        ),
        "storeys_2_3": _render_scene(
            cells[1:],
            palette=palette,
            cutaway_mode="standard",
            roofless_storeys=frozenset({2}),
            stack_plan=stack_plan,
        ),
        "stairs_only_2_3": _render_scene(
            cells[1:],
            palette=palette,
            cutaway_mode="standard",
            roofless_storeys=frozenset(),
            include_shell=False,
            stack_plan=stack_plan,
        ),
        "middle_landing": _render_scene(
            cells[1:2],
            palette=palette,
            cutaway_mode="landing_focus",
            roofless_storeys=frozenset({1}),
            stack_plan=stack_plan,
        ),
        "top_storey": _render_scene(
            cells[2:],
            palette=palette,
            cutaway_mode="standard",
            roofless_storeys=frozenset({2}),
            stack_plan=stack_plan,
        ),
    }


def main() -> int:
    scenes = build_preview_scenes()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for png_path in OUT_ROOT.glob("*.png"):
        png_path.unlink()
    scene_summaries: list[str] = []
    for scene_name, blocks in scenes.items():
        views = render_orthographic_views(blocks, width=720, height=540, backend="auto")
        for view_name, b64 in views.items():
            (OUT_ROOT / f"{scene_name}_{view_name}.png").write_bytes(base64.b64decode(b64))
        scene_summaries.append(f"{scene_name}={len(blocks)}")
    print(f"{' '.join(scene_summaries)} out_dir={OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
