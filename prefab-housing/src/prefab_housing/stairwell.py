"""Stairwell geometry shared by shell and interior passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prefab_housing.types import RoomComponentPlacement, RoomLayoutPlan, SemanticBlockDict, SemanticCell

TraversalKind = Literal["stair", "buffer"]
_INITIAL_PHASE = 0
_BUFFER_INDICES = frozenset({2, 5, 8, 11})


@dataclass(frozen=True, slots=True)
class StairTraversalElement:
    kind: TraversalKind
    x: int
    y: int
    z: int
    facing: str | None = None


@dataclass(frozen=True, slots=True)
class StairLightAnchor:
    x: int
    y: int
    z: int
    hanging: bool


@dataclass(frozen=True, slots=True)
class StairCellGeometry:
    elements: tuple[StairTraversalElement, ...]


@dataclass(frozen=True, slots=True)
class StairStackCellPlan:
    cell_index: tuple[int, int, int]
    geometry: StairCellGeometry
    local_blocks: tuple[SemanticBlockDict, ...]


@dataclass(frozen=True, slots=True)
class StairStackPlan:
    cells: tuple[StairStackCellPlan, ...]

    def for_cell(self, cell_index: tuple[int, int, int]) -> StairStackCellPlan | None:
        for cell in self.cells:
            if cell.cell_index == cell_index:
                return cell
        return None


def _centre_shaft(interior_size: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ix, _, iz = interior_size
    x0 = max(2, ix // 2)
    z0 = max(2, iz // 2)
    x1 = min(ix - 1, x0 + 1)
    z1 = min(iz - 1, z0 + 1)
    if x1 <= x0:
        x0, x1 = max(1, ix - 2), max(2, ix - 1)
    if z1 <= z0:
        z0, z1 = max(1, iz - 2), max(2, iz - 1)
    return (x0, x1, z0, z1)


def _movement_shaft(interior_size: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ix, _, iz = interior_size
    cycle = _cycle_positions(interior_size)
    x0 = max(1, min(x for x, _ in cycle))
    x1 = min(ix, max(x for x, _ in cycle))
    z0 = max(1, min(z for _, z in cycle))
    z1 = min(iz, max(z for _, z in cycle))
    return x0, x1, z0, z1


def stairwell_opening_rect(
    voxel_size: tuple[int, int, int],
    cell_index: tuple[int, int, int] | None,
    *,
    direction: str,
) -> tuple[int, int, int, int]:
    vx, _, vz = voxel_size
    del cell_index, direction
    interior_size = (max(1, vx - 2), 0, max(1, vz - 2))
    return _movement_shaft(interior_size)


def _cell_voxel_size(cell: SemanticCell) -> tuple[int, int, int]:
    (x0, y0, z0), (x1, y1, z1) = cell.voxel_bbox
    return (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def _cell_interior_size(cell: SemanticCell) -> tuple[int, int, int]:
    vx, vy, vz = _cell_voxel_size(cell)
    return (max(1, vx - 2), max(1, vy - 2), max(1, vz - 2))


def _cycle_positions(interior_size: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    x0, x1, z0, z1 = _centre_shaft(interior_size)
    return (
        (x1, z1 + 1),
        (x0, z1 + 1),
        (x0 - 1, z1 + 1),
        (x0 - 1, z1),
        (x0 - 1, z0),
        (x0 - 1, z0 - 1),
        (x0, z0 - 1),
        (x1, z0 - 1),
        (x1 + 1, z0 - 1),
        (x1 + 1, z0),
        (x1 + 1, z1),
        (x1 + 1, z1 + 1),
    )


def _aperture_cells(interior_size: tuple[int, int, int]) -> frozenset[tuple[int, int]]:
    x0, x1, z0, z1 = _movement_shaft(interior_size)
    return frozenset((x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1))


def _fallback_light_anchor(interior_size: tuple[int, int, int]) -> tuple[int, int, int]:
    x0, _, z0, _ = _movement_shaft(interior_size)
    return (max(1, x0 - 1), 1, max(1, z0 - 1))


def _is_turn_buffer(
    previous: StairTraversalElement,
    element: StairTraversalElement,
    next_element: StairTraversalElement,
) -> bool:
    if (
        previous.kind != "stair"
        or element.kind != "buffer"
        or next_element.kind != "stair"
    ):
        return False
    prev_step = (element.x - previous.x, element.z - previous.z)
    next_step = (next_element.x - element.x, next_element.z - element.z)
    return (
        abs(prev_step[0]) + abs(prev_step[1]) == 1
        and abs(next_step[0]) + abs(next_step[1]) == 1
        and prev_step != next_step
        and next_element.y == element.y + 1
    )


def _light_anchors(
    interior_size: tuple[int, int, int],
    elements: tuple[StairTraversalElement, ...],
) -> tuple[StairLightAnchor, ...]:
    anchors: list[StairLightAnchor] = []
    seen: set[tuple[int, int, int]] = set()

    for index in range(1, len(elements) - 1):
        previous = elements[index - 1]
        element = elements[index]
        next_element = elements[index + 1]
        if not _is_turn_buffer(previous, element, next_element):
            continue
        position = (next_element.x, next_element.y - 1, next_element.z)
        if position in seen:
            continue
        seen.add(position)
        anchors.append(
            StairLightAnchor(
                x=position[0],
                y=position[1],
                z=position[2],
                hanging=True,
            )
        )

    if anchors:
        return tuple(anchors)
    for element in elements:
        if element.kind == "buffer":
            return (
                StairLightAnchor(
                    x=element.x,
                    y=element.y + 1,
                    z=element.z,
                    hanging=False,
                ),
            )
    fallback_x, fallback_y, fallback_z = _fallback_light_anchor(interior_size)
    return (StairLightAnchor(x=fallback_x, y=fallback_y, z=fallback_z, hanging=False),)


def _top_exit_target(interior_size: tuple[int, int, int], current: tuple[int, int]) -> tuple[int, int]:
    x0, x1, z0, z1 = _centre_shaft(interior_size)
    mapping = {
        (x1, z1): (x1 + 1, z1),
        (x0, z1): (x0 - 1, z1),
        (x0, z0): (x0 - 1, z0),
        (x1, z0): (x1 + 1, z0),
    }
    return mapping.get(current, current)


def _step_facing(current: tuple[int, int], nxt: tuple[int, int] | None) -> str:
    if nxt is None:
        return "north"
    x, z = current
    nx, nz = nxt
    if nx > x:
        return "east"
    if nx < x:
        return "west"
    if nz > z:
        return "south"
    return "north"


def _stair_count_for_cell(cell: SemanticCell) -> int:
    _, cell_height, _ = _cell_voxel_size(cell)
    if "up" not in cell.open_faces:
        return 0
    return cell_height - 1


def _start_y_for_cell(cell: SemanticCell) -> int:
    return 1 if "up" in cell.open_faces else 0


def _next_stair_index(index: int, cycle_length: int) -> int:
    index %= cycle_length
    while index in _BUFFER_INDICES:
        index = (index + 1) % cycle_length
    return index


def _build_cell_elements(
    cycle: tuple[tuple[int, int], ...],
    *,
    phase: int,
    stair_count: int,
    start_y: int,
    has_down: bool,
) -> tuple[tuple[StairTraversalElement, ...], int]:
    elements: list[StairTraversalElement] = []
    current_y = start_y
    stairs_placed = 0
    index = phase

    if has_down:
        x, z = cycle[index]
        elements.append(StairTraversalElement(kind="buffer", x=x, y=0, z=z))
        index = (index + 1) % len(cycle)

    if stair_count <= 0:
        return tuple(elements), index

    while stairs_placed < stair_count:
        x, z = cycle[index]
        if index in _BUFFER_INDICES:
            elements.append(StairTraversalElement(kind="buffer", x=x, y=current_y - 1, z=z))
        else:
            elements.append(StairTraversalElement(kind="stair", x=x, y=current_y, z=z))
            current_y += 1
            stairs_placed += 1
        index = (index + 1) % len(cycle)

    return tuple(elements), index


def _apply_facing(
    interior_size: tuple[int, int, int],
    cells: list[tuple[SemanticCell, list[StairTraversalElement]]],
) -> list[tuple[SemanticCell, tuple[StairTraversalElement, ...]]]:
    flat: list[tuple[int, int, StairTraversalElement]] = []
    for cell_index, (_, elements) in enumerate(cells):
        for element_index, element in enumerate(elements):
            flat.append((cell_index, element_index, element))

    rebuilt: list[list[StairTraversalElement]] = [[] for _ in cells]
    for flat_index, (cell_index, element_index, element) in enumerate(flat):
        del element_index
        if element.kind == "buffer":
            rebuilt[cell_index].append(element)
            continue

        next_pos: tuple[int, int] | None = None
        if flat_index + 1 < len(flat):
            next_element = flat[flat_index + 1][2]
            next_pos = (next_element.x, next_element.z)
        else:
            next_pos = _top_exit_target(interior_size, (element.x, element.z))

        rebuilt[cell_index].append(
            StairTraversalElement(
                kind="stair",
                x=element.x,
                y=element.y,
                z=element.z,
                facing=_step_facing((element.x, element.z), next_pos),
            )
        )

    return [
        (cells[index][0], tuple(rebuilt[index]))
        for index in range(len(cells))
    ]


def _emit_cell_blocks(
    interior_size: tuple[int, int, int],
    elements: tuple[StairTraversalElement, ...],
) -> list[SemanticBlockDict]:
    light_anchors = _light_anchors(interior_size, elements)
    aperture = _aperture_cells(interior_size)
    blocks: list[SemanticBlockDict] = []

    def add(x: int, y: int, z: int, block_id: str, properties: dict[str, str] | None = None) -> None:
        block: SemanticBlockDict = {"x": x, "y": y, "z": z, "id": block_id}
        if properties is not None:
            block["properties"] = properties
        blocks.append(block)

    for element in elements:
        if element.kind == "stair":
            add(
                element.x,
                element.y,
                element.z,
                "minecraft:stone_brick_stairs",
                {"facing": element.facing or "south", "half": "bottom", "shape": "straight"},
            )
        else:
            add(element.x, element.y, element.z, "minecraft:smooth_stone")

        if (element.x, element.z) in aperture:
            continue
        for support_y in range(1, element.y):
            add(element.x, support_y, element.z, "minecraft:stone_bricks")

    for anchor in light_anchors:
        add(
            anchor.x,
            anchor.y,
            anchor.z,
            "minecraft:lantern",
            {"hanging": "true" if anchor.hanging else "false"},
        )
    return blocks


def build_stair_stack_plan(cells: tuple[SemanticCell, ...]) -> StairStackPlan:
    if not cells:
        return StairStackPlan(cells=())

    ordered = tuple(sorted(cells, key=lambda cell: cell.cell_index[1]))
    interior_size = _cell_interior_size(ordered[0])
    cycle = _cycle_positions(interior_size)
    phase = _INITIAL_PHASE
    raw: list[tuple[SemanticCell, list[StairTraversalElement]]] = []

    for cell in ordered:
        has_up = "up" in cell.open_faces
        has_down = "down" in cell.open_faces
        if not has_up and has_down:
            elements = (StairTraversalElement(kind="buffer", x=cycle[phase][0], y=0, z=cycle[phase][1]),)
        else:
            elements, phase = _build_cell_elements(
                cycle,
                phase=phase,
                stair_count=_stair_count_for_cell(cell),
                start_y=_start_y_for_cell(cell),
                has_down=has_down,
            )
        raw.append((cell, list(elements)))

    with_facing = _apply_facing(interior_size, raw)
    return StairStackPlan(
        cells=tuple(
            StairStackCellPlan(
                cell_index=cell.cell_index,
                geometry=StairCellGeometry(elements=elements),
                local_blocks=tuple(_emit_cell_blocks(interior_size, elements)),
            )
            for cell, elements in with_facing
        )
    )


def stairwell_layout_placements(
    interior_size: tuple[int, int, int],
    cell_index: tuple[int, int, int] | None,
    *,
    has_up: bool,
    has_down: bool,
) -> tuple[RoomComponentPlacement, ...]:
    del cell_index
    placements: list[RoomComponentPlacement] = []
    pseudo_cell = SemanticCell(
        cell_index=(0, 1 if has_down else 0, 0),
        voxel_bbox=(
            (0, 0, 0),
            (interior_size[0] + 1, interior_size[1] + 1, interior_size[2] + 1),
        ),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=0,
        door_faces=("south",),
        window_faces=(),
        open_faces=(("down",) if has_down else ()) + (("up",) if has_up else ()),
        opening_pattern="edge_only",
        interior_volume_voxels=0,
        pod_template_id="stairwell@preview",
    )
    elements = build_stair_stack_plan((pseudo_cell,)).cells[0].geometry.elements
    if has_up:
        xs = [element.x for element in elements]
        zs = [element.z for element in elements]
        ys = [element.y for element in elements] or [1]
        placements.append(
            RoomComponentPlacement(
                keyword="stair_flight",
                block_id="minecraft:stone_brick_stairs",
                category="core",
                origin=(min(xs), min(ys), min(zs)),
                footprint=(max(xs) - min(xs) + 1, max(zs) - min(zs) + 1),
                anchor="centre",
            )
        )
    for anchor in _light_anchors(interior_size, elements):
        placements.append(
            RoomComponentPlacement(
                keyword="landing_light",
                block_id="minecraft:lantern",
                category="lighting",
                origin=(anchor.x, anchor.y, anchor.z),
                footprint=(1, 1),
                anchor="centre",
            )
        )
    return tuple(placements)


def emit_stairwell_blocks(layout: RoomLayoutPlan) -> list[SemanticBlockDict]:
    pseudo_cell = SemanticCell(
        cell_index=layout.cell_index or (0, 0, 0),
        voxel_bbox=(
            (0, 0, 0),
            (
                layout.plan.signature.voxel_size[0] - 1,
                layout.plan.signature.voxel_size[1] - 1,
                layout.plan.signature.voxel_size[2] - 1,
            ),
        ),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=0,
        door_faces=layout.door_faces,
        window_faces=layout.window_faces,
        open_faces=layout.open_faces,
        opening_pattern=layout.opening_pattern,
        interior_volume_voxels=0,
        pod_template_id="stairwell@0",
    )
    stack_plan = build_stair_stack_plan((pseudo_cell,))
    cell_plan = stack_plan.for_cell(pseudo_cell.cell_index)
    if cell_plan is None:
        return []
    return list(cell_plan.local_blocks)


__all__ = [
    "StairCellGeometry",
    "StairStackCellPlan",
    "StairStackPlan",
    "StairTraversalElement",
    "build_stair_stack_plan",
    "emit_stairwell_blocks",
    "stairwell_layout_placements",
    "stairwell_opening_rect",
]
