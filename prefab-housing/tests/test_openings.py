from __future__ import annotations

from collections import Counter

from prefab_housing.annotate import annotate
from prefab_housing.connection_policy import derive_connection_policy
from prefab_housing.grid import CellGrid
from prefab_housing.housing_plan import HousingPlan, HousingPlanCell, HousingPlanMetadata, HousingPlanTuning, MassingProfile, StoreyDistributionPlan
from prefab_housing.layout import uniform_layout
from prefab_housing.materialise import materialise
from prefab_housing.palette import resolve_palette
from prefab_housing.programme import Programme
from prefab_housing.stairwell import stairwell_opening_rect
from prefab_housing.types import CellConnectionPolicy, ConnectionPolicy
from prefab_housing.wfc.solver import init_state
from prefab_housing.wfc.tiles import build_tile_set


def _tile_id(tiles, label: str) -> int:
    return tiles.tile_label.index(label)


def _make_plan(grid: CellGrid, tiles, assignments: dict[tuple[int, int, int], str]) -> HousingPlan:
    state = init_state(grid, tiles)
    state.domain[:, :] = False
    for cell_index, tile_label in assignments.items():
        tid = _tile_id(tiles, tile_label)
        flat = grid.flat_index(*cell_index)
        state.assignment[flat] = tid
        state.domain[flat, tid] = True
    state.entropy_count[:] = 1
    cells = tuple(
        HousingPlanCell(
            cell_index=cell_index,
            label=tile_label.split("@")[0],
            role="circulation" if tile_label.startswith(("entry", "corridor", "stairwell")) else "service" if tile_label.startswith("bathroom") else "habitable",
            tile_id=_tile_id(tiles, tile_label),
            tile_label=tile_label,
            rotation_quarters=0,
            occupancy_capacity=0,
            is_empty=False,
        )
        for cell_index, tile_label in assignments.items()
    )
    metadata = HousingPlanMetadata(
        seed=0,
        utility_type="residential",
        occupant_count=1,
        scale_class="compact",
        site_footprint_xz=(grid.cx * 8, grid.cz * 8),
        cell_grid_size=(grid.cx, grid.cy, grid.cz),
        cell_voxel_size=(8, 6, 8),
        score_total=1.0,
        score_breakdown={},
        stage_timings_ms={},
        rollouts=0,
        tuning=HousingPlanTuning(),
        storey_distribution=StoreyDistributionPlan(1, 1, 0, 0),
        massing_profile=MassingProfile(1, "x", 1, 0.0, 0.0),
    )
    programme = Programme(required_pods=tuple(Counter().items()), max_pods=tuple(Counter().items()), optional_pods=tuple(Counter().items()), target_min_cells=0)
    provisional = HousingPlan(
        state=state,
        programme=programme,
        cells=cells,
        metadata=metadata,
        connection_policy=ConnectionPolicy(),
    )
    return HousingPlan(
        state=state,
        programme=provisional.programme,
        cells=cells,
        metadata=metadata,
        connection_policy=derive_connection_policy(provisional),
    )


def test_annotate_preserves_open_faces_and_pattern() -> None:
    grid = CellGrid(cx=1, cy=1, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "corridor@0"})
    layout = uniform_layout(grid, (8, 6, 8))
    cells = annotate(plan.state, layout, plan.connection_policy)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.door_faces == ()
    assert cell.open_faces == ()
    assert cell.window_faces == ()
    assert cell.opening_pattern == "sealed"


def test_sequential_policy_marks_primary_circulation_links_open() -> None:
    grid = CellGrid(cx=2, cy=1, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "corridor@0", (1, 0, 0): "stairwell@0"})
    left = plan.connection_policy.for_cell((0, 0, 0))
    right = plan.connection_policy.for_cell((1, 0, 0))
    assert left is not None and right is not None
    assert "east" in left.open_faces
    assert "west" in right.open_faces
    assert left.opening_pattern == "edge_only"
    assert right.opening_pattern == "edge_only"


def test_materialise_only_carves_explicit_connection_faces() -> None:
    grid = CellGrid(cx=2, cy=1, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "bedroom@90", (1, 0, 0): "bedroom@270"})
    layout = uniform_layout(grid, (8, 6, 8))
    palette = resolve_palette("sci_fi_modular")
    policy = ConnectionPolicy(
        cells=(
            CellConnectionPolicy(cell_index=(0, 0, 0), opening_pattern="sealed"),
            CellConnectionPolicy(cell_index=(1, 0, 0), opening_pattern="sealed"),
        )
    )
    blocks = materialise(plan.state, layout, policy, palette)
    block_positions = {(block["x"], block["y"], block["z"]) for block in blocks}
    doorway_slice = {
        (8, y, z)
        for y in range(1, 4)
        for z in range(2, 6)
    }
    assert doorway_slice <= block_positions


def test_materialise_carves_when_faces_are_explicitly_connected() -> None:
    grid = CellGrid(cx=2, cy=1, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "bedroom@270", (1, 0, 0): "corridor@270"})
    layout = uniform_layout(grid, (8, 6, 8))
    palette = resolve_palette("sci_fi_modular")
    policy = ConnectionPolicy(
        cells=(
            CellConnectionPolicy(cell_index=(0, 0, 0), door_faces=("east",), opening_pattern="edge_only"),
            CellConnectionPolicy(cell_index=(1, 0, 0), door_faces=("west",), opening_pattern="edge_only"),
        )
    )
    blocks = materialise(plan.state, layout, policy, palette)
    block_positions = {(block["x"], block["y"], block["z"]) for block in blocks}
    doorway_slice = {
        (8, y, z)
        for y in range(1, 4)
        for z in range(2, 6)
    }
    assert doorway_slice.isdisjoint(block_positions)


def test_materialise_carves_boundary_entry_door_and_skips_facade_overlay() -> None:
    grid = CellGrid(cx=1, cy=1, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "entry@0"})
    layout = uniform_layout(grid, (8, 6, 8))
    palette = resolve_palette("sci_fi_modular")
    blocks = materialise(plan.state, layout, plan.connection_policy, palette)
    block_positions = {(block["x"], block["y"], block["z"]) for block in blocks}
    doorway_slice = {
        (x, y, z)
        for x in range(2, 6)
        for y in range(1, 4)
        for z in (-1, 0)
    }
    assert doorway_slice.isdisjoint(block_positions)


def test_materialise_carves_vertical_stairwell_aperture() -> None:
    grid = CellGrid(cx=1, cy=2, cz=1)
    tiles = build_tile_set()
    plan = _make_plan(grid, tiles, {(0, 0, 0): "stairwell@0", (0, 1, 0): "stairwell@0"})
    layout = uniform_layout(grid, (8, 6, 8))
    palette = resolve_palette("sci_fi_modular")
    blocks = materialise(plan.state, layout, plan.connection_policy, palette)
    block_positions = {(block["x"], block["y"], block["z"]) for block in blocks}
    rx0, rx1, rz0, rz1 = stairwell_opening_rect((8, 6, 8), (0, 0, 0), direction="up")
    aperture = {
        (x, 5, z)
        for x in range(rx0, rx1 + 1)
        for z in range(rz0, rz1 + 1)
    }
    assert aperture.isdisjoint(block_positions)


def test_stairwell_vertical_aperture_covers_stair_circulation_loop() -> None:
    assert stairwell_opening_rect((8, 6, 8), (0, 0, 0), direction="up") == (2, 5, 2, 5)


def test_stairwell_interior_translation_preserves_block_properties() -> None:
    from prefab_housing.interior import RoomInteriorCache, generate_room_interiors
    from prefab_housing.types import SemanticCell

    cells = [
        SemanticCell(
            cell_index=(0, 0, 0),
            voxel_bbox=((10, 0, 20), (17, 5, 27)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=1,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@0",
        ),
        SemanticCell(
            cell_index=(0, 1, 0),
            voxel_bbox=((10, 6, 20), (17, 11, 27)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=2,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up", "down"),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@1",
        ),
    ]

    cache = RoomInteriorCache()
    interiors, _ = generate_room_interiors(cells, utility_type="residential", cache=cache)
    interior = interiors[0]
    stair_blocks = [block for block in interior.blocks if block["id"] == "minecraft:stone_brick_stairs"]

    assert stair_blocks
    assert all("properties" in block for block in stair_blocks)
    assert {block["properties"].get("facing") for block in stair_blocks} != {None}
    assert {block["properties"].get("shape") for block in stair_blocks} == {"straight"}


def test_stair_stack_is_plan_contiguous_between_storeys() -> None:
    from prefab_housing.types import SemanticCell
    from prefab_housing.stairwell import build_stair_stack_plan

    cells = [
        SemanticCell(
            cell_index=(0, 0, 0),
            voxel_bbox=((0, 0, 0), (7, 5, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=1,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@0",
        ),
        SemanticCell(
            cell_index=(0, 1, 0),
            voxel_bbox=((0, 6, 0), (7, 11, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=2,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up", "down"),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@1",
        ),
        SemanticCell(
            cell_index=(0, 2, 0),
            voxel_bbox=((0, 12, 0), (7, 17, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=3,
            door_faces=("south",),
            window_faces=(),
            open_faces=("down",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@2",
        ),
    ]

    plan = build_stair_stack_plan(tuple(cells))
    traversal = [
        (element.x, element.y, element.z, element.kind)
        for cell in plan.cells
        for element in cell.geometry.elements
    ]

    assert traversal
    assert all(
        (
            abs(traversal[index][0] - traversal[index - 1][0])
            + abs(traversal[index][2] - traversal[index - 1][2])
            == 1
        )
        and traversal[index][1] >= traversal[index - 1][1] - 5
        for index in range(1, len(traversal))
    )
    assert any(kind == "buffer" for _, _, _, kind in traversal)
    middle = next(cell for cell in plan.cells if cell.cell_index == (0, 1, 0))
    assert any(element.kind == "buffer" and element.y == 0 for element in middle.geometry.elements)
    stairs = [
        element
        for cell in plan.cells
        for element in cell.geometry.elements
        if element.kind == "stair"
    ]
    assert stairs[0].facing == "west"
    assert any(element.facing == "east" for element in stairs)


def test_stair_stack_reaches_top_floor_cleanly() -> None:
    from prefab_housing.interior import generate_room_interiors
    from prefab_housing.types import SemanticCell

    cells = [
        SemanticCell(
            cell_index=(0, 0, 0),
            voxel_bbox=((0, 0, 0), (7, 5, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=1,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@0",
        ),
        SemanticCell(
            cell_index=(0, 1, 0),
            voxel_bbox=((0, 6, 0), (7, 11, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=2,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up", "down"),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@1",
        ),
        SemanticCell(
            cell_index=(0, 2, 0),
            voxel_bbox=((0, 12, 0), (7, 17, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=3,
            door_faces=("south",),
            window_faces=(),
            open_faces=("down",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@2",
        ),
    ]

    interiors, _ = generate_room_interiors(cells, utility_type="residential")
    lower_room = next(room for room in interiors if room.cell_index == (0, 0, 0))
    middle_room = next(room for room in interiors if room.cell_index == (0, 1, 0))
    top_room = next(room for room in interiors if room.cell_index == (0, 2, 0))
    top_stairs = [block for block in top_room.blocks if block["id"] == "minecraft:stone_brick_stairs"]

    assert not top_stairs
    lower_flights = [block for block in lower_room.blocks if block["id"] == "minecraft:stone_brick_stairs"]
    assert lower_flights
    assert max(int(block["y"]) for block in lower_flights) == 5
    assert any(block["id"] == "minecraft:smooth_stone" and int(block["y"]) == 6 for block in middle_room.blocks)
    top_flights = [
        block
        for room in interiors
        if room.cell_index == (0, 1, 0)
        for block in room.blocks
        if block["id"] == "minecraft:stone_brick_stairs"
    ]
    assert top_flights
    highest = max(int(block["y"]) for block in top_flights)
    assert highest == 11
    assert any(block["id"] == "minecraft:smooth_stone" and int(block["y"]) == 12 for block in top_room.blocks)


def test_stair_stack_lights_every_storey_on_turn_platform() -> None:
    from prefab_housing.interior import generate_room_interiors
    from prefab_housing.stairwell import build_stair_stack_plan
    from prefab_housing.types import SemanticCell

    cells = [
        SemanticCell(
            cell_index=(0, 0, 0),
            voxel_bbox=((0, 0, 0), (7, 5, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=1,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@0",
        ),
        SemanticCell(
            cell_index=(0, 1, 0),
            voxel_bbox=((0, 6, 0), (7, 11, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=2,
            door_faces=("south",),
            window_faces=(),
            open_faces=("up", "down"),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@1",
        ),
        SemanticCell(
            cell_index=(0, 2, 0),
            voxel_bbox=((0, 12, 0), (7, 17, 7)),
            label="stairwell",
            role="circulation",
            occupancy_capacity=0,
            daylight_score=0.0,
            privacy_depth=3,
            door_faces=("south",),
            window_faces=(),
            open_faces=("down",),
            opening_pattern="edge_only",
            interior_volume_voxels=144,
            pod_template_id="stairwell@2",
        ),
    ]

    interiors, _ = generate_room_interiors(cells, utility_type="residential")
    rooms_by_index = {room.cell_index: room for room in interiors}
    cells_by_index = {cell.cell_index: cell for cell in cells}
    stack_plan = build_stair_stack_plan(tuple(cells))

    for cell_plan in stack_plan.cells:
        room = rooms_by_index[cell_plan.cell_index]
        origin, _ = cells_by_index[cell_plan.cell_index].voxel_bbox
        lanterns = [block for block in room.blocks if block["id"] == "minecraft:lantern"]
        elements = cell_plan.geometry.elements
        expected_turn_positions: set[tuple[int, int, int]] = set()

        for index in range(1, len(elements) - 1):
            previous = elements[index - 1]
            element = elements[index]
            next_element = elements[index + 1]
            prev_step = (element.x - previous.x, element.z - previous.z)
            next_step = (next_element.x - element.x, next_element.z - element.z)
            is_turn = (
                previous.kind == "stair"
                and element.kind == "buffer"
                and next_element.kind == "stair"
                and abs(prev_step[0]) + abs(prev_step[1]) == 1
                and abs(next_step[0]) + abs(next_step[1]) == 1
                and prev_step != next_step
                and next_element.y == element.y + 1
            )
            if is_turn:
                expected_turn_positions.add(
                    (
                        origin[0] + next_element.x,
                        origin[1] + next_element.y - 1,
                        origin[2] + next_element.z,
                    )
                )

        lantern_positions = {
            (int(lantern["x"]), int(lantern["y"]), int(lantern["z"]))
            for lantern in lanterns
        }
        platform_positions = {
            (int(block["x"]), int(block["y"]), int(block["z"]))
            for block in room.blocks
            if block["id"] == "minecraft:smooth_stone"
        }
        stair_positions = {
            (int(block["x"]), int(block["y"]), int(block["z"]))
            for block in room.blocks
            if block["id"] == "minecraft:stone_brick_stairs"
        }

        if expected_turn_positions:
            assert lantern_positions == expected_turn_positions
            assert {(x, y + 1, z) for x, y, z in expected_turn_positions} <= stair_positions
            assert all(
                lantern["properties"] == {"hanging": "true"}
                for lantern in lanterns
            )
        else:
            assert len(lanterns) == 1
            lantern = lanterns[0]
            support_position = (
                int(lantern["x"]),
                int(lantern["y"]) - 1,
                int(lantern["z"]),
            )
            assert support_position in platform_positions
            assert lantern["properties"] == {"hanging": "false"}


def test_top_storey_stairwell_emits_only_aperture_carry_through() -> None:
    from prefab_housing.interior import _component_blocks, make_room_request, plan_room, plan_room_layout
    from prefab_housing.types import SemanticCell

    cell = SemanticCell(
        cell_index=(0, 2, 0),
        voxel_bbox=((0, 12, 0), (7, 17, 7)),
        label="stairwell",
        role="circulation",
        occupancy_capacity=0,
        daylight_score=0.0,
        privacy_depth=3,
        door_faces=("south",),
        window_faces=(),
        open_faces=("down",),
        opening_pattern="edge_only",
        interior_volume_voxels=144,
        pod_template_id="stairwell@2",
    )

    request = make_room_request(cell, utility_type="residential")
    plan = plan_room(request)
    layout = plan_room_layout(plan, request)
    blocks = _component_blocks(layout)

    assert {block["id"] for block in blocks} == {"minecraft:lantern", "minecraft:smooth_stone"}
    assert any(block["id"] == "minecraft:smooth_stone" and int(block["y"]) == 0 for block in blocks)
