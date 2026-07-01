from __future__ import annotations

from prefab_housing.entry_door import ENTRY_DOOR_BLOCK, generate_entry_door_blocks
from prefab_housing.grid import CellGrid
from prefab_housing.types import SemanticCell


def _entry_cell() -> SemanticCell:
    return SemanticCell(
        cell_index=(0, 0, 0),
        voxel_bbox=((0, 0, 0), (7, 5, 7)),
        label="entry",
        role="circulation",
        occupancy_capacity=1,
        daylight_score=0.0,
        privacy_depth=0,
        door_faces=("north", "east"),
        window_faces=(),
        interior_volume_voxels=144,
        pod_template_id="entry@0",
        opening_pattern="multi_direction_open",
    )


def test_entry_door_module_fills_boundary_aperture_only() -> None:
    grid = CellGrid(cx=2, cy=1, cz=1)

    blocks = generate_entry_door_blocks([_entry_cell()], grid, material_theme="sci_fi_modular")

    by_position = {
        (int(block["x"]), int(block["y"]), int(block["z"])): block
        for block in blocks
    }
    door_blocks = [block for block in blocks if block["id"] == ENTRY_DOOR_BLOCK]

    assert len(door_blocks) == 4
    assert {block["properties"]["facing"] for block in door_blocks} == {"north"}
    assert by_position[(3, 1, 0)]["properties"]["hinge"] == "right"
    assert by_position[(3, 2, 0)]["properties"]["hinge"] == "right"
    assert by_position[(4, 1, 0)]["properties"]["hinge"] == "left"
    assert by_position[(4, 2, 0)]["properties"]["hinge"] == "left"
    assert [block["properties"]["half"] for block in door_blocks].count("lower") == 2
    assert [block["properties"]["half"] for block in door_blocks].count("upper") == 2

    lower_and_upper_aperture = {
        (x, y, 0)
        for x in range(2, 6)
        for y in (1, 2)
    }
    lintel = {(x, 3, 0) for x in range(2, 6)}
    assert lower_and_upper_aperture <= set(by_position)
    assert lintel <= set(by_position)

    # The east face is listed as a door face but has a neighbour in this grid,
    # so the exterior-door module must not emit a second boundary door there.
    assert all(int(block["x"]) != 7 for block in blocks)
