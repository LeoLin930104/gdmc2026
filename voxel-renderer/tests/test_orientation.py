from __future__ import annotations

from voxel_renderer.orientation import (
    KNOWN_ROTATABLE_PROPERTIES,
    rotate_block,
    rotate_block_properties,
    rotate_y_property,
)
from voxel_renderer.prefab import rotate_y


def test_known_rotatable_properties_set() -> None:
    assert KNOWN_ROTATABLE_PROPERTIES == frozenset({"facing", "axis", "rotation"})


def test_rotate_facing_clockwise() -> None:
    assert rotate_y_property("facing", "north", 90) == "east"
    assert rotate_y_property("facing", "east", 90) == "south"
    assert rotate_y_property("facing", "south", 90) == "west"
    assert rotate_y_property("facing", "west", 90) == "north"


def test_rotate_facing_180_and_270() -> None:
    assert rotate_y_property("facing", "north", 180) == "south"
    assert rotate_y_property("facing", "north", 270) == "west"
    assert rotate_y_property("facing", "east", 270) == "north"


def test_facing_vertical_passthrough() -> None:
    # piston-style up/down facings are unaffected by Y rotation
    for deg in (0, 90, 180, 270):
        assert rotate_y_property("facing", "up", deg) == "up"
        assert rotate_y_property("facing", "down", deg) == "down"


def test_rotate_axis() -> None:
    assert rotate_y_property("axis", "x", 90) == "z"
    assert rotate_y_property("axis", "z", 90) == "x"
    assert rotate_y_property("axis", "x", 180) == "x"
    assert rotate_y_property("axis", "y", 90) == "y"


def test_rotate_rotation_16_steps() -> None:
    assert rotate_y_property("rotation", "0", 90) == "4"
    assert rotate_y_property("rotation", "12", 90) == "0"  # wraps mod 16
    assert rotate_y_property("rotation", "3", 180) == "11"


def test_rotate_unknown_property_returns_none() -> None:
    assert rotate_y_property("hinge", "left", 90) is None
    assert rotate_y_property("waterlogged", "true", 90) is None


def test_rotate_unparseable_value_returns_none() -> None:
    assert rotate_y_property("facing", "garbage", 90) is None
    assert rotate_y_property("axis", "diagonal", 90) is None
    assert rotate_y_property("rotation", "not_a_number", 90) is None


def test_rotate_block_properties_passthrough_unknown() -> None:
    props = {"facing": "north", "hinge": "left", "waterlogged": "true"}
    out = rotate_block_properties(props, 90)
    assert out == {"facing": "east", "hinge": "left", "waterlogged": "true"}
    # input not mutated
    assert props["facing"] == "north"


def test_rotate_block_strips_empty_properties() -> None:
    block = {"x": 0, "y": 0, "z": 0, "id": "minecraft:stone", "properties": {}}
    rotated = rotate_block(block, 90)
    assert "properties" not in rotated


def test_rotate_block_preserves_known_props() -> None:
    block = {
        "x": 0,
        "y": 0,
        "z": 0,
        "id": "minecraft:oak_stairs",
        "properties": {"facing": "north", "half": "bottom"},
    }
    rotated = rotate_block(block, 90)
    assert rotated["properties"] == {"facing": "east", "half": "bottom"}


def test_prefab_rotate_y_now_transforms_properties() -> None:
    blocks = [
        {
            "x": 0,
            "y": 0,
            "z": 0,
            "id": "minecraft:oak_stairs",
            "properties": {"facing": "north"},
        },
        {
            "x": 1,
            "y": 0,
            "z": 0,
            "id": "minecraft:oak_log",
            "properties": {"axis": "x"},
        },
    ]
    rotated = rotate_y(blocks, 90)
    by_id = {b["id"]: b for b in rotated}
    assert by_id["minecraft:oak_stairs"]["properties"]["facing"] == "east"
    assert by_id["minecraft:oak_log"]["properties"]["axis"] == "z"


def test_prefab_rotate_y_legacy_mode_skips_property_transform() -> None:
    blocks = [
        {
            "x": 0,
            "y": 0,
            "z": 0,
            "id": "minecraft:oak_stairs",
            "properties": {"facing": "north"},
        }
    ]
    rotated = rotate_y(blocks, 90, transform_properties=False)
    # property left untouched in legacy mode
    assert rotated[0]["properties"] == {"facing": "north"}
