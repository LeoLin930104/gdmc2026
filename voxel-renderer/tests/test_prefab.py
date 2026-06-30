from __future__ import annotations

from voxel_renderer.prefab import (
    face_signature,
    get_bounds,
    merge_prefabs,
    normalise_to_origin,
    opposite_face,
    rotate_y,
    translate_blocks,
)


def test_bounds_and_normalise_to_origin() -> None:
    blocks = [
        {"x": 5, "y": 2, "z": -1, "id": "custom:wall"},
        {"x": 6, "y": 3, "z": 1, "id": "custom:roof"},
    ]
    bounds = get_bounds(blocks)
    assert bounds is not None
    assert bounds.size == (2, 2, 3)

    normalised = normalise_to_origin(blocks)
    assert get_bounds(normalised).size == (2, 2, 3)  # type: ignore[union-attr]
    assert min(int(b["x"]) for b in normalised) == 0
    assert min(int(b["y"]) for b in normalised) == 0
    assert min(int(b["z"]) for b in normalised) == 0


def test_translate_and_merge_last_write_wins() -> None:
    base = [{"x": 0, "y": 0, "z": 0, "id": "custom:a"}]
    moved = translate_blocks(base, 1, 0, 0)
    assert moved == [{"x": 1, "y": 0, "z": 0, "id": "custom:a"}]

    merged = merge_prefabs(
        [{"x": 0, "y": 0, "z": 0, "id": "custom:first"}],
        [{"x": 0, "y": 0, "z": 0, "id": "custom:second"}],
    )
    assert merged == [{"x": 0, "y": 0, "z": 0, "id": "custom:second"}]


def test_rotate_y_renormalises_grid_coordinates() -> None:
    blocks = [
        {"x": 0, "y": 0, "z": 0, "id": "custom:a"},
        {"x": 1, "y": 0, "z": 0, "id": "custom:b"},
        {"x": 1, "y": 0, "z": 1, "id": "custom:c"},
    ]
    rotated = rotate_y(blocks, 90)
    assert {tuple((b["x"], b["y"], b["z"])) for b in rotated} == {
        (0, 0, 1),
        (0, 0, 0),
        (1, 0, 0),
    }


def test_face_signature_and_opposite_face() -> None:
    blocks = [
        {"x": 0, "y": 0, "z": 0, "id": "custom:door"},
        {"x": 1, "y": 0, "z": 0, "id": "custom:wall"},
        {"x": 1, "y": 1, "z": 2, "id": "custom:window"},
    ]
    assert face_signature(blocks, "north") == frozenset(
        {(0, 0, "custom:door"), (1, 0, "custom:wall")}
    )
    assert face_signature(blocks, "south") == frozenset({(1, 1, "custom:window")})
    assert opposite_face("north") == "south"
    assert opposite_face("up") == "down"
