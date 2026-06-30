from __future__ import annotations

import base64
import importlib
import io

from voxel_renderer.api import CAMERA_VIEWS, compose_comparison_strip, compose_gallery_grid, render_orthographic_views
from voxel_renderer.blockstate_resolver import resolve_block_models
from voxel_renderer.block_registry import create_coloured_block_mesh


def _image_module():
    return importlib.import_module("PIL.Image")


def test_render_returns_expected_views() -> None:
    views = render_orthographic_views(
        [{"x": 0, "y": 0, "z": 0, "id": "minecraft:stone"}],
        width=64,
        height=64,
    )
    assert set(views) == {name for name, _, _ in CAMERA_VIEWS}


def test_empty_render_returns_valid_pngs() -> None:
    image_module = _image_module()
    views = render_orthographic_views([], width=32, height=32)
    for encoded in views.values():
        img = image_module.open(io.BytesIO(base64.b64decode(encoded)))
        assert img.format == "PNG"
        assert img.size == (32, 32)


def test_composite_helpers_smoke() -> None:
    image_module = _image_module()
    views = render_orthographic_views([], width=32, height=32)

    strip = compose_comparison_strip(
        [("A", views["top"]), ("B", views["profile"])]
    )
    strip_img = image_module.open(io.BytesIO(strip))
    assert strip_img.size == (64, 32)

    gallery = compose_gallery_grid([
        ("ok", views["iso_right"]),
        ("missing", None),
    ], columns=2)
    gallery_img = image_module.open(io.BytesIO(gallery))
    assert gallery_img.size == (64, 32)


def test_stairs_without_shape_default_to_straight_variant() -> None:
    models = resolve_block_models(
        "minecraft:stone_brick_stairs",
        {"facing": "east", "half": "bottom"},
    )
    assert models is not None
    assert models[0].model == "minecraft:block/stone_brick_stairs"


def test_bed_blocks_use_low_special_geometry_not_cube_fallback() -> None:
    head = create_coloured_block_mesh(
        "minecraft:red_bed",
        {"part": "head", "facing": "north"},
    )
    foot = create_coloured_block_mesh(
        "minecraft:red_bed",
        {"part": "foot", "facing": "north"},
    )

    assert head.extents[1] < 0.7
    assert foot.extents[1] < 0.7
    assert head.extents[0] < 1.0
    assert head.extents[2] < 1.0
    assert len(head.faces) > len(foot.faces)


def test_two_part_bed_render_smoke() -> None:
    image_module = _image_module()
    views = render_orthographic_views(
        [
            {
                "x": 0,
                "y": 0,
                "z": 0,
                "id": "minecraft:red_bed",
                "properties": {"part": "head", "facing": "north"},
            },
            {
                "x": 0,
                "y": 0,
                "z": 1,
                "id": "minecraft:red_bed",
                "properties": {"part": "foot", "facing": "north"},
            },
        ],
        width=64,
        height=64,
    )
    assert set(views) == {name for name, _, _ in CAMERA_VIEWS}
    for encoded in views.values():
        img = image_module.open(io.BytesIO(base64.b64decode(encoded)))
        assert img.format == "PNG"
