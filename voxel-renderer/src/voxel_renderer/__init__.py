"""Neutral voxel renderer foundation package."""

from voxel_renderer.api import *  # noqa: F401,F403
from voxel_renderer.assets import get_asset_root
from voxel_renderer.orientation import (
    KNOWN_ROTATABLE_PROPERTIES,
    rotate_block,
    rotate_block_properties,
    rotate_y_property,
)
from voxel_renderer.palette import DEFAULT_MINECRAFT_PALETTE, build_minecraft_palette, is_minecraft_placeable
from voxel_renderer.prefab import (
    Bounds,
    face_signature,
    get_bounds,
    merge_prefabs,
    normalise_to_origin,
    opposite_face,
    rotate_y,
    translate_blocks,
)
from voxel_renderer.state import VoxelStore, canonicalise_block_array
from voxel_renderer.types import BlockEntry

__all__ = [
    "BlockEntry",
    "Bounds",
    "DEFAULT_MINECRAFT_PALETTE",
    "KNOWN_ROTATABLE_PROPERTIES",
    "VoxelStore",
    "build_minecraft_palette",
    "canonicalise_block_array",
    "face_signature",
    "get_bounds",
    "get_asset_root",
    "is_minecraft_placeable",
    "merge_prefabs",
    "normalise_to_origin",
    "opposite_face",
    "rotate_block",
    "rotate_block_properties",
    "rotate_y",
    "rotate_y_property",
    "translate_blocks",
]
