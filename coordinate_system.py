"""
Shared coordinate contract for the settlement pipeline.

Local simulation coordinates:
    x: east/west offset from the captured Minecraft slice origin
    y: Minecraft height
    z: north/south offset from the captured Minecraft slice origin

Array storage:
    2D terrain/mask arrays use [z, x]
    3D voxel arrays use [x, y, z]

World placement:
    world_x = origin_x + x
    world_y = y
    world_z = origin_z + z
"""

import numpy as np


def terrain_shape(heightmap: np.ndarray) -> tuple[int, int]:
    """Return (width_x, depth_z) for a terrain array stored as [z, x]."""
    depth_z, width_x = heightmap.shape
    return width_x, depth_z


def local_to_world(origin, x: int, y: int, z: int) -> tuple[int, int, int]:
    """Convert local settlement coordinates to Minecraft world coordinates."""
    ox, _, oz = origin
    return int(ox + x), int(y), int(oz + z)


def require_matching_terrain_and_blocks(heightmap: np.ndarray, blocks: np.ndarray) -> None:
    """Validate that heightmap[z, x] and blocks[x, y, z] describe the same footprint."""
    width_x, depth_z = terrain_shape(heightmap)
    block_width_x, _, block_depth_z = blocks.shape
    if (width_x, depth_z) != (block_width_x, block_depth_z):
        raise ValueError(
            "Coordinate mismatch: heightmap uses [z, x] with footprint "
            f"{width_x}x{depth_z}, but blocks use [x, y, z] with footprint "
            f"{block_width_x}x{block_depth_z}."
        )
