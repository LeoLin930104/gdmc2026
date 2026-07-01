"""Output boundary guards for generated block lists."""

from __future__ import annotations

from prefab_housing.types import SemanticBlockDict


def clip_blocks_to_site_footprint(
    blocks: list[SemanticBlockDict],
    *,
    site_footprint_xz: tuple[int, int] | None,
    origin_world: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    """Return only blocks inside the explicit x/z construction footprint.

    Planning cells may occupy less than the available site, but exported blocks
    must never exceed the upstream construction footprint. The y-axis is not
    clipped here because the current public request models height as a storey
    cap, not as an explicit voxel limit; roofs and foundations intentionally sit
    outside raw room-storey cells.
    """
    clipped, _removed = clip_blocks_to_site_footprint_with_removed(
        blocks,
        site_footprint_xz=site_footprint_xz,
        origin_world=origin_world,
    )
    return clipped


def clip_blocks_to_site_footprint_with_removed(
    blocks: list[SemanticBlockDict],
    *,
    site_footprint_xz: tuple[int, int] | None,
    origin_world: tuple[int, int, int],
) -> tuple[list[SemanticBlockDict], tuple[tuple[int, int, int], ...]]:
    """Clip blocks to the site footprint and report removed positions."""
    if site_footprint_xz is None:
        return blocks, ()
    sx, sz = site_footprint_xz
    if sx <= 0 or sz <= 0:
        raise ValueError("site_footprint_xz must be positive")

    ox, _, oz = origin_world
    x_min = ox
    x_max = ox + sx - 1
    z_min = oz
    z_max = oz + sz - 1
    clipped: list[SemanticBlockDict] = []
    removed: set[tuple[int, int, int]] = set()
    for block in blocks:
        position = (int(block["x"]), int(block["y"]), int(block["z"]))
        if x_min <= position[0] <= x_max and z_min <= position[2] <= z_max:
            clipped.append(block)
        else:
            removed.add(position)
    return clipped, tuple(sorted(removed))


__all__ = ["clip_blocks_to_site_footprint", "clip_blocks_to_site_footprint_with_removed"]
