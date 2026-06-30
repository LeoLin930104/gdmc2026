"""Stepped pyramidal roof generator (post-pipeline).

Architecture role
-----------------
Late swappable exterior detail stage invoked after structural shell, connection
carving, wall-face textures, foundation, and trim. Reads the spatial layout's
per-cell AABBs and emits a stepped roof above every contiguous region of cells
that share the same topmost-occupied storey. Decoupled from the WFC tile set so
roofs respond to whatever cell geometry the layout produced (banded, uniform,
future brick-bonded).

Algorithm
---------
1. **Column heights.** For each ``(ix, iz)`` column compute the topmost
   storey ``top_iy[ix, iz]`` whose cell is non-EMPTY, or ``-1`` if the
   column has no occupied cells.
2. **Region partition.** 4-connect cells on the ``(ix, iz)`` plane that
   share the same ``top_iy`` value (and have ``top_iy >= 0``). Each
   resulting region becomes one roof unit.
3. **Voxel mask.** For the region build a 2D ``bool`` mask over the
   bounding ``(x, z)`` voxel range — True wherever a region cell's AABB
   covers that voxel. Non-rectangular regions (L, T, …) are supported
   directly by the mask.
4. **Stepped erosion.** Course 0 paints ``roof_block`` on every masked
   voxel at ``y = base_y``; perimeter voxels (mask voxels with at least
   one orthogonal neighbour outside the mask) become ``roof_stair``.
   Erode the mask by one voxel on every side (a voxel survives iff all
   four orthogonal neighbours are also inside) and repeat at
   ``y = base_y + 1``. Continue until the mask is empty — the ridge.

The erosion-based ridge handles even/odd footprints uniformly: an even
side leaves a 2-voxel-wide ridge; an odd side narrows to 1.

Vulnerability — staircase aliasing on diagonals
-----------------------------------------------
Erosion is axis-aligned, so a diagonal hip on an odd-shape region steps
down in 1-voxel jumps. Acceptable in voxel idiom; smoothing would require
mid-voxel geometry the block format cannot represent.

Vulnerability — disconnected regions
------------------------------------
A region disconnected by an EMPTY notch produces independent stepped
pyramids on either side, with a clean valley between. Intended.
"""

from __future__ import annotations

import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.layout import SpatialLayout
from prefab_housing.palette import SLOT_ROOF_BLOCK, SLOT_ROOF_STAIR
from prefab_housing.types import SemanticBlockDict
from prefab_housing.wfc.solver import SolverState


def _block(x: int, y: int, z: int, block_id: str) -> SemanticBlockDict:
    return {"x": x, "y": y, "z": z, "id": block_id}


def _stair_block(
    x: int, y: int, z: int, block_id: str, facing: str, half: str = "bottom"
) -> SemanticBlockDict:
    """Stair voxel with explicit Minecraft orientation properties.

    ``facing`` is the cardinal name of the stair's full-height side. For roof
    eaves we want the full-height side on the *interior* of the roof so the
    step descends outward — i.e. ``facing`` points toward the ridge, opposite
    the outward perimeter direction.
    """
    return {
        "x": x, "y": y, "z": z, "id": block_id,
        "properties": {"facing": facing, "half": half, "shape": "straight"},
    }


def _outward_facing(
    x: int, z: int, mask: np.ndarray
) -> str | None:
    """Return the cardinal name of the *outward* edge for a perimeter voxel,
    or ``None`` if the voxel is fully interior. Coordinate convention:
    ``-z = north``, ``+z = south``, ``-x = west``, ``+x = east``. When a
    voxel sits at a convex corner (two outward edges) we pick the z-axis
    edge first — this gives roof corners a consistent miter direction in v1.
    """
    W, D = mask.shape
    n_out = z == 0 or not mask[x, z - 1]      # neighbour at z-1 is outside ⇒ north edge
    s_out = z == D - 1 or not mask[x, z + 1]  # neighbour at z+1 outside ⇒ south edge
    w_out = x == 0 or not mask[x - 1, z]
    e_out = x == W - 1 or not mask[x + 1, z]
    if not (n_out or s_out or w_out or e_out):
        return None
    # Priority order: prefer the z-axis perimeter when a voxel is on a corner
    # — keeps gable ends pointed along z by convention.
    if n_out:
        return "north"
    if s_out:
        return "south"
    if w_out:
        return "west"
    return "east"


def _is_occupied(state: SolverState, ix: int, iy: int, iz: int) -> bool:
    """True iff cell holds a non-EMPTY tile assignment."""
    flat = state.grid.flat_index(ix, iy, iz)
    tid = int(state.assignment[flat])
    if tid < 0:
        return False
    return not pt.is_void_pod_index(int(state.tiles.pod_index[tid]))


def _column_top_storey(state: SolverState) -> np.ndarray:
    """Return ``int8[cx, cz]`` with the highest occupied iy per column, or -1."""
    grid = state.grid
    out = np.full((grid.cx, grid.cz), -1, dtype=np.int8)
    for iz in range(grid.cz):
        for ix in range(grid.cx):
            for iy in range(grid.cy - 1, -1, -1):
                if _is_occupied(state, ix, iy, iz):
                    out[ix, iz] = iy
                    break
    return out


def _partition_regions(top_iy: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connect cells with equal top_iy values (and ``top_iy >= 0``).

    Returns a list of regions; each region is a list of ``(ix, iz)`` tuples.
    """
    cx, cz = top_iy.shape
    visited = np.zeros_like(top_iy, dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for sx in range(cx):
        for sz in range(cz):
            if visited[sx, sz] or top_iy[sx, sz] < 0:
                continue
            target = int(top_iy[sx, sz])
            stack = [(sx, sz)]
            region: list[tuple[int, int]] = []
            while stack:
                ix, iz = stack.pop()
                if visited[ix, iz]:
                    continue
                if int(top_iy[ix, iz]) != target:
                    continue
                visited[ix, iz] = True
                region.append((ix, iz))
                if ix + 1 < cx:
                    stack.append((ix + 1, iz))
                if ix - 1 >= 0:
                    stack.append((ix - 1, iz))
                if iz + 1 < cz:
                    stack.append((ix, iz + 1))
                if iz - 1 >= 0:
                    stack.append((ix, iz - 1))
            regions.append(region)
    return regions


def _largest_rect_in_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Return ``(x0, x1, z0, z1)`` of the largest axis-aligned rectangle of
    ``True`` cells in ``mask``, or ``None`` when ``mask`` has no ``True``.

    O(W^2 * D) brute force. Bounded by cell-grid extents (typically <= 8x8)
    so the upper bound is well under the per-house budget.
    """
    if not mask.any():
        return None
    W, D = mask.shape
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for x0 in range(W):
        for x1 in range(x0, W):
            # Pre-check the column strip is fully True for x in [x0, x1].
            strip_ok = True
            for z0 in range(D):
                column_ok = bool(mask[x0 : x1 + 1, z0].all())
                if not column_ok:
                    strip_ok = False
                    continue
                # Extend z1 as far as possible at this z0.
                z1 = z0
                while z1 + 1 < D and bool(mask[x0 : x1 + 1, z1 + 1].all()):
                    z1 += 1
                area = (x1 - x0 + 1) * (z1 - z0 + 1)
                if area > best_area:
                    best_area = area
                    best = (x0, x1, z0, z1)
            del strip_ok  # silence linter; loop kept self-contained
    return best


def _decompose_region_to_rectangles(
    region: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Greedy maximal-rectangle cover of a same-``top_iy`` cell region.

    Each output rectangle is returned as a flat list of ``(ix, iz)``
    cell-index tuples (rather than corner coords) so the caller can reuse
    ``_emit_region_roof`` unchanged.
    """
    if not region:
        return []
    xs = [ix for ix, _ in region]
    zs = [iz for _, iz in region]
    x_lo, x_hi = min(xs), max(xs)
    z_lo, z_hi = min(zs), max(zs)
    W = x_hi - x_lo + 1
    D = z_hi - z_lo + 1
    mask = np.zeros((W, D), dtype=bool)
    for ix, iz in region:
        mask[ix - x_lo, iz - z_lo] = True

    rectangles: list[list[tuple[int, int]]] = []
    while mask.any():
        rect = _largest_rect_in_mask(mask)
        if rect is None:
            break
        rx0, rx1, rz0, rz1 = rect
        cells = [
            (x_lo + xi, z_lo + zi)
            for xi in range(rx0, rx1 + 1)
            for zi in range(rz0, rz1 + 1)
        ]
        rectangles.append(cells)
        mask[rx0 : rx1 + 1, rz0 : rz1 + 1] = False
    return rectangles


def _emit_region_roof(
    region_cells: list[tuple[int, int, int]],
    layout: SpatialLayout,
    roof_block_id: str,
    roof_stair_id: str,
    *,
    max_courses: int | None = None,
) -> list[SemanticBlockDict]:
    """Mask-erosion stepped pyramid over the union of region cell AABBs.

    ``max_courses`` clamps the pyramid height so large rectangles get a
    flat hip plateau instead of a single tall apex. ``None`` ⇒ unbounded
    (preserves original behaviour for callers that haven't migrated).
    """
    # Compute voxel-bbox encompassing every cell in the region.
    x_lo = min(layout.bbox(ix, iy, iz)[0][0] for ix, iy, iz in region_cells)
    x_hi = max(layout.bbox(ix, iy, iz)[1][0] for ix, iy, iz in region_cells)
    z_lo = min(layout.bbox(ix, iy, iz)[0][2] for ix, iy, iz in region_cells)
    z_hi = max(layout.bbox(ix, iy, iz)[1][2] for ix, iy, iz in region_cells)
    base_y = max(layout.bbox(ix, iy, iz)[1][1] for ix, iy, iz in region_cells) + 1

    W = x_hi - x_lo + 1
    D = z_hi - z_lo + 1
    mask = np.zeros((W, D), dtype=bool)
    for ix, iy, iz in region_cells:
        (cx0, _, cz0), (cx1, _, cz1) = layout.bbox(ix, iy, iz)
        mask[cx0 - x_lo : cx1 - x_lo + 1, cz0 - z_lo : cz1 - z_lo + 1] = True

    out: list[SemanticBlockDict] = []
    course = 0
    while mask.any():
        y = base_y + course
        # Interior: voxels whose 4 orthogonal neighbours are all in mask.
        # Pad with False so border voxels are always "perimeter".
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        # When a course-cap is reached, every still-present voxel becomes a
        # full block (flat plateau). No further erosion — terminate loop.
        plateau = max_courses is not None and course + 1 >= max_courses
        # Inward-facing map: full-height stair side points toward the ridge so
        # the step descends outward. Convention: facing="north" places step on
        # +z (south), full-height block on -z (north) — see voxel_renderer
        # block_registry.mesh_stair.
        _inward = {"north": "south", "south": "north", "west": "east", "east": "west"}
        xs, zs = np.nonzero(mask)
        for xi, zi in zip(xs.tolist(), zs.tolist(), strict=False):
            if plateau or interior[xi, zi]:
                out.append(_block(x_lo + xi, y, z_lo + zi, roof_block_id))
            else:
                outward = _outward_facing(xi, zi, mask)
                # Defensive: any non-interior voxel must have an outward edge.
                facing = _inward[outward] if outward is not None else "north"
                out.append(
                    _stair_block(
                        x_lo + xi, y, z_lo + zi, roof_stair_id, facing, "bottom"
                    )
                )
        if plateau:
            break
        mask = interior
        course += 1
    return out


def _emit_parapet_cap(
    region_cells: list[tuple[int, int, int]],
    layout: SpatialLayout,
    roof_block_id: str,
) -> list[SemanticBlockDict]:
    """Single-course flat slab over the union of region cell AABBs.

    Used for ``wing`` regions whose top_iy is below the building maximum
    — visually reads as terrace / outdoor roof-deck rather than competing
    with the main pyramidal mass.
    """
    x_lo = min(layout.bbox(ix, iy, iz)[0][0] for ix, iy, iz in region_cells)
    x_hi = max(layout.bbox(ix, iy, iz)[1][0] for ix, iy, iz in region_cells)
    z_lo = min(layout.bbox(ix, iy, iz)[0][2] for ix, iy, iz in region_cells)
    z_hi = max(layout.bbox(ix, iy, iz)[1][2] for ix, iy, iz in region_cells)
    base_y = max(layout.bbox(ix, iy, iz)[1][1] for ix, iy, iz in region_cells) + 1

    W = x_hi - x_lo + 1
    D = z_hi - z_lo + 1
    mask = np.zeros((W, D), dtype=bool)
    for ix, iy, iz in region_cells:
        (cx0, _, cz0), (cx1, _, cz1) = layout.bbox(ix, iy, iz)
        mask[cx0 - x_lo : cx1 - x_lo + 1, cz0 - z_lo : cz1 - z_lo + 1] = True

    out: list[SemanticBlockDict] = []
    xs, zs = np.nonzero(mask)
    for xi, zi in zip(xs.tolist(), zs.tolist(), strict=False):
        out.append(_block(x_lo + xi, base_y, z_lo + zi, roof_block_id))
    return out


def generate_roof(
    state: SolverState,
    layout: SpatialLayout,
    palette: dict[str, str],
) -> list[SemanticBlockDict]:
    """Emit a tier-classified roof per contiguous same-height column region.

    Tier dispatch
    -------------
    The maximum ``top_iy`` across every region is the building's *crown
    tier*. Regions at the crown tier emit a stepped pyramid (course-clamped
    so wide rectangles flatten into a hip plateau rather than a single
    apex). Regions below the crown tier — typically ground-floor wings
    beneath a perimeter EMPTY top cell — emit a flat 1-course parapet
    cap so they read as terraces rather than competing pyramids.

    Adaptive max_courses
    --------------------
    For a rectangle of voxel dimensions ``(W, D)`` the natural pyramid
    fully resolves in ``ceil(min(W, D) / 2)`` courses. Clamping at
    ``max(2, min(W, D) // 3)`` halves the apex height on large rectangles
    while leaving small rectangles essentially untouched.

    Single-storey buildings: every region is the crown — parapet branch
    is unused (correct degenerate behaviour).

    Pure function. Returns an empty list if either roof palette slot is
    absent (deployment may opt out of roofs entirely).
    """
    if layout.grid is not state.grid:
        raise ValueError("layout.grid must be the same instance as state.grid")
    roof_block_id = palette.get(SLOT_ROOF_BLOCK)
    roof_stair_id = palette.get(SLOT_ROOF_STAIR)
    if roof_block_id is None or roof_stair_id is None:
        return []

    top_iy = _column_top_storey(state)
    regions = _partition_regions(top_iy)
    if not regions:
        return []

    # Crown tier = highest top_iy across all regions.
    crown_tier = max(int(top_iy[r[0][0], r[0][1]]) for r in regions)

    raw: list[SemanticBlockDict] = []
    for region in regions:
        iy = int(top_iy[region[0][0], region[0][1]])
        is_crown = iy == crown_tier
        for rect_cells in _decompose_region_to_rectangles(region):
            cells = [(ix, iy, iz) for (ix, iz) in rect_cells]
            if is_crown:
                # Compute rectangle voxel dimensions for adaptive clamp.
                x_lo = min(layout.bbox(cx, cy, cz)[0][0] for cx, cy, cz in cells)
                x_hi = max(layout.bbox(cx, cy, cz)[1][0] for cx, cy, cz in cells)
                z_lo = min(layout.bbox(cx, cy, cz)[0][2] for cx, cy, cz in cells)
                z_hi = max(layout.bbox(cx, cy, cz)[1][2] for cx, cy, cz in cells)
                W = x_hi - x_lo + 1
                D = z_hi - z_lo + 1
                max_courses = max(2, min(W, D) // 3)
                raw.extend(
                    _emit_region_roof(
                        cells, layout, roof_block_id, roof_stair_id,
                        max_courses=max_courses,
                    )
                )
            else:
                raw.extend(_emit_parapet_cap(cells, layout, roof_block_id))

    # Dedupe by (x, y, z) — later writes win. Adjacent sub-roofs may share
    # boundary voxels at their shared edge; keeping the *last* emission
    # ensures a deterministic, single-block-per-voxel output.
    by_pos: dict[tuple[int, int, int], SemanticBlockDict] = {}
    for blk in raw:
        by_pos[(int(blk["x"]), int(blk["y"]), int(blk["z"]))] = blk
    return list(by_pos.values())


__all__ = ["generate_roof"]
