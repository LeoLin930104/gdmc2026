import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter
import heapq
from coordinate_system import terrain_shape

def load_context():
    data = np.load('data/settlement_data.npz', allow_pickle=True)
    core_data = np.load('data/settlement_core.npz', allow_pickle=True)
    viz = np.load('data/settlement_viz.npz', allow_pickle=True)

    return {
        'seeds': data['seeds'],
        'heightmap': data['heightmap'].astype(float),
        'water_map': data['water_map'],
        'core_indices': set(core_data['core_indices'].tolist()),
        'palette': viz['palette'].tolist(),
        'origin': viz['origin'],
        'meta': viz['meta'],
        'num_drift': data['num_drift']
    }

def run_dijkstra(graph, start):
    """Computes shortest paths and tracks predecessors for path reconstruction."""
    distances = {start: 0}
    predecessors = {start: None}
    queue = [(0, start)]

    while queue:
        dist, current = heapq.heappop(queue)
        if dist > distances.get(current, float('inf')):
            continue
        for neighbor, weight in graph.get(current, {}).items():
            new_dist = dist + weight
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                predecessors[neighbor] = current
                heapq.heappush(queue, (new_dist, neighbor))
    return distances, predecessors

def check_connectivity(graph, start, end):
    """Lightweight check to see if a path still exists between two junctions."""
    queue = [(0, start)]
    distances = {start: 0}
    while queue:
        dist, current = heapq.heappop(queue)
        if current == end:
            return True
        if dist > distances.get(current, float('inf')):
            continue
        for neighbor, weight in graph.get(current, {}).items():
            new_dist = dist + weight
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return False


def relax_slope_constraints(heights, edges, tol=1e-3, max_iterations=300):
    """
    Enforce |h[i] - h[j]| <= bound for every (i, j, bound) edge via iterative
    pairwise relaxation (a difference-constraint projection, akin to
    Bellman-Ford consistency enforcement). Each violated edge pulls both
    endpoints halfway toward feasibility; repeating this over all edges
    converges monotonically for any feasible bound set (bounds >= 0 always
    admit at least the constant-height solution).
    """
    h = dict(heights)
    for _ in range(max_iterations):
        worst = 0.0
        for i, j, bound in edges:
            diff = h[i] - h[j]
            if diff > bound:
                excess = diff - bound
                h[i] -= excess / 2.0
                h[j] += excess / 2.0
                worst = max(worst, excess)
            elif diff < -bound:
                excess = -diff - bound
                h[i] += excess / 2.0
                h[j] -= excess / 2.0
                worst = max(worst, excess)
        if worst < tol:
            break
    return h


def level_paths_to_walkable(heightmap, free_mask, walkable_mask, max_step=1, max_iterations=400):
    """
    Adjust the free (path) columns of an integer heightmap so that every
    4-connected pair of walkable columns differs by at most `max_step` blocks.

    Columns in `walkable_mask & ~free_mask` (the leveled/graded core terrain)
    are held fixed as boundary conditions; only `free_mask` columns move. Each
    sweep clamps every free column into the interval
    [max(neighbour)-max_step, min(neighbour)+max_step] -- exactly the heights
    within max_step of all its walkable neighbours. When neighbours span more
    than 2*max_step the column snaps to their integer midpoint, which minimises
    the worst residual step at that column. This resolves essentially every
    path/plot seam; the only leftovers are the rare columns physically wedged
    between two plots that are close in the plane but far apart in height and
    not Voronoi-adjacent (so the cell-height relaxation never bounded them). On
    those it degrades gracefully to a single ~2-block step rather than
    oscillating. All arithmetic stays integer, so the result needs no rounding.
    """
    h = heightmap.astype(np.int64).copy()
    free = free_mask & walkable_mask
    if not np.any(free):
        return heightmap.astype(int)

    NEG = np.iinfo(np.int64).min // 4
    POS = np.iinfo(np.int64).max // 4

    for _ in range(max_iterations):
        # Neighbour extrema over walkable cells only (non-walkable read as
        # +/-inf so they never constrain).
        hi_src = np.where(walkable_mask, h, POS)   # for the min (upper bound)
        lo_src = np.where(walkable_mask, h, NEG)   # for the max (lower bound)

        nmin = np.full_like(h, POS)
        nmax = np.full_like(h, NEG)
        # North/South (z-1/z+1) and West/East (x-1/x+1) shifts.
        nmin[1:, :] = np.minimum(nmin[1:, :], hi_src[:-1, :]); nmax[1:, :] = np.maximum(nmax[1:, :], lo_src[:-1, :])
        nmin[:-1, :] = np.minimum(nmin[:-1, :], hi_src[1:, :]); nmax[:-1, :] = np.maximum(nmax[:-1, :], lo_src[1:, :])
        nmin[:, 1:] = np.minimum(nmin[:, 1:], hi_src[:, :-1]); nmax[:, 1:] = np.maximum(nmax[:, 1:], lo_src[:, :-1])
        nmin[:, :-1] = np.minimum(nmin[:, :-1], hi_src[:, 1:]); nmax[:, :-1] = np.maximum(nmax[:, :-1], lo_src[:, 1:])

        upper = nmin + max_step
        lower = nmax - max_step
        has_nbr = nmin < POS  # at least one walkable neighbour

        feasible = has_nbr & (lower <= upper)
        clamped = np.clip(h, lower, upper)
        midpoint = (nmin + nmax + 1) // 2  # integer midpoint, rounded up
        target = np.where(feasible, clamped, np.where(has_nbr, midpoint, h))

        new_h = np.where(free, target, h)
        if np.array_equal(new_h, h):
            break
        h = new_h

    return h.astype(int)


def _smooth_noise_field(shape, seed, sigma=6.0):
    """Deterministic smooth value-noise field normalised to roughly [-1, 1]."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random(shape), sigma=sigma)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-9:
        return np.zeros(shape, dtype=float)
    return (field - lo) / (hi - lo) * 2.0 - 1.0


def erode_boundary_band(
    heightmap,
    core_cell_mask,
    path_mask,
    water_map,
    *,
    talus_slope=0.7,
    slope_variation=0.5,
    roughness=2.0,
    noise_scale=20.0,
    seed=1234,
    max_reach=64,
):
    """
    Slump the sheer settlement<->exterior boundary into a natural eroded slope.

    Each exterior column is clamped into a talus cone measured from its nearest
    settlement anchor: its height may not exceed `anchor +/- talus * distance`.
    Where the natural land already fits inside the cone the height is kept (a
    seamless blend); where the land is a cliff it follows the cone, i.e. a talus
    rising/falling from the settlement edge. It is a direct clamp, not an
    iterative relaxation, so it never oscillates and degrades gracefully.

    "Full" erosion: there is no fixed band width -- the cone extends until it
    meets the natural grade (bounded only by `max_reach` for compute), so tall
    cliffs are blended all the way out instead of leaving a residual step.

    Naturalness:
    - The talus angle varies spatially by a LOW-frequency noise field
      (`slope_variation`, `noise_scale`) so slopes are steeper in some patches
      and gentler in others rather than one uniform cone.
    - A `roughness` height jitter (tapered to zero at the plot edge so it never
      re-introduces a wall there) breaks the clean concentric terraces.

    Returns (eroded_heightmap_int, band_mask).
    """
    hf = heightmap.astype(float)
    D, W = hf.shape
    settlement = core_cell_mask | path_mask
    water = np.asarray(water_map, dtype=bool)

    if not np.any(settlement):
        return heightmap.astype(int), np.zeros((D, W), dtype=bool)

    # Distance to the nearest settlement column, and that column's anchor height.
    dist, (iz, ix) = distance_transform_edt(~settlement, return_indices=True)
    anchor_h = hf[iz, ix]

    # Low-frequency spatial variation of the talus angle -> non-uniform slopes.
    angle_noise = _smooth_noise_field((D, W), seed, sigma=noise_scale)
    talus = np.clip(talus_slope * (1.0 + slope_variation * angle_noise), 0.3, 1.3)

    # Talus cone from the settlement edge; natural land inside the cone is kept.
    reach_h = talus * dist
    target = np.clip(hf, anchor_h - reach_h, anchor_h + reach_h)

    # The band is wherever the cone actually reshaped the land (cliffs), bounded
    # by max_reach so a gentle rise doesn't erode forever. No fixed width => the
    # slope blends fully out to the natural grade.
    band = (dist > 0) & (dist <= max_reach) & ~settlement & ~water & (np.abs(target - hf) >= 0.5)
    if not np.any(band):
        return heightmap.astype(int), band

    # Roughen the carved slope so it reads as eroded rock, not a clean cone.
    # Taper by distance so the plot edge stays clean (no re-introduced wall).
    rough_noise = _smooth_noise_field((D, W), seed + 1, sigma=max(4.0, noise_scale / 3.0))
    taper = np.clip(dist / 4.0, 0.0, 1.0)
    target = target + roughness * taper * rough_noise

    eroded = heightmap.astype(int).copy()
    eroded[band] = np.floor(target[band] + 0.5).astype(int)
    return eroded, band


def apply_terraforming(max_prune_score=1000, stair_run_blocks=2.0, plot_setback=2.0,
                       talus_slope=0.7, erosion_slope_variation=0.5, erosion_roughness=2.0,
                       erosion_noise_scale=20.0, erosion_max_reach=64, erosion_seed=1234):
    ctx = load_context()
    seeds, hmap, core_indices = ctx['seeds'], ctx['heightmap'], ctx['core_indices']
    palette, water_map = ctx['palette'], ctx['water_map']
    W, D = terrain_shape(hmap)
    vor = Voronoi(seeds)
    max_slope = 1.0 / float(stair_run_blocks)

    # 1. ESTABLISH INITIAL VERTEX HEIGHTS
    vertex_heights = {}
    for i, v_coord in enumerate(vor.vertices):
        vx, vz = int(np.clip(v_coord[0], 0, W-1)), int(np.clip(v_coord[1], 0, D-1))
        vertex_heights[i] = hmap[vz, vx]

    # 2. BUILD ROAD GRAPH FOR BETWEENNESS PRUNING
    # Moved ahead of cell-height leveling: this graph is purely topological
    # (Voronoi ridges + core_indices), so pruning can run before heights are
    # known.
    #
    # The graph keeps every ridge that touches at least one core cell -- this is
    # the original road topology, and it must stay that way so betweenness
    # pruning selects the same road network (internal + boundary) as before.
    # Boundary (core<->exterior) ridges are still rendered as roads in step 7;
    # they are just held flat at the core plot's height there instead of
    # conforming to the raw exterior terrain, so no road descends the outer
    # cliff. core_pair_ridges (both sides core) is collected separately for the
    # cell-height relaxation and border grading, which only apply internally.
    graph = {}
    all_edges = set()
    core_pair_ridges = []  # (p1, p2, edge_key, ridge_length) for cell-to-cell edges fully inside the core
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices: continue
        p1, p2 = ridge_points

        v1, v2 = ridge_vertices
        dist = np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])
        edge_key = tuple(sorted((v1, v2)))

        if p1 in core_indices and p2 in core_indices:
            core_pair_ridges.append((int(p1), int(p2), edge_key, float(dist)))

        # Keep any ridge with at least one core side in the betweenness graph.
        if p1 not in core_indices and p2 not in core_indices: continue

        if v1 not in graph: graph[v1] = {}
        if v2 not in graph: graph[v2] = {}

        graph[v1][v2] = dist
        graph[v2][v1] = dist
        all_edges.add(edge_key)

    # 3. RUN BETWEENNESS PRUNING SUITE
    cell_vertices = {}
    for p_idx in core_indices:
        region = [v for v in vor.regions[vor.point_region[p_idx]] if v != -1]
        if not region: continue
        best_v = min(region, key=lambda v: np.linalg.norm(vor.vertices[v] - seeds[p_idx]))
        cell_vertices[p_idx] = best_v

    edge_betweenness = {edge: 0 for edge in all_edges}
    active_cells = list(cell_vertices.keys())
    for i, p1 in enumerate(active_cells):
        s_vertex = cell_vertices[p1]
        distances, predecessors = run_dijkstra(graph, s_vertex)
        for p2 in active_cells[i+1:]:
            t_vertex = cell_vertices[p2]
            if t_vertex not in predecessors: continue
            curr = t_vertex
            while predecessors[curr] is not None:
                prev = predecessors[curr]
                edge = tuple(sorted((curr, prev)))
                edge_betweenness[edge] += 1
                curr = prev

    pruned_edges = set()
    for edge, score in list(edge_betweenness.items()):
        if score == 0:
            v1, v2 = edge
            if v2 in graph[v1]: del graph[v1][v2]
            if v1 in graph[v2]: del graph[v2][v1]
            pruned_edges.add(edge)

    remaining_edges = [(score, edge) for edge, score in edge_betweenness.items() if score > 0]
    remaining_edges.sort()
    for score, edge in remaining_edges:
        if score > max_prune_score: break
        v1, v2 = edge
        w = graph[v1][v2]
        del graph[v1][v2]
        del graph[v2][v1]
        if not check_connectivity(graph, v1, v2):
            graph[v1][v2] = w
            graph[v2][v1] = w
        else:
            pruned_edges.add(edge)

    # 4. PRE-CALCULATE ALL CELL HEIGHTS, THEN SLOPE-BOUND THEM (FIX A)
    cell_heights = {}
    for p_idx in range(len(seeds)):
        region_idx = vor.point_region[p_idx]
        region = [v for v in vor.regions[region_idx] if v != -1]
        if region:
            cell_heights[p_idx] = np.mean([vertex_heights[v] for v in region])
        else:
            cell_heights[p_idx] = hmap[int(seeds[p_idx][1]), int(seeds[p_idx][0])]

    # Bound every core-to-core cell-height gap before it gets baked into flat
    # plateaus. The transition between two adjacent plots is always absorbed
    # *across* their shared ridge, in the no-build setback gutter on each side
    # (the ring plotter.py excludes from buildable rectangles). That gutter is
    # `plot_setback` wide per side and absorbs `max_slope` of rise per block, so
    # it can hide up to 2*max_slope*plot_setback of gap -- the bound used for
    # every core-to-core edge here, whether or not it later carries a path.
    border_bound = 2.0 * max_slope * plot_setback
    cell_edges = [(p1, p2, border_bound) for p1, p2, _edge_key, _len in core_pair_ridges]
    cell_heights = relax_slope_constraints(cell_heights, cell_edges)

    # 5. LEVEL CELL INTERIORS (vectorized nearest-seed assignment via KD-tree;
    # matches the original per-cell "nearest seed among ALL seeds" rule, but
    # in one pass instead of a per-cell bounding-box scan, and reused below
    # for the border-grading pass).
    tree = cKDTree(seeds)
    grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(D))
    query_points = np.stack([grid_x.ravel(), grid_z.ravel()], axis=1).astype(float)
    neighbor_dist, neighbor_idx = tree.query(query_points, k=2)
    nearest_idx = neighbor_idx[:, 0].reshape(D, W)
    second_idx = neighbor_idx[:, 1].reshape(D, W)
    nearest_dist = neighbor_dist[:, 0].reshape(D, W)
    second_dist = neighbor_dist[:, 1].reshape(D, W)

    core_lookup = np.zeros(len(seeds), dtype=bool)
    core_lookup[np.array(sorted(core_indices), dtype=int)] = True
    core_cell_mask = core_lookup[nearest_idx]

    height_lookup = np.zeros(len(seeds), dtype=float)
    for idx, height in cell_heights.items():
        height_lookup[idx] = height
    hmap = np.where(core_cell_mask, height_lookup[nearest_idx], hmap)

    # 6. GRADE BARE PLOT-TO-PLOT BORDERS (FIX C)
    # Feather the flat plateaus into each other near any core-to-core ridge
    # that has no formal path, instead of leaving a raw step. Uses the
    # standard Voronoi bisector-distance identity: for a point whose two
    # nearest seeds are i (nearest) and j (second-nearest), its signed
    # distance to the i/j bisector is (dist_j - dist_i) / 2. The grading
    # margin is capped at plot_setback on each side, matching Fix A's tight
    # non-path bound above and the setback gutter plotter.py already
    # excludes from buildable rectangles, so this pass never touches the
    # flat buildable interior of any plot.
    second_is_core = core_lookup[second_idx]
    border_eligible = core_cell_mask & second_is_core & (nearest_idx != second_idx)

    height_i = height_lookup[nearest_idx]
    height_j = height_lookup[second_idx]
    gap = np.abs(height_i - height_j)
    bisector_dist = (second_dist - nearest_dist) / 2.0
    margin = np.minimum(plot_setback, gap / (2.0 * max_slope))
    grade_zone = border_eligible & (gap > 1e-9) & (bisector_dist <= margin)

    grade_rows, grade_cols = np.where(grade_zone)
    if grade_rows.size:
        d_g = bisector_dist[grade_rows, grade_cols]
        margin_g = margin[grade_rows, grade_cols]
        hi_g = height_i[grade_rows, grade_cols]
        hj_g = height_j[grade_rows, grade_cols]
        weight = np.clip((d_g + margin_g) / (2.0 * margin_g), 0.0, 1.0)
        hmap[grade_rows, grade_cols] = hj_g + weight * (hi_g - hj_g)

    # 7. RENDER PATHS (FIX B)
    # Every ridge that touches at least one core plot is laid as a road -- the
    # full internal network plus the settlement's boundary ridges. Path height,
    # though, is sourced differently for the two cases:
    #
    # * Interior ridge (both sides core): adopt the already-graded terrain
    #   height at each column. Step 4's relaxation bounds adjacent plots to
    #   within 2*max_slope*plot_setback and step 6 feathers the gutters, so the
    #   terrain field is a continuous, slope-bounded walkable surface; a path
    #   that conforms to it inherits that walkability, and this fixes the
    #   plot<->path cliffs the old junction-averaged heights produced.
    #
    # * Boundary ridge (one side exterior): the road stays FLAT at the core
    #   plot's own height and does not follow the raw exterior terrain. Making a
    #   border path chase the outside land is what produced the tall staircases
    #   down the settlement's edge; a flat rim ledge at plot height is the
    #   settlement's natural boundary treatment (the leveled plots already form
    #   the same retaining edge against exterior land).
    path_mask = np.zeros((D, W), dtype=bool)
    path_surface = np.full((D, W), np.nan, dtype=float)
    path_palette = [
        "minecraft:cobblestone",
        "minecraft:stone_bricks",
        "minecraft:polished_andesite",
        "minecraft:chiseled_stone_bricks",
        "minecraft:mossy_cobblestone",
    ]
    for block_name in path_palette:
        if block_name not in palette:
            palette.append(block_name)
    PATH_IDS = [palette.index(block_name) for block_name in path_palette]

    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices: continue
        v1_idx, v2_idx = ridge_vertices
        edge_key = tuple(sorted((v1_idx, v2_idx)))
        if edge_key in pruned_edges: continue

        p1, p2 = ridge_points
        p1_core = p1 in core_indices
        p2_core = p2 in core_indices
        if not p1_core and not p2_core: continue  # pure-exterior ridge: no road

        # Boundary ridges hold the core plot's flat height; interior ridges
        # conform to the graded terrain per column.
        is_boundary = not (p1_core and p2_core)
        boundary_height = cell_heights[p1 if p1_core else p2] if is_boundary else None

        v1, v2 = vor.vertices[v1_idx], vor.vertices[v2_idx]
        diff = v2 - v1
        length = np.linalg.norm(diff)
        if length < 0.1: continue

        dir_norm = diff / length
        perp = np.array([-dir_norm[1], dir_norm[0]])

        for t_dist in np.linspace(0, length, int(length * 2)):
            center_p = v1 + (t_dist / length) * diff

            for offset in [-1, 0, 1]:
                point = center_p + perp * offset
                px, pz = int(np.clip(point[0], 0, W-1)), int(np.clip(point[1], 0, D-1))
                if is_boundary:
                    path_surface[pz, px] = boundary_height
                else:
                    path_surface[pz, px] = hmap[pz, px]
                path_mask[pz, px] = True

    # 8. LEVEL PATHS FOR GUARANTEED WALKABILITY, THEN REBUILD VOXELS
    # Seed the path columns from the graded terrain they cover, then relax only
    # those columns against the fixed core terrain so every walkable 4-neighbour
    # step is <= 1. Conforming alone is not enough because some path columns
    # (perp offsets, clipped ridge ends) can fall on ungraded raw terrain
    # outside the core; the relaxation pulls those back onto walkable grades
    # without disturbing the flat buildable plots.
    hmap[path_mask] = np.rint(np.nan_to_num(path_surface, nan=0.0))[path_mask]
    final_hmap = np.floor(hmap + 0.5).astype(int)

    # The final integer step budget is always 1 block: a player can step up one
    # full block (and a stair, below, makes even that seamless). stair_run_blocks
    # only tunes the gentler pre-quantization cell grade (max_slope) upstream.
    #
    # Free the paths plus the setback gutter immediately around every core-core
    # ridge (bisector_dist <= plot_setback, from Fix C's KD-tree). A path
    # pinched between two flat plots that differ by up to 2 blocks cannot stay
    # within one block of both pinned pads at once; letting the surrounding
    # gutter feather resolves it. Capping the freed ring at exactly plot_setback
    # -- the same boundary plotter.py uses -- keeps every flat buildable pad
    # (which lies beyond the setback) untouched.
    gutter_mask = core_cell_mask & second_is_core & (bisector_dist <= plot_setback)
    free_mask = path_mask | gutter_mask
    walkable_mask = core_cell_mask | path_mask
    final_hmap = level_paths_to_walkable(final_hmap, free_mask, walkable_mask, max_step=1)
    path_base_y = np.where(path_mask, final_hmap, 0).astype(int)

    # FIX D: detect single-block path risers and orient a stair up each one.
    # A path column that has a 4-connected path neighbour exactly one block
    # higher gets a bottom stair on top, facing that higher neighbour (the
    # ascending direction), matching prefab_housing.stairwell._step_facing /
    # grid.FACE_DELTA (NORTH=0 -> (0,0,-1), EAST=1 -> (+1,0,0), SOUTH=2 ->
    # (0,0,+1), WEST=3 -> (-1,0,0)). Placed on the lower column, the stair
    # lets a player walk up the step instead of jumping it.
    path_stair_facing = np.full((D, W), -1, dtype=np.int8)
    face_dirs = ((0, -1, 0), (1, 0, 1), (0, 1, 2), (-1, 0, 3))  # (dx, dz, facing_code)
    path_rows, path_cols = np.where(path_mask)
    for z, x in zip(path_rows.tolist(), path_cols.tolist()):
        y = int(final_hmap[z, x])
        for dx, dz, facing_code in face_dirs:
            nx, nz = x + dx, z + dz
            if 0 <= nx < W and 0 <= nz < D and path_mask[nz, nx]:
                if int(final_hmap[nz, nx]) == y + 1:
                    path_stair_facing[z, x] = facing_code
                    break
    # Kept for on-disk schema compatibility with existing consumers; stairs now
    # supersede half-block slabs for path risers.
    path_slab_mask = np.zeros((D, W), dtype=bool)

    # 8b. ERODE THE SETTLEMENT<->EXTERIOR BOUNDARY INTO NATURAL TALUS SLOPES
    # Interior plots/paths are already finalised above and stay fixed; this only
    # reshapes a noise-jittered band of exterior land at the settlement edge so
    # the sheer cut/retaining walls slump into irregular walkable-looking slopes
    # that blend into the surrounding terrain. builder.py deploys this band
    # (clearing the old hill above cuts, filling talus below) with weathered
    # rock/gravel faces and revegetation.
    final_hmap, erosion_mask = erode_boundary_band(
        final_hmap,
        core_cell_mask,
        path_mask,
        water_map,
        talus_slope=talus_slope,
        slope_variation=erosion_slope_variation,
        roughness=erosion_roughness,
        noise_scale=erosion_noise_scale,
        max_reach=erosion_max_reach,
        seed=erosion_seed,
    )

    H = int(np.max(final_hmap)) + 15
    new_blocks = np.zeros((W, H, D), dtype=np.uint16)
    GRASS = palette.index("minecraft:grass_block") if "minecraft:grass_block" in palette else 1
    DIRT = palette.index("minecraft:dirt") if "minecraft:dirt" in palette else 2
    WATER = palette.index("minecraft:water") if "minecraft:water" in palette else 3

    for x in range(W):
        for z in range(D):
            new_y = int(final_hmap[z, x])
            min_neighbor_y = new_y
            for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, nz = x + dx, z + dz
                if 0 <= nx < W and 0 <= nz < D:
                    min_neighbor_y = min(min_neighbor_y, int(final_hmap[nz, nx]))

            fill_bottom = max(0, min_neighbor_y - 3)
            new_blocks[x, :new_y, z] = DIRT
            if fill_bottom < new_y:
                new_blocks[x, fill_bottom:new_y, z] = DIRT

            new_blocks[x, new_y, z] = GRASS
            if path_mask[z, x]:
                new_blocks[x, new_y, z] = PATH_IDS[(x * 31 + z * 17) % len(PATH_IDS)]
            if water_map[z, x] and not path_mask[z, x] and new_y < 10:
                new_blocks[x, new_y+1:12, z] = WATER

    # 9. SAVE BACK TO THE CORE
    np.savez('data/settlement_viz.npz', blocks=new_blocks, palette=np.array(palette), origin=ctx['origin'], meta=ctx['meta'])
    previous_data = np.load('data/settlement_data.npz', allow_pickle=True)
    preserved = {
        key: previous_data[key]
        for key in previous_data.files
        if key.startswith("zone_") or key == "zone_map"
    }
    np.savez('data/settlement_data.npz', seeds=seeds, heightmap=final_hmap, water_map=water_map,
             chasm_mask=previous_data['chasm_mask'], num_drift=ctx['num_drift'],
             origin=ctx['origin'], core_cell_mask=core_cell_mask, path_mask=path_mask,
             path_base_y=path_base_y, path_slab_mask=path_slab_mask,
             path_stair_facing=path_stair_facing, erosion_mask=erosion_mask, **preserved)
    print(f"Terraforming complete: plot borders slope-bounded and graded, path grades hard-capped, "
          f"{int(np.count_nonzero(erosion_mask))} boundary columns eroded.")

if __name__ == "__main__":
    apply_terraforming(max_prune_score=2)
