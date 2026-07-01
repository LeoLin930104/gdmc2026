import numpy as np
from scipy.spatial import Voronoi
import json
from coordinate_system import terrain_shape


def _validate_core_indices(core_indices, seeds):
    if len(core_indices) == 0:
        raise ValueError("settlement_core.npz contains no core indices.")

    max_index = max(core_indices)
    if max_index >= len(seeds):
        raise ValueError(
            "settlement_core.npz is stale: it references seed index "
            f"{max_index}, but settlement_data.npz only has {len(seeds)} seeds. "
            "Rerun isolate_buildable_plot() for the current settlement_data.npz."
        )


def _save_npz_with_updates(path, loaded, **updates):
    payload = {key: loaded[key] for key in loaded.files}
    payload.update(updates)
    np.savez(path, **payload)


def _core_adjacency(vor, core_indices):
    graph = {idx: set() for idx in core_indices}
    for p1, p2 in vor.ridge_points:
        if p1 in core_indices and p2 in core_indices:
            graph[p1].add(p2)
            graph[p2].add(p1)
    return graph


def _choose_zone_seeds(seeds, core_indices, num_zones):
    ordered = list(core_indices)
    if len(ordered) <= num_zones:
        return ordered

    center = np.mean(seeds[ordered], axis=0)
    first = max(ordered, key=lambda idx: np.linalg.norm(seeds[idx] - center))
    chosen = [first]

    while len(chosen) < num_zones:
        remaining = [idx for idx in ordered if idx not in chosen]
        next_idx = max(
            remaining,
            key=lambda idx: min(np.linalg.norm(seeds[idx] - seeds[c]) for c in chosen),
        )
        chosen.append(next_idx)

    return chosen


def generate_zones(num_zones=4):
    data = np.load('data/settlement_data.npz', allow_pickle=True)
    seeds = data['seeds']
    heightmap = data['heightmap']
    W, D = terrain_shape(heightmap)

    core_data = np.load('data/settlement_core.npz', allow_pickle=True)
    core_indices = set(core_data['core_indices'].tolist())
    if 'seeds' in core_data and not np.array_equal(core_data['seeds'], seeds):
        raise ValueError(
            "settlement_core.npz was generated from a different seed set. "
            "Rerun isolate_buildable_plot() before generate_zones()."
        )
    _validate_core_indices(core_indices, seeds)

    vor = Voronoi(seeds)
    graph = _core_adjacency(vor, core_indices)
    zone_seed_indices = _choose_zone_seeds(seeds, core_indices, min(num_zones, len(core_indices)))
    cell_zones = {}
    queue = []

    for zone_id, seed_idx in enumerate(zone_seed_indices):
        cell_zones[seed_idx] = zone_id
        queue.append(seed_idx)

    while queue:
        current = queue.pop(0)
        for neighbor in sorted(graph[current]):
            if neighbor in cell_zones:
                continue
            cell_zones[neighbor] = cell_zones[current]
            queue.append(neighbor)

    zone_map = np.full((D, W), -1, dtype=np.int16)
    for x in range(W):
        for z in range(D):
            nearest = int(np.argmin(np.linalg.norm(seeds - [x, z], axis=1)))
            if nearest in cell_zones:
                zone_map[z, x] = cell_zones[nearest]

    zone_seed_points = np.array([seeds[idx] for idx in zone_seed_indices])
    _save_npz_with_updates(
        'data/settlement_data.npz',
        data,
        zone_map=zone_map,
        zone_count=np.array(len(zone_seed_indices), dtype=np.int16),
        zone_seed_indices=np.array(zone_seed_indices, dtype=np.int32),
        zone_seed_points=zone_seed_points,
        zone_cell_assignments=np.array(sorted(cell_zones.items()), dtype=np.int32),
    )
    print(f"Generated {len(zone_seed_indices)} connected settlement zones. Outside zone is -1.")
    return zone_map

def isolate_buildable_plot():
    # 1. LOAD GEOMETRY DATA
    try:
        data = np.load('data/settlement_data.npz', allow_pickle=True)
        seeds = data['seeds']
        num_drift = data['num_drift']
        heightmap = data['heightmap']
        water_map = data['water_map']
        W, D = terrain_shape(heightmap)
    except FileNotFoundError:
        print("Error: settlement_data.npz not found. Run voronoi.py first.")
        return

    # 2. RECONSTRUCT VORONOI
    vor = Voronoi(seeds)

    # 3. FILTERING LOGIC
    # We only care about the cells belonging to 'House/Drift' seeds (0 to num_drift-1)
    buildable_indices = []

    # Thresholds
    min_area = 150  # Discard cells smaller than this (likely chasm/edge noise)

    for i in range(num_drift):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # A. Discard Infinite Regions (connected to the far border of the Voronoi space)
        if -1 in region:
            continue

        # B. Calculate Area and check for Map Border/Water contact
        vertices = vor.vertices[region]

        # Check if any vertex is outside or on the map border (256x256)
        on_border = np.any((vertices[:, 0] <= 5) | (vertices[:, 0] >= W-5) |
                           (vertices[:, 1] <= 5) | (vertices[:, 1] >= D-5))
        if on_border:
            continue

        # Calculate approximate area using shoelace formula
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        if area < min_area:
            continue

        # C. Water Check: Sample points inside the cell to ensure it's dry land
        # We sample the centroid and the vertices
        centroid = np.mean(vertices, axis=0).astype(int)
        if water_map[np.clip(centroid[1], 0, D-1), np.clip(centroid[0], 0, W-1)]:
            continue

        # If it passed all checks, it's a candidate
        buildable_indices.append(i)

    # 4. FIND CONNECTED REGIONS (Group touching cells)
    # Since you want the 'Single Largest Region', we find which candidate cells
    # share a Voronoi ridge.

    def get_neighbors(point_idx):
        # Find points that share a ridge with this point
        neighbors = []
        for ridge_points in vor.ridge_points:
            if point_idx in ridge_points:
                neighbor = ridge_points[0] if ridge_points[1] == point_idx else ridge_points[1]
                if neighbor in buildable_indices:
                    neighbors.append(neighbor)
        return neighbors

    # Simple BFS to find the largest connected component of buildable cells
    visited = set()
    clusters = []

    for idx in buildable_indices:
        if idx not in visited:
            current_cluster = []
            queue = [idx]
            visited.add(idx)
            while queue:
                curr = queue.pop(0)
                current_cluster.append(curr)
                for n in get_neighbors(curr):
                    if n not in visited:
                        visited.add(n)
                        queue.append(n)
            clusters.append(current_cluster)

    # 5. ISOLATE LARGEST REGION
    if not clusters:
        print("No buildable regions found with current constraints.")
        return False

    largest_cluster = max(clusters, key=len)
    print(f"Isolated settlement core: {len(largest_cluster)} cells.")

    # 6. UPDATE VISUALIZER (Visualizing the Settlement Perimeter)
    # We reload the viz file and 'paint' the chosen region to verify
    viz = np.load('data/settlement_viz.npz', allow_pickle=True)
    blocks = viz['blocks'].copy()
    palette = viz['palette'].tolist()

    if "minecraft:diamond_block" not in palette:
        palette.append("minecraft:diamond_block")
    DIAMOND = palette.index("minecraft:diamond_block")

    # Mark the cells of the largest cluster in the 3D view
    for seed_idx in largest_cluster:
        region = vor.regions[vor.point_region[seed_idx]]
        vertices = vor.vertices[region]
        # Mark the centroid with a gold pillar to show buildable status
        cx, cz = np.mean(vertices, axis=0).astype(int)
        cy = int(heightmap[cz, cx])
        blocks[cx, cy:cy+5, cz] = DIAMOND

    # Save the updated visualization
    np.savez('data/settlement_viz.npz',
             blocks=blocks,
             palette=np.array(palette),
             origin=viz['origin'],
             meta=viz['meta'])

    # Save the isolated cluster data for the building phase
    np.savez('data/settlement_core.npz',
             core_indices=np.array(largest_cluster),
             seeds=seeds)

    print("Settlement core isolated. Check visualizer for gold markers.")
    return True

import numpy as np
from scipy.spatial import Voronoi
import json
import heapq
from coordinate_system import terrain_shape

def mark_path_and_perimeter():
    # 1. LOAD DATA
    data = np.load('data/settlement_data.npz', allow_pickle=True)
    seeds = data['seeds']
    heightmap = data['heightmap']
    water_map = data['water_map']
    W, D = terrain_shape(heightmap)

    core_data = np.load('data/settlement_core.npz', allow_pickle=True)
    core_indices = set(core_data['core_indices'].tolist()) # Use set for O(1) lookup
    if 'seeds' in core_data and not np.array_equal(core_data['seeds'], seeds):
        raise ValueError(
            "settlement_core.npz was generated from a different seed set. "
            "Rerun isolate_buildable_plot() before mark_path_and_perimeter()."
        )
    _validate_core_indices(core_indices, seeds)

    vor = Voronoi(seeds)

    # 2. IDENTIFY CENTER
    core_seeds = seeds[list(core_indices)]
    avg_pos = np.mean(core_seeds, axis=0)
    center_idx = list(core_indices)[np.argmin(np.linalg.norm(core_seeds - avg_pos, axis=1))]
    center_point = seeds[center_idx]

    # Target vertex: Choose the vertex of the center cell closest to the center seed
    center_region = [v for v in vor.regions[vor.point_region[center_idx]] if v != -1]
    target_v_idx = center_region[np.argmin(np.linalg.norm(vor.vertices[center_region] - center_point, axis=1))]

    # 3. MAPPING INTERNAL GRAPH
    # Identify vertices that belong to at least one core cell
    internal_vertices = set()
    for i in core_indices:
        region = vor.regions[vor.point_region[i]]
        internal_vertices.update([v for v in region if v != -1])

    # Identify "Wall" ridges (ridges where one side is NOT in core)
    wall_ridges = set()
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        p1, p2 = ridge_points
        if (p1 in core_indices) != (p2 in core_indices):
            # Sort to ensure consistent tuple key
            wall_ridges.add(tuple(sorted(ridge_vertices)))

    # 4. PATHFINDING FUNCTION (Dijkstra)
    def get_internal_path(start_v):
        if start_v not in internal_vertices: return None

        q = [(0, start_v, [])]
        visited = {start_v: 0}

        while q:
            (cost, curr, path) = heapq.heappop(q)
            if curr == target_v_idx: return path + [curr]

            # Find neighbors via ridges
            for rv_list in vor.ridge_vertices:
                if curr in rv_list:
                    rv = tuple(sorted(rv_list))
                    # CONSTRAINT: Next vertex must be internal AND ridge must not be a wall
                    if rv in wall_ridges: continue

                    next_v = rv[0] if rv[1] == curr else rv[1]
                    if next_v == -1 or next_v not in internal_vertices: continue

                    new_cost = cost + np.linalg.norm(vor.vertices[curr] - vor.vertices[next_v])
                    if next_v not in visited or new_cost < visited[next_v]:
                        visited[next_v] = new_cost
                        heapq.heappush(q, (new_cost, next_v, path + [curr]))
        return None

    # 5. CARDINAL DOOR SELECTION WITH RECURSIVE FALLBACK
    perimeter_vertices = []
    for rv in wall_ridges:
        if -1 not in rv:
            perimeter_vertices.extend(rv)
    perimeter_vertices = list(set(perimeter_vertices))

    directions = {
        'North': [0, 1], 'South': [0, -1], 'East': [1, 0], 'West': [-1, 0]
    }

    final_paths = {}
    door_vertices = {}

    for d_name, vec in directions.items():
        # Sort all perimeter vertices by their distance in the cardinal direction
        candidates = []
        for v_idx in perimeter_vertices:
            v_coord = vor.vertices[v_idx]
            proj = np.dot(v_coord - center_point, vec)
            candidates.append((proj, v_idx))

        # Sort descending (furthest first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Try candidates until a valid internal path is found
        found = False
        for _, v_idx in candidates:
            path = get_internal_path(v_idx)
            if path:
                final_paths[d_name] = path
                door_vertices[d_name] = v_idx
                found = True
                break

        if not found:
            print(f"Warning: Could not find valid internal path for {d_name}")

    # 6. VOXEL UPDATES
    viz = np.load('data/settlement_viz.npz', allow_pickle=True)
    blocks = viz['blocks'].copy()
    palette = viz['palette'].tolist()

    def get_id(name):
        if name not in palette: palette.append(name)
        return palette.index(name)

    WALL = get_id("minecraft:red_wool")
    PATH = get_id("minecraft:yellow_concrete")
    DOOR = get_id("minecraft:iron_block")

    # A. Draw Walls (strictly on perimeter ridges)
    for rv in wall_ridges:
        if -1 in rv: continue
        v1, v2 = vor.vertices[rv[0]], vor.vertices[rv[1]]
        steps = int(np.linalg.norm(v1 - v2) * 2)
        for t in np.linspace(0, 1, steps):
            p = v1 + t * (v2 - v1)
            px, pz = int(p[0]), int(p[1])
            if 0 <= px < W and 0 <= pz < D:
                py = int(heightmap[pz, px])
                blocks[px, py:py+1, pz] = WALL

    # B. Draw Doors and Constraints-Aware Paths
    for d_name, path in final_paths.items():
        # Mark Door Location
        dv = vor.vertices[door_vertices[d_name]]
        dx, dz = int(dv[0]), int(dv[1])
        dy = int(heightmap[dz, dx])
        blocks[dx, dy:dy+5, dz] = DOOR

        # Mark Path
        for i in range(len(path)-1):
            va, vb = vor.vertices[path[i]], vor.vertices[path[i+1]]
            p_steps = int(np.linalg.norm(va - vb) * 2)
            for t in np.linspace(0, 1, p_steps):
                p = va + t * (vb - va)
                px, pz = int(p[0]), int(p[1])
                py = int(heightmap[pz, px])
                blocks[px, py, pz] = PATH

    # 7. SAVE
    np.savez('data/settlement_viz.npz', blocks=blocks, palette=np.array(palette),
             origin=viz['origin'], meta=viz['meta'])
    print("Wall erected and cardinal paths mapped.")

if __name__ == "__main__":
    isolate_buildable_plot()
    mark_path_and_perimeter()
