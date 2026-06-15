import numpy as np
from scipy.spatial import Voronoi
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


def quantized_stair_height(start_y, end_y, distance, length):
    """
    Interpolate in half-block units. A one-half-block change per forward block
    gives one full block of elevation over two blocks.
    """
    if length <= 0:
        return start_y

    start_half = int(round(start_y * 2))
    end_half = int(round(end_y * 2))
    delta_half = end_half - start_half
    max_half_steps = int(round(distance))

    if abs(delta_half) <= max_half_steps:
        step_half = np.sign(delta_half) * min(abs(delta_half), max_half_steps)
        return (start_half + step_half) / 2.0

    # Fallback for paths physically too short for the requested grade.
    t = distance / length
    return round((start_y + (end_y - start_y) * t) * 2) / 2.0


def apply_terraforming(max_prune_score=1000, stair_length_blocks=4.0):
    ctx = load_context()
    seeds, hmap, core_indices = ctx['seeds'], ctx['heightmap'], ctx['core_indices']
    palette, water_map = ctx['palette'], ctx['water_map']
    W, D = terrain_shape(hmap)
    vor = Voronoi(seeds)

    # 1. ESTABLISH INITIAL VERTEX HEIGHTS
    vertex_heights = {}
    for i, v_coord in enumerate(vor.vertices):
        vx, vz = int(np.clip(v_coord[0], 0, W-1)), int(np.clip(v_coord[1], 0, D-1))
        vertex_heights[i] = hmap[vz, vx]

    # 2. PRE-CALCULATE ALL CELL HEIGHTS
    cell_heights = {}
    for p_idx in range(len(seeds)):
        region_idx = vor.point_region[p_idx]
        region = [v for v in vor.regions[region_idx] if v != -1]
        if region:
            cell_heights[p_idx] = np.mean([vertex_heights[v] for v in region])
        else:
            cell_heights[p_idx] = hmap[int(seeds[p_idx][1]), int(seeds[p_idx][0])]

    # 3. LEVEL CELL INTERIORS FIRST
    core_cell_mask = np.zeros((D, W), dtype=bool)
    for p_idx in core_indices:
        region = [v for v in vor.regions[vor.point_region[p_idx]] if v != -1]
        if not region: continue

        target_y = cell_heights[p_idx]
        v_coords = vor.vertices[region]
        
        for x in range(int(np.min(v_coords[:,0])), int(np.max(v_coords[:,0])) + 1):
            for z in range(int(np.min(v_coords[:,1])), int(np.max(v_coords[:,1])) + 1):
                if 0 <= x < W and 0 <= z < D:
                    if np.argmin(np.linalg.norm(seeds - [x, z], axis=1)) == p_idx:
                        hmap[z, x] = target_y
                        core_cell_mask[z, x] = True

    # 4. BUILD ROAD GRAPH FOR BETWEENNESS PRUNING
    graph = {}
    all_edges = set()
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices: continue
        p1, p2 = ridge_points
        if p1 not in core_indices and p2 not in core_indices: continue
        
        v1, v2 = ridge_vertices
        dist = np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])
        
        if v1 not in graph: graph[v1] = {}
        if v2 not in graph: graph[v2] = {}
        
        graph[v1][v2] = dist
        graph[v2][v1] = dist
        all_edges.add(tuple(sorted((v1, v2))))

    # 5. RUN BETWEENNESS PRUNING SUITE
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

    # =========================================================================
    # NEW STEP: COMPUTE CONVERGING JUNCTION INTERSECTION HEIGHTS
    # =========================================================================
    # Find the flat floor baseline height for every single active path segment
    active_path_bases = {}
    junction_connected_paths = {v_idx: [] for v_idx in range(len(vor.vertices))}

    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices: continue
        v1_idx, v2_idx = ridge_vertices
        edge_key = tuple(sorted((v1_idx, v2_idx)))
        if edge_key in pruned_edges: continue
        
        p1, p2 = ridge_points
        if p1 not in core_indices and p2 not in core_indices: continue

        # The path baseline is derived from the adjacent leveled cell floors
        cell_y1 = cell_heights.get(p1, vertex_heights[v1_idx])
        cell_y2 = cell_heights.get(p2, vertex_heights[v2_idx])
        path_base_y = max(cell_y1, cell_y2)
        
        active_path_bases[edge_key] = path_base_y
        
        # Track which path bases meet at which vertices
        junction_connected_paths[v1_idx].append(path_base_y)
        junction_connected_paths[v2_idx].append(path_base_y)

    # Average the paths meeting at each junction vertex to form landing heights
    junction_landing_heights = {}
    for v_idx, converging_heights in junction_connected_paths.items():
        if len(converging_heights) > 0:
            # Settle on the average height of all active paths arriving here
            junction_landing_heights[v_idx] = np.mean(converging_heights)
        else:
            # Fallback to default structural height if no paths use it
            junction_landing_heights[v_idx] = vertex_heights[v_idx]
    # =========================================================================

    # 6. RENDER PATHS WITH INTERSECTION INTERPOLATION
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
        if p1 not in core_indices and p2 not in core_indices: continue

        v1, v2 = vor.vertices[v1_idx], vor.vertices[v2_idx]
        diff = v2 - v1
        length = np.linalg.norm(diff)
        if length < 0.1: continue
        
        dir_norm = diff / length
        perp = np.array([-dir_norm[1], dir_norm[0]]) 

        # Every path endpoint uses its junction height so all arriving paths meet.
        j_height1 = junction_landing_heights[v1_idx]
        j_height2 = junction_landing_heights[v2_idx]

        for t_dist in np.linspace(0, length, int(length * 2)):
            t = t_dist / length
            center_p = v1 + t * diff

            interp_y = quantized_stair_height(j_height1, j_height2, t_dist, length)

            for offset in [-1, 0, 1]:
                point = center_p + perp * offset
                px, pz = int(np.clip(point[0], 0, W-1)), int(np.clip(point[1], 0, D-1))
                hmap[pz, px] = interp_y
                path_mask[pz, px] = True
                if np.isnan(path_surface[pz, px]):
                    path_surface[pz, px] = interp_y
                else:
                    path_surface[pz, px] = max(path_surface[pz, px], interp_y)

    # 7. VOXEL RECONSTRUCTION WITH SOLID FOUNDATIONS
    # Terraforming uses averaged/interpolated heights. Do not floor them with
    # int()/astype(int), or surfaces get shaved down by one block.
    path_base_y = np.floor(np.nan_to_num(path_surface, nan=0.0)).astype(int)
    path_slab_mask = np.zeros((D, W), dtype=bool)
    path_slab_mask[path_mask] = np.isclose(path_surface[path_mask] % 1.0, 0.5)
    hmap[path_mask] = path_base_y[path_mask]

    final_hmap = np.floor(hmap + 0.5).astype(int)
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

    # 8. SAVE BACK TO THE CORE
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
             path_base_y=path_base_y, path_slab_mask=path_slab_mask, **preserved)
    print("Terraforming complete: Intersection landings unified across all converging paths.")

if __name__ == "__main__":
    apply_terraforming(max_prune_score=2)
