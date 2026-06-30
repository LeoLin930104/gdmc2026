import numpy as np
import json
from scipy.spatial import Voronoi, Delaunay
from scipy.ndimage import binary_dilation, gaussian_filter
from coordinate_system import terrain_shape


def generate_voronoi_diagram(
    grid_spacing=30,
    drift_steps=15,
    drift_speed=1.5,
    jitter_ratio=0.0,
    random_seed=42,
    buffer_stride=5,
):
    # 1. LOAD TERRAIN DATA
    data = np.load('data/data.npz')
    heightmap = data['heightmap']
    flat_mask = data['flat_mask']
    water_map = data['water_map']
    slope = data['slope']
    origin = data['origin'] if 'origin' in data else np.array([0, 0, 0])

    W, D = terrain_shape(heightmap)
    H = int(np.max(heightmap)) + 15 

    # 2. OBSTACLE & BUFFER LOGIC
    chasm_mask = slope > 5.0 
    obstacle_mask = water_map | chasm_mask
    edge_mask = binary_dilation(obstacle_mask) & ~obstacle_mask
    edge_coords = np.argwhere(edge_mask)
    if edge_coords.size:
        buffer_seeds = edge_coords[::max(1, int(buffer_stride))][:, [1, 0]].astype(float)
    else:
        buffer_seeds = np.empty((0, 2), dtype=float)

    # 3. SETTLEMENT SEED GENERATION (Slope-Drift)
    x_range = np.arange(grid_spacing // 2, W, grid_spacing)
    z_range = np.arange(grid_spacing // 2, D, grid_spacing)
    gx, gz = np.meshgrid(x_range, z_range)
    drift_seeds = np.vstack([gx.ravel(), gz.ravel()]).T.astype(float)

    if len(drift_seeds) and jitter_ratio > 0:
        rng = np.random.default_rng(random_seed)
        jitter = float(grid_spacing) * float(jitter_ratio)
        drift_seeds += rng.uniform(-jitter, jitter, size=drift_seeds.shape)
        drift_seeds[:, 0] = np.clip(drift_seeds[:, 0], 0, W - 1)
        drift_seeds[:, 1] = np.clip(drift_seeds[:, 1], 0, D - 1)

    smooth_h = gaussian_filter(heightmap.astype(float), sigma=1.5)
    dz, dx = np.gradient(smooth_h)

    for _ in range(drift_steps):
        for i in range(len(drift_seeds)):
            ix, iz = np.clip(drift_seeds[i], 0, [W-1, D-1]).astype(int)
            vx, vz = -dx[iz, ix], -dz[iz, ix]
            drift_seeds[i, 0] += vx * drift_speed
            drift_seeds[i, 1] += vz * drift_speed

    valid_drift = []
    for s in drift_seeds:
        ix, iz = np.clip(s, 0, [W-1, D-1]).astype(int)
        if not obstacle_mask[iz, ix]:
            valid_drift.append(s)
    drift_seeds = np.asarray(valid_drift, dtype=float).reshape(-1, 2)

    # 4. COMBINE SEEDS & GENERATE DIAGRAMS
    # We keep track of how many seeds are "House" seeds vs "Buffer" seeds
    seed_sets = [seeds for seeds in (drift_seeds, buffer_seeds) if len(seeds)]
    if not seed_sets:
        raise RuntimeError("Voronoi generation found no valid dry settlement seeds.")
    all_seeds = np.vstack(seed_sets)
    if len(all_seeds) < 4:
        raise RuntimeError(
            f"Voronoi generation needs at least 4 seeds; got {len(all_seeds)}. "
            "Use a larger land area or reduce obstacle filtering."
        )
    num_drift = len(drift_seeds)

    vor = Voronoi(all_seeds)
    tri = Delaunay(all_seeds)

    # 5. SAVE GEOMETRY FOR THE NEXT PROGRAM (settlement_data.npz)
    # This file stores the raw math needed to reconstruct the diagram
    np.savez('data/settlement_data.npz',
            seeds=all_seeds,
            num_drift=num_drift,
            heightmap=heightmap,
            water_map=water_map,
            chasm_mask=chasm_mask,
            origin=origin)

    # 6. VOXEL CONSTRUCTION (For Visualizer)
    palette = ["minecraft:air", "minecraft:grass_block", "minecraft:dirt", 
            "minecraft:water", "minecraft:cobblestone", "minecraft:oak_planks", "minecraft:stone"]
    blocks = np.zeros((W, H, D), dtype=np.uint16)
    AIR, GRASS, DIRT, WATER, ROAD, BRIDGE, STONE = 0, 1, 2, 3, 4, 5, 6

    for x in range(W):
        for z in range(D):
            y_s = int(heightmap[z, x])
            blocks[x, :y_s, z] = DIRT
            blocks[x, y_s, z] = GRASS if not chasm_mask[z, x] else STONE
            if water_map[z, x]:
                blocks[x, y_s:y_s+2, z] = WATER

    # Draw Roads
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            dist = np.linalg.norm(p2 - p1)
            for t in np.linspace(0, 1, int(dist * 2)):
                rx, rz = (p1 + t * (p2 - p1)).astype(int)
                if 0 <= rx < W and 0 <= rz < D and not water_map[rz, rx]:
                    blocks[rx, int(heightmap[rz, rx]), rz] = ROAD

    # 7. SAVE FOR VISUALIZER (settlement_viz.npz)
    meta_str = json.dumps({"world_name": "Geometry Exported", "size": [W, H, D]})
    np.savez('data/settlement_viz.npz', blocks=blocks, palette=np.array(palette), 
            origin=origin, meta=meta_str)

    print(f"Data saved in /data: 'settlement_data.npz' (Geometry) and 'settlement_viz.npz' (Voxels)")

if __name__ == "__main__":
    generate_voronoi_diagram()
