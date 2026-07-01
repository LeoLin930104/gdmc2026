import numpy as np

from coordinate_system import require_matching_terrain_and_blocks, terrain_shape


def validate_coordinate_artifacts():
    base = np.load('data/data.npz', allow_pickle=True)
    sim = np.load('data/settlement_data.npz', allow_pickle=True)
    viz = np.load('data/settlement_viz.npz', allow_pickle=True)

    base_heightmap = base['heightmap']
    sim_heightmap = sim['heightmap']
    blocks = viz['blocks']

    require_matching_terrain_and_blocks(sim_heightmap, blocks)

    base_width, base_depth = terrain_shape(base_heightmap)
    sim_width, sim_depth = terrain_shape(sim_heightmap)
    block_width, _, block_depth = blocks.shape

    print("Coordinate artifact validation")
    print(f"  terrain arrays: heightmap[z, x] = {sim_heightmap.shape}")
    print(f"  voxel array: blocks[x, y, z] = {blocks.shape}")
    print(f"  base footprint: {base_width}x{base_depth}")
    print(f"  sim footprint: {sim_width}x{sim_depth}")
    print(f"  block footprint: {block_width}x{block_depth}")
    missing_origins = [
        name for name, data in [('base', base), ('sim', sim), ('viz', viz)]
        if 'origin' not in data
    ]
    if missing_origins:
        raise ValueError(f"Missing origin in artifact(s): {', '.join(missing_origins)}")

    print(f"  base origin: {base['origin'].tolist()}")
    print(f"  sim origin: {sim['origin'].tolist()}")
    print(f"  viz origin: {viz['origin'].tolist()}")

    if not np.array_equal(base['origin'], sim['origin']):
        raise ValueError("Base and simulation origins do not match.")
    if not np.array_equal(sim['origin'], viz['origin']):
        raise ValueError("Simulation and visualization origins do not match.")

    print("  status: OK")


if __name__ == "__main__":
    validate_coordinate_artifacts()
