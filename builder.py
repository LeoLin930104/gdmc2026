import numpy as np
from gdpc import Editor, Block, Rect
from scipy.spatial import Voronoi
from map_manager import MapManager
from coordinate_system import local_to_world, require_matching_terrain_and_blocks, terrain_shape


PATH_BLOCKS = [
    "minecraft:cobblestone",
    "minecraft:stone_bricks",
    "minecraft:polished_andesite",
    "minecraft:chiseled_stone_bricks",
    "minecraft:mossy_cobblestone",
]
PATH_SLAB_BLOCKS = {
    "minecraft:cobblestone": "minecraft:cobblestone_slab",
    "minecraft:stone_bricks": "minecraft:stone_brick_slab",
    "minecraft:polished_andesite": "minecraft:polished_andesite_slab",
    "minecraft:chiseled_stone_bricks": "minecraft:stone_brick_slab",
    "minecraft:mossy_cobblestone": "minecraft:mossy_cobblestone_slab",
}
CELL_SURFACE_BLOCK = "minecraft:grass_block"
FOUNDATION_BLOCK = "minecraft:dirt"
AIR_BLOCK = "minecraft:air"
# House-footprint markers painted by the plotter (terracotta). When buildings
# are reserved rather than placed (prefab pipeline), these are read back as
# plain ground so the reserved plots don't show the marker block.
HOUSE_MARKER_BLOCKS = {
    "minecraft:terracotta",
}
FARM_BORDER_BLOCK = "minecraft:oak_log"
FARM_SOIL_BLOCK = "minecraft:farmland"
FARM_WATER_BLOCK = "minecraft:water"
# CHANGED FOR NARRATIVE: gate Phase-2C farm-field placement (classification untouched) so Premade Builds/farm_field.py is the sole crop-field source; True restores the generator's fields.
BUILD_FARM_FIELDS = False
CROP_BLOCKS = [
    ("minecraft:wheat", {"age": "7"}),
    ("minecraft:carrots", {"age": "7"}),
    ("minecraft:potatoes", {"age": "7"}),
    ("minecraft:beetroots", {"age": "3"}),
]
FLOWER_BLOCKS = [
    "minecraft:dandelion",
    "minecraft:poppy",
    "minecraft:azure_bluet",
    "minecraft:oxeye_daisy",
    "minecraft:cornflower",
    "minecraft:allium",
]
TREE_TRUNK_BLOCK = "minecraft:oak_log"
TREE_LEAF_BLOCK = "minecraft:oak_leaves"
BUSH_BLOCK = "minecraft:oak_leaves"
ZONE_BORDER_BLOCKS = [
    "minecraft:lime_stained_glass",
    "minecraft:cyan_stained_glass",
    "minecraft:magenta_stained_glass",
    "minecraft:orange_stained_glass",
    "minecraft:purple_stained_glass",
    "minecraft:yellow_stained_glass",
]

# Biome-adaptive (surface, foundation) for cell surface/plot pads/foundations; substring-matched on the sampled biome id, first hit wins, else DEFAULT_GROUND.
DEFAULT_GROUND = ("minecraft:grass_block", "minecraft:dirt")
GROUND_BLOCKS_BY_BIOME = [
    # (biome-id substrings, surface_block, foundation_block)
    # sandstone (not sand) on top too: sand is gravity-affected and would fall unpredictably over the foundation columns.
    (("desert",),                 "minecraft:sandstone",     "minecraft:sandstone"),
    (("badlands", "mesa"),        "minecraft:red_sandstone", "minecraft:red_sandstone"),
    (("beach", "shore"),          "minecraft:sandstone",     "minecraft:sandstone"),
    (("mushroom",),               "minecraft:mycelium",      "minecraft:dirt"),
    (("snow", "frozen", "ice", "grove"), "minecraft:snow_block", "minecraft:dirt"),
    (("mangrove", "swamp"),       "minecraft:grass_block",   "minecraft:dirt"),
]
# Generic simulated-terrain surfaces swapped for the biome surface outside the plots so the open area matches; other palette blocks (water, stone, natural sand) pass through.
REMAPPABLE_GROUND_SURFACES = {
    "minecraft:grass_block",
    "minecraft:dirt",
    "minecraft:coarse_dirt",
    "minecraft:podzol",
}


def ground_blocks_for_biome(biome):
    """(surface, foundation) ground blocks matched to `biome` (a biome id string).

    Falls back to grass_block/dirt for temperate or unknown biomes.
    """
    b = (biome or "").lower()
    for keys, surface, foundation in GROUND_BLOCKS_BY_BIOME:
        if any(k in b for k in keys):
            return surface, foundation
    return DEFAULT_GROUND


def sample_settlement_biome(editor, sim, origin):
    """Best-effort biome id at the settlement center; None if it can't be read."""
    try:
        W, _H, D = sim['blocks'].shape
        cx, cz = W // 2, D // 2
        ox, oz = int(origin[0]), int(origin[2])
        cy = int(sim['heightmap'][cz, cx])
        return editor.getBiome((ox + cx, cy, oz + cz))
    except Exception as exc:
        print(f"[warn] biome sample failed ({exc!r}); using default ground blocks.")
        return None


def load_simulated_data():
    try:
        viz_data = np.load('data/settlement_viz.npz', allow_pickle=True)
        sim_data = np.load('data/settlement_data.npz', allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find simulation data files.")
        
    if 'origin' not in viz_data:
        raise ValueError("settlement_viz.npz is missing origin. Regenerate the pipeline from map_manager.py.")
    if 'origin' not in sim_data:
        raise ValueError("settlement_data.npz is missing origin. Regenerate the pipeline from voronoi.py.")
    if not np.array_equal(viz_data['origin'], sim_data['origin']):
        raise ValueError("settlement_viz.npz and settlement_data.npz have different origins.")

    origin = viz_data['origin']

    return {
        'blocks': viz_data['blocks'],       
        'palette': viz_data['palette'].tolist(), 
        'origin': origin,
        'heightmap': sim_data['heightmap'],
        'seeds': sim_data['seeds'],
        'core_cell_mask': sim_data['core_cell_mask'] if 'core_cell_mask' in sim_data else None,
        'path_mask': sim_data['path_mask'] if 'path_mask' in sim_data else None,
        'path_base_y': sim_data['path_base_y'] if 'path_base_y' in sim_data else None,
        'path_slab_mask': sim_data['path_slab_mask'] if 'path_slab_mask' in sim_data else None,
        'zone_map': sim_data['zone_map'] if 'zone_map' in sim_data else None,
        'zone_count': int(sim_data['zone_count']) if 'zone_count' in sim_data else 0,
    }


def load_core_indices():
    core_data = np.load('data/settlement_core.npz', allow_pickle=True)
    return set(core_data['core_indices'].tolist())


def load_plot_data():
    try:
        plot_data = np.load('data/settlement_plots.npz', allow_pickle=True)
    except FileNotFoundError:
        return {}, {}, {}, 8

    module_size = int(plot_data['module_size']) if 'module_size' in plot_data else 8
    plots = dict(plot_data['plots']) if 'plots' in plot_data else {}
    farms = dict(plot_data['farms']) if 'farms' in plot_data else {}
    building_rects = dict(plot_data['building_rects']) if 'building_rects' in plot_data else {}
    return plots, farms, building_rects, module_size


def build_core_cell_mask(seeds, core_indices, heightmap):
    W, D = terrain_shape(heightmap)
    mask = np.zeros((D, W), dtype=bool)
    vor = Voronoi(seeds)

    for p_idx in core_indices:
        region = [v for v in vor.regions[vor.point_region[p_idx]] if v != -1]
        if not region:
            continue

        vertices = vor.vertices[region]
        min_x = max(0, int(np.floor(np.min(vertices[:, 0]))))
        max_x = min(W - 1, int(np.ceil(np.max(vertices[:, 0]))))
        min_z = max(0, int(np.floor(np.min(vertices[:, 1]))))
        max_z = min(D - 1, int(np.ceil(np.max(vertices[:, 1]))))

        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                if np.argmin(np.linalg.norm(seeds - [x, z], axis=1)) == p_idx:
                    mask[z, x] = True

    return mask


def infer_path_mask(blocks, palette, heightmap):
    W, D = terrain_shape(heightmap)
    mask = np.zeros((D, W), dtype=bool)
    path_blocks = {
        "minecraft:coarse_dirt",
        "minecraft:yellow_concrete",
        "minecraft:dirt_path",
        *PATH_BLOCKS,
    }

    for x in range(W):
        for z in range(D):
            y = int(heightmap[z, x])
            if y < blocks.shape[1] and palette[blocks[x, y, z]] in path_blocks:
                mask[z, x] = True

    return mask


def build_module_footprint(plots, module_size, heightmap):
    W, D = terrain_shape(heightmap)
    footprint = np.zeros((D, W), dtype=bool)

    for modules in plots.values():
        for mx, mz in modules:
            x0 = max(0, int(mx))
            z0 = max(0, int(mz))
            x1 = min(W, x0 + module_size)
            z1 = min(D, z0 + module_size)
            footprint[z0:z1, x0:x1] = True

    return footprint


def build_farm_footprint(farms, heightmap):
    W, D = terrain_shape(heightmap)
    footprint = np.zeros((D, W), dtype=bool)

    for cells in farms.values():
        for x, z in cells:
            if 0 <= x < W and 0 <= z < D:
                footprint[int(z), int(x)] = True

    return footprint


def build_rectangle_footprint(rects, heightmap):
    W, D = terrain_shape(heightmap)
    footprint = np.zeros((D, W), dtype=bool)

    for rect in rects.values():
        x0 = max(0, int(rect["x"]))
        z0 = max(0, int(rect["z"]))
        x1 = min(W, x0 + int(rect["width"]))
        z1 = min(D, z0 + int(rect["depth"]))
        footprint[z0:z1, x0:x1] = True

    return footprint


def rectangles_to_plots(rects):
    return {
        cell_id: [(int(rect["x"]), int(rect["z"]))]
        for cell_id, rect in rects.items()
    }


def compute_module_floors(plots, module_size, max_floors=3):
    floor_maps = {}

    for cell_id, modules in plots.items():
        floor_map = {tuple(module): 1 for module in modules}
        for floor in range(1, max_floors):
            for mx, mz in modules:
                key = (mx, mz)
                if floor_map[key] != floor:
                    continue

                neighbors = 0
                for dx, dz in [(module_size, 0), (-module_size, 0), (0, module_size), (0, -module_size)]:
                    neighbor_key = (mx + dx, mz + dz)
                    if neighbor_key in floor_map and floor_map[neighbor_key] >= floor:
                        neighbors += 1

                if neighbors > 2:
                    floor_map[key] += 1

        floor_maps[cell_id] = floor_map

    return floor_maps


def path_block_for(local_x, local_z):
    return PATH_BLOCKS[(local_x * 31 + local_z * 17) % len(PATH_BLOCKS)]


def path_slab_for(local_x, local_z):
    return PATH_SLAB_BLOCKS[path_block_for(local_x, local_z)]


def crop_for(local_x, local_z):
    return CROP_BLOCKS[(local_x * 13 + local_z * 7) % len(CROP_BLOCKS)]


def flower_for(local_x, local_z):
    return FLOWER_BLOCKS[(local_x * 11 + local_z * 5) % len(FLOWER_BLOCKS)]


def zone_border_block(zone_id):
    if zone_id < 0:
        return "minecraft:white_stained_glass"
    return ZONE_BORDER_BLOCKS[zone_id % len(ZONE_BORDER_BLOCKS)]


def neighbor_count(mask):
    counts = np.zeros(mask.shape, dtype=np.uint8)
    counts[1:, :] += mask[:-1, :]
    counts[:-1, :] += mask[1:, :]
    counts[:, 1:] += mask[:, :-1]
    counts[:, :-1] += mask[:, 1:]
    return counts


def dilate_mask(mask, radius=1):
    result = mask.copy()
    frontier = mask.copy()
    for _ in range(radius):
        expanded = frontier.copy()
        expanded[1:, :] |= frontier[:-1, :]
        expanded[:-1, :] |= frontier[1:, :]
        expanded[:, 1:] |= frontier[:, :-1]
        expanded[:, :-1] |= frontier[:, 1:]
        frontier = expanded
        result |= frontier
    return result


def select_spaced_points(candidates, min_spacing, rng):
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    selected = []
    min_spacing_sq = min_spacing * min_spacing

    for x, z in shuffled:
        if all((x - sx) ** 2 + (z - sz) ** 2 >= min_spacing_sq for sx, sz in selected):
            selected.append((int(x), int(z)))

    return selected


def plan_landscaping(
    core_mask,
    path_mask,
    reserved_mask,
    flower_density=0.015,
    bush_rate=0.35,
    tree_interval=11,
    tree_rate=0.7,
    seed=42,
):
    rng = np.random.default_rng(seed)
    open_mask = core_mask & ~reserved_mask

    adjacent_to_path = (neighbor_count(path_mask) > 0) & open_mask
    intersections = path_mask & (neighbor_count(path_mask) >= 3)
    near_intersections = dilate_mask(intersections, radius=2)

    path_near = dilate_mask(path_mask, radius=4) & ~dilate_mask(path_mask, radius=1)
    tree_candidates = np.argwhere(path_near & open_mask)
    tree_candidates = [(int(x), int(z)) for z, x in tree_candidates if rng.random() < tree_rate]
    trees = select_spaced_points(tree_candidates, tree_interval, rng)

    tree_mask = np.zeros(core_mask.shape, dtype=bool)
    for x, z in trees:
        tree_mask[z, x] = True

    bush_candidates = np.argwhere(adjacent_to_path & ~near_intersections & ~tree_mask)
    bush_candidates = [(int(x), int(z)) for z, x in bush_candidates if rng.random() < bush_rate]
    bush_mask = np.zeros(core_mask.shape, dtype=bool)
    for x, z in bush_candidates:
        bush_mask[z, x] = True

    blocked_by_trees = np.zeros(core_mask.shape, dtype=bool)
    for x, z in trees:
        z0, z1 = max(0, z - 2), min(blocked_by_trees.shape[0], z + 3)
        x0, x1 = max(0, x - 2), min(blocked_by_trees.shape[1], x + 3)
        blocked_by_trees[z0:z1, x0:x1] = True

    flower_candidates = open_mask & ~adjacent_to_path & ~blocked_by_trees & ~bush_mask
    flowers = [
        (int(x), int(z))
        for z, x in np.argwhere(flower_candidates)
        if rng.random() < flower_density
    ]

    return trees, bush_candidates, flowers

def deploy_settlement(
    flower_density=0.015,
    bush_rate=0.35,
    tree_interval=11,
    tree_rate=0.7,
    landscaping_seed=42,
    place_debug_frame=True,
    place_placeholders=True,
):
    # Defaults reproduce the narrative pipeline's behaviour (diagnostic frame +
    # placeholder buildings). The prefab pipeline calls with both False to
    # suppress the frame and reserve building footprints for its own prefabs.
    print("🚀 Initializing live settlement generation via GDPC...")
    manager = MapManager()
    if not manager.is_minecraft_available(): return
        
    editor = manager.editor
    sim = load_simulated_data()
    
    blocks = sim['blocks']
    palette = sim['palette']
    origin = sim['origin']
    require_matching_terrain_and_blocks(sim['heightmap'], blocks)
    
    W, H, D = blocks.shape

    # Shadow CELL_SURFACE_BLOCK/FOUNDATION_BLOCK with biome-matched locals; nested placement helpers close over them, so all pads/columns/foundations follow the biome.
    biome = sample_settlement_biome(editor, sim, origin)
    CELL_SURFACE_BLOCK, FOUNDATION_BLOCK = ground_blocks_for_biome(biome)
    print(f"🌍 Biome '{biome}' -> ground surface {CELL_SURFACE_BLOCK}, "
          f"foundation {FOUNDATION_BLOCK}")

    core_indices = load_core_indices()
    if core_indices and max(core_indices) >= len(sim['seeds']):
        raise ValueError(
            "settlement_core.npz is stale for the current settlement_data.npz. "
            "Rerun isolate_buildable_plot() before deploying."
        )
    plots, farms, building_rects, module_size = load_plot_data()
    core_cell_mask = sim['core_cell_mask']
    if core_cell_mask is None:
        core_cell_mask = build_core_cell_mask(sim['seeds'], core_indices, sim['heightmap'])
    path_mask = sim['path_mask']
    if path_mask is None:
        path_mask = infer_path_mask(blocks, palette, sim['heightmap'])
    path_base_y = sim['path_base_y'] if sim['path_base_y'] is not None else sim['heightmap']
    path_slab_mask = sim['path_slab_mask']
    if path_slab_mask is None:
        path_slab_mask = np.zeros_like(path_mask, dtype=bool)
    building_mask = build_module_footprint(plots, module_size, sim['heightmap'])
    building_rect_mask = build_rectangle_footprint(building_rects, sim['heightmap'])
    building_mask |= building_rect_mask
    farm_mask = build_farm_footprint(farms, sim['heightmap'])
    reserved_mask = path_mask | building_mask | farm_mask
    trees, bushes, flowers = plan_landscaping(
        core_cell_mask,
        path_mask,
        reserved_mask,
        flower_density=flower_density,
        bush_rate=bush_rate,
        tree_interval=tree_interval,
        tree_rate=tree_rate,
        seed=landscaping_seed,
    )
    greenery_mask = np.zeros_like(core_cell_mask, dtype=bool)
    for x, z in trees + bushes + flowers:
        greenery_mask[z, x] = True
    target_mask = core_cell_mask | path_mask | building_mask | farm_mask | greenery_mask
    
    print("🔌 Fetching current world state for differential updates...")
    ox, _, oz = origin
    world_slice = editor.loadWorldSlice(Rect((int(ox), int(oz)), (W, D)))
    base_hm = np.array(world_slice.heightmaps["MOTION_BLOCKING"], dtype=int)

    stats = {
        "placed": 0,
        "skipped": 0,
        "cleared": 0,
        "cells": 0,
        "paths": 0,
        "farms": 0,
        "buildings": 0,
        "trees": 0,
        "bushes": 0,
        "flowers": 0,
    }
    placed_blocks = {}

    def existing_block(local_x, y, local_z):
        cached = placed_blocks.get((local_x, y, local_z))
        if cached is not None:
            return cached
        return world_slice.getBlock((local_x, y, local_z)).id.split('[')[0]

    def make_block(block_spec):
        if isinstance(block_spec, tuple):
            block_name, states = block_spec
            return block_name, Block(block_name, states)
        return block_spec, Block(block_spec)

    def place_if_needed(local_x, y, local_z, block_spec):
        if y < 0 or y >= 320:
            return

        block_name, block = make_block(block_spec)
        if existing_block(local_x, y, local_z) != block_name:
            editor.placeBlock(local_to_world(origin, local_x, y, local_z), block)
            placed_blocks[(local_x, y, local_z)] = block_name
            if block_name == AIR_BLOCK:
                stats["cleared"] += 1
            else:
                stats["placed"] += 1
        else:
            stats["skipped"] += 1

    def place_zone_sky_borders(diag_y):
        zone_map = sim['zone_map']
        if zone_map is None:
            print("⚠️ No zone_map found; skipping sky zone borders.")
            return

        zone_map = np.asarray(zone_map)
        if zone_map.shape != sim['heightmap'].shape:
            print("⚠️ zone_map shape does not match heightmap; skipping sky zone borders.")
            return

        for z in range(D):
            for x in range(W):
                zone_id = int(zone_map[z, x])
                if zone_id < 0:
                    continue

                is_border = False
                for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, nz = x + dx, z + dz
                    neighbor_zone = -1
                    if 0 <= nx < W and 0 <= nz < D:
                        neighbor_zone = int(zone_map[nz, nx])
                    if neighbor_zone != zone_id:
                        is_border = True
                        break

                if is_border:
                    place_if_needed(x, diag_y, z, zone_border_block(zone_id))

    def surface_material(local_x, local_z, fallback):
        y = int(sim['heightmap'][local_z, local_x])
        if y >= H:
            return fallback

        block_idx = int(blocks[local_x, y, local_z])
        if block_idx == 0:
            return fallback
        block = palette[block_idx]
        # Suppress house-footprint markers so reserved building plots read as
        # plain ground (prefab/placeholder buildings fill them in Phase 2D).
        if block in HOUSE_MARKER_BLOCKS:
            return fallback
        # Remap generic ground (grass/dirt) to the biome surface so the open area matches; non-ground palette blocks pass through.
        if block in REMAPPABLE_GROUND_SURFACES:
            return CELL_SURFACE_BLOCK
        return block

    def clear_above_surface(local_x, local_z, extra_air=4):
        target_y = int(sim['heightmap'][local_z, local_x])
        top_y = int(base_hm[local_x, local_z])

        for y in range(top_y, target_y, -1):
            place_if_needed(local_x, y, local_z, AIR_BLOCK)
        for y in range(target_y + 1, target_y + extra_air + 1):
            place_if_needed(local_x, y, local_z, AIR_BLOCK)

    def place_terrain_column(local_x, local_z, surface_block):
        surface_y = int(sim['heightmap'][local_z, local_x])
        min_neighbor_y = surface_y
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, nz = local_x + dx, local_z + dz
            if 0 <= nx < W and 0 <= nz < D:
                min_neighbor_y = min(min_neighbor_y, int(sim['heightmap'][nz, nx]))

        fill_bottom = max(0, min_neighbor_y - 3)
        for y in range(fill_bottom, surface_y):
            place_if_needed(local_x, y, local_z, FOUNDATION_BLOCK)
        place_if_needed(local_x, surface_y, local_z, surface_block)

    def place_path_column(local_x, local_z):
        surface_y = int(path_base_y[local_z, local_x])
        min_neighbor_y = surface_y
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, nz = local_x + dx, local_z + dz
            if 0 <= nx < W and 0 <= nz < D:
                if path_mask[nz, nx]:
                    min_neighbor_y = min(min_neighbor_y, int(path_base_y[nz, nx]))
                else:
                    min_neighbor_y = min(min_neighbor_y, int(sim['heightmap'][nz, nx]))

        fill_bottom = max(0, min_neighbor_y - 3)
        for y in range(fill_bottom, surface_y):
            place_if_needed(local_x, y, local_z, FOUNDATION_BLOCK)

        path_block = path_block_for(local_x, local_z)
        place_if_needed(local_x, surface_y, local_z, path_block)
        if path_slab_mask[local_z, local_x]:
            place_if_needed(local_x, surface_y + 1, local_z, (path_slab_for(local_x, local_z), {"type": "bottom"}))

    def iter_mask(mask):
        for local_z, local_x in np.argwhere(mask):
            yield int(local_x), int(local_z)

    def farm_layout(cells):
        cell_set = {(int(x), int(z)) for x, z in cells}
        if not cell_set:
            return cell_set, set(), set(), set()

        xs = [x for x, _ in cell_set]
        zs = [z for _, z in cell_set]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        width = max_x - min_x + 1
        depth = max_z - min_z + 1

        border = set()
        for x, z in cell_set:
            for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (x + dx, z + dz) not in cell_set:
                    border.add((x, z))
                    break

        water = set()
        if width >= depth:
            center_z = (min_z + max_z) // 2
            water = {(x, center_z) for x in range(min_x, max_x + 1) if (x, center_z) in cell_set}
        else:
            center_x = (min_x + max_x) // 2
            water = {(center_x, z) for z in range(min_z, max_z + 1) if (center_x, z) in cell_set}

        crop_land = cell_set - border - water
        return cell_set, border, water, crop_land

    def place_tree(local_x, local_z):
        base_y = int(sim['heightmap'][local_z, local_x])
        trunk_height = 4 + ((local_x * 3 + local_z) % 2)
        place_terrain_column(local_x, local_z, CELL_SURFACE_BLOCK)

        for y in range(base_y + 1, base_y + trunk_height + 1):
            place_if_needed(local_x, y, local_z, TREE_TRUNK_BLOCK)

        canopy_y = base_y + trunk_height
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                dist = abs(dx) + abs(dz)
                if dist > 3:
                    continue
                x, z = local_x + dx, local_z + dz
                if not (0 <= x < W and 0 <= z < D):
                    continue
                for dy in range(0, 3):
                    if dist + dy > 4:
                        continue
                    place_if_needed(x, canopy_y + dy, z, TREE_LEAF_BLOCK)

    def place_bush(local_x, local_z):
        base_y = int(sim['heightmap'][local_z, local_x])
        place_terrain_column(local_x, local_z, CELL_SURFACE_BLOCK)
        place_if_needed(local_x, base_y + 1, local_z, BUSH_BLOCK)

    def place_flower(local_x, local_z):
        base_y = int(sim['heightmap'][local_z, local_x])
        place_terrain_column(local_x, local_z, CELL_SURFACE_BLOCK)
        place_if_needed(local_x, base_y + 1, local_z, flower_for(local_x, local_z))

    def place_bounded_placeholder_building(rect):
        x0 = max(0, int(rect["x"]))
        z0 = max(0, int(rect["z"]))
        x1 = min(W, x0 + int(rect["width"]))
        z1 = min(D, z0 + int(rect["depth"]))
        if x1 <= x0 or z1 <= z0:
            return

        placed_any = False

        for x in range(x0, x1):
            for z in range(z0, z1):
                place_terrain_column(x, z, FOUNDATION_BLOCK)
                surface_y = int(sim['heightmap'][z, x])
                for y in range(surface_y + 1, H):
                    block_idx = int(blocks[x, y, z])
                    if block_idx == 0:
                        continue
                    place_if_needed(x, y, z, palette[block_idx])
                    placed_any = True

        return placed_any

    def place_legacy_module_building(mx, mz, floor_count):
        base_y = int(sim['heightmap'][mz, mx])
        fallback_materials = [
            "minecraft:oak_planks",
            "minecraft:terracotta",
            "minecraft:yellow_concrete",
        ]

        for floor in range(floor_count):
            y0 = base_y + 1 + floor * module_size
            y1 = min(320, y0 + module_size)
            material = fallback_materials[min(floor, len(fallback_materials) - 1)]
            for local_x in range(max(0, mx), min(W, mx + module_size)):
                for local_z in range(max(0, mz), min(D, mz + module_size)):
                    for y in range(y0, y1):
                        place_if_needed(local_x, y, local_z, material)

    def tree_has_clearance(local_x, local_z):
        base_y = int(sim['heightmap'][local_z, local_x])
        trunk_height = 4 + ((local_x * 3 + local_z) % 2)

        for dx in range(-2, 3):
            for dz in range(-2, 3):
                x, z = local_x + dx, local_z + dz
                if not (0 <= x < W and 0 <= z < D):
                    return False
                if abs(int(sim['heightmap'][z, x]) - base_y) > 2:
                    return False

        for y in range(base_y + 1, base_y + trunk_height + 4):
            if existing_block(local_x, y, local_z) != AIR_BLOCK:
                return False

        canopy_y = base_y + trunk_height
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                dist = abs(dx) + abs(dz)
                if dist > 3:
                    continue
                x, z = local_x + dx, local_z + dz
                for dy in range(0, 3):
                    if dist + dy > 4:
                        continue
                    block = existing_block(x, canopy_y + dy, z)
                    if block not in (AIR_BLOCK, "minecraft:grass", "minecraft:tall_grass"):
                        return False

        return True

    editor.buffering = True 

    try:
        diag_y = 120
        if place_debug_frame:
            print("📍 Phase 0: Deploying diagnostic Sky-Frame at Y=120...")
            editor.placeBlock(local_to_world(origin, 0, diag_y, 0), Block("minecraft:emerald_block"))
            editor.placeBlock(local_to_world(origin, W - 1, diag_y, 0), Block("minecraft:redstone_block"))
            editor.placeBlock(local_to_world(origin, 0, diag_y, D - 1), Block("minecraft:lapis_block"))
            editor.placeBlock(local_to_world(origin, W - 1, diag_y, D - 1), Block("minecraft:gold_block"))
            for x in range(1, W - 1):
                editor.placeBlock(local_to_world(origin, x, diag_y, 0), Block("minecraft:red_stained_glass"))
                editor.placeBlock(local_to_world(origin, x, diag_y, D - 1), Block("minecraft:white_stained_glass"))
            for z in range(1, D - 1):
                editor.placeBlock(local_to_world(origin, 0, diag_y, z), Block("minecraft:blue_stained_glass"))
                editor.placeBlock(local_to_world(origin, W - 1, diag_y, z), Block("minecraft:white_stained_glass"))
            print("🗺️ Phase 0B: Drawing sky-level zone borders...")
            place_zone_sky_borders(diag_y)

        print(f"🧹 Phase 1: Clearing target settlement footprint...")
        for local_x, local_z in iter_mask(target_mask):
            clear_above_surface(local_x, local_z)

        print("🏗️ Phase 2A: Placing terraced Voronoi cells...")
        cell_mask = core_cell_mask & ~path_mask
        for local_x, local_z in iter_mask(cell_mask):
            place_terrain_column(
                local_x,
                local_z,
                surface_material(local_x, local_z, CELL_SURFACE_BLOCK),
            )
            stats["cells"] += 1

        print("🛤️ Phase 2B: Placing paths...")
        for local_x, local_z in iter_mask(path_mask):
            place_path_column(local_x, local_z)
            stats["paths"] += 1

        if BUILD_FARM_FIELDS:  # CHANGED FOR NARRATIVE: see flag definition above
            print("🌾 Phase 2C: Placing farms...")
            for farm_cells in farms.values():
                _, border_cells, water_cells, crop_cells = farm_layout(farm_cells)

                for local_x, local_z in border_cells:
                    y = int(sim['heightmap'][local_z, local_x])
                    place_terrain_column(local_x, local_z, FOUNDATION_BLOCK)
                    place_if_needed(local_x, y + 1, local_z, FARM_BORDER_BLOCK)
                    stats["farms"] += 1

                for local_x, local_z in water_cells:
                    y = int(sim['heightmap'][local_z, local_x])
                    place_terrain_column(local_x, local_z, FOUNDATION_BLOCK)
                    place_if_needed(local_x, y, local_z, FARM_WATER_BLOCK)
                    stats["farms"] += 1

                for local_x, local_z in crop_cells:
                    y = int(sim['heightmap'][local_z, local_x])
                    place_terrain_column(local_x, local_z, FARM_SOIL_BLOCK)
                    place_if_needed(local_x, y + 1, local_z, crop_for(local_x, local_z))
                    stats["farms"] += 1
        else:
            print("🌾 Phase 2C: Farm fields skipped (BUILD_FARM_FIELDS=False; "
                  "narrative layer renders them).")

        if place_placeholders:
            print("🏠 Phase 2D: Placing placeholder buildings in largest cell rectangles...")
        else:
            print("🏠 Phase 2D: Reserving building footprints for prefab placement...")
        if place_placeholders and building_rects:
            for rect in building_rects.values():
                place_bounded_placeholder_building(rect)
                stats["buildings"] += 1
        elif place_placeholders:
            module_floor_maps = compute_module_floors(plots, module_size)
            for floor_map in module_floor_maps.values():
                for (mx, mz), floor_count in floor_map.items():
                    place_legacy_module_building(mx, mz, floor_count)
                    stats["buildings"] += 1
        else:
            stats["buildings"] += len(building_rects)

        print("🌳 Phase 2E: Placing trees, bushes, and flowers...")
        for local_x, local_z in trees:
            if not tree_has_clearance(local_x, local_z):
                continue
            place_tree(local_x, local_z)
            stats["trees"] += 1

        for local_x, local_z in bushes:
            place_bush(local_x, local_z)
            stats["bushes"] += 1

        for local_x, local_z in flowers:
            place_flower(local_x, local_z)
            stats["flowers"] += 1

        print(
            "📊 Smart Build Stats: "
            f"{stats['cells']} cell columns, {stats['paths']} path columns, "
            f"{stats['farms']} farm columns, {stats['buildings']} modules. "
            f"{stats['trees']} trees, {stats['bushes']} bushes, {stats['flowers']} flowers. "
            f"Placed {stats['placed']} blocks, cleared {stats['cleared']}, "
            f"skipped {stats['skipped']}."
        )

    finally:
        print("📥 Flushing block placement buffers to server...")
        editor.flushBuffer()
        editor.buffering = False

    print("🎉 Generation Completed successfully!")

if __name__ == "__main__":
    deploy_settlement()
