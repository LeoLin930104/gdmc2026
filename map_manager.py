import os
import numpy as np
from gdpc import Editor, WorldSlice, Rect

# File path constants
DATA_DIR = "data"
SETTLEMENT_DATA_PATH = os.path.join(DATA_DIR, "data.npz")

# Solid ground layers that are allowed to define the baseline terrain
GROUND_BLOCKS = {
    "minecraft:grass_block", "minecraft:dirt", "minecraft:stone",
    "minecraft:sand", "minecraft:gravel", "minecraft:snow_block",
    "minecraft:clay", "minecraft:podzol", "minecraft:coarse_dirt",
    "minecraft:mud", "minecraft:rooted_dirt", "minecraft:mycelium",
    "minecraft:deepslate", "minecraft:calcite", "minecraft:tuff"
}

# Tree trunks and structural wood to be filtered out
TREE_BLOCKS = {
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
    "minecraft:mangrove_log", "minecraft:cherry_log", "minecraft:bamboo_block",
    "minecraft:mangrove_roots", "minecraft:stripped_oak_log", 
    "minecraft:stripped_birch_log", "minecraft:stripped_spruce_log",
    "minecraft:stripped_jungle_log", "minecraft:stripped_acacia_log", 
    "minecraft:stripped_dark_oak_log", "minecraft:stripped_mangrove_log", 
    "minecraft:stripped_cherry_log"
}

# Leaves and hanging canopy
LEAF_BLOCKS = {
    "minecraft:oak_leaves", "minecraft:birch_leaves", "minecraft:spruce_leaves",
    "minecraft:jungle_leaves", "minecraft:acacia_leaves", "minecraft:dark_oak_leaves",
    "minecraft:mangrove_leaves", "minecraft:cherry_leaves", "minecraft:azalea_leaves",
    "minecraft:flowering_azalea_leaves", "minecraft:bamboo_leaves"
}

# New: Explicit ephemeral foliage, grasses, and decorations to ignore
IGNORE_DECORATIONS = {
    "minecraft:grass", "minecraft:tall_grass", "minecraft:fern", "minecraft:large_fern",
    "minecraft:dandelion", "minecraft:poppy", "minecraft:blue_orchid", "minecraft:allium",
    "minecraft:azure_bluet", "minecraft:red_tulip", "minecraft:orange_tulip", "minecraft:white_tulip",
    "minecraft:pink_tulip", "minecraft:oxeye_daisy", "minecraft:cornflower", "minecraft:lily_of_the_valley",
    "minecraft:wither_rose", "minecraft:sunflower", "minecraft:lilac", "minecraft:rose_bush", "minecraft:peony",
    "minecraft:moss_carpet", "minecraft:snow", "minecraft:dead_bush", "minecraft:brown_mushroom", "minecraft:red_mushroom",
    "minecraft:sugar_cane", "minecraft:vines", "minecraft:glow_lichen"
}

WATER_BLOCKS = {"minecraft:water", "minecraft:flowing_water"}

# Test Coordinates
#(128, 128, 128)
#(128, 128, -128)
#(384, 128, 128)
#(384, 128, -128)

class MapManager:
    def __init__(self, area_size: int = 256, default_center=(384, 128, 128)):
        self.editor = Editor()
        self.area_size = area_size
        self.default_center = default_center

        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)

    def is_minecraft_available(self) -> bool:
        """Checks connection integrity with the Minecraft server instance."""
        try:
            self.editor.checkConnection()
            return True
        except Exception:
            return False

    def fetch_live_world_slice(
        self,
    ) -> tuple[WorldSlice, tuple[int, int, int, int]]:
        """
        Calculates player/default center and pulls chunk-aligned raw slice.
        Returns:
            (world_slice, (x1, z1, x2, z2))
        """
        try:
            center = self.editor.getPlayerPos()
            if center is None:
                center = self.default_center
        except Exception:
            center = self.default_center

        cx, _, cz = center
        half = self.area_size // 2

        # Align to chunk boundaries (16x16)
        x1 = (int(cx - half) // 16) * 16
        z1 = (int(cz - half) // 16) * 16
        x2 = x1 + self.area_size
        z2 = z1 + self.area_size

        # GDPC Rect usually expects (offset, size)
        rect = Rect((x1, z1), (self.area_size, self.area_size))

        print(f"🔌 Loading World Slice from bounds: {rect}")

        world_slice = self.editor.loadWorldSlice(rect)

        return world_slice, (x1, z1, x2, z2)

    def extract_and_orient_maps(self, world_slice: WorldSlice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes block columns into terrain arrays stored consistently as [z, x]."""
        # Use standard height limit safely as a maximum starting point
        base_hm = np.array(world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"], dtype=int)
        width, depth = base_hm.shape

        hm = np.zeros((depth, width), dtype=base_hm.dtype)
        tree_map = np.zeros((depth, width), dtype=bool)
        water_map = np.zeros((depth, width), dtype=bool)

        for x in range(width):
            for z in range(depth):
                # Start at the raw surface height and drop downward until finding true ground
                top_y = base_hm[x, z]
                found_ground = False

                for y in range(top_y, 0, -1):
                    block = world_slice.getBlock((x, y, z))
                    bid = block.id

                    # 1. Skip over air or passing structural fluids instantly
                    if bid.endswith("air"):
                        continue

                    # 2. Handle canopy footprints
                    if bid in LEAF_BLOCKS or bid in TREE_BLOCKS:
                        tree_map[z, x] = True
                        continue # Bypass logs entirely so we look underneath them

                    # 3. Skip wild decorations, carpets, and temporary plant layers
                    if bid in IGNORE_DECORATIONS:
                        continue # Keep falling down the column

                    # 4. Track passing water tables without stopping the ground check
                    if bid in WATER_BLOCKS:
                        water_map[z, x] = True
                        continue 

                    # 5. Lock onto certified solid ground
                    if bid in GROUND_BLOCKS:
                        hm[z, x] = y
                        found_ground = True
                        break

                    # 6. Fallback fallback safety mechanism
                    if not found_ground:
                        hm[z, x] = y
                        found_ground = True
                        break

        return hm, tree_map, water_map

    def compute_slopes(
        self,
        heightmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates slope steepness matrices and corresponding flat masks.
        """

        dz = np.gradient(heightmap, axis=0)
        dx = np.gradient(heightmap, axis=1)

        slope = np.sqrt(dx**2 + dz**2)

        flat_mask = slope < 2

        return slope, flat_mask

    def load_environment_dataset(self, force_refresh: bool = False) -> dict:
        """
        Master retrieval command.
        Pulls fresh matrices from Minecraft if online (saving it to cache), 
        otherwise gracefully pulls from local compressed data storage arrays.
        """
        if self.is_minecraft_available() and not force_refresh:
            print("✅ Minecraft detected. Parsing live environment context...")
            world_slice, bounds = self.fetch_live_world_slice()
            hm, tree_map, water_map = self.extract_and_orient_maps(world_slice)
            slope, flat_mask = self.compute_slopes(hm)

            # Preserve coordinates relative to server map location
            origin = np.array([bounds[0], 0, bounds[1]])

            # Pack local cache archive
            np.savez_compressed(
                SETTLEMENT_DATA_PATH,
                seeds=np.array([]), 
                heightmap=hm,
                slope=slope,
                flat_mask=flat_mask,
                tree_map=tree_map,
                water_map=water_map,
                origin=origin,
                num_drift=0
            )
            print(f"💾 Fresh map context cached to: {SETTLEMENT_DATA_PATH}")
            
            return {
                "heightmap": hm,
                "slope": slope,
                "flat_mask": flat_mask,
                "tree_map": tree_map,
                "water_map": water_map,
                "origin": origin
            }

        else:
            print("⚠️ Minecraft server unavailable or bypass active. Opening backup archives...")
            if not os.path.exists(SETTLEMENT_DATA_PATH):
                raise FileNotFoundError(
                    f"Fatal error: No cached file array found at {SETTLEMENT_DATA_PATH}. "
                    "Start Minecraft client instance once to initialize the map baseline data files."
                )
            
            loaded = np.load(SETTLEMENT_DATA_PATH, allow_pickle=True)
            print(f"📖 Loaded environment snapshot matrix from: {SETTLEMENT_DATA_PATH}")
            
            # Extract heightmap from cache
            hm = loaded["heightmap"]
            
            # Look for slope and flat_mask; if missing from old cache, calculate on the fly!
            if "slope" in loaded and "flat_mask" in loaded:
                slope = loaded["slope"]
                flat_mask = loaded["flat_mask"]
            else:
                print("🔄 Legacy cache detected (missing slope maps). Computing on the fly...")
                slope, flat_mask = self.compute_slopes(hm)
            
            return {
                "heightmap": hm,
                "slope": slope,
                "flat_mask": flat_mask,
                "tree_map": loaded["tree_map"],
                "water_map": loaded["water_map"],
                "origin": loaded["origin"] if "origin" in loaded else np.array([0, 0, 0])
            }

# Quick validation runtime execution block
def load_current_map():

    manager = MapManager()

    dataset = manager.load_environment_dataset()

    print(
        "Dataset confirmation keys available:",
        list(dataset.keys()),
    )
