import os
import math
import re
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

# Narrative default capture centre, used as the last-resort fallback when no
# explicit region/center/origin is provided and no player position is readable.
NARRATIVE_DEFAULT_CENTER = (384, 128, 128)


def _parse_env_int_tuple(name, expected_len):
    raw = os.environ.get(name)
    if not raw:
        return None
    parts = [part.strip() for part in raw.replace(",", " ").split()]
    if len(parts) != expected_len:
        raise ValueError(
            f"{name} must contain {expected_len} integer values; got {raw!r}"
        )
    return tuple(int(part) for part in parts)


def _normalise_host(host):
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}"


def _parse_player_pose(data):
    pos_match = re.search(r"Pos:\[([^\]]+)\]", data)
    if not pos_match:
        return None
    pos_values = [part.strip().rstrip("dD") for part in pos_match.group(1).split(",")]
    if len(pos_values) < 3:
        return None
    return (
        int(math.floor(float(pos_values[0]))),
        int(math.floor(float(pos_values[1]))),
        int(math.floor(float(pos_values[2]))),
    )


def _get_player_position_from_http(host):
    import requests

    response = requests.get(
        f"{_normalise_host(host).rstrip('/')}/players",
        params={"includeData": "true"},
        timeout=1.0,
    )
    response.raise_for_status()
    players = response.json()
    if not players:
        raise RuntimeError("GDMC server returned no players")
    pose = _parse_player_pose(players[0].get("data", ""))
    if pose is None:
        raise RuntimeError("could not parse player position from GDMC player data")
    return pose


class MapManager:
    def __init__(
        self,
        area_size: int | None = None,
        region_center=None,
        region_origin=None,
        default_center=None,
        host=None,
    ):
        if host is None:
            self.editor = Editor()
        else:
            self.editor = Editor(host=_normalise_host(host))
        # Defaults preserve the narrative pipeline's original behaviour
        # (256-wide capture centred on (384, 128, 128)) while still allowing
        # the prefab pipeline to override via args or GDMC_* env vars.
        self.area_size = int(area_size or os.environ.get("GDMC_AREA_SIZE", 256))
        self.region_center = region_center or _parse_env_int_tuple("GDMC_REGION_CENTER", 3)
        self.region_origin = region_origin or _parse_env_int_tuple("GDMC_REGION_ORIGIN", 2)
        self.default_center = (
            default_center
            or _parse_env_int_tuple("GDMC_DEFAULT_CENTER", 3)
            or NARRATIVE_DEFAULT_CENTER
        )

        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)

    def is_minecraft_available(self) -> bool:
        """Checks connection integrity with the Minecraft server instance."""
        try:
            self.editor.checkConnection()
            return True
        except Exception:
            return False

    def resolve_center(self):
        """Resolve the capture centre from override, player position, or explicit fallback."""
        if self.region_center is not None:
            return tuple(int(value) for value in self.region_center)
        if self.region_origin is not None:
            ox, oz = (int(value) for value in self.region_origin)
            half = self.area_size // 2
            return (ox + half, 0, oz + half)

        try:
            get_player_pos = getattr(self.editor, "getPlayerPos", None)
            if callable(get_player_pos):
                center = get_player_pos()
                if center is not None:
                    return tuple(int(value) for value in center)
            return _get_player_position_from_http(self.editor.host)
        except Exception as exc:
            player_error = exc
        else:
            player_error = None

        if self.default_center is not None:
            return tuple(int(value) for value in self.default_center)

        raise RuntimeError(
            "No capture region is available. Join the Minecraft world so the "
            "player position can be read, or provide GDMC_REGION_CENTER / "
            "GDMC_REGION_ORIGIN, or pass --region-center / --region-origin "
            "through the wrapper script."
        ) from player_error

    def fetch_live_world_slice(
        self,
        center=None,
        origin=None,
    ) -> tuple[WorldSlice, tuple[int, int, int, int]]:
        """
        Calculates the requested region and pulls a raw slice.
        Returns:
            (world_slice, (x1, z1, x2, z2))
        """
        if origin is not None:
            x1, z1 = (int(value) for value in origin)
        else:
            if center is None:
                center = self.resolve_center()

            cx, _, cz = center
            half = self.area_size // 2

            # Align player/centre driven captures to chunk boundaries (16x16).
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

    def load_environment_dataset(self, force_refresh: bool = False, center=None, origin=None) -> dict:
        """
        Master retrieval command.
        Pulls fresh matrices from Minecraft if online (saving it to cache),
        otherwise gracefully pulls from local compressed data storage arrays.
        """
        if self.is_minecraft_available() and not force_refresh:
            print("✅ Minecraft detected. Parsing live environment context...")
            world_slice, bounds = self.fetch_live_world_slice(center=center, origin=origin)
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
