import sys
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so Unicode symbols print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow running from this directory without installing
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Resolve the target datapack output directory
# ---------------------------------------------------------------------------

def _find_datapack_dir(world_name: str = "New World") -> Path:
    import platform
    home = Path.home()
    if platform.system() == "Darwin":  # macOS
        app_support = home / "Library" / "Application Support"
        candidates = [
            app_support / "ModrinthApp" / "profiles" / "GDMC" / "saves",
            app_support / "minecraft" / "saves",
        ]
    else:  # Windows (and Linux fallback)
        appdata = home / "AppData" / "Roaming"
        candidates = [
            appdata / "ModrinthApp" / "profiles" / "GDMC" / "saves",
            appdata / ".minecraft" / "saves",
        ]
    for saves in candidates:
        world = saves / world_name
        if world.exists():
            return world / "datapacks" / "area_discovery"

    fallback = Path("./generated_datapack")
    print(f"[warn] Could not find Minecraft world '{world_name}' in any known location. "
          f"Writing to {fallback.resolve()} instead.")
    return fallback


# Change this if your world has a different name
MINECRAFT_WORLD = "New World"

from area_discovery_gen import (
    AABB,
    MCColor,
    SoundConfig,
    TitleConfig,
    Zone,
    GeneratorConfig,
    DatapackGenerator,
    zone_from_corners,
    aabb_from_footprint,
    aabb_from_square_footprint,
)


# =============================================================================
# METHOD 1 — Convenience helper (recommended for most GDMC use cases)
#            Provide two world-space corners + a named style preset.
# =============================================================================

town_center = zone_from_corners(
    zone_id      = "town_center",
    display_name = "Town Center",
    subtitle     = "A bustling hub of trade and adventure",
    x1=-50, y1=60, z1=-50,
    x2= 50, y2=99, z2= 50,
    preset       = "town",
    notes        = "Main settlement plaza. Expand y2 if buildings grow taller.",
)

ancient_ruins = zone_from_corners(
    zone_id      = "ancient_ruins",
    display_name = "Ancient Ruins",
    subtitle     = "Something stirs in the deep...",
    x1=200, y1=40, z1=150,
    x2=350, y2=99, z2=230,
    preset       = "ruins",
)

dark_forest = zone_from_corners(
    zone_id      = "dark_forest",
    display_name = "The Dark Forest",
    subtitle     = "Tread carefully — you are not alone",
    x1=-300, y1=60, z1=100,
    x2=-100, y2=120, z2=400,
    preset       = "nature",
    enabled      = False,   # ← disabled: won't appear in the datapack yet
    notes        = "Algorithm not finalised — disable until boundaries are confirmed.",
)


# =============================================================================
# METHOD 2 — Direct Zone + TitleConfig construction (full control)
# =============================================================================

market_district = Zone(
    zone_id      = "market_district",
    display_name = "Market District",
    aabb         = AABB.from_corners(
        -10, 60, 60,
         60, 90, 140,
    ),
    title = TitleConfig(
        main_title   = "Market District",
        subtitle     = "Coin and commerce await",
        main_color   = MCColor.GOLD,
        sub_color    = MCColor.YELLOW,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "✦ ",
        prefix_color = MCColor.WHITE,
        fade_in      = 15,
        stay         = 70,
        fade_out     = 15,
    ),
    sound = SoundConfig(
        sound_id = "minecraft:block.note_block.harp",
        volume   = 0.9,
        pitch    = 1.4,
    ),
)


# =============================================================================
# METHOD 3 — AABB built from a square footprint (useful for circular/radial
#            layout algorithms that output a center point + radius)
# =============================================================================

harbor = Zone(
    zone_id      = "harbor",
    display_name = "The Harbor",
    aabb         = aabb_from_square_footprint(
        cx=500, cz=500,   # center of the harbor area
        radius=75,        # 151×_×151 footprint
        y_min=55,
        y_max=100,
    ),
    title = TitleConfig(
        main_title   = "The Harbor",
        subtitle     = "Salt air and the sound of gulls",
        main_color   = MCColor.AQUA,
        sub_color    = MCColor.WHITE,
        sub_italic   = True,
        prefix       = "⚓ ",
        prefix_color = MCColor.DARK_AQUA,
    ),
    sound = SoundConfig(
        "minecraft:ambient.underwater.enter",
        volume=0.7, pitch=1.0,
    ),
)


# =============================================================================
# Generate the datapack
# =============================================================================

def main() -> None:
    config = GeneratorConfig(
        namespace        = "area_discovery",
        pack_description = "§6Area Discovery§r — GDMC zone title system",
        pack_format      = 26,
        output_dir       = _find_datapack_dir(MINECRAFT_WORLD),
        overwrite        = True,
        write_templates  = True,
    )

    gen = DatapackGenerator(config)
    gen.add_zones([
        town_center,
        ancient_ruins,
        dark_forest,      # disabled — will be listed in summary but not generated
        market_district,
        harbor,
    ])

    output_path = gen.generate()
    gen.summary()

    print(f"\nDatapack written to: {output_path.resolve()}")
    print("Then run in-game:  /function area_discovery:setup")


if __name__ == "__main__":
    main()
