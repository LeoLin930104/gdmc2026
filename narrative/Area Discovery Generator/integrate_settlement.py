import sys
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so Unicode symbols print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow running from this directory without installing.
sys.path.insert(0, str(Path(__file__).parent))
# Put LLM Narrative on the path so we can generate the settlement here directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "LLM Narrative"))

import numpy as np

from settlement_generator import generate_settlement
from settlement_goal import generate_settlement_goal
from shared_events import generate_shared_events
from district_namer import generate_districts
from biome_context import biome_hint

from area_discovery_gen import (
    GeneratorConfig,
    DatapackGenerator,
    aabb_from_zone_map,
    zone_descriptors_from_zone_map,
    zone_from_aabb,
)


# --- config ---------------------------------------------------------------
MINECRAFT_WORLD = "New World"
SETTLEMENT_THEME = "Fantasy"

# Where the gdmc2026 generator writes its output (it uses a relative 'data/'
# path, so the .npz lands inside the generator's own folder).
# In <repo>/narrative/Area Discovery Generator/, so the generator's data/ is two levels up (the repo root).
_DEFAULT_NPZ = (
    Path(__file__).parent.parent.parent / "data" / "settlement_data.npz"
)


def _find_settlement_npz() -> Path:
    if _DEFAULT_NPZ.exists():
        return _DEFAULT_NPZ
    # Fall back to a recursive search in case the folder layout differs.
    root = Path(__file__).parent.parent.parent
    matches = list(root.glob("**/data/settlement_data.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find settlement_data.npz. Run the area generator first:\n"
        "  python main.py   (from the repo root)\n"
        f"(looked for {_DEFAULT_NPZ})"
    )


def _find_datapack_dir(world_name: str = "New World") -> Path:
    import platform
    home = Path.home()
    if platform.system() == "Darwin":
        app_support = home / "Library" / "Application Support"
        candidates = [
            app_support / "ModrinthApp" / "profiles" / "GDMC" / "saves",
            app_support / "minecraft" / "saves",
        ]
    else:
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
    print(f"[warn] Could not find Minecraft world '{world_name}'. "
          f"Writing to {fallback.resolve()} instead.")
    return fallback


def _detect_biome() -> str | None:
    try:
        from gdpc import Editor
        from biome_context import sample_biome_at_player
        biome, _player_pos = sample_biome_at_player(Editor())
        return biome
    except Exception as exc:
        print(f"[info] biome detection skipped ({exc!r}); generating without biome grounding.")
        return None


def main(
    world: str = MINECRAFT_WORLD,
    theme: str = SETTLEMENT_THEME,
    npz: str | None = None,
    settlement=None,
    biome: str | None = None,
) -> None:
    npz_path = Path(npz) if npz else _find_settlement_npz()
    if npz and not npz_path.exists():
        raise FileNotFoundError(f"--npz path does not exist: {npz_path}")
    print(f"Loading settlement layout: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    for key in ("zone_map", "origin", "heightmap"):
        if key not in data.files:
            raise KeyError(
                f"settlement_data.npz is missing '{key}'. Run the full generator "
                "pipeline (through generate_zones) before integrating."
            )
    zone_map = data["zone_map"]
    origin = data["origin"]
    heightmap = data["heightmap"]

    descriptors = zone_descriptors_from_zone_map(zone_map, origin)
    if not descriptors:
        raise RuntimeError(
            "zone_map has no zones (all cells are -1). Did generate_zones() run?"
        )
    print(f"Found {len(descriptors)} settlement zones at origin "
          f"({int(origin[0])}, {int(origin[2])}).")

    # --- narrative identity (one shared instance, threaded everywhere) ----
    if settlement is None:
        biome = biome or _detect_biome()
        if biome:
            print(f"Biome: {biome_hint(biome)}")
        settlement = generate_settlement(theme, biome=biome)
        print(f"Settlement: {settlement.name} — {settlement.era}")
        # Pre-passes, mutated onto the SAME instance (goal first, per CLAUDE.md).
        settlement.goal = generate_settlement_goal(settlement)
        settlement.shared_events = generate_shared_events(settlement)
    else:
        biome = biome or getattr(settlement, "biome", None)
        print(f"Settlement (shared): {settlement.name} — "
              f"{getattr(settlement, 'era', '?')}")

    # --- name the districts -----------------------------------------------
    districts = generate_districts(settlement, descriptors, biome=biome)

    zones = []
    for d in districts:
        zid = d["zone_index"]
        aabb = aabb_from_zone_map(zone_map, zid, origin, heightmap)
        zone = zone_from_aabb(
            zone_id      = f"district_{zid}",
            display_name = d["name"],
            aabb         = aabb,
            preset       = d["preset"],          # styling
            settlement   = settlement,           # subtitle auto-generated
        )
        zones.append(zone)
        print(f"  zone {zid}: {d['name']!r} [{d['preset']}] "
              f"@ {aabb.x},{aabb.y},{aabb.z} ({aabb.width}x{aabb.height}x{aabb.depth})")

    # --- write the datapack -----------------------------------------------
    config = GeneratorConfig(
        namespace        = "area_discovery",
        pack_description = f"§6{settlement.name}§r — Area Discovery",
        output_dir       = _find_datapack_dir(world),
        overwrite        = True,
    )
    gen = DatapackGenerator(config)
    gen.add_zones(zones)
    output_path = gen.generate()
    print(f"\nDatapack written to: {output_path}")
    gen.summary()
    print(
        "\nIn-game: /reload, then run once: /function area_discovery:setup\n"
        "Then walk between districts to see the titles."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wire the narrative datapack onto a gdmc2026 settlement layout."
    )
    parser.add_argument(
        "--world", default=MINECRAFT_WORLD,
        help=f"Minecraft world/save name to write the datapack into (default: {MINECRAFT_WORLD!r}).",
    )
    parser.add_argument(
        "--theme", default=SETTLEMENT_THEME,
        help=f"Settlement theme passed to generate_settlement (default: {SETTLEMENT_THEME!r}).",
    )
    parser.add_argument(
        "--npz", default=None,
        help="Path to settlement_data.npz (default: auto-locate under the repo's data/).",
    )
    args = parser.parse_args()
    main(world=args.world, theme=args.theme, npz=args.npz)
