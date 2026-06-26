import sys
import time
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so Unicode symbols print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow running from this directory without installing
sys.path.insert(0, str(Path(__file__).parent))

# Put LLM Narrative on the path so we can generate_settlement() here directly.
# (The lazy import inside gdmc_bridge would also handle this, but we need the
# module up front for the settlement itself.)
sys.path.insert(0, str(Path(__file__).parent.parent / "LLM Narrative"))

# Item Relic Generator hosts the chest builders we reuse for the relic step.
sys.path.insert(0, str(Path(__file__).parent.parent / "Item Relic Generator"))

from biome_context import biome_hint, sample_biome_at_player
from diary_generator import generate_diaries
from settlement_generator import generate_settlement
from settlement_goal import generate_settlement_goal
from shared_events import generate_shared_events
from tool_generator import generate_tools
from place_diary_lectern import (
    build_lectern_snbt,
    build_tool_chest_snbt,
    glint_for_rarity,
    match_diaries_to_zones,
    match_tools_to_zones,
    place_lectern,
    tool_chest_pos,
    zone_center_floor,
)
from place_relic_chest import build_chest_snbt, load_relics_from_llm, place_chest

from gdpc import Editor

from area_discovery_gen import (
    GeneratorConfig,
    DatapackGenerator,
    zone_from_corners,
)


# Change this if your world has a different name
MINECRAFT_WORLD = "New World"
SETTLEMENT_THEME = "Fantasy"
RELIC_COUNT = 3


def _find_datapack_dir(world_name: str = "New World") -> Path:
    """Resolve <saves>/<world_name>/datapacks/area_discovery (same logic as example_usage.py)."""
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


def main() -> None:
    print(f'Generating settlement for theme:\n  "{SETTLEMENT_THEME}"')

    editor = Editor()
    biome, player_pos = sample_biome_at_player(editor)
    print(f'Detected player at {player_pos}.')
    print(f'Biome: {biome_hint(biome)}\n')

    start = time.perf_counter()
    settlement = generate_settlement(SETTLEMENT_THEME, biome=biome)
    print(f"[{time.perf_counter() - start:4.1f}s]  {settlement.name} — {settlement.era}")
    print(f"          {settlement.founding_story}\n")

    # Goal pre-pass. One cheap LLM call invents the single objective the whole
    # settlement is working toward, grounded in the just-generated identity and
    # mutated onto the SAME Settlement instance. Runs BEFORE shared_events so
    # the events can be grounded in the goal; every downstream generator then
    # inherits it via goal_hint(settlement).
    print("Generating settlement goal via LLM...")
    goal_start = time.perf_counter()
    settlement.goal = generate_settlement_goal(settlement)
    if settlement.goal:
        print(f"[{time.perf_counter() - goal_start:4.1f}s]  Goal: {settlement.goal.summary}")
        if settlement.goal.stakes:
            print(f"          Stakes: {settlement.goal.stakes}")
    else:
        print(f"[{time.perf_counter() - goal_start:4.1f}s]  (no goal generated)")
    print()

    # Shared-history pre-pass. One cheap LLM call grounds 3-5 concrete events
    # in the just-generated identity, mutated onto the SAME Settlement instance
    # so every downstream generator inherits them via events_hint(settlement).
    print("Generating shared history via LLM...")
    events_start = time.perf_counter()
    settlement.shared_events = generate_shared_events(settlement)
    print(f"[{time.perf_counter() - events_start:4.1f}s]  "
          f"{len(settlement.shared_events)} event(s):")
    for event in settlement.shared_events:
        print(f"  - {event}")
    print()

    # Four zones. Each gets subtitle="" so zone_from_corners calls
    # generate_zone_subtitle(preset, settlement) under the hood.
    print("Generating zones with auto subtitles...")
    # Flat-world demo layout: four 10x10 footprints arranged in a 2x2 grid on
    # the surface (y=-60), each 10 blocks tall so jumping/walking still
    # triggers zone entry. 20-block gaps between zones for clear transitions.
    # Generic settlement components (not coastal-specific) so the layout fits
    # whatever theme the LLM rolled.
    zone_specs = [
        ("town_square", "Town Square", "town",   (10, -60, 10,  19, -50, 19)),
        ("farm",        "Farm",        "nature", (30, -60, 10,  39, -50, 19)),
        ("barracks",    "Barracks",    "town",   (10, -60, 30,  19, -50, 39)),
        ("residential", "Residential", "town",   (30, -60, 30,  39, -50, 39)),
    ]
    zones = []
    for zone_id, display_name, preset, (x1, y1, z1, x2, y2, z2) in zone_specs:
        t0 = time.perf_counter()
        zone = zone_from_corners(
            zone_id      = zone_id,
            display_name = display_name,
            x1=x1, y1=y1, z1=z1,
            x2=x2, y2=y2, z2=z2,
            preset       = preset,
            settlement   = settlement,
        )
        print(f"  [{preset:9}] ({time.perf_counter() - t0:4.1f}s)  "
              f"{display_name}: \"{zone.title.subtitle}\"")
        zones.append(zone)

    print("\nWriting datapack...")
    config = GeneratorConfig(
        namespace        = "area_discovery",
        pack_description = f"§6{settlement.name}§r — Area Discovery",
        pack_format      = 26,
        output_dir       = _find_datapack_dir(MINECRAFT_WORLD),
        overwrite        = True,
        write_templates  = True,
    )
    gen = DatapackGenerator(config)
    gen.add_zones(zones)
    output_path = gen.generate()
    gen.summary()

    print(f"\nDatapack written to: {output_path.resolve()}")
    print("Then run in-game:  /function area_discovery:setup")

    print(f"\nGenerating {len(zones)} NPC diary entries via LLM...")
    diary_start = time.perf_counter()
    diaries = generate_diaries(
        settlement=settlement,
        zone_specs=zone_specs,
        biome=biome,
    )
    for diary, zone in match_diaries_to_zones(diaries, zones):
        place_lectern(editor, zone_center_floor(zone), build_lectern_snbt(diary))
        print(
            f"  [{diary.author_role:14}] {diary.author_name} -> "
            f"{zone.zone_id} ({len(diary.pages)} page"
            f"{'s' if len(diary.pages) != 1 else ''})"
        )
    print(f"[{time.perf_counter() - diary_start:4.1f}s]  "
          f"Placed {len(diaries)} lectern(s) at zone centers.")

    print(f"\nGenerating {len(zones)} thematic tool(s) via LLM...")
    tool_start = time.perf_counter()
    tools = generate_tools(settlement=settlement, zone_specs=zone_specs, biome=biome)
    for tool, zone in match_tools_to_zones(tools, zones):
        glint = glint_for_rarity(tool.rarity)
        snbt = build_tool_chest_snbt(tool, glint=glint)
        place_chest(editor, tool_chest_pos(zone), snbt)
        print(
            f"  [{tool.rarity:9}] {tool.name} ({tool.item_type}) -> "
            f"{zone.zone_id}{' *glint*' if glint else ''}"
        )
    print(f"[{time.perf_counter() - tool_start:4.1f}s]  "
          f"Placed {len(tools)} tool chest(s) beside lecterns.")

    print(f"\nGenerating {RELIC_COUNT} relic(s) via LLM...")
    relic_start = time.perf_counter()
    relics = load_relics_from_llm(
        SETTLEMENT_THEME, count=RELIC_COUNT, settlement=settlement, biome=biome
    )
    chest_snbt = build_chest_snbt(relics)
    place_chest(editor, player_pos, chest_snbt)
    print(
        f"[{time.perf_counter() - relic_start:4.1f}s]  "
        f"Placed chest with {len(relics)} relic(s) at {player_pos}:"
    )
    for relic in relics:
        print(f"          - {relic['name']} ({relic['item_type']})")


if __name__ == "__main__":
    main()
