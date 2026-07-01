from __future__ import annotations

from typing import Any


BIOME_TRAITS: dict[str, str] = {
    "plains":         "wide grassland dotted with oaks, temperate and open",
    "forest":         "dense oak and birch, shafts of filtered light",
    "dark_forest":    "looming dark oak canopy, perpetual gloom, moss underfoot",
    "birch_forest":   "pale birch columns, cool shade, rustling undergrowth",
    "desert":         "bone-dry dunes, sun-bleached stone, no running water",
    "taiga":          "cold spruce highlands, thin snow, wolves at the tree line",
    "snowy_taiga":    "deep snow, frozen spruce, silence broken by ice",
    "snowy_plains":   "flat white expanse, bitter wind, frozen lakes",
    "jungle":         "hot tangled vines, parrots, three canopies deep",
    "savanna":        "golden acacia plains, dry wind, flat-topped horizons",
    "swamp":          "dark stagnant water, lily pads, hanging vines",
    "mangrove_swamp": "warm rooted shallows, heavy humidity, ochre mud",
    "badlands":       "layered red mesa stone, cracked clay, rare cacti",
    "ocean":          "endless saltwater, kelp forests far below",
    "beach":          "pale sand, driftwood, steady tide",
    "river":          "slow freshwater cut between banks, reeds, silt",
    "mushroom_fields":"muted mycelium, towering red and brown mushrooms, no hostile mobs",
    "deep_dark":      "sculk-fringed abyss, absolute quiet, something listens",
    "cherry_grove":   "pale pink petals drifting, serene uplands",
    "meadow":         "wildflower slopes, warm sun, distant peaks",
    "stony_peaks":    "bare stone ridges, hard wind, thin air",
}


def biome_hint(biome: str | None) -> str:
    if not biome:
        return ""
    key = biome.strip().lower()
    if key.startswith("minecraft:"):
        key = key.split(":", 1)[1]
    traits = BIOME_TRAITS.get(key)
    return f"{key} — {traits}" if traits else key


def sample_biome(editor: Any, pos: tuple[int, int, int]) -> str:
    return editor.getBiome(pos)


def get_player_position(
    host: str = "localhost",
    port: int = 9000,
    timeout: float = 5.0,
) -> tuple[int, int, int]:
    import math
    import re
    import requests

    url = f"http://{host}:{port}/players"
    resp = requests.get(url, params={"includeData": "true"}, timeout=timeout)
    resp.raise_for_status()
    players = resp.json()
    if not players:
        raise RuntimeError(
            f"No players found at {url}. Make sure Minecraft is running, "
            f"the GDMC HTTP Interface mod is loaded, and you are in the world."
        )
    snbt = players[0].get("data", "")
    match = re.search(r"Pos:\s*\[\s*([^\]]+)\]", snbt)
    if not match:
        raise RuntimeError(f"Could not find Pos in player data: {snbt!r}")
    parts = [p.strip().rstrip("dD") for p in match.group(1).split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"Expected 3 Pos values, got {len(parts)}: {parts}")
    x, y, z = (int(math.floor(float(p))) for p in parts)
    return x, y, z


def sample_biome_at_player(
    editor: Any,
    host: str = "localhost",
    port: int = 9000,
) -> tuple[str, tuple[int, int, int]]:
    pos = get_player_position(host=host, port=port)
    return sample_biome(editor, pos), pos
