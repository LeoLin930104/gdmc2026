from __future__ import annotations

import json
import os
from typing import Iterable

_JSON_PATH = os.path.join(os.path.dirname(__file__), "fallback_content.json")

# Last-ditch content if fallback_content.json is missing or unparsable. Keeps
# every builder non-raising even when its data source is gone.
_MINIMAL = {
    "settlement": {
        "name": "Hollowmere",
        "era": "an age after the founding",
        "founding_story": "A handful of families settled by still water and stayed.",
    },
    "goal": {"summary": "lay in stores before the season turns", "stakes": ""},
    "shared_events": ["the hard winter", "the founding feast", "the long drought"],
    "mood_tier": "strained",
    "subtitles": {"default": "a quiet place that keeps its own counsel"},
    "district_names": [{"name": "The Commons", "preset": "town"}],
    "relics": [{
        "name": "Founder's Plain Token",
        "item_type": "minecraft:book",
        "description": "A keepsake from the founding.",
        "lore": "Kept by the first family to settle here.",
        "color": "yellow",
        "rarity": "Common",
    }],
    "tools": [{
        "preset": "town",
        "name": "Old Tann's Worn Hoe",
        "item_type": "minecraft:wooden_hoe",
        "description": "A farmhand's well-used tool.",
        "lore": "Tann worked this ground for forty seasons.",
        "color": "yellow",
        "rarity": "Old",
    }],
    "diaries": [{
        "preset": "town",
        "author_name": "A Settler",
        "author_role": "settler",
        "book_title": "Day Book",
        "page": "Another day's work done. The water is low but the stores hold. We keep on.",
    }],
}

_cache: dict | None = None

DEFAULT_VARIANT = "1"


def _select_variant(variants: dict) -> str:
    """Pick the fallback lore variant from NARRATIVE_FALLBACK_VARIANT (default '1').

    Judges set this in the repo-root .env (loaded by lm_client at import). Each
    variant is a wholly different authored settlement; the number just chooses
    which offline place the pipeline reads as. An unknown value warns and falls
    back to '1' (or the first available variant).
    """
    key = os.environ.get("NARRATIVE_FALLBACK_VARIANT", DEFAULT_VARIANT).strip() or DEFAULT_VARIANT
    if key in variants:
        return key
    print(f"[warn] fallback_content: variant {key!r} not found "
          f"(have {sorted(variants)}); using {DEFAULT_VARIANT!r}.")
    return DEFAULT_VARIANT if DEFAULT_VARIANT in variants else next(iter(variants))


def _load() -> dict:
    """Return the authored fallback content for the selected variant, cached.

    The JSON file holds several numbered lore variants under "variants"; the
    active one is chosen by NARRATIVE_FALLBACK_VARIANT so judges can swap the
    offline settlement's whole flavor without touching code. A file with no
    "variants" key is treated as a single flat variant (backward compatible).
    Never raises.
    """
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # missing file, bad JSON, permissions...
        print(f"[warn] fallback_content: could not load {_JSON_PATH} ({exc}); "
              f"using minimal built-in content.")
        _cache = _MINIMAL
        return _cache
    variants = data.get("variants")
    if isinstance(variants, dict) and variants:
        _cache = variants[_select_variant(variants)]
    else:
        _cache = data  # flat/legacy format (no numbered variants)
    return _cache


# ---------------------------------------------------------------------------
# Per-zone assignment helper
# ---------------------------------------------------------------------------

def _pick_by_preset(pool: list[dict], preset: str | None, used: set[int]) -> dict:
    """Pick a pool entry for a zone, preferring one tagged with `preset`.

    Greedy + uniqueness-aware: tries an unused preset match first, then any
    unused entry, then cycles. This keeps distinct zones getting distinct
    authors/items even when several zones share a preset.
    """
    want = (preset or "").strip().lower()
    for i, entry in enumerate(pool):
        if i not in used and str(entry.get("preset", "")).strip().lower() == want:
            used.add(i)
            return entry
    for i, entry in enumerate(pool):
        if i not in used:
            used.add(i)
            return entry
    # Everything used at least once -- cycle deterministically.
    idx = len(used) % len(pool)
    used.add(idx)
    return pool[idx]


# ---------------------------------------------------------------------------
# Builders -- one per generator, matching that generator's return shape
# ---------------------------------------------------------------------------

def fallback_settlement(theme: str, biome: str | None = None):
    """Return the authored Settlement, tagged with the caller's theme/biome."""
    from settlement_generator import Settlement  # lazy: avoids circular import

    s = _load()["settlement"]
    return Settlement(
        name=s["name"],
        era=s["era"],
        founding_story=s["founding_story"],
        theme=theme,
        biome=biome,
        primary_industry=s.get("primary_industry"),
        central_virtue=s.get("central_virtue"),
        collective_fear=s.get("collective_fear"),
        historical_wound=s.get("historical_wound"),
        motif=s.get("motif"),
        worldview=s.get("worldview"),
        social_structure=s.get("social_structure"),
        outsider_reputation=s.get("outsider_reputation"),
    )


def fallback_goal(settlement):
    """Return the authored SettlementGoal."""
    from settlement_goal import SettlementGoal  # lazy: avoids circular import

    g = _load()["goal"]
    return SettlementGoal(summary=g["summary"], stakes=g.get("stakes", ""))


def fallback_shared_events(settlement, count: int = 4) -> list[str]:
    """Return up to `count` authored event phrases (cycling if asked for more)."""
    events = list(_load()["shared_events"])
    if not events:
        return []
    if count <= len(events):
        return events[:count]
    out = []
    while len(out) < count:
        out.extend(events)
    return out[:count]


def fallback_mood_tier(settlement) -> str:
    """Return the authored mood tier (a valid VALID_TIERS string)."""
    return _load().get("mood_tier", "strained")


def fallback_subtitle(zone_type: str, settlement, biome: str | None = None) -> str:
    """Return an authored subtitle for the zone type, or the default line."""
    subs = _load().get("subtitles", {})
    key = (zone_type or "").strip().lower()
    return subs.get(key) or subs.get("default", "")


def fallback_districts(settlement, descriptors: list[dict],
                       biome: str | None = None) -> list[dict]:
    """Name each descriptor from the authored pool, one entry per district."""
    pool = list(_load().get("district_names", []))
    if not pool:
        # Mirror district_namer's own generic fallback shape.
        return [{"zone_index": d["zone_index"],
                 "name": "Settlement Core", "preset": "town"} for d in descriptors]
    out = []
    for i, d in enumerate(descriptors):
        entry = pool[i % len(pool)]
        out.append({
            "zone_index": d["zone_index"],
            "name": entry["name"],
            "preset": entry.get("preset", "town"),
        })
    return out


def fallback_relics(theme: str, count: int = 3, settlement=None,
                    biome: str | None = None) -> list[dict]:
    """Return up to `count` authored relic dicts (cycling if asked for more)."""
    pool = _load().get("relics", [])
    if not pool:
        return []
    out: list[dict] = []
    i = 0
    while len(out) < count:
        # Shallow copy so callers can mutate freely without touching the cache.
        out.append(dict(pool[i % len(pool)]))
        i += 1
    return out[:count]


def fallback_tools(settlement, zone_specs: Iterable[tuple],
                   biome: str | None = None, per_zone: int = 1) -> list:
    """Return `per_zone` authored Tool(s) per zone spec, themed to the preset.

    `_pick_by_preset` is uniqueness-aware, so a zone asking for several tools
    gets distinct pool entries before any repeat.
    """
    from tool_generator import Tool  # lazy: avoids circular import

    specs = [(z[0], z[1], z[2]) for z in zone_specs]
    pool = _load().get("tools", [])
    if not specs or not pool:
        return []
    used: set[int] = set()
    tools = []
    for zid, _name, preset in specs:
        for _ in range(max(1, per_zone)):
            entry = _pick_by_preset(pool, preset, used)
            tools.append(Tool(
                zone_id=zid,
                name=entry["name"],
                item_type=entry["item_type"],
                description=entry.get("description", ""),
                lore=entry.get("lore", ""),
                color=entry.get("color", "yellow"),
                rarity=entry.get("rarity", "Common"),
            ))
    return tools


def fallback_diaries(settlement, zone_specs: Iterable[tuple],
                     biome: str | None = None) -> list:
    """Return one authored Diary per zone spec, themed to the zone's preset."""
    from diary_generator import Diary  # lazy: avoids circular import

    specs = [(z[0], z[1], z[2]) for z in zone_specs]
    pool = _load().get("diaries", [])
    if not specs or not pool:
        return []
    used: set[int] = set()
    diaries = []
    for zid, _name, preset in specs:
        entry = _pick_by_preset(pool, preset, used)
        diaries.append(Diary(
            author_name=entry["author_name"],
            author_role=entry.get("author_role", "settler"),
            book_title=entry["book_title"],
            zone_id=zid,
            pages=[entry["page"]],
        ))
    return diaries
