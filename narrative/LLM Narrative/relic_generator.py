from __future__ import annotations
import json
import re
from biome_context import biome_hint
from fallback_content import fallback_relics
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_generator import Settlement
from settlement_goal import goal_hint
from shared_events import events_hint

# Items the LLM commonly invents but which do not exist in vanilla Minecraft.
# Exact-match against the normalized "minecraft:foo" string; the SYSTEM_PROMPT
# already lists these as INVALID but a 3B model will occasionally pick one
# anyway (observed: "minecraft:amulet" in a Riversend run). Anything in this
# set is dropped with a [warn] at validate time so the chest only contains
# real items. Extend this list as new hallucinations surface.
_KNOWN_FAKE_RELIC_ITEMS = frozenset({
    "minecraft:amulet",
    "minecraft:medallion",
    "minecraft:hook",
    "minecraft:relic",
    "minecraft:idol",
    "minecraft:charm",
    "minecraft:rune",
    "minecraft:scroll",
    "minecraft:orb",
    "minecraft:talisman",
    "minecraft:fragment",
    "minecraft:emblem",
    "minecraft:badge",
    "minecraft:sigil",
    "minecraft:token",
    "minecraft:locket",
    "minecraft:ring",
    "minecraft:bracelet",
    "minecraft:gem",
    "minecraft:crystal",      # end_crystal is real; bare crystal is not
    "minecraft:shard",        # amethyst_shard/echo_shard/prismarine_shard are real; bare shard is not
    "minecraft:loot_table",
    "minecraft:effigy",
    "minecraft:phylactery",
    "minecraft:reliquary",
    "minecraft:trinket",
    "minecraft:tool",
    "minecraft:gear",
})


SYSTEM_PROMPT = """You generate relic items for a Minecraft Java Edition 1.21 settlement.
Output ONLY a JSON array. No markdown fences, no prose before or after.

Each object must have these fields:
  "name":         evocative item name, 3-6 words
  "item_type":    a REAL vanilla Minecraft Java Edition item ID ("minecraft:xxx").
                  CRITICAL: if you are not 100% certain the item exists in vanilla
                  Minecraft, DO NOT use it. Pick a safer, well-known alternative.
                  Minecraft has no generic "relic" items — every relic is reskinned
                  from an existing vanilla item via custom_name and lore.
                  Thematically useful vanilla items include:
                    natural  — feather, bone, leather, rabbit_foot, nautilus_shell,
                               amethyst_shard, echo_shard, heart_of_the_sea,
                               prismarine_shard, turtle_scute, honeycomb
                    magic    — ender_pearl, blaze_rod, ghast_tear, dragon_breath,
                               experience_bottle, totem_of_undying, end_crystal
                    tools    — compass, recovery_compass, clock, spyglass, map,
                               goat_horn, lantern, soul_lantern
                    books    — book, written_book, enchanted_book
                    food     — golden_apple, enchanted_golden_apple, chorus_fruit
                    weapons  — bow, crossbow, trident, mace, shield
                    music    — music_disc_pigstep, music_disc_otherside, music_disc_13
                    heads    — skeleton_skull, wither_skeleton_skull, zombie_head
                  INVALID examples that DO NOT exist and must NEVER be used:
                    minecraft:medallion, minecraft:amulet, minecraft:hook,
                    minecraft:relic, minecraft:loot_table, minecraft:idol,
                    minecraft:charm, minecraft:rune, minecraft:scroll,
                    minecraft:orb, minecraft:crystal, minecraft:gem
  "description":  one-line flavor, under 15 words
  "lore":         1-2 sentences of story as ONE string, NOT an array
  "color":        one of: gold, dark_purple, dark_red, aqua, green, yellow, red, blue, light_purple, white
  "rarity":       one of: Common, Uncommon, Rare, Epic, Legendary

When a settlement's identity axes are provided, the relic set as a whole
should echo the historical wound, motif, or central virtue at least once —
not every relic, but the collection should feel like it remembers.

When shared history is provided, a relic may explicitly reference one event
from the list — the relic was carried through that day, or it remembers it
in its inscription. At most one or two relics in the set should do this;
the rest stay implicit.

When a "Current struggle" is provided, at least one relic should ORIGINATE
from that objective — an artifact recovered, forged, or sought because of the
village's present effort (the survey-stone for the rebuilding, the warhorn for
the wolf-watch)."""


def _extract_json_array(text: str) -> list:
    """Parse the first JSON array in `text`, tolerating surrounding junk."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON array found in model output:\n{text}")


def _salvage_json_objects(text: str) -> list[dict]:
    """Best-effort fallback: pull out each top-level {...} object and parse it
    individually, skipping any that fail.

    Used when strict array parsing fails on a stochastic LLM glitch (a trailing
    comma, a dropped value, an unescaped char in ONE entry). Brace-matching that
    respects strings/escapes lets us recover the well-formed siblings instead of
    throwing the whole batch away. Returns [] if nothing parses.
    """
    objs: list[dict] = []
    depth = 0
    obj_start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and obj_start != -1:
                try:
                    obj = json.loads(text[obj_start : i + 1])
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = -1
    return objs


def _normalize(relic: dict) -> None:
    """Coerce common LLM drift back into the schema shape."""
    for field in ("lore", "description"):
        value = relic.get(field)
        if isinstance(value, list):
            relic[field] = " ".join(str(v).strip() for v in value)

    item_type = relic.get("item_type")
    if isinstance(item_type, str):
        item_type = item_type.strip().lower()
        if item_type and ":" not in item_type:
            item_type = f"minecraft:{item_type}"
        relic["item_type"] = item_type


def _validate(relic: dict) -> None:
    if not isinstance(relic, dict):
        raise ValueError(f"Expected dict, got {type(relic).__name__}: {relic}")
    if not relic.get("name"):
        raise ValueError(f"Relic missing 'name': {relic}")
    item_type = relic.get("item_type", "")
    if not re.match(r"^[a-z0-9_]+:[a-z0-9_]+$", item_type):
        raise ValueError(f"Relic has malformed item_type {item_type!r}: {relic}")


def generate_relics(
    theme: str,
    count: int = 3,
    max_tokens: int = 600,
    settlement: Settlement | None = None,
    biome: str | None = None,
) -> list[dict]:
    """Generate `count` relic objects themed around `theme`.

    If `settlement` is provided, its name/era/founding story are prepended to
    the user prompt so relic lore coheres with the rest of the settlement's
    narrative (see CLAUDE.md — "Consistency constraint").

    If `biome` is provided (or inherited from `settlement.biome`), its trait
    hint is prepended to ground relic materials/imagery in the Minecraft
    world location.

    Returns a list of dicts matching the relics.json schema
    (ready to drop into {"relics": [...]} and load via place_relic_chest.load_relics).
    """
    effective_biome = biome or (settlement.biome if settlement is not None else None)
    hint = biome_hint(effective_biome)
    biome_block = f"Biome: {hint}\n\n" if hint else ""

    context_block = ""
    axes_block = ""
    events_block = ""
    goal_block = ""
    if settlement is not None:
        context_block = (
            f"Settlement context:\n"
            f"  Name: {settlement.name}\n"
            f"  Era: {settlement.era}\n"
            f"  Story: {settlement.founding_story}\n\n"
        )
        axes = axes_hint(settlement)
        axes_block = f"{axes}\n\n" if axes else ""
        events = events_hint(settlement)
        events_block = f"{events}\n\n" if events else ""
        goal = goal_hint(settlement)
        goal_block = f"{goal}\n\n" if goal else ""
    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"{context_block}"
        f"Generate {count} relics themed around: {theme}\n\n"
        f"Return ONLY the JSON array, starting with [ and ending with ]."
    )
    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=max_tokens,
            timeout=120,
        )
    except LLMUnavailable as exc:
        print(f"[warn] generate_relics: LLM unavailable ({exc}); using offline fallback relics.")
        return fallback_relics(theme, count, settlement, effective_biome)
    except Exception as exc:
        print(f"[warn] generate_relics: LLM call failed ({exc!r}); using offline fallback relics.")
        return fallback_relics(theme, count, settlement, effective_biome)
    # Strict parse first; on a stochastic JSON glitch, fall back to salvaging
    # individual objects so one malformed entry doesn't waste the whole run.
    try:
        relics = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        relics = None

    if not isinstance(relics, list):
        relics = _salvage_json_objects(raw)
        if relics:
            print(f"[warn] Relic JSON was malformed; salvaged "
                  f"{len(relics)} object(s) individually.")
        else:
            print("[warn] generate_relics: could not parse any relic object; "
                  "returning []. Pipeline continues with an empty/partial chest.")
            return []

    # Warn-and-recover loop: drop individual relics that fail validation or
    # name an invented item, but never abort the whole call — a 3-relic
    # request that returns 2 good and 1 fake should still ship the 2.
    validated: list[dict] = []
    for relic in relics:
        if not isinstance(relic, dict):
            print(f"[warn] Relic entry is not a dict ({type(relic).__name__}); dropping.")
            continue
        _normalize(relic)
        try:
            _validate(relic)
        except ValueError as exc:
            print(f"[warn] Relic failed validation: {exc}; dropping.")
            continue
        item_type = relic.get("item_type", "")
        if item_type in _KNOWN_FAKE_RELIC_ITEMS:
            print(
                f"[warn] Relic {relic.get('name')!r} uses non-vanilla item_type "
                f"{item_type!r}; dropping. (Add it to _KNOWN_FAKE_RELIC_ITEMS "
                f"if the LLM keeps inventing it.)"
            )
            continue
        validated.append(relic)
    return validated
