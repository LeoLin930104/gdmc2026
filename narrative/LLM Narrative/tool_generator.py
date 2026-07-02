from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from biome_context import biome_hint
from fallback_content import fallback_tools
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_generator import Settlement
from settlement_goal import goal_hint
from shared_events import events_hint


RARITIES = ("Old", "Common", "Uncommon", "Rare", "Epic", "Legendary")
_RARITY_LOOKUP = {r.lower(): r for r in RARITIES}

VALID_COLORS = {"yellow", "aqua", "light_purple", "gold", "white", "green"}

# What counts as a tool/gear. Prefixed families (any wooden/stone/iron/golden/
# diamond/netherite/leather/chainmail variant) match by suffix; unprefixed
# tools/gear match by exact id. Anything outside this set is "not a tool"
# (compass, bell, lantern, written_book, etc. -- those belong to relics).
_TOOL_SUFFIXES = (
    "_sword", "_axe", "_pickaxe", "_shovel", "_hoe",
    "_helmet", "_chestplate", "_leggings", "_boots",
)
_TOOL_STANDALONE = {
    "bow", "crossbow", "trident",
    "fishing_rod", "shears", "flint_and_steel",
    "shield", "turtle_helmet",
}


def _is_tool_or_gear(item_type: str) -> bool:
    """True if `item_type` is a functional tool/weapon or wearable gear piece."""
    bare = item_type.split(":", 1)[1] if ":" in item_type else item_type
    if bare in _TOOL_STANDALONE:
        return True
    return any(bare.endswith(s) for s in _TOOL_SUFFIXES)


SYSTEM_PROMPT = """You generate FUNCTIONAL TOOLS OR WEARABLE GEAR for the zones of a Minecraft Java Edition 1.21 settlement.
Output ONLY a JSON array of tool objects. No markdown fences, no prose before or after.
The user says how many tools each zone needs. When a zone needs more than one,
make them DISTINCT — vary the item type, the invented owner name, and the rarity
so the zone reads as several different workers' and guardians' gear, not copies.

CRITICAL — what counts as a "tool" or "gear":
  Tools (item id MUST include a material prefix from the list below):
    *_sword, *_axe, *_pickaxe, *_shovel, *_hoe
  Tools with no material prefix (use the bare id):
    bow, crossbow, trident, fishing_rod, shears, flint_and_steel
  Gear (item id MUST include a material prefix from the list below):
    *_helmet, *_chestplate, *_leggings, *_boots
  Gear with no material prefix (use the bare id):
    shield, turtle_helmet
  Material prefixes for tools: wooden_, stone_, iron_, golden_, diamond_, netherite_
  Material prefixes for armor: leather_, chainmail_, iron_, golden_, diamond_, netherite_

CRITICAL — there is NO bare "minecraft:sword", "minecraft:hoe", "minecraft:axe",
"minecraft:helmet", "minecraft:chestplate", etc. in Minecraft. Every prefix-required
tool/gear MUST start with a material. ALWAYS write the full id. Examples of
VALID item_types:
  minecraft:wooden_hoe, minecraft:iron_hoe, minecraft:stone_axe,
  minecraft:iron_sword, minecraft:diamond_pickaxe, minecraft:netherite_axe,
  minecraft:leather_helmet, minecraft:chainmail_chestplate,
  minecraft:iron_boots, minecraft:golden_leggings,
  minecraft:bow, minecraft:crossbow, minecraft:fishing_rod,
  minecraft:shears, minecraft:shield, minecraft:turtle_helmet
Examples of INVALID item_types that DO NOT EXIST and will be dropped:
  minecraft:sword, minecraft:hoe, minecraft:axe, minecraft:pickaxe,
  minecraft:helmet, minecraft:chestplate, minecraft:boots,
  minecraft:leather, minecraft:iron.

DO NOT pick decorative, ceremonial, ammunition, food, utility, or trinket
items. Those belong in the separate relic system, NOT here. The following
are NOT tools and MUST NEVER appear in your output:
  bell, compass, recovery_compass, lantern, soul_lantern, clock, spyglass,
  goat_horn, lodestone, end_crystal, totem_of_undying, written_book,
  enchanted_book, book, music_disc_*, bucket, bone_meal, wheat_seeds,
  tipped_arrow, arrow, golden_apple, ender_pearl, blaze_rod, nautilus_shell,
  amethyst_shard, echo_shard, heart_of_the_sea, prismarine_shard,
  turtle_scute, honeycomb, feather, leather, bone, rabbit_foot,
  ghast_tear, dragon_breath, experience_bottle, mace, skeleton_skull,
  wither_skeleton_skull, zombie_head.
If you are tempted to pick one of these for a zone where "no tool fits"
(e.g. a town square or landmark), pick a tool/weapon a townsperson or
guardian of that zone would carry: a watch-captain's sword, a stonemason's
pickaxe, a steward's iron chestplate, a herald's golden axe.

Each object must have these fields:
  "zone_id":     MUST exactly match one of the zone_id values listed by the user.
  "name":        the in-world item name. MUST include both:
                   (a) an owner-name or epithet you INVENT for this settlement
                       (a first name, an "Old <Name>" epithet, or a "Title Name"
                       form) — fitting its biome/era. CRITICAL: write a REAL
                       invented name. NEVER output a placeholder, a role word, or
                       any token in angle brackets — do NOT write "<owner>",
                       "<Steward>", "<Farmer>", "<Name>" or the literal word
                       "owner". Write an actual name like "Maren" or "Old Tobin".
                       Never reuse a name from these instructions. AND
                   (b) a descriptive name-adjective hinting at age/quality.
                       This is PROSE, not the rarity field.
                       For low-tier items: "Old", "Worn", "Plain", "Battered".
                       For mid-tier items: "Trusty", "Sturdy", "Well-Kept".
                       For high-tier items: "Polished", "Sharp", "Heirloom",
                       "Storied", "Chosen".
                 Examples of the FORM only — invent your own owner, never copy
                 these names verbatim:
                           "Maren's Old Hoe" (rarity Old/Common),
                           "Old Tobin's Trusty Axe" (rarity Uncommon),
                           "Captain Yorvald's Heirloom Sword" (rarity Rare/Epic).
  "item_type":   a REAL vanilla Minecraft Java Edition 1.21 item ID
                 ("minecraft:xxx") drawn ONLY from the tool/gear families above.
                 MUST include the material prefix where required.
  "description": one-line flavor under 15 words.
  "lore":        1-2 sentences of story as ONE string, NOT an array.
  "color":       one of: yellow, aqua, light_purple, gold, white, green.
  "rarity":      MUST be EXACTLY one of these six strings (case-sensitive):
                   "Old", "Common", "Uncommon", "Rare", "Epic", "Legendary".
                 Do NOT put name-adjectives like "Heirloom", "Worn", or "Trusty"
                 here -- those go in the `name` field. The rarity field is a
                 strict enum.

Zone-preset -> recommended tool/gear (pick something a worker, guardian, or
resident of that zone would actually use):
  town       -> wooden_axe, iron_axe (carpenters); iron_sword, iron_helmet
                (watchmen); stone_pickaxe (stonemasons); golden_chestplate
                (heralds/stewards). NEVER bell/lantern/clock/book.
  nature     -> wooden_hoe, iron_hoe, stone_hoe (farmers); shears (shepherds);
                fishing_rod (anglers); wooden_axe, iron_axe (woodcutters);
                leather_chestplate, leather_boots (rangers).
  ruins      -> chipped/old gear: wooden_sword, stone_axe, wooden_pickaxe,
                leather_helmet, leather_chestplate, bow.
  dungeon    -> iron_sword, diamond_sword, iron_axe, iron_chestplate,
                iron_helmet, crossbow, shield.
  landmark   -> ceremonial gear: golden_sword, golden_helmet,
                golden_chestplate, diamond_sword, netherite_sword (rare).
                NEVER spyglass/lodestone/end_crystal.

Rarity -> material mapping (PICK item_type CONSISTENT WITH RARITY):
  Old        -> wooden_*, leather_*, stone_*. Plain, well-used.
  Common     -> wooden_*, leather_*, stone_*. Decent condition.
  Uncommon   -> iron_*, chainmail_*. Solid working gear.
  Rare       -> iron_* (implied enchantment), golden_*.
  Epic       -> diamond_*.
  Legendary  -> netherite_*.

Color guidance (display name color follows rarity):
  Old/Common -> yellow or white
  Uncommon   -> green
  Rare       -> aqua
  Epic       -> light_purple
  Legendary  -> gold

Rarity distribution: most should be Old/Common/Uncommon (these are NPCs'
working tools, not heroes' loot). At most ONE Legendary across the whole
settlement, and only if a zone clearly warrants it.

NEVER use these item_types - they do not exist:
  minecraft:medallion, minecraft:amulet, minecraft:relic, minecraft:charm,
  minecraft:scroll, minecraft:tool, minecraft:gear.

When the settlement's identity axes are provided, the tool names and lore
should reflect the primary industry and central virtue (a discipline-focused
mining town's hoe reads differently from a hospitality-focused farming
town's hoe). The motif may also surface as a name or carving detail.

When shared history is provided, one or two tools in the set may tie back
to a specific event ("carried downstream the year the rain wouldn't stop",
"the only blade that came back from the ridge expedition"). Not every tool
needs to — most are working gear, not heirlooms.

When a "Current struggle" is provided, at least one tool should read as MADE
FOR that objective — its name or lore should make plain it serves the village's
present effort (the reforged axe for the rebuilding, the long-bow for the
wolf-watch)."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    zone_id: str
    name: str
    item_type: str
    description: str
    lore: str
    color: str
    rarity: str


# ---------------------------------------------------------------------------
# JSON parsing / normalization
# ---------------------------------------------------------------------------

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


def _normalize_tool(raw: dict) -> dict:
    """Coerce common LLM drift back into the schema shape (in-place)."""
    # Defensive net for the prompt's name-shape guidance: small models sometimes
    # echo a placeholder token literally ("<Steward>'s Old Hoe"). Strip the angle
    # brackets so the inner word survives ("Steward's Old Hoe") rather than
    # letting literal "<...>" reach the in-game item name. Belt-and-suspenders to
    # the prompt rule that forbids placeholders outright.
    name = raw.get("name")
    if isinstance(name, str) and "<" in name and ">" in name:
        cleaned = re.sub(r"<\s*([^<>]*?)\s*>", r"\1", name).strip()
        if cleaned and cleaned != name:
            print(f"[warn] Tool name {name!r} contained a placeholder; "
                  f"sanitized to {cleaned!r}.")
        raw["name"] = cleaned or name

    for field in ("lore", "description"):
        value = raw.get(field)
        if isinstance(value, list):
            raw[field] = " ".join(str(v).strip() for v in value)
        elif value is None:
            raw[field] = ""

    item_type = raw.get("item_type")
    if isinstance(item_type, str):
        item_type = item_type.strip().lower()
        if item_type and ":" not in item_type:
            item_type = f"minecraft:{item_type}"
        raw["item_type"] = item_type

    color = raw.get("color")
    if not isinstance(color, str) or color.strip().lower() not in VALID_COLORS:
        raw["color"] = "yellow"
    else:
        raw["color"] = color.strip().lower()

    rarity_raw = raw.get("rarity")
    if isinstance(rarity_raw, str) and rarity_raw.strip().lower() in _RARITY_LOOKUP:
        raw["rarity"] = _RARITY_LOOKUP[rarity_raw.strip().lower()]
    else:
        name_label = raw.get("name") or raw.get("zone_id") or "?"
        print(f"[warn] Tool {name_label!r} had invalid rarity {rarity_raw!r}; defaulting to 'Common'.")
        raw["rarity"] = "Common"

    return raw


def _validate_tool(raw: dict) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected tool dict, got {type(raw).__name__}: {raw}")
    for key in ("name", "zone_id"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Tool missing or empty {key!r}: {raw}")
    item_type = raw.get("item_type", "")
    if not isinstance(item_type, str) or not re.match(r"^[a-z0-9_]+:[a-z0-9_]+$", item_type):
        raise ValueError(f"Tool has malformed item_type {item_type!r}: {raw}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tools(
    settlement: Settlement,
    zone_specs: Iterable[tuple],
    biome: str | None = None,
    max_tokens: int = 1000,
    per_zone: int = 1,
) -> list[Tool]:
    """Generate `per_zone` Tool(s) per entry in `zone_specs`.

    `zone_specs` is the same shape used by `example_settlement_demo.py`:
    each entry is `(zone_id, display_name, preset, *rest)`. Only the first
    three fields are sent to the LLM; any AABB tuple at position 3 is ignored.

    `per_zone` asks the model for that many DISTINCT tools per zone (they share
    the zone's `zone_id`, so the caller groups them by zone). The returned list
    is flat and may hold several tools with the same `zone_id`.

    Settlement identity + biome are threaded in so tools share narrative
    context with zones, diaries, and relics. Each Tool carries a `zone_id`
    used by `match_tools_to_zones` to pair it with the right zone for chest
    placement.

    Temperature is held lower than `generate_diaries` (0.7 vs 0.9) — for
    tools we want consistent material/rarity choices, not flowery prose.
    """
    specs = [(z[0], z[1], z[2]) for z in zone_specs]
    if not specs:
        raise ValueError("zone_specs is empty - no tools to generate")
    per_zone = max(1, per_zone)
    # Room for every requested tool: ~150 tokens of JSON each, never below the
    # caller's floor. Without this a big per_zone request truncates mid-array.
    max_tokens = max(max_tokens, len(specs) * per_zone * 150)

    effective_biome = biome or settlement.biome
    hint = biome_hint(effective_biome)
    biome_block = f"Biome: {hint}\n\n" if hint else ""

    axes = axes_hint(settlement)
    axes_block = f"{axes}\n\n" if axes else ""

    events = events_hint(settlement)
    events_block = f"{events}\n\n" if events else ""

    goal = goal_hint(settlement)
    goal_block = f"{goal}\n\n" if goal else ""

    context_block = (
        f"Settlement context:\n"
        f"  Name: {settlement.name}\n"
        f"  Era: {settlement.era}\n"
        f"  Story: {settlement.founding_story}\n\n"
    )

    zones_block = "\n".join(
        f'  - zone_id="{zid}"  display_name="{name}"  preset="{preset}"'
        for zid, name, preset in specs
    )

    if per_zone > 1:
        count_line = f"Zones in this settlement (generate {per_zone} DISTINCT tools per zone):\n"
        total_line = (
            f"Generate exactly {per_zone} tools for EACH zone "
            f"({len(specs) * per_zone} tools total). Repeat each zone_id "
            f"{per_zone} times, once per tool, and within a zone vary the item "
            f"type, the invented owner name, and the rarity. Each tool's "
        )
    else:
        count_line = "Zones in this settlement (generate ONE tool per zone):\n"
        total_line = f"Generate exactly {len(specs)} tools (one per zone). Each tool's "

    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"{context_block}"
        f"{count_line}"
        f"{zones_block}\n\n"
        f"{total_line}"
        f"item_type must be thematically native to its zone's preset and its "
        f"material must match its rarity tier. Return ONLY the JSON array."
    )

    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=max_tokens,
            timeout=120,
        )
    except LLMUnavailable as exc:
        print(f"[warn] generate_tools: LLM unavailable ({exc}); using offline fallback tools.")
        return fallback_tools(settlement, specs, effective_biome, per_zone=per_zone)

    # Strict parse first; on a stochastic JSON glitch, fall back to salvaging
    # individual objects so one malformed entry doesn't waste the whole run.
    try:
        parsed = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        candidates = parsed
    else:
        candidates = _salvage_json_objects(raw)
        if candidates:
            print(f"[warn] Tool JSON was malformed; salvaged "
                  f"{len(candidates)} object(s) individually.")
        else:
            print("[warn] generate_tools: could not parse any tool object; "
                  "returning []. Pipeline continues with no tool chests placed.")
            return []

    known_zone_ids = {zid for zid, _, _ in specs}
    tools: list[Tool] = []
    for entry in candidates:
        # Per-entry warn-and-recover: a single malformed tool must not abort
        # the batch — settlement/goal/events/zones/diaries already succeeded.
        try:
            _normalize_tool(entry)
            _validate_tool(entry)
        except (ValueError, TypeError) as exc:
            print(f"[warn] Skipping malformed tool object ({exc}).")
            continue
        if not _is_tool_or_gear(entry["item_type"]):
            print(
                f"[warn] Tool {entry.get('name')!r} for zone "
                f"{entry.get('zone_id')!r} picked non-tool item_type "
                f"{entry['item_type']!r}; dropping. (Tools must be "
                f"swords/axes/pickaxes/shovels/hoes, bows/crossbows/tridents, "
                f"fishing_rod/shears/flint_and_steel, or armor/shield.)"
            )
            continue
        # Drop hallucinated zone_ids here rather than letting
        # match_tools_to_zones raise and abort the run.
        if entry["zone_id"].strip() not in known_zone_ids:
            print(f"[warn] Tool {entry.get('name')!r} has unknown "
                  f"zone_id {entry['zone_id']!r}; skipping.")
            continue
        tools.append(Tool(
            zone_id=entry["zone_id"].strip(),
            name=entry["name"].strip(),
            item_type=entry["item_type"],
            description=entry.get("description", "").strip(),
            lore=entry.get("lore", "").strip(),
            color=entry["color"],
            rarity=entry["rarity"],
        ))
    return tools
