from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from biome_context import biome_hint
from fallback_content import fallback_districts
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_goal import goal_hint
from shared_events import events_hint

if TYPE_CHECKING:
    from settlement_generator import Settlement


# Presets the Area Discovery datapack knows how to style (see gdmc_bridge._PRESETS).
VALID_PRESETS = ("town", "ruins", "nature", "landmark", "dungeon")
DEFAULT_PRESET = "town"


SYSTEM_PROMPT = """You name the districts of a Minecraft settlement.
Output ONLY a JSON array of objects. No markdown fences, no prose before or after.

Each object names ONE district and has exactly these keys:
  "zone_index" : the integer id of the district you are naming (copy it back)
  "name"       : the district's display name, 1-4 words (e.g. "Tanner's Row",
                 "The Drowned Quarter", "Highmarket", "Saltgate")
  "preset"     : one visual style, one of: town, ruins, nature, landmark, dungeon

Rules:
  - every district name must be DISTINCT and evocative of THIS settlement's
    identity (industry, virtue, fear, wound, motif) — not generic ("Zone 1")
  - let the district's position and relative size shape its name and preset
    (a large central district reads as a market/town core; an outlying one may
    be farmland=nature, a watchpost=landmark, or an abandoned edge=ruins)
  - prefer "town" and "nature"; reserve "ruins"/"dungeon" for districts the
    lore would actually treat as decayed or dangerous
  - if a current struggle or shared history is given, one district name may
    glance at it
  - no emoji, no trailing punctuation, no quotes inside the name
  - return one object per district you are given, and only those"""


def _extract_json_array(text: str) -> list:
    """Parse the first JSON array in `text`, tolerating surrounding junk.

    Same shape as the extractor the other generators carry — copied here to
    keep this module self-contained.
    """
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


def _fallback_name(descriptor: dict) -> str:
    """Generic, never-empty district name from a descriptor's position."""
    position = str(descriptor.get("position", "")).strip().lower()
    if position and position != "central":
        return f"{position.title()} District"
    return "Settlement Core"


def _fallback_districts(descriptors: list[dict]) -> list[dict]:
    return [
        {
            "zone_index": d["zone_index"],
            "name": _fallback_name(d),
            "preset": DEFAULT_PRESET,
        }
        for d in descriptors
    ]


def generate_districts(
    settlement: "Settlement",
    descriptors: list[dict],
    biome: str | None = None,
    max_tokens: int = 400,
) -> list[dict]:
    """Name each district in `descriptors`, grounded in `settlement`.

    `descriptors` comes from `gdmc_bridge.zone_descriptors_from_zone_map` — one
    dict per zone with keys "zone_index", "cell_count", "position".

    Returns one dict per descriptor: {"zone_index", "name", "preset"}, in the
    same order. `preset` is always one of VALID_PRESETS (defaulting to "town").

    Warn-and-recover: on any failure (transport error, malformed JSON, missing
    entries) the affected districts fall back to a position-based generic name
    so the caller always gets a complete, usable list.
    """
    if not descriptors:
        return []

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

    district_lines = "\n".join(
        f"  - zone_index {d['zone_index']}: {d.get('position', 'unknown')} side, "
        f"{d.get('cell_count', '?')} cells"
        for d in descriptors
    )

    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"{context_block}"
        f"Districts to name ({len(descriptors)} total):\n{district_lines}\n\n"
        f"Return ONLY the JSON array — one object per district above."
    )

    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.9,
            max_tokens=max_tokens,
            timeout=60,
        )
    except LLMUnavailable as exc:
        print(f"[warn] district_namer: LLM unavailable ({exc}); using offline fallback names.")
        return fallback_districts(settlement, descriptors, effective_biome)
    except Exception as exc:
        print(f"[warn] district_namer: LLM call failed ({exc!r}); using generic names.")
        return _fallback_districts(descriptors)

    try:
        parsed = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] district_namer: failed to parse JSON array ({exc}); using generic names.")
        return _fallback_districts(descriptors)

    if not isinstance(parsed, list):
        print(f"[warn] district_namer: expected list, got {type(parsed).__name__}; using generic names.")
        return _fallback_districts(descriptors)

    # Index the model's objects by zone_index so we can match them back to the
    # descriptors we asked about (the model may reorder or drop entries).
    by_index: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            zid = int(item.get("zone_index"))
        except (TypeError, ValueError):
            continue
        by_index[zid] = item

    districts: list[dict] = []
    for d in descriptors:
        zid = d["zone_index"]
        item = by_index.get(zid)
        name = ""
        preset = DEFAULT_PRESET
        if item is not None:
            raw_name = item.get("name")
            if isinstance(raw_name, str):
                name = raw_name.strip().strip('"').strip("'").strip()
            raw_preset = str(item.get("preset", "")).strip().lower()
            if raw_preset in VALID_PRESETS:
                preset = raw_preset
        if not name:
            name = _fallback_name(d)
            print(f"[warn] district_namer: zone {zid} had no usable name; using {name!r}.")
        districts.append({"zone_index": zid, "name": name, "preset": preset})

    return districts
