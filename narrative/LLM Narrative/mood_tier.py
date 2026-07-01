from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from biome_context import biome_hint
from fallback_content import fallback_mood_tier
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_goal import goal_hint
from shared_events import events_hint

if TYPE_CHECKING:
    from settlement_generator import Settlement


VALID_TIERS = ("thriving", "strained", "struggling")
DEFAULT_TIER = "strained"


SYSTEM_PROMPT = """You judge the CURRENT material condition of a Minecraft settlement —
how its buildings would actually look right now: prosperous and well-kept,
merely getting by, or declining and neglected.

Output ONLY a JSON object. No markdown fences, no prose before or after.
The object must have exactly these two fields:
  "tier":   exactly one of "thriving", "strained", "struggling"
  "reason": one short sentence justifying the choice from the settlement's condition

Tier meanings (about physical UPKEEP and prosperity, not personality or mood):
  "thriving":   prosperous, growing, well-maintained — fresh timber, clean
                dressed stone, bright cloth, lanterns lit
  "strained":   getting by — weathered but functional, the ordinary working
                state. This is the DEFAULT; choose it unless the settlement
                clearly leans prosperous OR clearly leans toward decline.
  "struggling": declining, neglected, in crisis — moss and cracks, faded cloth,
                broken fences, things let go

How to weigh it:
  - the current struggle: is the settlement winning it (lean thriving) or losing
    it / overwhelmed by it (lean struggling)?
  - the historical_wound: fresh and severe -> struggling; old and healed -> fine
  - how much the collective_fear grips daily life
  - the primary_industry and outsider_reputation: prosperous and respected ->
    thriving; failing and reviled -> struggling
Do NOT reach for the extremes — most settlements are "strained". Only pick
thriving or struggling when the context clearly points that way."""


def _extract_json_object(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating surrounding junk.

    Self-contained copy, matching the other pre-pass/generator modules.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in model output:\n{text}")


def generate_mood_tier(
    settlement: "Settlement",
    max_tokens: int = 150,
) -> str:
    """Return the settlement's mood tier — one of `VALID_TIERS`.

    Grounds the judgment in the settlement's biome + axes + shared events +
    ongoing goal, exactly like the other generators (all four hints injected in
    the canonical order). Runs as the LAST pre-pass so the goal and events are
    available to weigh.

    Warn-and-recover: on any failure (transport error, malformed JSON, an
    unrecognized tier) logs [warn] and returns DEFAULT_TIER ("strained"). Never
    returns None and never raises — placement always needs a valid tier.
    """
    biome_block = ""
    hint = biome_hint(settlement.biome)
    if hint:
        biome_block = f"Biome: {hint}\n\n"

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

    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"{context_block}"
        f"Judge this settlement's current material condition. "
        f"Return ONLY the JSON object with keys tier and reason."
    )

    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.6,
            max_tokens=max_tokens,
            timeout=60,
        )
    except LLMUnavailable as exc:
        tier = fallback_mood_tier(settlement)
        print(f"[warn] mood_tier: LLM unavailable ({exc}); using offline fallback tier {tier!r}.")
        return tier if tier in VALID_TIERS else DEFAULT_TIER
    except Exception as exc:
        print(f"[warn] mood_tier: LLM call failed ({exc!r}); using {DEFAULT_TIER!r}.")
        return DEFAULT_TIER

    try:
        data = _extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] mood_tier: failed to parse JSON object ({exc}); using {DEFAULT_TIER!r}.")
        return DEFAULT_TIER

    if not isinstance(data, dict):
        print(f"[warn] mood_tier: expected object, got {type(data).__name__}; using {DEFAULT_TIER!r}.")
        return DEFAULT_TIER

    tier = data.get("tier")
    if isinstance(tier, str):
        tier = tier.strip().strip('"').strip("'").lower()
    if tier not in VALID_TIERS:
        print(f"[warn] mood_tier: unrecognized tier {data.get('tier')!r}; using {DEFAULT_TIER!r}.")
        return DEFAULT_TIER

    return tier
