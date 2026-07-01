from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biome_context import biome_hint
from fallback_content import fallback_settlement
from identity_axes import AXIS_OPTIONS
from lm_client import LLMUnavailable, chat

if TYPE_CHECKING:
    from settlement_goal import SettlementGoal


# Eight identity axes the LLM picks at settlement-generation time and that
# downstream prompts (zone subtitles, relics, diaries, tools) inject via
# `identity_axes.axes_hint`. Order matches `_AXIS_LABELS` for display parity.
AXIS_FIELDS = (
    "primary_industry",
    "central_virtue",
    "collective_fear",
    "historical_wound",
    "motif",
    "worldview",
    "social_structure",
    "outsider_reputation",
)


def _join_options(key: str) -> str:
    return ", ".join(AXIS_OPTIONS[key])


SYSTEM_PROMPT = f"""You invent a single Minecraft settlement's identity for a procedural world.
Output ONLY a JSON object. No markdown fences, no prose before or after.

The object must have exactly these eleven fields:

  Core identity:
    "name":           the settlement's proper name, 1-3 words, evocative
    "era":            a short phrase placing it in time, e.g. "late bronze age"
                      or "three centuries after the Sundering"
    "founding_story": 2-3 sentences describing how the settlement came to be,
                      its character, and what haunts or defines it today

  Identity axes (each a short phrase, 1-6 words):
    "primary_industry":    what the settlement DOES, e.g. one of: {_join_options("primary_industry")}
    "central_virtue":      what it ADMIRES above all, e.g. one of: {_join_options("central_virtue")}
    "collective_fear":     what HAUNTS it, e.g. one of: {_join_options("collective_fear")}
    "historical_wound":    one defining traumatic event in its past
                           (free-form short phrase; these show SHAPE only, do not
                            copy them — "the deep shaft cave-in of '47",
                            "the betrayal at the river ford", "the long winter of 312")
    "motif":               a repeated symbol that recurs across its culture
                           (free-form short phrase, e.g. "red lanterns",
                            "spiral carvings", "ravens", "iron bells")
    "worldview":           a short spiritual saying it lives by
                           (free-form short phrase, e.g. "the mountain remembers",
                            "the sea gives and takes", "ancestors watch the door")
    "social_structure":    how power works, e.g. one of: {_join_options("social_structure")}
    "outsider_reputation": what nearby settlements think of it,
                           e.g. one of: {_join_options("outsider_reputation")}

Write in the tone of a fantasy setting. All eleven fields must be internally
consistent — name, era, founding story, and the eight axes describe ONE place
and reinforce each other (a mining town with fear=collapse and motif=spiral
carvings, not a mining town with fear=flooding and motif=seashells)."""


def _extract_json_object(text: str) -> dict:
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


@dataclass
class Settlement:
    name: str
    era: str
    founding_story: str
    theme: str
    biome: str | None = None
    # Identity axes — see identity_axes.py. All optional so hand-constructed
    # Settlements still work; downstream `axes_hint` skips None/empty values.
    primary_industry:    str | None = None
    central_virtue:      str | None = None
    collective_fear:     str | None = None
    historical_wound:    str | None = None
    motif:               str | None = None
    worldview:           str | None = None
    social_structure:    str | None = None
    outsider_reputation: str | None = None
    # Concrete history. Populated by `shared_events.generate_shared_events()`
    # as a pre-pass after `generate_settlement()` returns; mutated onto the
    # same Settlement instance so all downstream generators inherit it.
    shared_events:       list[str] | None = None
    # Ongoing objective. Populated by `settlement_goal.generate_settlement_goal()`
    # as a pre-pass (runs BEFORE shared_events so events can reference it);
    # mutated onto the same Settlement instance so all downstream generators
    # inherit it via `settlement_goal.goal_hint`.
    goal:                "SettlementGoal | None" = None
    # Current material condition: "thriving" / "strained" / "struggling".
    # Populated by `mood_tier.generate_mood_tier()` as the LAST pre-pass (after
    # goal + shared_events, which it weighs). Drives the premade-build palette
    # swap (Premade Builds/families.py via premade_placer.mood_tier_for); unlike
    # the other pre-pass fields it has no `*_hint` (it selects blocks, not prose).
    mood_tier:           str | None = None


def generate_settlement(
    theme: str,
    biome: str | None = None,
    max_tokens: int = 700,
) -> Settlement:
    hint = biome_hint(biome)
    biome_line = f"Biome: {hint}. Let this shape mood, materials, and threats.\n\n" if hint else ""
    user_prompt = (
        f"{biome_line}"
        f"Invent a settlement themed around: {theme}\n\n"
        f"Return ONLY a JSON object with keys name, era, founding_story, "
        f"primary_industry, central_virtue, collective_fear, historical_wound, "
        f"motif, worldview, social_structure, outsider_reputation."
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
        print(f"[warn] generate_settlement: LLM unavailable ({exc}); "
              f"using offline fallback settlement.")
        return fallback_settlement(theme, biome)
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    for field in ("name", "era", "founding_story"):
        value = data.get(field)
        if isinstance(value, list):
            value = " ".join(str(v).strip() for v in value)
            data[field] = value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Settlement missing or empty {field!r}: {data}")

    # Soft normalization for the eight identity axes: coerce lists to strings,
    # accept any non-empty string, leave anything else as None. Never raise —
    # a partial axis set is still useful downstream, and the single LLM call
    # is too expensive to throw away over one missing axis.
    axes: dict[str, str | None] = {}
    missing: list[str] = []
    for field in AXIS_FIELDS:
        value = data.get(field)
        if isinstance(value, list):
            value = " ".join(str(v).strip() for v in value if str(v).strip())
        if isinstance(value, str) and value.strip():
            axes[field] = value.strip()
        else:
            axes[field] = None
            missing.append(field)
    if missing:
        print(f"[warn] Settlement missing axes {missing}; defaulting to None.")

    return Settlement(
        name=data["name"].strip(),
        era=data["era"].strip(),
        founding_story=data["founding_story"].strip(),
        theme=theme,
        biome=biome,
        **axes,
    )
