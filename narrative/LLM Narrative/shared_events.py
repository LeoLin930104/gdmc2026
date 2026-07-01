from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from biome_context import biome_hint
from fallback_content import fallback_shared_events
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_goal import goal_hint

if TYPE_CHECKING:
    from settlement_generator import Settlement


SYSTEM_PROMPT = """You generate concrete historical events for a Minecraft settlement.
Output ONLY a JSON array of strings. No markdown fences, no prose before or after.

Each event is ONE short noun phrase the settlement remembers — 3-8 words,
specific, grounded in the settlement's identity (industry, virtue, fear,
wound, biome). Past tense or undated, like a chapter heading in local
memory. These show the SHAPE only — invent your own, never copy them:
  "the long rain of '12"
  "the downstream expedition that never returned"
  "the mill fire"
  "the winter the wolves crossed the ridge"
  "the failed pilgrimage to the high shrine"
  "the smith's daughter's wedding"

Rules:
  - 3-8 words per event, prose noun phrase, no full sentences
  - mix scales: at least one disaster or hardship AND at least one human-scale
    moment (a wedding, a feast, a quarrel, a birth, a falling-out)
  - no overlap with the settlement's historical_wound — that one is already
    on the record; these are the OTHER things the village remembers
  - if a "Current struggle" is given, at least one event should be a setback or
    a milestone tied to that struggle (the events shape what the village faces)
  - no fantasy proper nouns the player can't decode (no "the Vesperal Pact")
  - invent every event from THIS settlement's own identity; never reuse a name,
    place, or phrase from the example list above
  - lowercase except proper names; no trailing punctuation
  - no markdown, no quotes around individual entries"""


def _extract_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    if start != -1:
        # Decode the FIRST complete array and ignore any trailing junk. This
        # tolerates models that emit two arrays back-to-back (the greedy
        # first-[ .. last-] slice would span both and raise "Extra data").
        try:
            value, _end = json.JSONDecoder().raw_decode(text, start)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass
        # Last resort: greedy first-[ .. last-] (handles a single array with
        # stray brackets inside string values).
        end = text.rfind("]")
        if end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON array found in model output:\n{text}")


def generate_shared_events(
    settlement: "Settlement",
    count: int = 4,
    max_tokens: int = 400,
) -> list[str]:
    biome_block = ""
    hint = biome_hint(settlement.biome)
    if hint:
        biome_block = f"Biome: {hint}\n\n"

    axes = axes_hint(settlement)
    axes_block = f"{axes}\n\n" if axes else ""

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
        f"{goal_block}"
        f"{context_block}"
        f"Generate exactly {count} concrete events this settlement remembers. "
        f"Return ONLY the JSON array of strings."
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
        print(f"[warn] shared_events: LLM unavailable ({exc}); using offline fallback events.")
        return fallback_shared_events(settlement, count)
    except Exception as exc:
        print(f"[warn] shared_events: LLM call failed ({exc!r}); returning [].")
        return []

    try:
        parsed = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] shared_events: failed to parse JSON array ({exc}); returning [].")
        return []

    if not isinstance(parsed, list):
        print(f"[warn] shared_events: expected list, got {type(parsed).__name__}; returning [].")
        return []

    events: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            value = item.strip().strip('"').strip("'").strip()
        elif isinstance(item, dict):
            # Some models emit {"event": "..."} or {"name": "..."}; salvage.
            for key in ("event", "name", "text"):
                if isinstance(item.get(key), str) and item[key].strip():
                    value = item[key].strip()
                    break
            else:
                continue
        else:
            continue
        if value:
            events.append(value)
        if len(events) >= count:
            break

    if not events:
        print(f"[warn] shared_events: parsed array had no usable strings; returning [].")
        return []

    return events


def events_hint(settlement: "Settlement") -> str:
    events = getattr(settlement, "shared_events", None)
    if not events:
        return ""
    lines = [f"  - {e.strip()}" for e in events if isinstance(e, str) and e.strip()]
    if not lines:
        return ""
    return "Shared history:\n" + "\n".join(lines)
