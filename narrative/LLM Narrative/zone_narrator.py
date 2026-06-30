from __future__ import annotations

from biome_context import biome_hint
from fallback_content import fallback_subtitle
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_generator import Settlement
from settlement_goal import goal_hint
from shared_events import events_hint


SYSTEM_PROMPT = """You write one short subtitle for a Minecraft zone-entry title card.
Output ONLY the subtitle text. No quotes, no markdown, no prose around it.

Rules:
  - exactly one line, 6-12 words
  - italic flavor tone (a fragment of lore, not a full sentence)
  - reference the settlement's identity when natural — do not simply name-drop
  - let the settlement's primary industry and motif color the subtitle's imagery
  - if shared history is provided, the imagery may glance at one remembered event
  - if a current struggle is provided, the imagery may lean toward that objective
  - do not address the player directly, do not use second person
  - no emoji, no trailing punctuation beyond a period or ellipsis"""


def generate_zone_subtitle(
    zone_type: str,
    settlement: Settlement,
    biome: str | None = None,
    max_tokens: int = 60,
) -> str:
    """Return a one-line subtitle for the given zone type + settlement context.

    `biome` falls back to `settlement.biome` when not passed; when truthy,
    its trait hint is injected into the prompt so the subtitle references
    biome-appropriate imagery.
    """
    effective_biome = biome or settlement.biome
    hint = biome_hint(effective_biome)
    biome_line = f"Biome: {hint}.\n" if hint else ""
    axes = axes_hint(settlement)
    axes_block = f"{axes}\n" if axes else ""
    events = events_hint(settlement)
    events_block = f"{events}\n" if events else ""
    goal = goal_hint(settlement)
    goal_block = f"{goal}\n" if goal else ""
    user_prompt = (
        f"Settlement: {settlement.name} ({settlement.era}).\n"
        f"Story: {settlement.founding_story}\n"
        f"{biome_line}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"Zone type: {zone_type}\n\n"
        f"Write one short subtitle shown when a player enters this zone."
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
        print(f"[warn] generate_zone_subtitle: LLM unavailable ({exc}); using offline fallback subtitle.")
        return fallback_subtitle(zone_type, settlement, effective_biome)
    except Exception as exc:
        print(f"[warn] generate_zone_subtitle: LLM call failed ({exc!r}); using offline fallback subtitle.")
        return fallback_subtitle(zone_type, settlement, effective_biome)
    # Strip outer quotes/whitespace, then collapse any internal whitespace
    # (including stray newlines if the LLM emits a two-line subtitle) into
    # single spaces. The actionbar renders as one line and a literal \n
    # mid-string breaks the title text-component.
    cleaned = raw.strip().strip('"').strip("'").strip()
    return " ".join(cleaned.split())
