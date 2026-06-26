from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biome_context import biome_hint
from fallback_content import fallback_goal
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat

if TYPE_CHECKING:
    from settlement_generator import Settlement


SYSTEM_PROMPT = """You invent ONE ongoing goal for a Minecraft settlement.
Output ONLY a JSON object. No markdown fences, no prose before or after.

The object must have exactly these two fields:
  "summary": the objective the WHOLE settlement is working toward right now,
             a short present-tense phrase, 4-10 words. Concrete and active.
             Examples that show the SHAPE only — never copy one verbatim:
               "rebuild the river-mill before flood season"
               "lay in stores before the long freeze"
               "drive the wolves back past the ridge"
               "reopen the collapsed eastern shaft"
               "win back the trade road through the high pass"
  "stakes": one sentence on what is at risk or why it matters now.

Rules:
  - the goal MUST grow out of THIS settlement's identity: ground it in the
    historical_wound, collective_fear, or primary_industry (e.g. a wound about
    a flood -> rebuild the mill; a fear of winter -> lay in stores; an industry
    of mining -> reopen a shaft)
  - do NOT introduce a new industry the settlement doesn't have, and do NOT name
    a place, person, or neighbor that is not already in the settlement context
  - the example phrases above are SHAPE templates, not content — invent your own
    objective from this settlement; never reuse an example's words or places
  - present tense, forward-looking, achievable — something underway, not a
    vague wish and not a finished triumph
  - no fantasy proper nouns the player can't decode (no "the Vesperal Pact")
  - plain language, no markdown, no labels inside the values"""


def _extract_json_object(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating surrounding junk.

    Same shape as the extractor `settlement_generator` carries — copied here
    rather than imported to keep this module self-contained (matches how the
    relic/diary/tool/shared-events modules each keep their own copy).
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


@dataclass
class SettlementGoal:
    summary: str
    stakes: str = ""


def generate_settlement_goal(
    settlement: "Settlement",
    max_tokens: int = 300,
) -> SettlementGoal | None:
    """Generate one ongoing goal/struggle grounded in `settlement`'s identity.

    The call grounds the goal in the settlement's biome + identity axes + core
    context so it reads as a natural consequence of who the place is. Runs
    before the shared-events pre-pass so events can reference it.

    Warn-and-recover: on any failure (transport error from `chat`, malformed
    JSON, missing `summary`) logs [warn] and returns None. The downstream
    pipeline continues with `goal_hint(settlement)` collapsing to "" — same
    behavior as a no-goal Settlement. `stakes` is soft (defaults to "").
    """
    biome_block = ""
    hint = biome_hint(settlement.biome)
    if hint:
        biome_block = f"Biome: {hint}\n\n"

    axes = axes_hint(settlement)
    axes_block = f"{axes}\n\n" if axes else ""

    context_block = (
        f"Settlement context:\n"
        f"  Name: {settlement.name}\n"
        f"  Era: {settlement.era}\n"
        f"  Story: {settlement.founding_story}\n\n"
    )

    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{context_block}"
        f"Invent the single goal this settlement is working toward right now. "
        f"Return ONLY the JSON object with keys summary and stakes."
    )

    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=max_tokens,
            timeout=60,
        )
    except LLMUnavailable as exc:
        print(f"[warn] settlement_goal: LLM unavailable ({exc}); using offline fallback goal.")
        return fallback_goal(settlement)
    except Exception as exc:
        print(f"[warn] settlement_goal: LLM call failed ({exc!r}); returning None.")
        return None

    try:
        data = _extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] settlement_goal: failed to parse JSON object ({exc}); returning None.")
        return None

    if not isinstance(data, dict):
        print(f"[warn] settlement_goal: expected object, got {type(data).__name__}; returning None.")
        return None

    summary = data.get("summary")
    if isinstance(summary, list):
        summary = " ".join(str(v).strip() for v in summary if str(v).strip())
    if not isinstance(summary, str) or not summary.strip():
        print(f"[warn] settlement_goal: missing or empty 'summary'; returning None.")
        return None

    stakes = data.get("stakes")
    if isinstance(stakes, list):
        stakes = " ".join(str(v).strip() for v in stakes if str(v).strip())
    if not isinstance(stakes, str):
        stakes = ""

    return SettlementGoal(summary=summary.strip(), stakes=stakes.strip())


def goal_hint(settlement: "Settlement") -> str:
    """Return a compact current-struggle block for prompt injection, or "" if empty.

    Skips entirely when `settlement.goal` is None, so hand-constructed or
    pre-pre-pass Settlements add zero prompt noise. Same defensive contract as
    `biome_context.biome_hint`, `identity_axes.axes_hint`, and
    `shared_events.events_hint`. The stakes line is skipped when empty.

    Example output:

      Current struggle:
        Working toward: rebuild the river-mill before flood season.
        What's at stake: without it the grain stores won't last the winter.
    """
    goal = getattr(settlement, "goal", None)
    if goal is None:
        return ""
    summary = getattr(goal, "summary", "")
    if not isinstance(summary, str) or not summary.strip():
        return ""
    lines = [f"  Working toward: {summary.strip()}"]
    stakes = getattr(goal, "stakes", "")
    if isinstance(stakes, str) and stakes.strip():
        lines.append(f"  What's at stake: {stakes.strip()}")
    return "Current struggle:\n" + "\n".join(lines)
