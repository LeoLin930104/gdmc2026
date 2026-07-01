from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from biome_context import biome_hint
from fallback_content import fallback_diaries
from identity_axes import axes_hint
from lm_client import LLMUnavailable, chat
from settlement_generator import Settlement
from settlement_goal import goal_hint
from shared_events import events_hint


# Vanilla Minecraft renders ~14 lines x ~19 chars on a written-book page.
# 256 is a comfortable cap with a small visual margin.
PAGE_CHAR_CAP = 256
MAX_PAGES = 3
TITLE_CHAR_CAP = 30


SYSTEM_PROMPT = f"""You write short in-character diary entries for NPCs in a Minecraft Java Edition 1.21 settlement.
Output ONLY a JSON array of diary objects. No markdown fences, no prose before or after.

Each object must have these fields:
  "author_name":  the NPC's full name, evocative and biome/era-appropriate,
                  INVENTED for this settlement (a familiar elder's name, a
                  titled officer, a temple sibling — but your own words, never
                  a name copied from these instructions)
  "author_role":  one or two words for what they do in the settlement, fitting
                  the zone they belong to and the biome
                  (e.g. "baker", "fisher", "watch-captain", "herbalist")
  "book_title":   short title shown on the book cover, <= {TITLE_CHAR_CAP} characters
  "zone_id":      MUST exactly match one of the zone_id values listed by the user
  "pages":        a JSON ARRAY of strings. Length 1 is preferred.
                  Each page MUST be <= {PAGE_CHAR_CAP} characters - this is what
                  fits on a single Minecraft book page. Aim for ~150-200 chars
                  per page (2-4 short sentences). NEVER exceed 256 chars.
                  Up to {MAX_PAGES} pages allowed if the entry runs long, but
                  always prefer fewer.

Write each diary as ONE TERSE self-contained journal entry:
  - 2-4 short sentences, total ~150-200 chars; long-winded entries are wrong
  - first person, undated or with in-world time markers ("the third frost",
    "after the long rain")
  - one concrete biome-grounded sensory detail per entry, no padding
  - reference shared events, names, or rumours across at least two diaries so
    the village reads as one place, but each entry stays readable on its own
  - when identity axes are provided, let the motif and historical wound surface
    across multiple diaries (different authors notice them differently — a
    baker sees the motif on her shutters, a watchman sees it on a banner)
  - tone each entry to the settlement's central virtue and collective fear
  - when a "Current struggle" is provided, at least two diaries should mention
    that ongoing objective from their own role's angle (the baker frets over the
    grain it depends on; the watch-captain counts the hands it will take)
  - when a "Shared history" list is provided, EACH diary should reference 1-2
    events from that list by name or close paraphrase, FROM THIS AUTHOR'S
    PERSPECTIVE — different authors notice the same event differently (the
    baker sees the rain ruining flour; the watch-captain sees the rain hiding
    an expedition's departure). At least two diaries in the set should pick
    the SAME event from different angles. This is the cohesion contract.
  - cut anything that isn't doing work - no preamble, no closing reflection
  - no second-person address to the reader, no modern idiom, no quotation marks
  - no markdown, no headings, no labels - just the diary text"""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Diary:
    author_name: str
    author_role: str
    book_title: str
    zone_id: str
    pages: list[str]


# ---------------------------------------------------------------------------
# JSON parsing / normalization
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list:
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


_SENTENCE_ENDS = (". ", "! ", "? ", "… ", "; ")


def _find_split(text: str, max_chars: int) -> int:
    window = text[: max_chars + 1]
    best = -1
    for ending in _SENTENCE_ENDS:
        idx = window.rfind(ending)
        if idx > best:
            best = idx + len(ending)
    if best > 0:
        return best
    space = window.rfind(" ")
    if space > 0:
        return space + 1
    return max_chars


def _split_to_pages(text: str, max_chars: int, max_pages: int) -> list[str]:
    pages: list[str] = []
    remaining = text.strip()
    while remaining and len(pages) < max_pages:
        if len(remaining) <= max_chars:
            pages.append(remaining)
            remaining = ""
            break
        cut = _find_split(remaining, max_chars)
        pages.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        print(
            f"[warn] Diary entry exceeded {max_pages} pages of {max_chars} "
            f"chars; truncating {len(remaining)} trailing char(s)."
        )
    return pages


def _normalize_pages(value) -> list[str]:
    if isinstance(value, str):
        chunks = [value]
    elif isinstance(value, list):
        chunks = [str(p).strip() for p in value if str(p).strip()]
    else:
        raise ValueError(f"Diary 'pages' must be string or list, got {type(value).__name__}")
    if not chunks:
        raise ValueError("Diary 'pages' is empty")

    # If every chunk already fits, keep them as-is up to MAX_PAGES.
    if all(len(c) <= PAGE_CHAR_CAP for c in chunks) and len(chunks) <= MAX_PAGES:
        return chunks

    # Otherwise rejoin and split at sentence boundaries.
    joined = " ".join(chunks).strip()
    return _split_to_pages(joined, max_chars=PAGE_CHAR_CAP, max_pages=MAX_PAGES)


def _normalize_diary(raw: dict) -> Diary:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected diary dict, got {type(raw).__name__}: {raw}")

    # Hard-required: identity + placement. Missing these makes the diary unplaceable.
    for key in ("author_name", "book_title", "zone_id"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Diary missing or empty {key!r}: {raw}")

    # author_role is just a console label; LLMs occasionally emit null.
    # Default rather than crash the pipeline.
    role_raw = raw.get("author_role")
    if isinstance(role_raw, str) and role_raw.strip():
        role = role_raw.strip()
    else:
        role = "settler"
        print(f"[warn] Diary by {raw['author_name']!r} had no author_role; defaulting to 'settler'.")

    title = raw["book_title"].strip()
    if len(title) > TITLE_CHAR_CAP:
        title = title[:TITLE_CHAR_CAP].rstrip()

    pages = _normalize_pages(raw.get("pages"))

    return Diary(
        author_name=raw["author_name"].strip(),
        author_role=role,
        book_title=title,
        zone_id=raw["zone_id"].strip(),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_diaries(
    settlement: Settlement,
    zone_specs: Iterable[tuple],
    biome: str | None = None,
    max_tokens: int = 1200,
) -> list[Diary]:
    specs = [(z[0], z[1], z[2]) for z in zone_specs]
    if not specs:
        raise ValueError("zone_specs is empty - no diaries to generate")

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

    user_prompt = (
        f"{biome_block}"
        f"{axes_block}"
        f"{events_block}"
        f"{goal_block}"
        f"{context_block}"
        f"Zones in this settlement (write ONE diary per zone):\n"
        f"{zones_block}\n\n"
        f"Generate exactly {len(specs)} diaries (one per zone), each from a different NPC. "
        f"Cross-reference shared events or people across at least two diaries so the "
        f"village reads as one place. Return ONLY the JSON array."
    )

    try:
        raw = chat(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
            temperature=0.9,
            max_tokens=max_tokens,
            timeout=120,
        )
    except LLMUnavailable as exc:
        print(f"[warn] generate_diaries: LLM unavailable ({exc}); using offline fallback diaries.")
        return fallback_diaries(settlement, specs, effective_biome)
    except Exception as exc:
        print(f"[warn] generate_diaries: LLM call failed ({exc!r}); using offline fallback diaries.")
        return fallback_diaries(settlement, specs, effective_biome)

    # Strict parse first; on a stochastic JSON glitch, fall back to salvaging
    # individual objects so one malformed entry doesn't waste the whole run
    # (settlement/goal/events/zones/datapack already succeeded by this point).
    try:
        parsed = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        candidates = parsed
    else:
        candidates = _salvage_json_objects(raw)
        if candidates:
            print(f"[warn] Diary JSON was malformed; salvaged "
                  f"{len(candidates)} object(s) individually.")
        else:
            print("[warn] generate_diaries: could not parse any diary object; "
                  "returning []. Pipeline continues with no lecterns placed.")
            return []

    known_zone_ids = {zid for zid, _, _ in specs}
    diaries: list[Diary] = []
    for d in candidates:
        try:
            diary = _normalize_diary(d)
        except (ValueError, TypeError) as exc:
            print(f"[warn] Skipping malformed diary object ({exc}).")
            continue
        # Drop hallucinated zone_ids here rather than letting
        # match_diaries_to_zones raise and abort the run.
        if diary.zone_id not in known_zone_ids:
            print(f"[warn] Diary by {diary.author_name!r} has unknown "
                  f"zone_id {diary.zone_id!r}; skipping.")
            continue
        diaries.append(diary)

    if not diaries:
        print("[warn] generate_diaries: no usable diaries after normalization; "
              "returning []. Pipeline continues with no lecterns placed.")
    return diaries
