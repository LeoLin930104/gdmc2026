from __future__ import annotations

import json
from typing import Iterable

from gdpc import Editor, Block


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def zone_center_floor(zone) -> tuple[int, int, int]:
    """Return the floor-level center coordinate of `zone` for lectern placement.

    `zone` is an Area Discovery `Zone` (it has a `.aabb` field with min-corner
    `(x, y, z)` and extents `(dx, dy, dz)`). The lectern lands on the AABB
    floor (`aabb.y`) at the horizontal midpoint.
    """
    aabb = zone.aabb
    cx = aabb.x + aabb.dx // 2
    cz = aabb.z + aabb.dz // 2
    return (cx, aabb.y, cz)


# ---------------------------------------------------------------------------
# SNBT construction
# ---------------------------------------------------------------------------

def build_lectern_snbt(diary) -> str:
    """Build the lectern block-entity SNBT carrying `diary` as a written book.

    Targets MC 1.21.11: written books use the `minecraft:written_book_content`
    component (post-1.20.5 format) with `title`, `author`, and `pages`.
    Equivalent to `/give @s minecraft:written_book[minecraft:written_book_content=
    {title:"...",author:"...",pages:[[["..."]]]}]`.
    """
    title  = json.dumps(diary.book_title,  ensure_ascii=False)
    author = json.dumps(diary.author_name, ensure_ascii=False)
    # MC 1.21.9+ (we target 1.21.11): each page is an INLINE text-component list
    # [["text"]], NOT the pre-1.21.9 single-quoted string-of-JSON form. Verified:
    #   /give @a written_book[written_book_content={pages:[[["test text"]]],...}]
    # (Pre-1.21.9 wanted pages:['[["test text"]]']; the {text:"..."}/{raw:"..."}
    # compound forms still render blank via gdpc's data= path.)
    pages = "[" + ",".join(_format_page(p) for p in diary.pages) + "]"

    book_snbt = (
        f'{{id:"minecraft:written_book",'
        f'count:1,'
        f'components:{{'
        f'"minecraft:written_book_content":{{'
        f'title:{title},'
        f'author:{author},'
        f'pages:{pages},'
        f'generation:0'
        f'}}}}}}'
    )
    return f"{{Book:{book_snbt},Page:0}}"


def _format_page(text: str) -> str:
    """Encode one page as an inline SNBT text-component list: `[["text"]]`.

    MC 1.21.9+ (we target 1.21.11) takes each page as a raw list value
    (`pages:[[["text"]]]`), NOT the pre-1.21.9 single-quoted string-of-JSON
    form (`pages:['[["text"]]']`). json.dumps escapes the inner string exactly
    as an SNBT double-quoted string expects, so no extra escaping/quoting is
    needed — we emit the array literal directly.
    """
    return f"[[{json.dumps(text, ensure_ascii=False)}]]"


def build_book_item_nbt(diary, slot: int) -> str:
    """Build the SNBT for `diary` as a written-book ITEM in a chest slot.

    The lectern form (`build_lectern_snbt`) wraps the book as a block-entity's
    held `Book`; this is the same `minecraft:written_book` stack but as a chest
    `Items` entry (with a `Slot`), so a diary can live inside a premade build's
    chest alongside its tool. Same 1.21.11 `written_book_content` component and
    the inline `pages:[[["text"]]]` list page form.
    """
    title  = json.dumps(diary.book_title,  ensure_ascii=False)
    author = json.dumps(diary.author_name, ensure_ascii=False)
    pages = "[" + ",".join(_format_page(p) for p in diary.pages) + "]"
    return (
        f'{{Slot:{slot}b,'
        f'id:"minecraft:written_book",'
        f'count:1,'
        f'components:{{"minecraft:written_book_content":{{'
        f'title:{title},author:{author},pages:{pages},generation:0}}}}}}'
    )


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def place_lectern(editor: Editor, pos: tuple[int, int, int], snbt: str) -> None:
    """Place a lectern at `pos` holding the book described by `snbt`.

    Block state `has_book=true` is set inline so the lectern renders the book
    immediately on placement. `facing=south` is an arbitrary default — the
    book is readable from any side.
    """
    block = Block("minecraft:lectern[has_book=true,facing=south]", data=snbt)
    editor.placeBlock(pos, block)


# ---------------------------------------------------------------------------
# Diary <-> Zone matching
# ---------------------------------------------------------------------------

def match_diaries_to_zones(diaries: Iterable, zones: Iterable) -> list[tuple]:
    """Pair each diary with its matching `Zone` by `zone_id`.

    Raises `ValueError` if the LLM emits a `zone_id` that doesn't appear in
    the zone list — surfaces hallucinations immediately rather than silently
    dropping diaries.
    """
    zones_by_id = {z.zone_id: z for z in zones}
    pairs: list[tuple] = []
    for diary in diaries:
        zone = zones_by_id.get(diary.zone_id)
        if zone is None:
            raise ValueError(
                f"Diary by {diary.author_name!r} references unknown "
                f"zone_id {diary.zone_id!r}. Known zones: {list(zones_by_id)}"
            )
        pairs.append((diary, zone))
    return pairs


# ---------------------------------------------------------------------------
# Per-zone tool chest helpers
# ---------------------------------------------------------------------------

_GLINT_RARITIES = {"Rare", "Epic", "Legendary"}


def tool_chest_pos(zone) -> tuple[int, int, int]:
    """Return the chest position one block east of the zone's lectern.

    Lectern lands at `zone_center_floor`; the chest sits at (cx + 1, y, cz).
    West (cx - 1) is equally valid — east chosen for consistency.
    """
    aabb = zone.aabb
    cx = aabb.x + aabb.dx // 2
    cz = aabb.z + aabb.dz // 2
    return (cx + 1, aabb.y, cz)


def glint_for_rarity(rarity: str) -> bool:
    """True for Rare/Epic/Legendary tools; False otherwise.

    Glint signals "this is special," not "this is non-wooden" — Uncommon
    iron gear should remain matte.
    """
    return rarity in _GLINT_RARITIES


def build_tool_chest_snbt(tool, glint: bool) -> str:
    """Build a single-item chest SNBT carrying `tool` in slot 0.

    Reuses `build_item_nbt` from place_relic_chest so the item rendering
    (custom_name + lore + optional glint) stays consistent with relic chests.
    """
    from place_relic_chest import build_item_nbt  # avoid import cycle at module load
    item_dict = {
        "name":        tool.name,
        "item_type":   tool.item_type,
        "description": tool.description,
        "lore":        tool.lore,
        "color":       tool.color,
    }
    item_nbt = build_item_nbt(item_dict, slot=0, glint=glint)
    return "{Items:[" + item_nbt + "]}"


def match_tools_to_zones(tools: Iterable, zones: Iterable) -> list[tuple]:
    """Pair each tool with its matching `Zone` by `zone_id`.

    Raises `ValueError` if the LLM emits a `zone_id` that doesn't appear in
    the zone list — same hallucination-surfacing pattern as
    `match_diaries_to_zones`.
    """
    zones_by_id = {z.zone_id: z for z in zones}
    pairs: list[tuple] = []
    for tool in tools:
        zone = zones_by_id.get(tool.zone_id)
        if zone is None:
            raise ValueError(
                f"Tool {tool.name!r} references unknown "
                f"zone_id {tool.zone_id!r}. Known zones: {list(zones_by_id)}"
            )
        pairs.append((tool, zone))
    return pairs
