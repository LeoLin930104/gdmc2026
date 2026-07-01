from __future__ import annotations

import json
from typing import Iterable

from gdpc import Editor, Block


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def zone_center_floor(zone) -> tuple[int, int, int]:
    aabb = zone.aabb
    cx = aabb.x + aabb.dx // 2
    cz = aabb.z + aabb.dz // 2
    return (cx, aabb.y, cz)


# ---------------------------------------------------------------------------
# SNBT construction
# ---------------------------------------------------------------------------

def build_lectern_snbt(diary) -> str:
    title  = json.dumps(diary.book_title,  ensure_ascii=False)
    author = json.dumps(diary.author_name, ensure_ascii=False)
    # Each page is a single-quoted SNBT string containing a JSON text-component
    # array of the form [["text"]]. Verified in-game with:
    #   /give @a written_book[written_book_content={pages:['[["test text"]]'],...}]
    # The {text:"..."} and {raw:"..."} compound forms both rendered blank when
    # written via gdpc's data= path; this string-of-JSON form is what works.
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
    component_json = f"[[{json.dumps(text, ensure_ascii=False)}]]"
    escaped = component_json.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def build_book_item_nbt(diary, slot: int) -> str:
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
    block = Block("minecraft:lectern[has_book=true,facing=south]", data=snbt)
    editor.placeBlock(pos, block)


# ---------------------------------------------------------------------------
# Diary <-> Zone matching
# ---------------------------------------------------------------------------

def match_diaries_to_zones(diaries: Iterable, zones: Iterable) -> list[tuple]:
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
    aabb = zone.aabb
    cx = aabb.x + aabb.dx // 2
    cz = aabb.z + aabb.dz // 2
    return (cx + 1, aabb.y, cz)


def glint_for_rarity(rarity: str) -> bool:
    return rarity in _GLINT_RARITIES


def build_tool_chest_snbt(tool, glint: bool) -> str:
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
