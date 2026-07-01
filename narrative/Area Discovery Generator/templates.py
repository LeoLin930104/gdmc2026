from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Zone, SoundConfig, TitleConfig


# ---------------------------------------------------------------------------
# pack.mcmeta
# ---------------------------------------------------------------------------

def pack_mcmeta(description: str, pack_format: int = 26) -> str:
    data = {
        "pack": {
            "pack_format": pack_format,
            "description": description,
        }
    }
    return json.dumps(data, indent=2) + "\n"


# ---------------------------------------------------------------------------
# tags/functions/tick.json
# ---------------------------------------------------------------------------

def tick_tag_json(namespace: str) -> str:
    data = {"values": [f"{namespace}:tick"]}
    return json.dumps(data, indent=2) + "\n"


# ---------------------------------------------------------------------------
# functions/setup.mcfunction
# ---------------------------------------------------------------------------

def setup_function(namespace: str) -> str:
    return f"""\
# =============================================================================
# {namespace}:setup
# Run once to initialize the scoreboard objective used by all zones.
# Add to data/minecraft/tags/functions/load.json to auto-run on datapack load.
# =============================================================================

scoreboard objectives add ad_zone dummy
tellraw @a [\\
  {{"text":"[Area Discovery] ","color":"gold"}},\\
  {{"text":"System initialized.","color":"white"}}\\
]
"""


# ---------------------------------------------------------------------------
# functions/tick.mcfunction
# ---------------------------------------------------------------------------

def tick_function(namespace: str, zone_ids: list[str]) -> str:
    header = f"""\
# =============================================================================
# {namespace}:tick
# Registered in tags/functions/tick.json — runs every game tick.
#
# HOW TO ADD A NEW ZONE:
#   1. Run the generator with your new Zone() added to the zone list.
#      The generator rewrites this file automatically.
#   2. Or manually append:
#        execute as @a at @s run function {namespace}:zones/YOUR_ZONE_ID
# =============================================================================

"""
    lines = [
        f"execute as @a at @s run function {namespace}:zones/{zid}"
        for zid in zone_ids
    ]
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# functions/zones/<zone_id>.mcfunction
# ---------------------------------------------------------------------------

def zone_function(namespace: str, zone: "Zone") -> str:
    sel   = zone.aabb.selector_args()
    tag   = zone.tag_name
    notes = f"# Notes: {zone.notes}\n#\n" if zone.notes else ""

    return f"""\
# =============================================================================
# {namespace}:zones/{zone.zone_id}
# Zone:   {zone.display_name}
# Volume: {zone.aabb}
# Tag:    {tag}
# {notes}\
# =============================================================================

# --- ENTRY -------------------------------------------------------------------
# Fires on the FIRST tick the player steps inside the bounding box.
execute \\
  if entity @s[{sel}] \\
  unless entity @s[tag={tag}] \\
  run function {namespace}:titles/{zone.zone_id}

execute \\
  if entity @s[{sel}] \\
  unless entity @s[tag={tag}] \\
  run tag @s add {tag}

# --- EXIT --------------------------------------------------------------------
# Fires on the first tick the player is outside and the tag still exists.
execute \\
  if entity @s[tag={tag}] \\
  unless entity @s[{sel}] \\
  run tag @s remove {tag}
"""


# ---------------------------------------------------------------------------
# functions/titles/<zone_id>.mcfunction
# ---------------------------------------------------------------------------

def title_function(namespace: str, zone: "Zone") -> str:
    tc: TitleConfig = zone.title

    # Actionbar is a single line — combine main title + separator + subtitle parts
    # into one JSON array. `title @s times` does not affect actionbar timing.
    components: list[dict] = [tc._main_component().to_json_dict()]
    sub_parts = tc._subtitle_components()
    if sub_parts:
        components.append({"text": " — ", "color": "gray"})
        components.extend(p.to_json_dict() for p in sub_parts)
    actionbar_json = json.dumps(components)

    # Build optional sound line
    sc: SoundConfig = zone.sound
    sound_line = (
        f"\nplaysound {sc.sound_id} {sc.source.value} @s ~ ~ ~ {sc.volume} {sc.pitch}"
        if not sc.is_silent
        else "\n# (no sound configured for this zone)"
    )

    return f"""\
# =============================================================================
# {namespace}:titles/{zone.zone_id}
# Called by zones/{zone.zone_id}.mcfunction on the first tick of entry.
# Context: @s is the entering player.
# =============================================================================

title @s actionbar {actionbar_json}
{sound_line}
"""


# ---------------------------------------------------------------------------
# functions/zones/_template.mcfunction
# ---------------------------------------------------------------------------

def zone_template(namespace: str) -> str:
    return f"""\
# =============================================================================
# {namespace}:zones/_template
# Copy this file to  zones/YOUR_ZONE_ID.mcfunction  to add a new zone.
# Also copy  titles/_template.mcfunction  to  titles/YOUR_ZONE_ID.mcfunction
#
# STEP 1  Replace all occurrences of ZONE_ID with your snake_case zone id
# STEP 2  Fill in the AABB coordinates (x, y, z = minimum corner; dx/dy/dz = extent - 1)
# STEP 3  Add one line to tick.mcfunction:
#           execute as @a at @s run function {namespace}:zones/ZONE_ID
# STEP 4  Customise titles/ZONE_ID.mcfunction with your area name and colors
# =============================================================================

# --- ENTRY -------------------------------------------------------------------
execute \\
  if entity @s[x=0,y=0,z=0,dx=9,dy=9,dz=9] \\
  unless entity @s[tag=ad_in_ZONE_ID] \\
  run function {namespace}:titles/ZONE_ID

execute \\
  if entity @s[x=0,y=0,z=0,dx=9,dy=9,dz=9] \\
  unless entity @s[tag=ad_in_ZONE_ID] \\
  run tag @s add ad_in_ZONE_ID

# --- EXIT --------------------------------------------------------------------
execute \\
  if entity @s[tag=ad_in_ZONE_ID] \\
  unless entity @s[x=0,y=0,z=0,dx=9,dy=9,dz=9] \\
  run tag @s remove ad_in_ZONE_ID
"""


# ---------------------------------------------------------------------------
# functions/titles/_template.mcfunction
# ---------------------------------------------------------------------------

def title_template(namespace: str) -> str:
    return f"""\
# =============================================================================
# {namespace}:titles/_template
# Called by zones/ZONE_ID.mcfunction on the first tick of zone entry.
# Context: @s is the entering player.
# =============================================================================

# Actionbar — single small line above the hotbar (combine name + flavor here).
# Colors: gold  aqua  green  red  light_purple  white  yellow  dark_red ...
# Note: `title ... actionbar` ignores `title ... times`; duration is fixed by MC.
title @s actionbar [{{"text":"<AREA NAME>","color":"gold","bold":true}},{{"text":" — ","color":"gray"}},{{"text":"<Flavor text>","color":"white","italic":true}}]

# Sound on entry (optional — delete line if not needed)
# playsound minecraft:block.note_block.bell master @s ~ ~ ~ 0.8 1.2
"""
