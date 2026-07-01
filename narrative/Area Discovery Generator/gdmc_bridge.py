from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from models import (
    AABB,
    MCColor,
    SoundConfig,
    TitleConfig,
    Zone,
)


def _auto_subtitle(preset: str, settlement: Any) -> str:
    llm_path = Path(__file__).parent.parent / "LLM Narrative"
    if str(llm_path) not in sys.path:
        sys.path.insert(0, str(llm_path))
    from zone_narrator import generate_zone_subtitle
    return generate_zone_subtitle(preset, settlement)


# ---------------------------------------------------------------------------
# AABB builders
# ---------------------------------------------------------------------------

def aabb_from_corners(
    x1: int, y1: int, z1: int,
    x2: int, y2: int, z2: int,
) -> AABB:
    return AABB.from_corners(x1, y1, z1, x2, y2, z2)


def aabb_from_footprint(
    x1: int, z1: int,
    x2: int, z2: int,
    y_min: int,
    y_max: int,
) -> AABB:
    return AABB.from_corners(x1, y_min, z1, x2, y_max, z2)


def aabb_from_square_footprint(
    cx: int, cz: int,
    radius: int,
    y_min: int,
    y_max: int,
) -> AABB:
    return AABB.from_corners(
        cx - radius, y_min, cz - radius,
        cx + radius, y_max, cz + radius,
    )


def aabb_from_gdpc_box(box: Any) -> AABB:
    try:
        # Try 3-D Box first (has .offset with z component)
        o = box.offset
        s = box.size
        return AABB(
            x=int(o.x), y=int(o.y), z=int(o.z),
            dx=int(s.x) - 1,
            dy=int(s.y) - 1,
            dz=int(s.z) - 1,
        )
    except AttributeError:
        pass

    try:
        # Fall back to 2-D Rect (x, z only)
        o = box.offset
        s = box.size
        return AABB(
            x=int(o.x), y=0, z=int(o.y),
            dx=int(s.x) - 1,
            dy=255,
            dz=int(s.y) - 1,
        )
    except AttributeError as exc:
        raise TypeError(
            f"Cannot convert {type(box).__name__!r} to AABB. "
            "Expected a GDPC Box or Rect object."
        ) from exc


def _relative_position(dx: float, dz: float, deadzone: float) -> str:
    if abs(dx) < deadzone and abs(dz) < deadzone:
        return "central"
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def aabb_from_zone_map(
    zone_map: Any,
    zone_id: int,
    origin: Any,
    heightmap: Any,
    y_pad_below: int = 2,
    y_pad_above: int = 30,
) -> AABB:
    import numpy as np

    mask = np.asarray(zone_map) == zone_id
    if not mask.any():
        raise ValueError(f"zone_map has no cells for zone_id={zone_id}")

    zs, xs = np.where(mask)  # zone_map is [z, x]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_z, max_z = int(zs.min()), int(zs.max())

    ox, oz = int(origin[0]), int(origin[2])

    surf = np.asarray(heightmap)[mask]
    min_y = int(surf.min()) - int(y_pad_below)
    max_y = int(surf.max()) + int(y_pad_above)

    return AABB.from_corners(
        ox + min_x, min_y, oz + min_z,
        ox + max_x, max_y, oz + max_z,
    )


def zone_descriptors_from_zone_map(zone_map: Any, origin: Any) -> list[dict]:
    import numpy as np

    grid = np.asarray(zone_map)
    ids = sorted(int(v) for v in np.unique(grid) if v >= 0)
    if not ids:
        return []

    all_zs, all_xs = np.where(grid >= 0)
    cx_all, cz_all = float(all_xs.mean()), float(all_zs.mean())

    # A zone counts as "central" only when its centroid sits well inside the
    # core. Size the threshold to the core's extent so it scales from a tiny
    # test grid to a full 256-block settlement.
    extent = max(int(np.ptp(all_xs)), int(np.ptp(all_zs))) + 1
    deadzone = 0.15 * extent

    descriptors: list[dict] = []
    for zid in ids:
        zs, xs = np.where(grid == zid)
        cx, cz = float(xs.mean()), float(zs.mean())
        descriptors.append({
            "zone_index": zid,
            "cell_count": int(zs.size),
            "position": _relative_position(cx - cx_all, cz - cz_all, deadzone),
        })
    return descriptors


def aabb_from_dict(d: dict[str, int]) -> AABB:
    if "xFrom" in d:
        return AABB.from_corners(
            d["xFrom"], d["yFrom"], d["zFrom"],
            d["xTo"],   d["yTo"],   d["zTo"],
        )
    if "sizeX" in d:
        return AABB(
            x=d["x"], y=d["y"], z=d["z"],
            dx=d["sizeX"] - 1,
            dy=d["sizeY"] - 1,
            dz=d["sizeZ"] - 1,
        )
    raise KeyError(
        f"Dict does not match any known AABB format. Keys: {list(d.keys())}"
    )


# ---------------------------------------------------------------------------
# Zone builders
# ---------------------------------------------------------------------------

# Preset palettes — opinionated title styles for common settlement types
_PRESETS: dict[str, dict] = {
    "town": dict(
        main_color   = MCColor.GOLD,
        sub_color    = MCColor.WHITE,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "❧ ",
        prefix_color = MCColor.YELLOW,
        fade_in=20, stay=80, fade_out=20,
        sound        = SoundConfig.town(),
    ),
    "ruins": dict(
        main_color   = MCColor.DARK_RED,
        sub_color    = MCColor.GRAY,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "",
        prefix_color = MCColor.DARK_GRAY,
        fade_in=30, stay=100, fade_out=30,
        sound        = SoundConfig.ruins(),
    ),
    "dungeon": dict(
        main_color   = MCColor.DARK_PURPLE,
        sub_color    = MCColor.DARK_GRAY,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "☠ ",
        prefix_color = MCColor.LIGHT_PURPLE,
        fade_in=10, stay=80, fade_out=20,
        sound        = SoundConfig.dungeon(),
    ),
    "nature": dict(
        main_color   = MCColor.GREEN,
        sub_color    = MCColor.DARK_GREEN,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "❀ ",
        prefix_color = MCColor.AQUA,
        fade_in=20, stay=70, fade_out=20,
        sound        = SoundConfig(
            "minecraft:block.azalea_leaves.place",
            SoundConfig.town().source, 0.5, 1.1
        ),
    ),
    "landmark": dict(
        main_color   = MCColor.AQUA,
        sub_color    = MCColor.WHITE,
        main_bold    = True,
        sub_italic   = True,
        prefix       = "★ ",
        prefix_color = MCColor.YELLOW,
        fade_in=20, stay=80, fade_out=20,
        sound        = SoundConfig.victory(),
    ),
}


def _title_for_preset(
    preset: str,
    display_name: str,
    subtitle: str,
    settlement: Any,
) -> tuple[TitleConfig, SoundConfig]:
    p = _PRESETS.get(preset)
    if p is None:
        raise ValueError(
            f"Unknown preset {preset!r}. "
            f"Valid presets: {list(_PRESETS.keys())}"
        )
    if not subtitle and settlement is not None:
        subtitle = _auto_subtitle(preset, settlement)
    title = TitleConfig(
        main_title   = display_name,
        subtitle     = subtitle,
        main_color   = p["main_color"],
        sub_color    = p["sub_color"],
        main_bold    = p["main_bold"],
        sub_italic   = p["sub_italic"],
        prefix       = p["prefix"],
        prefix_color = p["prefix_color"],
        fade_in      = p["fade_in"],
        stay         = p["stay"],
        fade_out     = p["fade_out"],
    )
    return title, p["sound"]


def zone_from_aabb(
    zone_id:      str,
    display_name: str,
    aabb:         AABB,
    subtitle:     str                = "",
    preset:       str                = "town",
    notes:        str                = "",
    enabled:      bool               = True,
    settlement:   "Settlement | None" = None,
) -> Zone:
    title, sound = _title_for_preset(preset, display_name, subtitle, settlement)
    return Zone(
        zone_id      = zone_id,
        display_name = display_name,
        aabb         = aabb,
        title        = title,
        sound        = sound,
        notes        = notes,
        enabled      = enabled,
    )


def zone_from_corners(
    zone_id:      str,
    display_name: str,
    subtitle:     str                = "",
    x1: int = 0, y1: int = 0, z1: int = 0,
    x2: int = 0, y2: int = 0, z2: int = 0,
    preset:       str                = "town",
    notes:        str                = "",
    enabled:      bool               = True,
    settlement:   "Settlement | None" = None,
) -> Zone:
    return zone_from_aabb(
        zone_id      = zone_id,
        display_name = display_name,
        aabb         = AABB.from_corners(x1, y1, z1, x2, y2, z2),
        subtitle     = subtitle,
        preset       = preset,
        notes        = notes,
        enabled      = enabled,
        settlement   = settlement,
    )


def zone_from_gdpc_box(
    zone_id:      str,
    display_name: str,
    subtitle:     str                 = "",
    box:          Any                 = None,
    preset:       str                 = "town",
    notes:        str                 = "",
    enabled:      bool                = True,
    settlement:   "Settlement | None" = None,
) -> Zone:
    if box is None:
        raise ValueError("zone_from_gdpc_box requires a GDPC Box or Rect via `box=`.")
    return zone_from_aabb(
        zone_id      = zone_id,
        display_name = display_name,
        aabb         = aabb_from_gdpc_box(box),
        subtitle     = subtitle,
        preset       = preset,
        notes        = notes,
        enabled      = enabled,
        settlement   = settlement,
    )


def available_presets() -> list[str]:
    return list(_PRESETS.keys())
