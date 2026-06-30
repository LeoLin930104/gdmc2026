"""
Block Geometry Registry — Parametric mesh generators, colour palette, and
texture-mapped rendering.

Phase 1.5 (current): actual Minecraft 1.21 textures applied to full-cube
    faces via UV mapping.  Partial-geometry blocks (stairs, slabs, fences)
    retain procedural geometry with texture-mapped faces where applicable.
Phase 2 (deferred): parsed Minecraft .json block models for accurate geometry.

The registry auto-discovers any block encountered at render time.  Blocks
with a matching texture on disk get texture-mapped cubes; blocks with a
colour palette entry get flat-shaded procedural geometry; everything else
gets a visible fallback magenta cube.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import lru_cache
from typing import Callable

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Texture asset path resolution
# ---------------------------------------------------------------------------

from voxel_renderer.assets import get_asset_root

_ASSETS_DIR = str(get_asset_root() / "textures" / "block")

_TEXTURES_ENABLED: bool = True
try:
    from PIL import Image as PILImage
except ImportError:
    _TEXTURES_ENABLED = False
    logger.info("PIL not available — texture mapping disabled, using flat colours.")


# ---------------------------------------------------------------------------
# Helpers: colour generation from 16-colour palette
# ---------------------------------------------------------------------------

_ALL_COLOURS = (
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "light_gray",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
)

_WOOD_TYPES = (
    "oak",
    "spruce",
    "birch",
    "jungle",
    "acacia",
    "dark_oak",
    "mangrove",
    "cherry",
    "bamboo",
    "crimson",
    "warped",
)

_STANDARD_WOOD = (
    "oak",
    "spruce",
    "birch",
    "jungle",
    "acacia",
    "dark_oak",
    "mangrove",
    "cherry",
)

# ---------------------------------------------------------------------------
# Colour Palette (flat RGB fallbacks)
# ---------------------------------------------------------------------------
# Used when texture files are missing.  Covers the full block universe
# encountered in the GDMC-2026-ICELAB project.

_WOOD_PLANK_COLOURS = {
    "oak": (180, 144, 90),
    "spruce": (115, 85, 48),
    "birch": (216, 201, 155),
    "jungle": (160, 115, 80),
    "acacia": (168, 90, 50),
    "dark_oak": (67, 43, 20),
    "mangrove": (118, 54, 48),
    "cherry": (226, 178, 172),
    "bamboo": (194, 176, 81),
    "crimson": (101, 48, 70),
    "warped": (43, 105, 99),
}

_WOOL_COLOURS = {
    "white": (234, 236, 236),
    "orange": (241, 118, 20),
    "magenta": (189, 68, 179),
    "light_blue": (58, 175, 217),
    "yellow": (249, 198, 40),
    "lime": (112, 185, 26),
    "pink": (238, 141, 172),
    "gray": (63, 68, 72),
    "light_gray": (142, 142, 135),
    "cyan": (21, 138, 145),
    "purple": (122, 42, 173),
    "blue": (53, 57, 157),
    "brown": (114, 72, 41),
    "green": (85, 110, 27),
    "red": (161, 39, 35),
    "black": (20, 21, 26),
}

_CONCRETE_COLOURS = {
    "white": (207, 213, 214),
    "orange": (224, 97, 1),
    "magenta": (170, 48, 159),
    "light_blue": (36, 137, 199),
    "yellow": (241, 175, 21),
    "lime": (94, 169, 25),
    "pink": (214, 101, 143),
    "gray": (55, 58, 62),
    "light_gray": (125, 125, 115),
    "cyan": (21, 119, 136),
    "purple": (100, 32, 156),
    "blue": (45, 47, 143),
    "brown": (96, 60, 32),
    "green": (73, 91, 36),
    "red": (142, 33, 33),
    "black": (8, 10, 15),
}

_CONCRETE_POWDER_COLOURS = {
    "white": (225, 228, 229),
    "orange": (228, 131, 32),
    "magenta": (193, 84, 185),
    "light_blue": (74, 181, 213),
    "yellow": (233, 199, 55),
    "lime": (125, 189, 41),
    "pink": (229, 153, 181),
    "gray": (77, 82, 87),
    "light_gray": (155, 155, 148),
    "cyan": (37, 148, 157),
    "purple": (132, 56, 178),
    "blue": (70, 73, 167),
    "brown": (127, 83, 51),
    "green": (97, 119, 44),
    "red": (168, 54, 51),
    "black": (25, 27, 32),
}

_STAINED_GLASS_COLOURS = {
    "white": (210, 220, 225),
    "orange": (216, 127, 51),
    "magenta": (178, 76, 216),
    "light_blue": (102, 153, 216),
    "yellow": (229, 229, 51),
    "lime": (127, 204, 25),
    "pink": (242, 127, 165),
    "gray": (76, 76, 76),
    "light_gray": (153, 153, 153),
    "cyan": (76, 127, 153),
    "purple": (127, 63, 178),
    "blue": (51, 76, 178),
    "brown": (102, 76, 51),
    "green": (102, 127, 51),
    "red": (153, 51, 51),
    "black": (25, 25, 25),
}

_TERRACOTTA_COLOURS = {
    "white": (210, 178, 161),
    "orange": (162, 84, 38),
    "magenta": (150, 88, 109),
    "light_blue": (113, 109, 138),
    "yellow": (186, 133, 35),
    "lime": (103, 118, 53),
    "pink": (162, 78, 79),
    "gray": (57, 42, 36),
    "light_gray": (135, 107, 98),
    "cyan": (87, 91, 91),
    "purple": (118, 70, 86),
    "blue": (74, 60, 91),
    "brown": (77, 51, 36),
    "green": (76, 83, 42),
    "red": (143, 61, 47),
    "black": (37, 23, 16),
}


def _build_colour_palette() -> dict[str, tuple[int, int, int]]:
    """Construct the complete colour palette programmatically."""
    c: dict[str, tuple[int, int, int]] = {}

    # Planks + derived (stairs, slabs, fences, logs)
    for wood, rgb in _WOOD_PLANK_COLOURS.items():
        c[f"minecraft:{wood}_planks"] = rgb
        c[f"minecraft:{wood}_stairs"] = rgb
        c[f"minecraft:{wood}_slab"] = rgb
        c[f"minecraft:{wood}_fence"] = rgb
        c[f"minecraft:{wood}_fence_gate"] = rgb
        c[f"minecraft:{wood}_door"] = rgb
        c[f"minecraft:{wood}_trapdoor"] = rgb
        c[f"minecraft:{wood}_button"] = rgb
        c[f"minecraft:{wood}_pressure_plate"] = rgb
        c[f"minecraft:{wood}_sign"] = rgb
        c[f"minecraft:{wood}_wall_sign"] = rgb

    # Logs — slightly darker than planks
    for wood in _STANDARD_WOOD:
        rgb = _WOOD_PLANK_COLOURS.get(wood, (128, 100, 70))
        darker = (max(0, rgb[0] - 30), max(0, rgb[1] - 30), max(0, rgb[2] - 30))
        c[f"minecraft:{wood}_log"] = darker
        c[f"minecraft:stripped_{wood}_log"] = rgb
        c[f"minecraft:{wood}_wood"] = darker
        c[f"minecraft:stripped_{wood}_wood"] = rgb
        c[f"minecraft:{wood}_leaves"] = (60, 140, 50)

    c["minecraft:bamboo_block"] = (120, 130, 50)
    c["minecraft:crimson_stem"] = (93, 26, 29)
    c["minecraft:warped_stem"] = (22, 124, 105)

    # Wool, concrete, concrete powder, stained glass, terracotta
    for colour_name in _ALL_COLOURS:
        c[f"minecraft:{colour_name}_wool"] = _WOOL_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_carpet"] = _WOOL_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_concrete"] = _CONCRETE_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_concrete_powder"] = _CONCRETE_POWDER_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_stained_glass"] = _STAINED_GLASS_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_stained_glass_pane"] = _STAINED_GLASS_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_terracotta"] = _TERRACOTTA_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_bed"] = _WOOL_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_banner"] = _WOOL_COLOURS[colour_name]
        c[f"minecraft:{colour_name}_wall_banner"] = _WOOL_COLOURS[colour_name]

    # Stone family
    stone_blocks = {
        "stone": (125, 125, 125),
        "cobblestone": (127, 127, 127),
        "stone_bricks": (122, 122, 122),
        "mossy_stone_bricks": (105, 122, 95),
        "cracked_stone_bricks": (118, 118, 118),
        "chiseled_stone_bricks": (120, 120, 120),
        "deepslate_bricks": (70, 70, 76),
        "deepslate": (65, 65, 70),
        "polished_deepslate": (72, 72, 78),
        "deepslate_tiles": (68, 68, 74),
        "cobbled_deepslate": (73, 73, 79),
        "bricks": (150, 97, 76),
        "smooth_stone": (160, 160, 160),
        "polished_andesite": (132, 135, 134),
        "polished_diorite": (190, 190, 195),
        "polished_granite": (154, 107, 89),
        "andesite": (136, 136, 136),
        "diorite": (188, 188, 190),
        "granite": (149, 103, 86),
        "mossy_cobblestone": (110, 127, 100),
        "sandstone": (218, 210, 158),
        "red_sandstone": (186, 99, 29),
        "cut_sandstone": (215, 207, 152),
        "cut_red_sandstone": (183, 96, 26),
        "smooth_sandstone": (220, 212, 160),
        "end_stone": (219, 223, 158),
        "end_stone_bricks": (218, 224, 162),
        "purpur_block": (170, 126, 170),
        "purpur_pillar": (172, 130, 172),
        "prismarine": (99, 172, 158),
        "prismarine_bricks": (99, 171, 140),
        "dark_prismarine": (51, 91, 75),
        "calcite": (224, 225, 221),
        "tuff": (108, 109, 102),
        "dripstone_block": (134, 107, 92),
        "mud": (60, 58, 54),
    }
    for name, rgb in stone_blocks.items():
        c[f"minecraft:{name}"] = rgb

    # Stone stairs/slabs/walls inherit from base
    for base_name in (
        "stone",
        "cobblestone",
        "stone_brick",
        "brick",
        "deepslate_brick",
        "mossy_cobblestone",
        "sandstone",
        "red_sandstone",
        "prismarine",
        "prismarine_brick",
        "dark_prismarine",
        "end_stone_brick",
        "purpur",
        "polished_andesite",
        "polished_diorite",
        "polished_granite",
    ):
        # Find base colour (handle brick vs bricks naming)
        base_key = f"minecraft:{base_name}"
        if base_key.endswith("_brick"):
            base_key = f"minecraft:{base_name}s"
        rgb = c.get(base_key, (128, 128, 128))
        c[f"minecraft:{base_name}_stairs"] = rgb
        c[f"minecraft:{base_name}_slab"] = rgb
        c[f"minecraft:{base_name}_wall"] = rgb

    c["minecraft:smooth_stone_slab"] = (160, 160, 160)
    c["minecraft:cobblestone_slab"] = (127, 127, 127)

    # Natural
    c.update(
        {
            "minecraft:dirt": (134, 96, 67),
            "minecraft:coarse_dirt": (119, 85, 59),
            "minecraft:grass_block": (124, 189, 107),
            "minecraft:podzol": (91, 63, 24),
            "minecraft:mycelium": (111, 99, 105),
            "minecraft:gravel": (131, 127, 126),
            "minecraft:sand": (219, 207, 163),
            "minecraft:red_sand": (190, 102, 33),
            "minecraft:clay": (160, 166, 179),
            "minecraft:soul_sand": (81, 62, 50),
            "minecraft:soul_soil": (75, 57, 46),
            "minecraft:netherrack": (97, 38, 38),
            "minecraft:snow_block": (249, 254, 254),
            "minecraft:ice": (145, 183, 253),
            "minecraft:packed_ice": (141, 180, 250),
            "minecraft:blue_ice": (116, 168, 250),
            "minecraft:moss_block": (89, 109, 45),
            "minecraft:water": (62, 118, 200),
            "minecraft:lava": (207, 92, 15),
            "minecraft:short_grass": (90, 150, 60),
            "minecraft:fern": (75, 130, 55),
        }
    )

    # Metal / mineral
    c.update(
        {
            "minecraft:iron_block": (220, 220, 220),
            "minecraft:gold_block": (246, 208, 62),
            "minecraft:diamond_block": (99, 236, 228),
            "minecraft:emerald_block": (42, 183, 75),
            "minecraft:lapis_block": (31, 67, 140),
            "minecraft:coal_block": (16, 15, 15),
            "minecraft:redstone_block": (171, 27, 8),
            "minecraft:copper_block": (192, 107, 79),
            "minecraft:exposed_copper": (154, 121, 89),
            "minecraft:weathered_copper": (109, 145, 107),
            "minecraft:oxidized_copper": (82, 162, 132),
            "minecraft:obsidian": (15, 11, 25),
            "minecraft:crying_obsidian": (32, 10, 60),
            "minecraft:glowstone": (171, 131, 84),
        }
    )

    # Nether
    c.update(
        {
            "minecraft:nether_bricks": (44, 22, 26),
            "minecraft:red_nether_bricks": (69, 7, 9),
            "minecraft:basalt": (73, 72, 77),
            "minecraft:polished_basalt": (88, 88, 91),
            "minecraft:blackstone": (42, 36, 40),
            "minecraft:polished_blackstone": (53, 48, 56),
            "minecraft:polished_blackstone_bricks": (48, 42, 50),
            "minecraft:quartz_block": (236, 230, 223),
            "minecraft:smooth_quartz": (236, 230, 223),
            "minecraft:quartz_bricks": (234, 228, 220),
            "minecraft:quartz_pillar": (235, 229, 222),
        }
    )

    # Decorative / functional
    c.update(
        {
            "minecraft:bookshelf": (111, 88, 55),
            "minecraft:crafting_table": (130, 95, 55),
            "minecraft:furnace": (120, 120, 120),
            "minecraft:dispenser": (120, 120, 120),
            "minecraft:chest": (160, 120, 50),
            "minecraft:barrel": (130, 100, 60),
            "minecraft:anvil": (72, 72, 72),
            "minecraft:chain": (55, 60, 67),
            "minecraft:iron_bars": (165, 165, 165),
            "minecraft:glass_pane": (200, 220, 230),
            "minecraft:glass": (200, 220, 230),
            "minecraft:lantern": (200, 160, 60),
            "minecraft:soul_lantern": (80, 200, 200),
            "minecraft:torch": (255, 200, 50),
            "minecraft:wall_torch": (255, 200, 50),
            "minecraft:soul_torch": (80, 200, 200),
            "minecraft:ladder": (160, 130, 80),
            "minecraft:flower_pot": (140, 73, 53),
            "minecraft:skeleton_skull": (200, 200, 190),
            "minecraft:terracotta": (152, 94, 68),
            "minecraft:hay_block": (186, 161, 39),
            "minecraft:melon": (110, 145, 30),
            "minecraft:pumpkin": (198, 118, 24),
            "minecraft:stone_button": (125, 125, 125),
            "minecraft:redstone_wire": (200, 0, 0),
        }
    )

    return c


BLOCK_COLOURS: dict[str, tuple[int, int, int]] = _build_colour_palette()
FALLBACK_MISSING_BLOCK_COLOUR = (200, 100, 200)


def _auto_sample_missing_colours() -> None:
    """Fill colour gaps by sampling the average opaque pixel from block textures.

    For every blockstate on disk that lacks a manual colour entry, this
    function attempts to locate a matching texture PNG and computes the
    mean RGB of its opaque (alpha > 128) pixels.  Blocks with no matching
    texture or with fewer than 8 opaque pixels retain the magenta fallback.

    Resolution strategy for texture name → block name mismatches:
      1. Exact name (e.g., ``amethyst_block.png``)
      2. Name with ``_top`` / ``_side`` / ``_front`` suffix
      3. Strip ``_slab`` / ``_stairs`` / ``_wall`` suffix (inherit from base)
      4. Strip suffix + try ``s`` plural (e.g., ``brick_slab`` → ``bricks``)
      5. Inherit from a sibling block that already has a colour entry

    Runs once at import time.  Typical cost: ~200 ms for ~600 textures.
    """
    if not _TEXTURES_ENABLED:
        return

    blockstates_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets",
        "blockstates",
    )
    if not os.path.isdir(blockstates_dir):
        return

    # Suffixes that derive from a base material
    _DERIVATIVE_SUFFIXES = (
        "_slab",
        "_stairs",
        "_wall",
        "_fence",
        "_fence_gate",
        "_button",
        "_pressure_plate",
        "_door",
        "_trapdoor",
        "_sign",
        "_wall_sign",
        "_hanging_sign",
        "_wall_hanging_sign",
    )

    def _sample_texture(tex_name: str) -> tuple[int, int, int] | None:
        tex_path = os.path.join(_ASSETS_DIR, f"{tex_name}.png")
        if not os.path.exists(tex_path):
            return None
        try:
            img = PILImage.open(tex_path).convert("RGBA")
            arr = np.array(img)
            opaque = arr[:, :, 3] > 128
            if opaque.sum() < 8:
                return None
            rgb_mean = arr[:, :, :3][opaque].mean(axis=0).astype(int)
            return (int(rgb_mean[0]), int(rgb_mean[1]), int(rgb_mean[2]))
        except Exception:
            return None

    def _try_sample_block(block_name: str) -> tuple[int, int, int] | None:
        """Try multiple texture name strategies for a block."""
        # Strategy 1-2: exact name or with suffixes
        for suffix in ("", "_top", "_side", "_front"):
            result = _sample_texture(f"{block_name}{suffix}")
            if result is not None:
                return result

        # Strategy 3-4: strip derivative suffix → base material
        for ds in _DERIVATIVE_SUFFIXES:
            if block_name.endswith(ds):
                base = block_name[: -len(ds)]
                # Try base directly
                for suffix in ("", "_top", "_side", "_planks"):
                    result = _sample_texture(f"{base}{suffix}")
                    if result is not None:
                        return result
                # Try plural (e.g., "brick" → "bricks")
                result = _sample_texture(f"{base}s")
                if result is not None:
                    return result
                break

        return None

    # --- Pass 1: sample from textures ---
    sampled = 0
    still_missing: list[str] = []
    for filename in os.listdir(blockstates_dir):
        if not filename.endswith(".json"):
            continue
        block_name = filename[:-5]
        block_id = f"minecraft:{block_name}"
        if block_id in BLOCK_COLOURS:
            continue

        result = _try_sample_block(block_name)
        if result is not None:
            BLOCK_COLOURS[block_id] = result
            sampled += 1
        else:
            still_missing.append(block_id)

    # --- Pass 2: inherit from sibling blocks already in the palette ---
    # e.g., "minecraft:blackstone_slab" inherits from "minecraft:blackstone"
    inherited = 0
    remaining_after_p2: list[str] = []
    for block_id in still_missing:
        block_name = block_id.replace("minecraft:", "")
        found = False
        for ds in _DERIVATIVE_SUFFIXES:
            if block_name.endswith(ds):
                base = block_name[: -len(ds)]
                # Try base_id directly, or with "s" suffix
                for candidate in (f"minecraft:{base}", f"minecraft:{base}s"):
                    if candidate in BLOCK_COLOURS:
                        BLOCK_COLOURS[block_id] = BLOCK_COLOURS[candidate]
                        inherited += 1
                        found = True
                        break
                break
        if not found:
            remaining_after_p2.append(block_id)

    # --- Pass 3: prefix-based inheritance for remaining blocks ---
    # Waxed copper variants → unwaxed counterpart
    # Potted plants → flower_pot colour
    # Candle cakes → cake colour
    prefix_inherited = 0
    for block_id in remaining_after_p2:
        block_name = block_id.replace("minecraft:", "")

        # waxed_ → strip prefix, look up unwaxed
        if block_name.startswith("waxed_"):
            unwaxed_id = f"minecraft:{block_name[6:]}"
            if unwaxed_id in BLOCK_COLOURS:
                BLOCK_COLOURS[block_id] = BLOCK_COLOURS[unwaxed_id]
                prefix_inherited += 1
                continue

        # potted_ → flower_pot
        if block_name.startswith("potted_"):
            BLOCK_COLOURS[block_id] = BLOCK_COLOURS.get("minecraft:flower_pot", (140, 73, 53))
            prefix_inherited += 1
            continue

        # *_candle_cake → cake-ish colour
        if block_name.endswith("_candle_cake") or block_name == "candle_cake":
            BLOCK_COLOURS[block_id] = (210, 180, 140)
            prefix_inherited += 1
            continue

        # *_wall_fan → base coral block
        if block_name.endswith("_wall_fan"):
            coral_base = block_name.replace("_wall_fan", "_block")
            if f"minecraft:{coral_base}" in BLOCK_COLOURS:
                BLOCK_COLOURS[block_id] = BLOCK_COLOURS[f"minecraft:{coral_base}"]
                prefix_inherited += 1
                continue

        # *_wall_head / *_wall_skull → strip "wall_"
        if "_wall_head" in block_name or "_wall_skull" in block_name:
            non_wall = block_name.replace("_wall_head", "_head").replace("_wall_skull", "_skull")
            if f"minecraft:{non_wall}" in BLOCK_COLOURS:
                BLOCK_COLOURS[block_id] = BLOCK_COLOURS[f"minecraft:{non_wall}"]
                prefix_inherited += 1
                continue

    total_filled = sampled + inherited + prefix_inherited
    if total_filled > 0:
        logger.debug(
            "Auto-filled fallback colours: %d sampled, %d inherited, %d prefix-matched.",
            sampled,
            inherited,
            prefix_inherited,
        )


_auto_sample_missing_colours()


def get_block_colour(block_id: str) -> tuple[int, int, int]:
    """Return RGB colour tuple for a block ID."""
    return BLOCK_COLOURS.get(block_id, FALLBACK_MISSING_BLOCK_COLOUR)


# ---------------------------------------------------------------------------
# Texture Loading & UV-Mapped Mesh Generation
# ---------------------------------------------------------------------------


def _build_texture_map() -> dict[str, tuple[str, ...]]:
    """Construct the complete texture mapping programmatically."""
    t: dict[str, tuple[str, ...]] = {}

    # Uniform-texture full cubes
    for wood in _WOOD_TYPES:
        t[f"minecraft:{wood}_planks"] = (f"{wood}_planks",)
        t[f"minecraft:{wood}_stairs"] = (f"{wood}_planks",)
        t[f"minecraft:{wood}_slab"] = (f"{wood}_planks",)

    for colour_name in _ALL_COLOURS:
        t[f"minecraft:{colour_name}_wool"] = (f"{colour_name}_wool",)
        # Carpets use wool textures
        t[f"minecraft:{colour_name}_carpet"] = (f"{colour_name}_wool",)
        t[f"minecraft:{colour_name}_concrete"] = (f"{colour_name}_concrete",)
        t[f"minecraft:{colour_name}_concrete_powder"] = (f"{colour_name}_concrete_powder",)
        t[f"minecraft:{colour_name}_stained_glass"] = (f"{colour_name}_stained_glass",)
        t[f"minecraft:{colour_name}_terracotta"] = (f"{colour_name}_terracotta",)

    # Simple uniform stones
    for name in (
        "stone",
        "cobblestone",
        "stone_bricks",
        "mossy_stone_bricks",
        "cracked_stone_bricks",
        "chiseled_stone_bricks",
        "deepslate_bricks",
        "polished_deepslate",
        "deepslate_tiles",
        "cobbled_deepslate",
        "bricks",
        "smooth_stone",
        "polished_andesite",
        "polished_diorite",
        "polished_granite",
        "andesite",
        "diorite",
        "granite",
        "mossy_cobblestone",
        "terracotta",
        "sandstone",
        "red_sandstone",
        "cut_sandstone",
        "cut_red_sandstone",
        "end_stone",
        "end_stone_bricks",
        "prismarine",
        "prismarine_bricks",
        "dark_prismarine",
        "calcite",
        "tuff",
        "dripstone_block",
        "nether_bricks",
        "red_nether_bricks",
        "blackstone",
        "polished_blackstone",
        "polished_blackstone_bricks",
        "quartz_bricks",
        "dirt",
        "coarse_dirt",
        "gravel",
        "sand",
        "red_sand",
        "netherrack",
        "soul_sand",
        "soul_soil",
        "clay",
        "mud",
        "moss_block",
        "iron_block",
        "gold_block",
        "diamond_block",
        "emerald_block",
        "lapis_block",
        "coal_block",
        "redstone_block",
        "copper_block",
        "exposed_copper",
        "weathered_copper",
        "oxidized_copper",
        "obsidian",
        "crying_obsidian",
        "glowstone",
        "glass",
        "white_stained_glass",
        "ice",
        "packed_ice",
        "blue_ice",
        "snow",
        "bookshelf",
    ):
        t[f"minecraft:{name}"] = (name,)

    # Stone stairs/slabs
    stair_slab_base = {
        "stone": "stone",
        "cobblestone": "cobblestone",
        "stone_brick": "stone_bricks",
        "brick": "bricks",
        "deepslate_brick": "deepslate_bricks",
        "smooth_stone": "smooth_stone",
        "mossy_cobblestone": "mossy_cobblestone",
        "sandstone": "sandstone",
        "red_sandstone": "red_sandstone",
        "prismarine": "prismarine",
        "prismarine_brick": "prismarine_bricks",
        "dark_prismarine": "dark_prismarine",
        "end_stone_brick": "end_stone_bricks",
        "purpur": "purpur_block",
        "polished_andesite": "polished_andesite",
        "polished_diorite": "polished_diorite",
        "polished_granite": "polished_granite",
        "nether_brick": "nether_bricks",
        "red_nether_brick": "red_nether_bricks",
        "polished_blackstone_brick": "polished_blackstone_bricks",
        "cobblestone": "cobblestone",
    }
    for prefix, tex in stair_slab_base.items():
        t[f"minecraft:{prefix}_stairs"] = (tex,)
        t[f"minecraft:{prefix}_slab"] = (tex,)

    # Top/side/bottom logs
    for wood in _STANDARD_WOOD:
        t[f"minecraft:{wood}_log"] = (f"{wood}_log_top", f"{wood}_log", f"{wood}_log_top")
        if os.path.exists(os.path.join(_ASSETS_DIR, f"stripped_{wood}_log.png")):
            t[f"minecraft:stripped_{wood}_log"] = (
                f"stripped_{wood}_log_top",
                f"stripped_{wood}_log",
                f"stripped_{wood}_log_top",
            )
    t["minecraft:bamboo_block"] = ("bamboo_block",)
    t["minecraft:crimson_stem"] = ("crimson_stem_top", "crimson_stem", "crimson_stem_top")
    t["minecraft:warped_stem"] = ("warped_stem_top", "warped_stem", "warped_stem_top")

    # Top/side blocks
    t["minecraft:grass_block"] = ("grass_block_top", "grass_block_side", "dirt")
    t["minecraft:podzol"] = ("podzol_top", "podzol_side", "dirt")
    t["minecraft:mycelium"] = ("mycelium_top", "mycelium_side", "dirt")
    t["minecraft:crafting_table"] = ("crafting_table_top", "crafting_table_front", "oak_planks")
    t["minecraft:furnace"] = ("furnace_top", "furnace_front", "furnace_side")
    t["minecraft:hay_block"] = ("hay_block_top", "hay_block_side", "hay_block_top")
    t["minecraft:basalt"] = ("basalt_top", "basalt_side", "basalt_top")
    t["minecraft:polished_basalt"] = (
        "polished_basalt_top",
        "polished_basalt_side",
        "polished_basalt_top",
    )
    t["minecraft:quartz_block"] = ("quartz_block_top", "quartz_block_side", "quartz_block_bottom")
    t["minecraft:quartz_pillar"] = ("quartz_pillar_top", "quartz_pillar", "quartz_pillar_top")
    t["minecraft:purpur_pillar"] = ("purpur_pillar_top", "purpur_pillar", "purpur_pillar_top")
    t["minecraft:deepslate"] = ("deepslate_top", "deepslate", "deepslate_top")
    t["minecraft:barrel"] = ("barrel_top", "barrel_side", "barrel_bottom")
    t["minecraft:melon"] = ("melon_top", "melon_side", "melon_top")
    t["minecraft:pumpkin"] = ("pumpkin_top", "pumpkin_side", "pumpkin_top")
    t["minecraft:sandstone"] = ("sandstone_top", "sandstone", "sandstone_bottom")
    t["minecraft:red_sandstone"] = ("red_sandstone_top", "red_sandstone", "red_sandstone_bottom")

    # Leaves
    for wood in _STANDARD_WOOD:
        t[f"minecraft:{wood}_leaves"] = (f"{wood}_leaves",)

    return t


BLOCK_TEXTURE_MAP: dict[str, tuple[str, ...]] = _build_texture_map()


@lru_cache(maxsize=512)
def _load_texture(name: str) -> PILImage.Image | None:
    """Load a texture PNG from the assets directory.  Returns None if missing."""
    if not _TEXTURES_ENABLED:
        return None
    path = os.path.join(_ASSETS_DIR, f"{name}.png")
    if not os.path.exists(path):
        return None
    try:
        img = PILImage.open(path).convert("RGBA")
        return img
    except Exception as e:
        logger.debug("Failed to load texture '%s': %s", name, e)
        return None


def _build_atlas_3face(top_name: str, side_name: str, bottom_name: str) -> PILImage.Image | None:
    """Build a 3x1 texture atlas: [top | side | bottom]."""
    top = _load_texture(top_name)
    side = _load_texture(side_name)
    bottom = _load_texture(bottom_name)
    if top is None or side is None or bottom is None:
        return None
    top = top.resize((16, 16), PILImage.Resampling.NEAREST)
    side = side.resize((16, 16), PILImage.Resampling.NEAREST)
    bottom = bottom.resize((16, 16), PILImage.Resampling.NEAREST)
    atlas = PILImage.new("RGBA", (48, 16))
    atlas.paste(top, (0, 0))
    atlas.paste(side, (16, 0))
    atlas.paste(bottom, (32, 0))
    return atlas


def _build_unit_cube_geometry() -> tuple[np.ndarray, np.ndarray]:
    """
    Build a unit cube centred at the origin with unshared quad-based vertices.

    Returns (vertices, faces) where:
      - vertices: (24, 3) — 4 per face × 6 faces, CCW winding from outside.
      - faces: (12, 3) — 2 triangles per face × 6 faces.

    Face order: down (-Y), up (+Y), north (-Z), south (+Z), west (-X), east (+X).
    This matches Minecraft's face conventions and the UV generators below.
    """
    h = 0.5  # half-extent

    # Each face: 4 vertices wound CCW when viewed from outside
    face_verts = np.array(
        [
            # down (-Y)
            [-h, -h, -h],
            [+h, -h, -h],
            [+h, -h, +h],
            [-h, -h, +h],
            # up (+Y)
            [-h, +h, +h],
            [+h, +h, +h],
            [+h, +h, -h],
            [-h, +h, -h],
            # north (-Z)
            [+h, +h, -h],
            [-h, +h, -h],
            [-h, -h, -h],
            [+h, -h, -h],
            # south (+Z)
            [-h, +h, +h],
            [+h, +h, +h],
            [+h, -h, +h],
            [-h, -h, +h],
            # west (-X)
            [-h, +h, -h],
            [-h, +h, +h],
            [-h, -h, +h],
            [-h, -h, -h],
            # east (+X)
            [+h, +h, +h],
            [+h, +h, -h],
            [+h, -h, -h],
            [+h, -h, +h],
        ],
        dtype=np.float64,
    )

    # Two triangles per quad.
    # Down (f=0) and Up (f=1) vertices are already CCW from outside →
    # standard [0,1,2],[0,2,3].  The four side faces (f=2..5) were authored
    # in Minecraft's CW convention → reverse to [0,2,1],[0,3,2] for OpenGL.
    tri_faces = np.empty((12, 3), dtype=np.int64)
    for f in range(6):
        base = f * 4
        if f < 2:  # down, up — correct CCW winding
            tri_faces[f * 2] = [base, base + 1, base + 2]
            tri_faces[f * 2 + 1] = [base, base + 2, base + 3]
        else:  # north, south, west, east — CW from outside, flip
            tri_faces[f * 2] = [base, base + 2, base + 1]
            tri_faces[f * 2 + 1] = [base, base + 3, base + 2]

    return face_verts, tri_faces


def _uv_cube_uniform() -> np.ndarray:
    """UV coords for uniform-texture cube (all 6 faces → full [0,1]²).

    Returns (24, 2) — 4 per face, matching ``_build_unit_cube_geometry`` vertex order.
    V is flipped (1 - v) because Minecraft has V=0 at top, OpenGL at bottom.
    """
    face = [[0, 1], [1, 1], [1, 0], [0, 0]]
    return np.array(face * 6, dtype=np.float64)


def _uv_cube_3face() -> np.ndarray:
    """UV coords for 3-face atlas cube: top=[0,1/3], side=[1/3,2/3], bottom=[2/3,1].

    Face order: down, up, north, south, west, east.
    Returns (24, 2).
    V is flipped (1 - v) because Minecraft has V=0 at top, OpenGL at bottom.
    """
    top_u, side_u, bot_u = (0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)

    def face_uvs(u_range: tuple[float, float]) -> list[list[float]]:
        u0, u1 = u_range
        return [[u0, 1], [u1, 1], [u1, 0], [u0, 0]]

    uvs = []
    uvs.extend(face_uvs(bot_u))  # down  (-Y)
    uvs.extend(face_uvs(top_u))  # up    (+Y)
    uvs.extend(face_uvs(side_u))  # north (-Z)
    uvs.extend(face_uvs(side_u))  # south (+Z)
    uvs.extend(face_uvs(side_u))  # west  (-X)
    uvs.extend(face_uvs(side_u))  # east  (+X)
    return np.array(uvs, dtype=np.float64)


def create_textured_cube(block_id: str) -> trimesh.Trimesh | None:
    """
    Create a texture-mapped unit cube for the given block ID.

    Uses explicit quad-based geometry (24 vertices, 12 triangles, 24 UVs)
    so that UV count equals vertex count — no ``_unshare_vertices`` needed.
    Returns None if textures are unavailable, falling back to flat colour.
    """
    if not _TEXTURES_ENABLED:
        return None

    tex_info = BLOCK_TEXTURE_MAP.get(block_id)
    if tex_info is None:
        return None

    if len(tex_info) == 1:
        tex_img = _load_texture(tex_info[0])
        if tex_img is None:
            return None
        tex_img = tex_img.resize((16, 16), PILImage.Resampling.NEAREST)
        uv = _uv_cube_uniform()
    elif len(tex_info) == 3:
        tex_img = _build_atlas_3face(tex_info[0], tex_info[1], tex_info[2])
        if tex_img is None:
            return None
        uv = _uv_cube_3face()
    else:
        return None

    verts, faces = _build_unit_cube_geometry()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    material = trimesh.visual.material.SimpleMaterial(image=tex_img)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


# ---------------------------------------------------------------------------
# Parametric Mesh Generators
# ---------------------------------------------------------------------------


def apply_flat_colour_to_faces(
    mesh: trimesh.Trimesh, colour: tuple[int, int, int]
) -> trimesh.Trimesh:
    """Apply a flat colour to every face of a mesh via face_colors."""
    rgba = np.array([colour[0], colour[1], colour[2], 255], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
    return mesh


def _coloured_box(
    extents: tuple[float, float, float],
    translation: tuple[float, float, float],
    colour: tuple[int, int, int],
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    return apply_flat_colour_to_faces(mesh, colour)


def mesh_full_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def mesh_slab(half: str = "bottom") -> trimesh.Trimesh:
    slab = trimesh.creation.box(extents=(1.0, 0.5, 1.0))
    slab.apply_translation([0.0, 0.25 if half == "top" else -0.25, 0.0])
    return slab


def mesh_stair(facing: str = "north", half: str = "bottom") -> trimesh.Trimesh:
    base = trimesh.creation.box(extents=(1.0, 0.5, 1.0))
    base.apply_translation([0.0, -0.25, 0.0])

    offsets = {
        "north": [0.0, 0.25, 0.25],
        "south": [0.0, 0.25, -0.25],
        "east": [-0.25, 0.25, 0.0],
        "west": [0.25, 0.25, 0.0],
    }
    if facing in ("east", "west"):
        step = trimesh.creation.box(extents=(0.5, 0.5, 1.0))
    else:
        step = trimesh.creation.box(extents=(1.0, 0.5, 0.5))
    step.apply_translation(offsets.get(facing, offsets["north"]))
    combined = trimesh.util.concatenate([base, step])

    if half == "top":
        # Rotate 180° around X through the origin — preserves winding order
        # (negative Y scale would invert normals, causing backface culling).
        rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0], point=[0, 0, 0])
        combined.apply_transform(rot)
    return combined


def mesh_fence_post() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.25, 1.0, 0.25))


def mesh_fence(props: dict[str, str] | None = None) -> trimesh.Trimesh:
    """Fence post with optional horizontal connecting rails.

    Connection properties (``north``, ``south``, ``east``, ``west``) are
    expected as ``"true"``/``"false"`` strings, as produced by the adjacency
    resolver.  Each connected direction adds two horizontal rails (upper
    and lower) matching Minecraft's 1.21 fence geometry.

    Rail dimensions (per Minecraft model):
      - Cross-section: 0.125 × 0.1875  (2px × 3px)
      - Length: 0.375 (from post edge to cell boundary)
      - Upper rail Y centre: +0.34375  (Y=12–15 in pixel coords)
      - Lower rail Y centre: −0.03125  (Y=6–9 in pixel coords)
    """
    post = mesh_fence_post()
    if props is None:
        return post

    # Direction → (arm_centre_x, arm_centre_z, extents_x, extents_z)
    _ARM_OFFSETS: dict[str, tuple[float, float, float, float]] = {
        "north": (0.0, -0.3125, 0.125, 0.375),
        "south": (0.0, 0.3125, 0.125, 0.375),
        "west": (-0.3125, 0.0, 0.375, 0.125),
        "east": (0.3125, 0.0, 0.375, 0.125),
    }
    _UPPER_Y = 0.34375
    _LOWER_Y = -0.03125
    _RAIL_HEIGHT = 0.1875

    parts = [post]
    for direction, (cx, cz, ex, ez) in _ARM_OFFSETS.items():
        if props.get(direction) == "true":
            for y_centre in (_UPPER_Y, _LOWER_Y):
                rail = trimesh.creation.box(extents=(ex, _RAIL_HEIGHT, ez))
                rail.apply_translation([cx, y_centre, cz])
                parts.append(rail)

    if len(parts) == 1:
        return post
    return trimesh.util.concatenate(parts)


def mesh_wall_post() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.5, 1.0, 0.5))


def mesh_thin_pane() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.125, 1.0, 0.125))


def mesh_torch() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.125, 0.625, 0.125))


def mesh_lantern() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.375, 0.5625, 0.375))


def mesh_bed(
    colour: tuple[int, int, int],
    *,
    facing: str = "south",
    part: str = "foot",
) -> trimesh.Trimesh:
    """Procedural bed part for Minecraft's special-rendered bed block.

    The vanilla block model for beds has no cuboid elements because the game
    renders beds through a special renderer.  Without this factory, the
    renderer falls through to a full-cube fallback and beds read as solid red
    blocks.  This mesh intentionally approximates the in-game footprint: a low
    mattress slab, wooden legs, and a pillow on the head part.
    """

    wood = _WOOD_PLANK_COLOURS["dark_oak"]
    pillow = (235, 235, 225)
    parts: list[trimesh.Trimesh] = [
        _coloured_box((0.9375, 0.4375, 0.9375), (0.0, -0.25, 0.0), colour),
    ]

    for x in (-0.34375, 0.34375):
        for z in (-0.34375, 0.34375):
            parts.append(_coloured_box((0.125, 0.375, 0.125), (x, -0.3125, z), wood))

    if part == "head":
        pillow_extents: tuple[float, float, float]
        pillow_translation: tuple[float, float, float]
        if facing in {"east", "west"}:
            pillow_extents = (0.25, 0.125, 0.75)
            pillow_translation = (0.25 if facing == "east" else -0.25, 0.03125, 0.0)
        else:
            pillow_extents = (0.75, 0.125, 0.25)
            pillow_translation = (0.0, 0.03125, 0.25 if facing == "south" else -0.25)
        parts.append(_coloured_box(pillow_extents, pillow_translation, pillow))

    return trimesh.util.concatenate(parts)


def mesh_carpet() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 0.0625, 1.0))


def mesh_door(half: str = "lower") -> trimesh.Trimesh:
    """Door — thin full-height panel."""
    door = trimesh.creation.box(extents=(1.0, 1.0, 0.1875))
    if half == "upper":
        door.apply_translation([0.0, 0.5, 0.0])
    return door


def mesh_trapdoor() -> trimesh.Trimesh:
    """Trapdoor — thin horizontal panel (closed position)."""
    return trimesh.creation.box(extents=(1.0, 0.1875, 1.0))


def mesh_button() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.375, 0.125, 0.25))


def mesh_pressure_plate() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.875, 0.0625, 0.875))


def mesh_ladder() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 0.0625))


def mesh_chain() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.0625, 1.0, 0.0625))


def mesh_sign() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 0.5, 0.0625))


# ---------------------------------------------------------------------------
# Factory functions (receive props dict, return Trimesh)
# ---------------------------------------------------------------------------

MeshFactory = Callable[[dict[str, str]], trimesh.Trimesh]


def _factory_full_cube(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_full_cube()


def _factory_slab(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_slab(half=props.get("type", "bottom"))


def _factory_stair(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_stair(facing=props.get("facing", "north"), half=props.get("half", "bottom"))


def _factory_fence(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_fence(props)


def _factory_wall(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_wall_post()


def _factory_pane(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_thin_pane()


def _factory_torch(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_torch()


def _factory_lantern(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_lantern()


def _make_bed_factory(colour: tuple[int, int, int]) -> MeshFactory:
    def _factory_bed(props: dict[str, str]) -> trimesh.Trimesh:
        return mesh_bed(
            colour,
            facing=props.get("facing", "south"),
            part=props.get("part", "foot"),
        )

    return _factory_bed


def _factory_carpet(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_carpet()


def _factory_door(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_door(half=props.get("half", "lower"))


def _factory_trapdoor(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_trapdoor()


def _factory_button(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_button()


def _factory_pressure_plate(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_pressure_plate()


def _factory_ladder(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_ladder()


def _factory_chain(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_chain()


def _factory_sign(props: dict[str, str]) -> trimesh.Trimesh:
    return mesh_sign()


# ---------------------------------------------------------------------------
# Registry: block_id -> mesh factory
# ---------------------------------------------------------------------------

BLOCK_MESH_FACTORIES: dict[str, MeshFactory] = {}


def _populate_factories() -> None:
    """Register all known blocks to their geometry factories."""
    f = BLOCK_MESH_FACTORIES

    # Full cubes — everything in BLOCK_COLOURS that isn't specialised gets full cube
    # We register specialised ones explicitly; unknown IDs fall through to full cube.

    # Stairs
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_stairs"] = _factory_stair
    for prefix in (
        "stone",
        "cobblestone",
        "stone_brick",
        "brick",
        "deepslate_brick",
        "mossy_cobblestone",
        "sandstone",
        "red_sandstone",
        "smooth_sandstone",
        "prismarine",
        "prismarine_brick",
        "dark_prismarine",
        "end_stone_brick",
        "purpur",
        "polished_andesite",
        "polished_diorite",
        "polished_granite",
        "nether_brick",
        "red_nether_brick",
        "polished_blackstone_brick",
    ):
        f[f"minecraft:{prefix}_stairs"] = _factory_stair

    # Slabs
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_slab"] = _factory_slab
    for prefix in (
        "stone",
        "cobblestone",
        "stone_brick",
        "brick",
        "deepslate_brick",
        "smooth_stone",
        "mossy_cobblestone",
        "sandstone",
        "red_sandstone",
        "prismarine",
        "prismarine_brick",
        "dark_prismarine",
        "end_stone_brick",
        "purpur",
        "polished_andesite",
        "polished_diorite",
        "polished_granite",
        "nether_brick",
        "red_nether_brick",
        "polished_blackstone_brick",
    ):
        f[f"minecraft:{prefix}_slab"] = _factory_slab

    # Fences
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_fence"] = _factory_fence
        f[f"minecraft:{wood}_fence_gate"] = _factory_fence

    # Walls
    for prefix in (
        "cobblestone",
        "mossy_cobblestone",
        "stone_brick",
        "mossy_stone_brick",
        "brick",
        "deepslate_brick",
        "sandstone",
        "red_sandstone",
        "nether_brick",
        "red_nether_brick",
        "prismarine",
        "end_stone_brick",
        "polished_blackstone_brick",
        "blackstone",
        "deepslate_tile",
        "cobbled_deepslate",
    ):
        f[f"minecraft:{prefix}_wall"] = _factory_wall

    # Panes
    f["minecraft:iron_bars"] = _factory_pane
    f["minecraft:glass_pane"] = _factory_pane
    for colour_name in _ALL_COLOURS:
        f[f"minecraft:{colour_name}_stained_glass_pane"] = _factory_pane

    # Torches
    f["minecraft:torch"] = _factory_torch
    f["minecraft:wall_torch"] = _factory_torch
    f["minecraft:soul_torch"] = _factory_torch

    # Lanterns
    f["minecraft:lantern"] = _factory_lantern
    f["minecraft:soul_lantern"] = _factory_lantern

    # Beds use a special renderer in Minecraft; the JSON model is intentionally
    # empty, so they need a procedural fallback here.
    for colour_name in _ALL_COLOURS:
        f[f"minecraft:{colour_name}_bed"] = _make_bed_factory(_WOOL_COLOURS[colour_name])

    # Carpets
    for colour_name in _ALL_COLOURS:
        f[f"minecraft:{colour_name}_carpet"] = _factory_carpet

    # Doors
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_door"] = _factory_door
    f["minecraft:iron_door"] = _factory_door

    # Trapdoors
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_trapdoor"] = _factory_trapdoor
    f["minecraft:iron_trapdoor"] = _factory_trapdoor

    # Buttons
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_button"] = _factory_button
    f["minecraft:stone_button"] = _factory_button
    f["minecraft:polished_blackstone_button"] = _factory_button

    # Pressure plates
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_pressure_plate"] = _factory_pressure_plate
    f["minecraft:stone_pressure_plate"] = _factory_pressure_plate
    f["minecraft:polished_blackstone_pressure_plate"] = _factory_pressure_plate

    # Misc
    f["minecraft:ladder"] = _factory_ladder
    f["minecraft:chain"] = _factory_chain
    for wood in _WOOD_TYPES:
        f[f"minecraft:{wood}_sign"] = _factory_sign
        f[f"minecraft:{wood}_wall_sign"] = _factory_sign


_populate_factories()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_coloured_block_mesh(
    block_id: str, properties: dict[str, str] | None = None
) -> trimesh.Trimesh:
    """
    Return a Trimesh for the given block ID and properties.

    Resolution order:
    1. Phase 2: blockstate resolver + model parser (accurate JSON geometry).
    2. Phase 1.5: registered factory with texture mapping.
    3. Flat-colour procedural fallback.
    """
    props = properties or {}

    # --- Phase 2: JSON model pipeline ---
    mesh = _try_json_model(block_id, props)
    if mesh is not None:
        return mesh

    # --- Phase 1.5 / fallback ---
    factory = BLOCK_MESH_FACTORIES.get(block_id)

    if factory is None:
        # Unknown block — try textured cube, then flat-colour cube
        if _TEXTURES_ENABLED:
            textured = create_textured_cube(block_id)
            if textured is not None:
                return textured
        mesh = mesh_full_cube()
        colour = get_block_colour(block_id)
        apply_flat_colour_to_faces(mesh, colour)
        return mesh

    # Check if this is a full-cube factory (not specialised geometry)
    is_full_cube = factory is _factory_full_cube

    # For full cubes, attempt texture mapping
    if is_full_cube and _TEXTURES_ENABLED:
        textured = create_textured_cube(block_id)
        if textured is not None:
            return textured

    # Specialised or non-textured: procedural with flat colour
    mesh = factory(props)
    if block_id.endswith("_bed"):
        return mesh
    colour = get_block_colour(block_id)
    apply_flat_colour_to_faces(mesh, colour)
    return mesh


def _try_json_model(block_id: str, props: dict[str, str]) -> trimesh.Trimesh | None:
    """
    Attempt to build a mesh via the Phase 2 JSON model pipeline.

    Returns None if blockstate/model is unavailable, allowing fallback.
    """
    try:
        from voxel_renderer.blockstate_resolver import resolve_block_models
        from voxel_renderer.model_parser import build_model_mesh

        applications = resolve_block_models(block_id, props)
        if applications is None or not applications:
            return None

        meshes: list[trimesh.Trimesh] = []
        for app in applications:
            m = build_model_mesh(app.model, y_rot=app.y_rotation, x_rot=app.x_rotation)
            if m is not None:
                meshes.append(m)

        if not meshes:
            return None

        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)

    except Exception as e:
        logger.debug("Phase 2 model pipeline failed for %s: %s", block_id, e)
        return None
