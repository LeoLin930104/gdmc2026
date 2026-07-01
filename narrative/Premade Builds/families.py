from __future__ import annotations

from typing import Literal

Tier = Literal["thriving", "strained", "struggling"]
TIERS: tuple[Tier, ...] = ("thriving", "strained", "struggling")

# ---------------------------------------------------------------------------
# Remap families: baseline block id -> per-tier target.
# "strained" is always the authored baseline (identity) so an unswapped build
# is correct. Edit the thriving/struggling columns to taste.
# ---------------------------------------------------------------------------
MATERIAL_FAMILIES: dict[str, dict[Tier, str]] = {
    # --- stone (struggling = mossy; cracked has no slab/wall variant) ---
    "minecraft:stone_bricks": {
        "thriving": "minecraft:stone_bricks",
        "strained": "minecraft:stone_bricks",
        "struggling": "minecraft:mossy_stone_bricks",
    },
    "minecraft:stone_brick_slab": {
        "thriving": "minecraft:stone_brick_slab",
        "strained": "minecraft:stone_brick_slab",
        "struggling": "minecraft:mossy_stone_brick_slab",
    },
    "minecraft:stone_brick_wall": {
        "thriving": "minecraft:stone_brick_wall",
        "strained": "minecraft:stone_brick_wall",
        "struggling": "minecraft:mossy_stone_brick_wall",
    },

    # --- cobblestone (gold family: swings BOTH ways, shape-complete) ---
    # thriving = dressed into stone brick, struggling = moss reclaims it.
    "minecraft:cobblestone": {
        "thriving": "minecraft:stone_bricks",
        "strained": "minecraft:cobblestone",
        "struggling": "minecraft:mossy_cobblestone",
    },
    "minecraft:cobblestone_stairs": {
        "thriving": "minecraft:stone_brick_stairs",
        "strained": "minecraft:cobblestone_stairs",
        "struggling": "minecraft:mossy_cobblestone_stairs",
    },
    "minecraft:cobblestone_slab": {
        "thriving": "minecraft:stone_brick_slab",
        "strained": "minecraft:cobblestone_slab",
        "struggling": "minecraft:mossy_cobblestone_slab",
    },
    "minecraft:cobblestone_wall": {
        "thriving": "minecraft:stone_brick_wall",
        "strained": "minecraft:cobblestone_wall",
        "struggling": "minecraft:mossy_cobblestone_wall",
    },

    # --- deepslate brick (gold family: swings BOTH ways, shape-complete) ---
    # thriving = polished/dressed, struggling = crumbled back to cobbled rubble.
    "minecraft:deepslate_bricks": {
        "thriving": "minecraft:polished_deepslate",
        "strained": "minecraft:deepslate_bricks",
        "struggling": "minecraft:cobbled_deepslate",
    },
    "minecraft:deepslate_brick_stairs": {
        "thriving": "minecraft:polished_deepslate_stairs",
        "strained": "minecraft:deepslate_brick_stairs",
        "struggling": "minecraft:cobbled_deepslate_stairs",
    },
    "minecraft:deepslate_brick_slab": {
        "thriving": "minecraft:polished_deepslate_slab",
        "strained": "minecraft:deepslate_brick_slab",
        "struggling": "minecraft:cobbled_deepslate_slab",
    },
    "minecraft:deepslate_brick_wall": {
        "thriving": "minecraft:polished_deepslate_wall",
        "strained": "minecraft:deepslate_brick_wall",
        "struggling": "minecraft:cobbled_deepslate_wall",
    },

    # --- wood: Framing A (brightness/freshness = health) ---
    # birch = fresh/cared-for, oak = neutral baseline, spruce = grayed/worn.
    # Wood is shape-complete across species, so add stairs/slabs/doors/etc.
    # here later with the same birch/oak/spruce columns.
    # Per-settlement "rot" alt: swap the struggling column to mangrove_* for a
    # damp, waterlogged-rot look (see STRUGGLING_WOOD_ALT below).
    "minecraft:oak_planks": {
        "thriving": "minecraft:birch_planks",
        "strained": "minecraft:oak_planks",
        "struggling": "minecraft:spruce_planks",
    },
    "minecraft:oak_fence": {
        "thriving": "minecraft:birch_fence",
        "strained": "minecraft:oak_fence",
        "struggling": "minecraft:spruce_fence",
    },
    "minecraft:oak_fence_gate": {
        "thriving": "minecraft:birch_fence_gate",
        "strained": "minecraft:oak_fence_gate",
        "struggling": "minecraft:spruce_fence_gate",
    },
    "minecraft:oak_stairs": {
        "thriving": "minecraft:birch_stairs",
        "strained": "minecraft:oak_stairs",
        "struggling": "minecraft:spruce_stairs",
    },
    "minecraft:oak_slab": {
        "thriving": "minecraft:birch_slab",
        "strained": "minecraft:oak_slab",
        "struggling": "minecraft:spruce_slab",
    },
    "minecraft:oak_trapdoor": {
        "thriving": "minecraft:birch_trapdoor",
        "strained": "minecraft:oak_trapdoor",
        "struggling": "minecraft:spruce_trapdoor",
    },

    # --- wool awning (color fade = condition) ---
    "minecraft:white_wool": {
        "thriving": "minecraft:white_wool",
        "strained": "minecraft:white_wool",   # authored baseline preserved (white can't brighten)
        "struggling": "minecraft:gray_wool",
    },
    "minecraft:red_wool": {
        "thriving": "minecraft:red_wool",
        "strained": "minecraft:red_wool",
        "struggling": "minecraft:brown_wool",
    },

    # --- light (warm/plentiful -> cold/failing) ---
    "minecraft:lantern": {
        "thriving": "minecraft:lantern",
        "strained": "minecraft:lantern",
        "struggling": "minecraft:soul_lantern",
    },
    "minecraft:sea_lantern": {
        "thriving": "minecraft:sea_lantern",
        "strained": "minecraft:sea_lantern",
        "struggling": "minecraft:dark_prismarine",
    },

    # --- accents (residential_7); full-block use only, so block-only variants ok ---
    # white plaster: crisp -> grimy. Concrete has no stairs/slab, used as a cube.
    "minecraft:white_concrete": {
        "thriving": "minecraft:white_concrete",
        "strained": "minecraft:white_concrete",
        "struggling": "minecraft:light_gray_concrete",
    },
    # blackstone accent: SAFE only because it's used as a full block here —
    # cracked_polished_blackstone_bricks is block-only (no stairs/slab/wall).
    # If a future build uses polished_blackstone_brick_stairs, struggling breaks.
    "minecraft:polished_blackstone_bricks": {
        "thriving": "minecraft:polished_blackstone_bricks",
        "strained": "minecraft:polished_blackstone_bricks",
        "struggling": "minecraft:cracked_polished_blackstone_bricks",
    },
}

# Blocks seen in builds that intentionally PASS THROUGH unchanged (no good mood
# variant): metal/functional access blocks. They fall through resolve_block()'s
# graceful default, so they need no entry — listed here only as documentation
# so a future reader knows the omission is deliberate, not an oversight.
#   chain, iron_bars, ladder  -> metal/access, no decay variant
#   dirt                      -> baked ground; should be deleted from the .nbt
#                                (foundation pass generates grade), not swapped
# acacia_trapdoor (residential_7): a 2nd wood species used as a warm accent.
# DECIDED (2026-06-17): keep as a FIXED accent (passthrough, unchanged across all
# tiers) so the warm orange contrast stays constant regardless of mood.

# Optional per-settlement override: a stronger, damp "rot" read for struggling
# wood. To use, copy these into the struggling column of the wood families
# (e.g. for a poor waterside village). Kept separate so the default stays spruce.
STRUGGLING_WOOD_ALT: dict[str, str] = {
    "minecraft:oak_planks": "minecraft:mangrove_planks",
    "minecraft:oak_fence": "minecraft:mangrove_fence",
    "minecraft:oak_fence_gate": "minecraft:mangrove_fence_gate",
}

# ---------------------------------------------------------------------------
# Drop on struggling: these blocks become air at the struggling tier
# (the fountain shrub dies back, the basin runs dry). Thriving/strained keep them.
# ---------------------------------------------------------------------------
DROP_ON_STRUGGLING: set[str] = {
    "minecraft:oak_leaves",
    "minecraft:water",
}

# ---------------------------------------------------------------------------
# Liquids: cannot be reproduced by naive per-block placement (flowing states
# re-flow). The placement code must place source blocks / let them re-fill, or
# treat the liquid region specially. Flagged here so the resolver can warn.
# ---------------------------------------------------------------------------
LIQUIDS: set[str] = {
    "minecraft:water",
    "minecraft:lava",
}

# ---------------------------------------------------------------------------
# Fixed-functional: carry function, never remapped. Substring match against the
# de-namespaced id. Kept in sync with dump_palette.py's _FIXED_FUNCTIONAL.
# ---------------------------------------------------------------------------
FIXED_FUNCTIONAL: tuple[str, ...] = (
    "chest", "barrel", "furnace", "smoker", "blast_furnace", "lectern",
    "crafting_table", "loom", "smithing_table", "cartography_table",
    "fletching_table", "grindstone", "stonecutter", "brewing_stand",
    "anvil", "bell", "campfire", "bookshelf", "composter", "cauldron",
    "beehive", "bee_nest", "jukebox", "note_block", "flower_pot",
    "armor_stand", "item_frame", "sign", "banner", "head", "skull",
    "spawner", "beacon", "conduit",
)

_AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")


def _short(block_id: str) -> str:
    return block_id.split(":", 1)[-1]


def is_fixed_functional(block_id: str) -> bool:
    s = _short(block_id)
    return any(tok in s for tok in FIXED_FUNCTIONAL)


def is_liquid(block_id: str) -> bool:
    return block_id in LIQUIDS


def resolve_block(block_id: str, tier: Tier) -> str | None:
    if block_id in _AIR:
        return block_id
    if tier == "struggling" and block_id in DROP_ON_STRUGGLING:
        return None
    family = MATERIAL_FAMILIES.get(block_id)
    if family is not None:
        return family[tier]
    return block_id


def _demo() -> None:
    width = max(len(k) for k in MATERIAL_FAMILIES)
    print(f"{'baseline'.ljust(width)}  {'thriving':<28}{'strained':<28}struggling")
    print("-" * (width + 2 + 28 + 28 + 12))
    for base, tiers in MATERIAL_FAMILIES.items():
        print(
            f"{base.ljust(width)}  "
            f"{tiers['thriving']:<28}{tiers['strained']:<28}{tiers['struggling']}"
        )
    print()
    print("drop on struggling (-> air):", sorted(DROP_ON_STRUGGLING))
    print("liquids (special placement):", sorted(LIQUIDS))


if __name__ == "__main__":
    _demo()
