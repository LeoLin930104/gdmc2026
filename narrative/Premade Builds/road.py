from __future__ import annotations

import hashlib
import random

import numpy as np

# --- surface palette by mood: weighted (block, weight) -----------------------
# strained mirrors the generator's own PATH_BLOCKS mix so the baseline road
# reads the same as before; thriving dresses it up, struggling lets it go.
_SURFACE_PALETTE: dict[str, list[tuple[str, int]]] = {
    "thriving": [
        ("minecraft:stone_bricks", 4),
        ("minecraft:polished_andesite", 3),
        ("minecraft:smooth_stone", 2),
        ("minecraft:chiseled_stone_bricks", 1),
        ("minecraft:stone", 1),
    ],
    "strained": [
        ("minecraft:cobblestone", 4),
        ("minecraft:stone_bricks", 3),
        ("minecraft:polished_andesite", 2),
        ("minecraft:chiseled_stone_bricks", 1),
        ("minecraft:mossy_cobblestone", 1),
    ],
    "struggling": [
        ("minecraft:mossy_cobblestone", 4),
        ("minecraft:cobblestone", 3),
        ("minecraft:cracked_stone_bricks", 2),
        ("minecraft:gravel", 2),
        ("minecraft:coarse_dirt", 1),
        ("minecraft:andesite", 1),
    ],
}

# Slab variant for each surface block, used on the generator's slab cells (the
# half-step it places on slopes). Blocks with no matching slab (gravel,
# coarse_dirt, chiseled) are absent -> a slab cell that rolled one falls back to
# plain cobblestone + cobblestone_slab so the elevation step stays intact.
_SLAB_FOR: dict[str, str] = {
    "minecraft:cobblestone": "minecraft:cobblestone_slab",
    "minecraft:mossy_cobblestone": "minecraft:mossy_cobblestone_slab",
    "minecraft:stone_bricks": "minecraft:stone_brick_slab",
    "minecraft:cracked_stone_bricks": "minecraft:stone_brick_slab",  # no cracked slab
    "minecraft:polished_andesite": "minecraft:polished_andesite_slab",
    "minecraft:andesite": "minecraft:andesite_slab",
    "minecraft:smooth_stone": "minecraft:smooth_stone_slab",
    "minecraft:stone": "minecraft:stone_slab",
}
_SLAB_FALLBACK = ("minecraft:cobblestone", "minecraft:cobblestone_slab")

_SLAB_STATE = {"type": "bottom"}


def _cell_rng(seed_name: str, x: int, z: int) -> random.Random:
    digest = hashlib.sha256(f"road:{seed_name}:{int(x)}:{int(z)}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _weighted_choice(pool, rng: random.Random):
    total = sum(w for _, w in pool)
    r = rng.uniform(0, total)
    upto = 0.0
    for item, w in pool:
        upto += w
        if r <= upto:
            return item
    return pool[-1][0]


def place_roads(
    editor,
    path_mask,
    path_base_y,
    path_slab_mask,
    heightmap,
    origin,
    mood: str,
    seed_name: str,
) -> dict:
    """Re-skin every road cell by mood. Returns stats.

    `path_mask` / `path_base_y` / `path_slab_mask` are the [z, x] arrays from
    settlement_data.npz; `path_base_y` (the road surface height) falls back to
    `heightmap` when the generator didn't export it. Only the surface block (and
    its slab) is overwritten — the dirt foundation the generator already laid
    underneath is left untouched.
    """
    from gdpc import Block  # lazy

    ox, oz = int(origin[0]), int(origin[2])
    surf_pool = _SURFACE_PALETTE.get(mood, _SURFACE_PALETTE["strained"])
    base_y = path_base_y if path_base_y is not None else heightmap

    stats = {"surface": 0, "slabs": 0}

    for z, x in np.argwhere(path_mask):
        x, z = int(x), int(z)
        wx, wz = ox + x, oz + z
        surface_y = int(base_y[z, x])
        is_slab = path_slab_mask is not None and bool(path_slab_mask[z, x])
        rng = _cell_rng(seed_name, x, z)

        # Ordinary road surface (+ slab on the generator's slope cells).
        surface = _weighted_choice(surf_pool, rng)
        slab = _SLAB_FOR.get(surface)
        if is_slab and slab is None:
            surface, slab = _SLAB_FALLBACK
        editor.placeBlock((wx, surface_y, wz), Block(surface))
        stats["surface"] += 1
        if is_slab:
            editor.placeBlock((wx, surface_y + 1, wz), Block(slab, _SLAB_STATE))
            stats["slabs"] += 1

    return stats


if __name__ == "__main__":
    # gdpc-free preview: surface mix per mood over a sample road.
    import collections

    # A straight 60-cell road strip (z fixed), plus a couple of branch cells.
    sample = [(x, 0) for x in range(60)] + [(30, 1), (30, 2), (45, 1)]
    for mood in ("thriving", "strained", "struggling"):
        counts = collections.Counter()
        for x, z in sample:
            rng = _cell_rng("Emberwell", x, z)
            counts[_weighted_choice(_SURFACE_PALETTE[mood], rng).split(":")[-1]] += 1
        top = ", ".join(f"{k}:{v}" for k, v in counts.most_common(6))
        print(f"{mood:<11} {len(sample)} cells")
        print(f"            surface: {top}")
