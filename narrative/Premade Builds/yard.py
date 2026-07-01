from __future__ import annotations

import hashlib
import random

import families

GRASS = "minecraft:grass_block"

# --- biome-adaptive base ground -------------------------------------------
# The "healthy" ground block that stands in for grass, so a desert town sits on
# sand, a badlands town on red sand, a taiga on podzol, etc. Keyword-matched
# from the biome id (mirrors wallface_narrative.biome_family, kept local to
# avoid a cross-package import). Anything unmatched stays temperate grass.
_BIOME_GROUND = (
    (("desert",), "minecraft:sand"),
    (("badlands", "mesa"), "minecraft:red_sand"),
    (("snow", "frozen", "ice"), "minecraft:snow_block"),
    (("taiga", "grove", "old_growth"), "minecraft:podzol"),
    (("mushroom",), "minecraft:mycelium"),
)


def biome_ground(biome: str | None) -> str:
    b = (biome or "").lower()
    for keys, block in _BIOME_GROUND:
        if any(k in b for k in keys):
            return block
    return GRASS


# --- ground: weighted (block, weight) -------------------------------------
_MOOD_GROUND = {
    "thriving":   [(GRASS, 12)],
    "strained":   [(GRASS, 9), ("minecraft:coarse_dirt", 1)],
    "struggling": [(GRASS, 3), ("minecraft:coarse_dirt", 4),
                   ("minecraft:gravel", 2), ("minecraft:podzol", 1)],
}
_ROLE_GROUND = {
    "town_square": [("minecraft:gravel", 2), ("minecraft:dirt_path", 2)],
    "residential": [("minecraft:dirt_path", 1)],
    "barracks":    [("minecraft:gravel", 3), ("minecraft:coarse_dirt", 2)],
}

# --- props: weighted (token, weight); None = empty; "@flower" = a random flower
_MOOD_PROPS = {
    "thriving":   [(None, 6), ("@flower", 3), ("minecraft:short_grass", 2)],
    "strained":   [(None, 10), ("minecraft:short_grass", 2), ("@flower", 1)],
    "struggling": [(None, 7), ("minecraft:dead_bush", 2),
                   ("minecraft:fern", 2), ("minecraft:short_grass", 1)],
}
_ROLE_PROPS = {
    "town_square": [("@flower", 2)],
    "residential": [("minecraft:short_grass", 2), ("@flower", 1), ("minecraft:oak_fence", 1)],
    "barracks":    [("minecraft:oak_fence", 1)],
}

_CATEGORIES = {
    "@flower": [
        "minecraft:dandelion", "minecraft:poppy", "minecraft:azure_bluet",
        "minecraft:oxeye_daisy", "minecraft:cornflower", "minecraft:allium",
    ],
}

# --- perimeter fence: P(a perimeter cell gets a fence) by role + mood ------
# town_square is absent -> open public space (no border). Lower P when
# struggling reads as a broken/gappy fence; high P when thriving reads as kept.
_FENCE_BASE = "minecraft:oak_fence"   # mood-swapped (birch/oak/spruce) like the builds
_ROLE_FENCE = {
    "residential": {"thriving": 0.75, "strained": 0.45, "struggling": 0.20},
    "barracks":    {"thriving": 0.65, "strained": 0.40, "struggling": 0.20},
}

# Ground a plant can sit on without popping off; props NOT in _NON_PLANT need it.
_PLANTABLE = {
    "minecraft:grass_block", "minecraft:dirt", "minecraft:coarse_dirt",
    "minecraft:podzol", "minecraft:farmland", "minecraft:moss_block",
}
_NON_PLANT = {"minecraft:oak_fence"}   # placeable on any ground


def _pool(table_mood: dict, table_role: dict, mood: str, role: str) -> list:
    base = table_mood.get(mood, table_mood["strained"])
    return base + table_role.get(role, [])


def _weighted_choice(pool: list, rng: random.Random):
    total = sum(w for _, w in pool)
    r = rng.uniform(0, total)
    upto = 0.0
    for item, w in pool:
        upto += w
        if r <= upto:
            return item
    return pool[-1][0]


def _resolve(token: str, rng: random.Random) -> str:
    if token in _CATEGORIES:
        return rng.choice(_CATEGORIES[token])
    return token


def _cell_rng(seed_name: str, x: int, z: int) -> random.Random:
    digest = hashlib.sha256(f"{seed_name}:{int(x)}:{int(z)}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def place_yard(
    editor,
    cells,
    occupied,
    origin,
    ground_y,
    role: str,
    mood: str,
    seed_name: str,
    biome: str | None = None,
    clear_height: int = 5,
) -> dict:
    from gdpc import Block  # lazy

    ox, oz = int(origin[0]), int(origin[2])
    occupied = {(int(x), int(z)) for x, z in occupied}
    ground_pool = _pool(_MOOD_GROUND, _ROLE_GROUND, mood, role)
    # Swap the temperate grass base for the biome's ground (sand/podzol/snow/...)
    # so the yard matches the surrounding terrain instead of always reading grass.
    base_ground = biome_ground(biome)
    plantable = _PLANTABLE
    if base_ground != GRASS:
        ground_pool = [(base_ground if blk == GRASS else blk, w) for blk, w in ground_pool]
        # podzol/mycelium hold plants; sand/snow don't, so props on them are
        # skipped below (no floating flowers) rather than forced.
        if base_ground in ("minecraft:podzol", "minecraft:mycelium"):
            plantable = _PLANTABLE | {base_ground}
    prop_pool = _pool(_MOOD_PROPS, _ROLE_PROPS, mood, role)
    air = Block("minecraft:air")

    cell_set = {(int(x), int(z)) for x, z in cells}
    fence_p = _ROLE_FENCE.get(role, {}).get(mood, 0.0)
    fence_block = families.resolve_block(_FENCE_BASE, mood) or _FENCE_BASE
    stats = {"ground": 0, "props": 0, "fences": 0}

    def _is_perimeter(x: int, z: int) -> bool:
        return any((x + dx, z + dz) not in cell_set
                   for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    for x, z in cell_set:
        if (x, z) in occupied:
            continue  # under a build — build_premade handles it
        wx, wz = ox + x, oz + z
        g = ground_y(wx, wz)
        rng = _cell_rng(seed_name, x, z)

        ground_block = _weighted_choice(ground_pool, rng)
        editor.placeBlock((wx, g, wz), Block(ground_block))
        for y in range(g + 1, g + 1 + clear_height):
            editor.placeBlock((wx, y, wz), air)
        stats["ground"] += 1

        # Perimeter cells may get a fence (mood-scaled density = broken vs kept),
        # which takes the slot a prop would have used.
        if fence_p and _is_perimeter(x, z) and rng.random() < fence_p:
            editor.placeBlock((wx, g + 1, wz), Block(fence_block))
            stats["fences"] += 1
            continue

        token = _weighted_choice(prop_pool, rng)
        if token is None:
            continue
        prop = _resolve(token, rng)
        if prop not in _NON_PLANT and ground_block not in plantable:
            continue  # would pop off gravel/path/sand — skip
        editor.placeBlock((wx, g + 1, wz), Block(prop))
        stats["props"] += 1

    return stats


if __name__ == "__main__":
    # gdpc-free inspection: show resolved pools + a sample scatter per (role, mood).
    import collections

    roles = ("town_square", "residential", "barracks")
    moods = ("thriving", "strained", "struggling")
    for role in roles:
        for mood in moods:
            gp = _pool(_MOOD_GROUND, _ROLE_GROUND, mood, role)
            pp = _pool(_MOOD_PROPS, _ROLE_PROPS, mood, role)
            sample = collections.Counter()
            for i in range(400):
                rng = _cell_rng("Verdant Spire", i % 20, i // 20)
                g = _weighted_choice(gp, rng)
                t = _weighted_choice(pp, rng)
                p = "none" if t is None else _resolve(t, rng).split(":")[-1]
                sample[(g.split(":")[-1], p)] += 1
            top = ", ".join(f"{k[0]}/{k[1]}:{v}" for k, v in sample.most_common(4))
            print(f"{role:<12} {mood:<11} -> {top}")
