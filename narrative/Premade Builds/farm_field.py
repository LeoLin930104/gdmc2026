from __future__ import annotations

import hashlib
import random

import families

# --- block palette (mirrors builder.py's farm constants) -------------------
FOUNDATION_BLOCK = "minecraft:dirt"
BORDER_BLOCK = "minecraft:oak_log"          # mood-swapped (oak/spruce/mossy) like the builds
SOIL_BLOCK = "minecraft:farmland"
WATER_BLOCK = "minecraft:water"
DRY_CHANNEL_BLOCK = "minecraft:coarse_dirt"  # struggling: a dried-up irrigation channel

# crop id + the "ripe" age its block uses (builder.py CROP_BLOCKS)
_CROPS = [
    ("minecraft:wheat", 7),
    ("minecraft:carrots", 7),
    ("minecraft:potatoes", 7),
    ("minecraft:beetroots", 3),
]

# --- mood knobs ------------------------------------------------------------
# Border block by mood — struggling weathers oak logs to stripped (bark fallen
# off), the rest keep BORDER_BLOCK. Still run through families.resolve_block so a
# wood-species swap (if any) applies on top.
_BORDER_BY_MOOD = {
    "thriving":   BORDER_BLOCK,
    "strained":   BORDER_BLOCK,
    "struggling": "minecraft:stripped_oak_log",
}
# P(a crop-land cell actually grows a crop); the rest become bare/weedy farmland.
_CROP_DENSITY = {"thriving": 0.95, "strained": 0.6, "struggling": 0.22}
# P(a border cell keeps its log) — low when struggling reads as a broken fence.
_BORDER_KEEP = {"thriving": 1.0, "strained": 0.92, "struggling": 0.55}
# P(a channel cell holds water vs. dries to coarse dirt).
_WATER_KEEP = {"thriving": 1.0, "strained": 0.95, "struggling": 0.5}
# Farmland moisture state (0 dry .. 7 wet) — dry farmland renders pale/cracked.
_SOIL_MOISTURE = {"thriving": "7", "strained": "7", "struggling": "0"}

# What fills a crop-land cell that DIDN'T grow a crop: weighted (token, weight).
# None = bare farmland. Plants sit at y+1 on the farmland.
_EMPTY_FILL = {
    "thriving":   [(None, 1)],
    "strained":   [(None, 3), ("minecraft:short_grass", 1)],
    "struggling": [(None, 3), ("minecraft:dead_bush", 3), ("minecraft:short_grass", 1)],
}


def farm_layout(cells):
    cell_set = {(int(x), int(z)) for x, z in cells}
    if not cell_set:
        return cell_set, set(), set(), set()

    xs = [x for x, _ in cell_set]
    zs = [z for _, z in cell_set]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    width = max_x - min_x + 1
    depth = max_z - min_z + 1

    border = set()
    for x, z in cell_set:
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, z + dz) not in cell_set:
                border.add((x, z))
                break

    if width >= depth:
        center_z = (min_z + max_z) // 2
        water = {(x, center_z) for x in range(min_x, max_x + 1) if (x, center_z) in cell_set}
    else:
        center_x = (min_x + max_x) // 2
        water = {(center_x, z) for z in range(min_z, max_z + 1) if (center_x, z) in cell_set}

    crop_land = cell_set - border - water
    return cell_set, border, water, crop_land


def _cell_rng(seed_name: str, x: int, z: int) -> random.Random:
    digest = hashlib.sha256(f"farm:{seed_name}:{int(x)}:{int(z)}".encode("utf-8")).hexdigest()
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


def _crop_state(crop: str, ripe_age: int, mood: str, rng: random.Random) -> dict:
    if mood == "struggling":
        age = rng.randint(1, max(1, ripe_age // 2))
    elif mood == "strained":
        age = rng.randint(max(1, ripe_age // 2), ripe_age)
    else:
        age = ripe_age
    return {"age": str(age)}


def place_farm_field(
    editor,
    cells,
    origin,
    ground_y,
    mood: str,
    seed_name: str,
    clear_height: int = 5,
) -> dict:
    from gdpc import Block  # lazy

    ox, oz = int(origin[0]), int(origin[2])
    _, border, water, crop_land = farm_layout(cells)
    air = Block("minecraft:air")

    base_border = _BORDER_BY_MOOD.get(mood, BORDER_BLOCK)
    border_block = families.resolve_block(base_border, mood) or base_border
    border_keep = _BORDER_KEEP.get(mood, 1.0)
    water_keep = _WATER_KEEP.get(mood, 1.0)
    moisture = _SOIL_MOISTURE.get(mood, "7")
    empty_pool = _EMPTY_FILL.get(mood, _EMPTY_FILL["strained"])
    crop_density = _CROP_DENSITY.get(mood, 0.6)

    stats = {"border": 0, "water": 0, "crops": 0, "empty": 0}

    def _clear_above(wx, g, wz):
        for y in range(g + 1, g + 1 + clear_height):
            editor.placeBlock((wx, y, wz), air)

    # Border ring (mood-gated: a broken border just leaves the foundation).
    for x, z in border:
        wx, wz = ox + x, oz + z
        g = ground_y(wx, wz)
        editor.placeBlock((wx, g, wz), Block(FOUNDATION_BLOCK))
        _clear_above(wx, g, wz)
        if _cell_rng(seed_name, x, z).random() < border_keep:
            editor.placeBlock((wx, g + 1, wz), Block(border_block))
            stats["border"] += 1

    # Central irrigation channel (struggling may dry parts to coarse dirt).
    for x, z in water:
        wx, wz = ox + x, oz + z
        g = ground_y(wx, wz)
        _clear_above(wx, g, wz)
        wet = _cell_rng(seed_name, x, z).random() < water_keep
        editor.placeBlock((wx, g, wz), Block(WATER_BLOCK if wet else DRY_CHANNEL_BLOCK))
        stats["water"] += 1

    # Crop land: farmland everywhere, a crop or weed/empty on top per mood.
    for x, z in crop_land:
        wx, wz = ox + x, oz + z
        g = ground_y(wx, wz)
        editor.placeBlock((wx, g, wz), Block(SOIL_BLOCK, {"moisture": moisture}))
        _clear_above(wx, g, wz)
        rng = _cell_rng(seed_name, x, z)
        if rng.random() < crop_density:
            crop, ripe = rng.choice(_CROPS)
            editor.placeBlock((wx, g + 1, wz), Block(crop, _crop_state(crop, ripe, mood, rng)))
            stats["crops"] += 1
        else:
            token = _weighted_choice(empty_pool, rng)
            if token is not None:
                editor.placeBlock((wx, g + 1, wz), Block(token))
            stats["empty"] += 1

    return stats


if __name__ == "__main__":
    # gdpc-free preview: layout split + a sample crop/empty mix per mood.
    import collections

    # A 12x8 rectangular cell to exercise border/water/crop split.
    sample_cells = [(x, z) for x in range(12) for z in range(8)]
    _, border, water, crop = farm_layout(sample_cells)
    print(f"layout: {len(sample_cells)} cells -> "
          f"{len(border)} border, {len(water)} water, {len(crop)} crop_land")

    for mood in ("thriving", "strained", "struggling"):
        density = _CROP_DENSITY[mood]
        counts = collections.Counter()
        for x, z in sorted(crop):
            rng = _cell_rng("Verdant Spire", x, z)
            if rng.random() < density:
                c, ripe = rng.choice(_CROPS)
                counts[c.split(":")[-1] + f"@{_crop_state(c, ripe, mood, rng)['age']}"] += 1
            else:
                t = _weighted_choice(_EMPTY_FILL[mood], rng)
                counts["empty" if t is None else t.split(":")[-1]] += 1
        base_border = _BORDER_BY_MOOD.get(mood, BORDER_BLOCK)
        border_block = families.resolve_block(base_border, mood) or base_border
        top = ", ".join(f"{k}:{v}" for k, v in counts.most_common(6))
        print(f"{mood:<11} border={border_block.split(':')[-1]} "
              f"keep={_BORDER_KEEP[mood]} moist={_SOIL_MOISTURE[mood]} -> {top}")
