from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import families
from nbt_structure import Structure, StructureBlock, parse_structure

_AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")

# Mood tiers the lore->tier resolver can pick; "strained" is the authored baseline.
TIERS = families.TIERS


def chest_local_pos(structure: Structure) -> tuple[int, int, int] | None:
    chest = barrel = None
    for b in structure.blocks:
        short = b.name.split(":")[-1]
        if "chest" in short and chest is None:
            chest = b.pos
        elif "barrel" in short and barrel is None:
            barrel = b.pos
    return chest if chest is not None else barrel


def mood_tier_for(settlement) -> str:
    tier = getattr(settlement, "mood_tier", None)
    if isinstance(tier, str) and tier in TIERS:
        return tier
    return "strained"


def place_premade(
    editor,
    structure: Structure,
    anchor: tuple[int, int, int],
    tier: str = "strained",
    rotation: int = 0,
    place_air: bool = False,
    decay: bool = True,
    decay_seed: str | None = None,
    chest_payloads: dict[tuple[int, int, int], str] | None = None,
) -> dict:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    from gdpc import Block, Transform  # lazy: keeps the module gdpc-free to import

    stats = {"placed": 0, "dropped": 0, "liquids": 0, "liquid_flow_skipped": 0,
             "skipped_air": 0, "removed": 0, "cobwebs": 0, "chests_filled": 0}

    decay_plan = None
    if decay and tier == "struggling":
        import decay as decay_mod  # lazy: pure-stdlib, but keep the import local
        decay_plan = decay_mod.plan_decay(
            structure, seed=decay_seed if decay_seed is not None else str(anchor)
        )

    with editor.pushTransform(Transform(anchor, rotation=rotation)):
        for sb in structure.blocks:
            if decay_plan is not None and sb.pos in decay_plan.remove:
                stats["removed"] += 1               # leave the cleared air -> a hole
                continue

            resolved = families.resolve_block(sb.name, tier)

            if resolved is None:                       # DROP (e.g. struggling shrub/water)
                stats["dropped"] += 1
                continue
            if resolved in _AIR and not place_air:     # don't bother overwriting cleared space
                stats["skipped_air"] += 1
                continue

            if families.is_liquid(resolved):
                # Place only authored SOURCE cells (level 0 / absent); skip the
                # flowing cells (level 1-8) and let Minecraft re-flow them from
                # the source. Placing every cell as a source overfills/overflows
                # (e.g. a fountain) — one authored spout source fills it naturally.
                if sb.properties.get("level", "0") != "0":
                    stats["liquid_flow_skipped"] += 1
                    continue
                block = Block(resolved)
                stats["liquids"] += 1
            else:
                data = sb.nbt
                if chest_payloads is not None and sb.pos in chest_payloads:
                    data = chest_payloads[sb.pos]   # fill the build's chest
                    stats["chests_filled"] += 1
                block = Block(resolved, sb.properties, data=data)

            editor.placeBlock(sb.pos, block)
            stats["placed"] += 1

        if decay_plan is not None:
            web = Block("minecraft:cobweb")
            for pos in decay_plan.cobwebs:
                editor.placeBlock(pos, web)
                stats["cobwebs"] += 1

    return stats


def place_premade_file(
    editor,
    nbt_path: str | Path,
    anchor: tuple[int, int, int],
    tier: str = "strained",
    rotation: int = 0,
    place_air: bool = False,
    decay: bool = True,
    decay_seed: str | None = None,
) -> dict:
    structure = parse_structure(nbt_path)
    return place_premade(editor, structure, anchor, tier=tier, rotation=rotation,
                         place_air=place_air, decay=decay, decay_seed=decay_seed)


# ---------------------------------------------------------------------------
# Foundation / leveling pass
# ---------------------------------------------------------------------------
# Premades are authored flat-bottomed with NO terrain (CLAUDE.md authoring
# contract): the floor is the build's lowest layer, and everything below grade
# is generated here at place-time. We pick one flat pad level for the whole
# footprint (median of the ground under it), clear any terrain above it, and
# fill a foundation skirt down to grade so edges on a slope don't float. This is
# what makes the build "match the surrounding area" vertically; the yard pass
# (later) blends the horizontal seam.


def _aggregate(values, mode: str) -> int:
    vals = sorted(int(v) for v in values)
    if not vals:
        return 0
    if mode == "min":
        return vals[0]
    if mode == "max":
        return vals[-1]
    return vals[len(vals) // 2]  # median (robust to a few outlier columns)


def _rotation_offset(rotation: int, sx: int, sz: int) -> tuple[int, int]:
    r = rotation % 4
    if r == 0:
        return (0, 0)
    if r == 1:
        return (sz - 1, 0)
    if r == 2:
        return (sx - 1, sz - 1)
    return (0, sx - 1)  # r == 3


def heightmap_ground_y(heightmap, origin):
    ox, oz = int(origin[0]), int(origin[2])
    depth = len(heightmap)
    width = len(heightmap[0])

    def ground_y(world_x, world_z) -> int:
        lx = max(0, min(width - 1, int(world_x) - ox))
        lz = max(0, min(depth - 1, int(world_z) - oz))
        return int(heightmap[lz][lx])

    return ground_y


def level_pad(
    editor,
    anchor_xz: tuple[int, int],
    footprint: tuple[int, int],
    floor_y: int,
    build_height: int,
    ground_y,
    foundation_block: str,
    surface_block: str = "minecraft:grass_block",
    clearance: int = 4,
) -> dict:
    from gdpc import Block  # lazy

    ax, az = anchor_xz
    fw, fd = footprint
    air = Block("minecraft:air")
    found = Block(foundation_block)
    cap = Block(surface_block)
    ceil = floor_y + build_height + clearance
    stats = {"cleared": 0, "filled": 0, "capped": 0}

    for dx in range(fw):
        for dz in range(fd):
            wx, wz = ax + dx, az + dz
            for y in range(floor_y + 1, ceil + 1):   # clear ABOVE the floor only
                editor.placeBlock((wx, y, wz), air)
                stats["cleared"] += 1
            grade = ground_y(wx, wz)
            for y in range(grade, floor_y):          # foundation skirt (empty if grade >= floor_y)
                editor.placeBlock((wx, y, wz), found)
                stats["filled"] += 1
            editor.placeBlock((wx, floor_y, wz), cap)  # ground cap at the floor level
            stats["capped"] += 1

    return stats


def build_premade(
    editor,
    structure: Structure,
    anchor_xz: tuple[int, int],
    ground_y,
    tier: str = "strained",
    rotation: int = 0,
    sink: int = 0,
    foundation_block: str = "minecraft:cobblestone",
    surface_block: str = "minecraft:grass_block",
    base_y_mode: str = "median",
    clearance: int = 4,
    decay: bool = True,
    decay_seed: str | None = None,
    chest_payload: str | None = None,
) -> dict:
    sx, sy, sz = structure.size
    fw, fd = (sz, sx) if rotation % 2 == 1 else (sx, sz)
    ax, az = anchor_xz

    samples = [ground_y(ax + dx, az + dz) for dx in range(fw) for dz in range(fd)]
    base_y = _aggregate(samples, base_y_mode)
    floor_y = base_y - sink

    # Foundation matches the build's mood (struggling -> mossy skirt, etc.).
    found_block = families.resolve_block(foundation_block, tier) or foundation_block
    pad_stats = level_pad(editor, (ax, az), (fw, fd), floor_y, sy, ground_y,
                          found_block, surface_block=surface_block, clearance=clearance)

    chest_payloads = None
    if chest_payload is not None:
        cpos = chest_local_pos(structure)
        if cpos is not None:
            chest_payloads = {cpos: chest_payload}

    odx, odz = _rotation_offset(rotation, sx, sz)
    place_stats = place_premade(editor, structure, (ax + odx, floor_y, az + odz),
                                tier=tier, rotation=rotation,
                                decay=decay, decay_seed=decay_seed,
                                chest_payloads=chest_payloads)

    return {**place_stats, **pad_stats, "base_y": base_y, "floor_y": floor_y, "tier": tier}


# ---------------------------------------------------------------------------
# Dry-run inspector: parse + apply the swap for all three tiers WITHOUT gdpc.
# Verifies the families table against a real build (counts + every swap), so
# you can confirm coverage before ever connecting to a world.
# ---------------------------------------------------------------------------

def dry_run(nbt_path: str | Path) -> None:
    structure = parse_structure(nbt_path)
    names = structure.distinct_names()
    path = Path(nbt_path)

    print("=" * 72)
    print(f"{path.name}   size={structure.size}   {structure.block_count} blocks   "
          f"{len(names)} distinct")
    print("=" * 72)
    print(f"{'block':<40}{'thriving':<10}{'strained':<10}struggling")
    print("-" * 72)

    def cell(name: str, tier: str) -> str:
        r = families.resolve_block(name, tier)
        if r is None:
            return "DROP"
        if families.is_liquid(r):
            return "~src"           # placed as a liquid source
        short = r.split(":", 1)[-1]
        return short if r != name else "·"   # · = unchanged from baseline

    for name in names:
        short = name.split(":", 1)[-1]
        fixed = " (fixed)" if families.is_fixed_functional(name) else ""
        print(f"{short + fixed:<40}"
              f"{cell(name,'thriving'):<10}{cell(name,'strained'):<10}{cell(name,'struggling')}")

    print("-" * 72)
    for tier in TIERS:
        placed = dropped = liquids = air = 0
        for b in structure.blocks:
            r = families.resolve_block(b.name, tier)
            if r is None:
                dropped += 1
            elif r in _AIR:
                air += 1                       # skipped at place-time (place_air=False)
            elif families.is_liquid(r):
                liquids += 1
            else:
                placed += 1
        print(f"{tier:<10} place={placed:<5} drop={dropped:<4} "
              f"liquid-source={liquids:<4} air-skipped={air}")
    print("legend: · = unchanged | DROP = removed (air) | ~src = liquid source | "
          "names = swapped block\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dry-run the mood palette swap on premade .nbt files "
                    "(no gdpc / no Minecraft needed)."
    )
    parser.add_argument(
        "target", nargs="?", default="nbt",
        help="A .nbt file or a directory of them (default: ./nbt).",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not target.is_absolute():
        target = _HERE / target
    files = sorted(target.glob("*.nbt")) if target.is_dir() else [target]

    if not files:
        print("No .nbt files found.")
    for f in files:
        dry_run(f)
