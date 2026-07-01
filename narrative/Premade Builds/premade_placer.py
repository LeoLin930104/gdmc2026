from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import families
from nbt_structure import Structure, StructureBlock, parse_structure

_AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")
SERVER_COMPAT_BLOCKS: dict[str, tuple[str, dict[str, str]]] = {
    # Some GDMC server versions reject the post-1.16 chain block. Iron bars
    # preserve the metal support read and exist in older targets. Chain's axis
    # property is not valid for iron bars, so properties are deliberately
    # cleared.
    "minecraft:chain": ("minecraft:iron_bars", {}),
}

# Mood tiers the lore->tier resolver can pick; "strained" is the authored baseline.
TIERS = families.TIERS


def chest_local_pos(structure: Structure) -> tuple[int, int, int] | None:
    """Local position of a fillable container in `structure`, or None.

    Prefers a `chest` over a `barrel` (both store an `Items` list the same way).
    Used by the narrative layer to drop a district's diary + tool into the build
    that already stands at its center, instead of spawning a separate chest.
    """
    chest = barrel = None
    for b in structure.blocks:
        short = b.name.split(":")[-1]
        if "chest" in short and chest is None:
            chest = b.pos
        elif "barrel" in short and barrel is None:
            barrel = b.pos
    return chest if chest is not None else barrel


def mood_tier_for(settlement) -> str:
    """Read the settlement's mood tier (one of `TIERS`), defaulting to baseline.

    The tier is decided by the `mood_tier.generate_mood_tier()` pre-pass in
    LLM Narrative (from settlement.goal + historical_wound / collective_fear,
    etc.) and stored on `settlement.mood_tier`. This reader keeps the placer
    decoupled from the LLM layer — it only consumes the attribute. Falls back to
    "strained" (the authored baseline) when the pre-pass wasn't run or left an
    unrecognized value, so a hand-built Settlement still places correctly.
    """
    tier = getattr(settlement, "mood_tier", None)
    if isinstance(tier, str) and tier in TIERS:
        return tier
    return "strained"


def _server_compatible_block(
    block_id: str,
    properties: dict[str, str],
) -> tuple[str, dict[str, str], bool]:
    replacement = SERVER_COMPAT_BLOCKS.get(block_id)
    if replacement is None:
        return block_id, dict(properties), False
    replacement_id, replacement_props = replacement
    return replacement_id, dict(replacement_props), True


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
    """Place a parsed `structure` at world `anchor`, swapping blocks for `tier`.

    `anchor` is the world position the structure's local (0,0,0) maps to.
    `rotation` is in 90° steps about the Y axis (gdpc convention). NOTE: rotation
    rotates about the anchor, so for rotation != 0 the caller must offset `anchor`
    to keep the footprint where intended — that footprint/road-edge anchoring is
    a later step; rotation=0 places exactly at `anchor`.

    `decay` (struggling tier only): on top of the families material swap, knock a
    few exposed blocks out of the walls/roof and string cobwebs in the corners
    (decay.plan_decay). Seeded by `decay_seed` (defaults to the anchor) so a
    settlement decays each build the same way. No-op at thriving/strained.

    `chest_payloads`: {local_pos: block-entity SNBT} — overrides the `data=` of
    the block at each local position (used to fill a build's existing chest with
    narrative items). The block keeps its id + properties (so a chest stays a
    chest, facing intact); only its contents change. Placement happens under the
    gdpc Transform, so the caller supplies LOCAL positions and gdpc maps them.

    Returns a stats dict: {placed, dropped, liquids, skipped_air, removed,
    cobwebs, chests_filled}.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    from gdpc import Block, Transform  # lazy: keeps the module gdpc-free to import

    stats = {"placed": 0, "dropped": 0, "liquids": 0, "liquid_flow_skipped": 0,
             "skipped_air": 0, "removed": 0, "cobwebs": 0, "chests_filled": 0,
             "compat_remapped": 0}

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
            resolved, properties, remapped = _server_compatible_block(
                resolved,
                sb.properties,
            )
            if remapped:
                stats["compat_remapped"] += 1

            if families.is_liquid(resolved):
                # Place only authored SOURCE cells (level 0 / absent); skip the
                # flowing cells (level 1-8) and let Minecraft re-flow them from
                # the source. Placing every cell as a source overfills/overflows
                # (e.g. a fountain) — one authored spout source fills it naturally.
                if properties.get("level", "0") != "0":
                    stats["liquid_flow_skipped"] += 1
                    continue
                block = Block(resolved)
                stats["liquids"] += 1
            else:
                data = sb.nbt
                if chest_payloads is not None and sb.pos in chest_payloads:
                    data = chest_payloads[sb.pos]   # fill the build's chest
                    stats["chests_filled"] += 1
                block = Block(resolved, properties, data=data)

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
    """Convenience: parse a `.nbt` then place it. See place_premade for args."""
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
    """Reduce a list of ground heights to one pad level (median/min/max)."""
    vals = sorted(int(v) for v in values)
    if not vals:
        return 0
    if mode == "min":
        return vals[0]
    if mode == "max":
        return vals[-1]
    return vals[len(vals) // 2]  # median (robust to a few outlier columns)


def _rotation_offset(rotation: int, sx: int, sz: int) -> tuple[int, int]:
    """Translation (dx, dz) that keeps a rotated structure's footprint anchored
    at the same corner, so the build lands on the pad we carved.

    Assumes gdpc's rotation is clockwise about +Y (Minecraft CLOCKWISE_90),
    i.e. local (x,z) -> (-z, x) for one step. Footprint dims swap on odd steps.
    NOTE: handedness is unverified in-world — if a rotated build lands mirrored,
    swap the r==1 and r==3 cases.
    """
    r = rotation % 4
    if r == 0:
        return (0, 0)
    if r == 1:
        return (sz - 1, 0)
    if r == 2:
        return (sx - 1, sz - 1)
    return (0, sx - 1)  # r == 3


def heightmap_ground_y(heightmap, origin):
    """Build a ground_y(world_x, world_z) -> int callable from a gdmc2026 npz.

    `heightmap` is the generator's [z, x] surface array (top SOLID block y);
    `origin` is [ox, _, oz]. The generator lays farm-cell grass exactly at this
    height, so it's the right grade truth. Out-of-range columns clamp to the
    edge. No numpy import needed — plain indexing works on the passed array.
    """
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
    """Carve a flat pad and fill its foundation under one build's footprint.

    `anchor_xz` is the world (x, z) of the footprint's min corner; `footprint`
    is (fw, fd) AFTER any rotation. For each column:
      - clear from `floor_y + 1` up through the build volume + `clearance`
        (removes terrain bumps and leaves the build's interior air empty) — but
        NOT the floor level itself;
      - fill `foundation_block` from grade up to just below `floor_y` (the skirt
        that keeps slope edges from floating; empty on flat ground);
      - cap the floor level with `surface_block` so footprint cells the build
        doesn't fill read as ground, not air pits.
    The build is placed afterward and overwrites the cap wherever it has a floor;
    gaps keep the cap. This is what fixes the "ground turned to air / build
    floats" artifact: we never leave the floor level as air.
    """
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
    """Full place-time pipeline for one premade: level a pad, then place it.

    Steps: sample grade over the (rotation-aware) footprint -> pick one pad
    level -> floor_y = pad - `sink` (use sink=1 for builds whose floor sits one
    block below grade) -> carve + foundation + ground cap (`surface_block`) ->
    place the build with the rotation anchor offset so it lands on the pad. The
    build overwrites the cap where it has a floor; gaps stay ground. Returns
    merged place + pad stats plus base_y/floor_y.

    `decay`/`decay_seed` forward to place_premade (struggling-tier holes +
    cobwebs); see place_premade. The default seed is the build's world anchor.

    `chest_payload` (optional): block-entity SNBT to drop into THIS build's chest
    (located via chest_local_pos). No-op if the structure has no chest.
    """
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
