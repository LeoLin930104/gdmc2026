from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import families

_AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")

# Never punch these out — removing half of a multi-block / functional piece
# reads as a bug, not as decay. (Fixed-functional blocks are excluded via
# families.is_fixed_functional; this covers the structural multi-blocks.)
_PROTECTED_SUBSTRINGS = ("door", "bed", "ladder")

COBWEB = "minecraft:cobweb"

# Defaults — "minimal" by request.
MIN_MISSING = 3
MAX_MISSING = 5
MIN_COBWEBS = 2
MAX_COBWEBS = 4

_NEIGHBORS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@dataclass
class DecayPlan:
    """Place-time decay for one struggling build, in structure-local coords.

    `remove`  : positions whose block should be skipped (left as the cleared air
                the foundation pass already carved) -> a visible hole.
    `cobwebs` : air positions to fill with a cobweb.
    """

    remove: set[tuple[int, int, int]]
    cobwebs: set[tuple[int, int, int]]


def _seed_rng(seed: str) -> random.Random:
    """Stable RNG from an arbitrary string (hashlib, not built-in hash())."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest, 16))


def _is_removable(name: str) -> bool:
    if name in _AIR:
        return False
    if families.is_fixed_functional(name):
        return False
    if families.is_liquid(name):
        return False
    return not any(tok in name for tok in _PROTECTED_SUBSTRINGS)


def plan_decay(
    structure,
    seed: str,
    *,
    min_missing: int = MIN_MISSING,
    max_missing: int = MAX_MISSING,
    min_cobwebs: int = MIN_COBWEBS,
    max_cobwebs: int = MAX_COBWEBS,
) -> DecayPlan:
    """Plan the struggling-tier decay for one parsed `structure`.

    Removal candidates are EXPOSED (a face open to air/outside), ABOVE the floor
    layer, and non-functional, so each hole is visible and the build keeps
    standing. Cobwebs prefer air cells tucked into corners (>=3 solid neighbours
    after removal), relaxing the threshold only if too few corners exist; the
    fresh holes are seeded in as web candidates too. Returns empty sets if the
    structure has nothing safe to touch (warn-free; never raises).
    """
    rng = _seed_rng(seed)

    solid: set[tuple[int, int, int]] = {b.pos for b in structure.blocks if b.name not in _AIR}
    air: set[tuple[int, int, int]] = {b.pos for b in structure.blocks if b.name in _AIR}
    if not solid:
        return DecayPlan(remove=set(), cobwebs=set())

    floor_y = min(p[1] for p in solid)

    # Exposed = has at least one neighbour that is NOT solid (open face / shell).
    candidates = [
        b.pos
        for b in structure.blocks
        if _is_removable(b.name)
        and b.pos[1] > floor_y
        and any((b.pos[0] + dx, b.pos[1] + dy, b.pos[2] + dz) not in solid
                for dx, dy, dz in _NEIGHBORS6)
    ]

    remove: set[tuple[int, int, int]] = set()
    if candidates:
        n = rng.randint(min_missing, max_missing)
        remove = set(rng.sample(candidates, min(n, len(candidates))))

    # Corners are scored against the post-removal solids so a fresh hole reads
    # as a webbed gap, and removed cells themselves become web candidates.
    solid_after = solid - remove
    web_candidates = (air | remove)

    def solid_neighbours(pos: tuple[int, int, int]) -> int:
        x, y, z = pos
        return sum((x + dx, y + dy, z + dz) in solid_after for dx, dy, dz in _NEIGHBORS6)

    cobwebs: set[tuple[int, int, int]] = set()
    if web_candidates:
        want = rng.randint(min_cobwebs, max_cobwebs)
        for threshold in (3, 2, 1):                 # corners first, then relax
            pool = [p for p in web_candidates if p not in cobwebs and solid_neighbours(p) >= threshold]
            rng.shuffle(pool)
            for p in pool:
                if len(cobwebs) >= want:
                    break
                cobwebs.add(p)
            if len(cobwebs) >= want:
                break

    return DecayPlan(remove=remove, cobwebs=cobwebs)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from nbt_structure import parse_structure

    ap = argparse.ArgumentParser(
        description="Preview the struggling-tier decay plan for premade .nbt files "
                    "(no gdpc / no Minecraft needed)."
    )
    ap.add_argument("target", nargs="?", default="nbt",
                    help="A .nbt file or a directory of them (default: ./nbt).")
    ap.add_argument("--seed", default="preview", help="Decay seed string.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    target = Path(args.target)
    if not target.is_absolute():
        target = here / target
    files = sorted(target.glob("*.nbt")) if target.is_dir() else [target]

    for f in files:
        s = parse_structure(f)
        plan = plan_decay(s, seed=f"{args.seed}:{f.name}")
        print(f"{f.name:<22} size={s.size}  -> remove {len(plan.remove)} block(s), "
              f"{len(plan.cobwebs)} cobweb(s)")
        for p in sorted(plan.remove):
            print(f"    hole   {p}")
        for p in sorted(plan.cobwebs):
            print(f"    cobweb {p}")
