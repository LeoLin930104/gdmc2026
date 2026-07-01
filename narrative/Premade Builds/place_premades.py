from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "LLM Narrative"))
sys.path.insert(0, str(_HERE.parent / "Item Relic Generator"))  # diary/tool chest builders

import numpy as np

from district_roles import assign_district_roles
from nbt_structure import Structure, parse_structure
from premade_placer import (
    TIERS, build_premade, chest_local_pos, heightmap_ground_y, mood_tier_for,
)
from yard import place_yard

# role -> premade file prefix in ./nbt
ROLE_PREFIX = {
    "town_square": "town_center",
    "residential": "residential",
    "barracks": "barrack",
}
# role -> foundation skirt block (mood-swapped at place-time); default cobblestone
ROLE_FOUNDATION = {"barracks": "minecraft:deepslate_bricks"}
DEFAULT_FOUNDATION = "minecraft:cobblestone"

# role -> (display name, zone preset) used to prompt the diary/tool generators.
# preset steers the LLM's item choice (barracks -> martial gear, farm -> tools).
ROLE_NARRATIVE = {
    "town_square": ("Town Square", "town"),
    "residential": ("Residential Quarter", "town"),
    "barracks":    ("Barracks", "dungeon"),
    "farm":        ("Farmstead", "nature"),
}

NBT_DIR = _HERE / "nbt"
SIZES = (11, 7)   # try the larger build first; both are square edges

# In <repo>/narrative/Premade Builds/, so the generator's data/ is two levels up (the repo root).
_GEN_DATA = _HERE.parent.parent / "data"
_DEFAULT_DATA_NPZ = _GEN_DATA / "settlement_data.npz"
_DEFAULT_PLOTS_NPZ = _GEN_DATA / "settlement_plots.npz"


# ---------------------------------------------------------------------------
# npz helpers
# ---------------------------------------------------------------------------

def _items_to_dict(arr) -> dict:
    """Round-trip np.array(list(d.items()), dtype=object) back to a dict."""
    out = {}
    for pair in arr:
        if len(pair) == 2:
            out[pair[0]] = pair[1]
    return out


def largest_rect_with_origin(points) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle covering only `points` (list of (x, z)).

    Returns (x0, z0, w, d) in the SAME local coords as the points — like
    analyze_plots._largest_rect_from_points but also reporting the rectangle's
    min corner so we can anchor a build inside it. (0,0,0,0) if empty.
    Max-rectangle-in-histogram over the cell's boolean mask.
    """
    pts = [(int(x), int(z)) for x, z in points]
    if not pts:
        return (0, 0, 0, 0)
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    W = max_x - min_x + 1
    D = max_z - min_z + 1
    mask = np.zeros((D, W), dtype=bool)
    for x, z in pts:
        mask[z - min_z, x - min_x] = True

    heights = np.zeros(W, dtype=int)
    best = (0, 0, 0, 0)
    best_area = 0
    for z in range(D):
        heights = np.where(mask[z], heights + 1, 0)
        stack: list[int] = []
        for x in range(W + 1):
            cur = heights[x] if x < W else 0
            while stack and heights[stack[-1]] > cur:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                w = x - left
                if w * h > best_area:
                    best_area = w * h
                    best = (min_x + left, min_z + (z - h + 1), w, h)
            stack.append(x)
    return best


def _zone_of_cell(cells, zone_map) -> int | None:
    """The district id a farm cell belongs to (mode of zone_map over its cells)."""
    D, W = zone_map.shape
    counts: Counter[int] = Counter()
    for x, z in cells:
        x, z = int(x), int(z)
        if 0 <= z < D and 0 <= x < W:
            zid = int(zone_map[z, x])
            if zid >= 0:
                counts[zid] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# build selection
# ---------------------------------------------------------------------------

def _variants_for(role: str, size: int, cache: dict) -> list[Structure]:
    """All authored builds for (role, size): base + numbered variants, parsed.

    The base file is `{prefix}_{size}.nbt`; alternates carry a digit before the
    `_{size}` (e.g. `barrack2_7.nbt`, `town_center2_7.nbt`) so the placer can vary
    appearance and avoid repetitive-looking builds. Returns an empty list if no
    file exists for the (role, size). Parsed structures are cached per key.
    """
    key = (role, size)
    if key not in cache:
        prefix = ROLE_PREFIX[role]
        paths = []
        base = NBT_DIR / f"{prefix}_{size}.nbt"
        if base.exists():
            paths.append(base)
        # numbered alternates: prefix2_size, prefix3_size, ... (sorted for stability)
        paths.extend(sorted(NBT_DIR.glob(f"{prefix}[0-9]*_{size}.nbt")))
        cache[key] = [parse_structure(p) for p in paths]
    return cache[key]


def _pick_variant(structs: list[Structure], seed_name: str, cell_key) -> Structure:
    """Deterministically choose one variant for a build, seeded by (settlement, cell).

    Independent of `_rotation_for` (separate seed namespace) so variant and
    rotation vary independently while staying reproducible for a given settlement.
    """
    if len(structs) == 1:
        return structs[0]
    digest = hashlib.sha256(f"{seed_name}:variant:{cell_key}".encode("utf-8")).hexdigest()
    return structs[int(digest, 16) % len(structs)]


def _pick_size(fit_square: int, role: str, cache: dict) -> int | None:
    """Largest authored size that fits the cell's fit-square, or None if too small."""
    for size in SIZES:                       # 11 then 7
        if fit_square >= size and _variants_for(role, size, cache):
            return size
    return None


# ---------------------------------------------------------------------------
# settlement identity (graceful without LM Studio)
# ---------------------------------------------------------------------------

def _make_settlement(theme: str, biome: str | None):
    """Generate the shared Settlement + 3 pre-passes; fall back to neutral identity."""
    try:
        from mood_tier import generate_mood_tier
        from settlement_generator import generate_settlement
        from settlement_goal import generate_settlement_goal
        from shared_events import generate_shared_events

        s = generate_settlement(theme, biome=biome)
        s.goal = generate_settlement_goal(s)          # pre-pass 1
        s.shared_events = generate_shared_events(s)   # pre-pass 2
        s.mood_tier = generate_mood_tier(s)           # pre-pass 3 (last)
        return s
    except Exception as exc:  # noqa: BLE001 - LLM optional; keep geometry working
        from types import SimpleNamespace
        print(f"[warn] settlement generation failed ({exc!r}); "
              f"using neutral identity (mood tier 'strained').")
        return SimpleNamespace(name=theme or "Settlement", mood_tier=None)


def _detect_biome() -> str | None:
    try:
        from gdpc import Editor
        from biome_context import sample_biome_at_player
        biome, _pos = sample_biome_at_player(Editor())
        return biome
    except Exception as exc:  # noqa: BLE001
        print(f"[info] biome detection skipped ({exc!r}).")
        return None


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

CLUSTER_FIT = 15          # cells with a fit-square this big get a 2-build cluster
CLUSTER_AXIS_MIN = 14     # ...only if the rect's longer axis fits two size-7 builds


def _rotation_for(seed_name: str, cell_key) -> int:
    """Deterministic 0..3 rotation per (settlement, cell key) — reproducible, varied.

    Seeded from the settlement name + a cell key (hashlib, not built-in hash; the
    key is the cell id, or "<cell>.<i>" for clustered sub-builds), so a given
    settlement always orients each build the same way while builds vary. All
    premades are square, so rotation never changes the footprint — only the
    facing of oriented blocks (gdpc Transform handles it).
    """
    digest = hashlib.sha256(f"{seed_name}:{cell_key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 4


def _footprint_cells(ax: int, az: int, size: int) -> set:
    return {(ax + dx, az + dz) for dx in range(size) for dz in range(size)}


def _road_dir(fit_rect, path_mask) -> str | None:
    """Which side of the fit-rect borders the most path cells ('n'/'s'/'e'/'w'), or None."""
    if path_mask is None:
        return None
    x0, z0, w, d = fit_rect
    D, W = path_mask.shape

    def count(ring) -> int:
        return sum(1 for x, z in ring
                   if 0 <= z < D and 0 <= x < W and path_mask[z, x])

    sides = {
        "w": count((x0 - 1, z) for z in range(z0, z0 + d)),
        "e": count((x0 + w, z) for z in range(z0, z0 + d)),
        "n": count((x, z0 - 1) for x in range(x0, x0 + w)),
        "s": count((x, z0 + d) for x in range(x0, x0 + w)),
    }
    best = max(sides, key=sides.get)
    return best if sides[best] > 0 else None


def _anchor_to_edge(fit_rect, size: int, road: str | None) -> tuple[int, int]:
    """Anchor a size×size build flush to the road side of the fit-rect (else center)."""
    x0, z0, w, d = fit_rect
    cx = x0 + max(0, (w - size) // 2)
    cz = z0 + max(0, (d - size) // 2)
    if road == "e":
        return (x0 + (w - size), cz)
    if road == "w":
        return (x0, cz)
    if road == "s":
        return (cx, z0 + (d - size))
    if road == "n":
        return (cx, z0)
    return (cx, cz)


def _cluster_anchors(fit_rect, size: int) -> list[tuple[int, int]]:
    """Two size×size anchors at the ends of the fit-rect's longer axis."""
    x0, z0, w, d = fit_rect
    if w >= d:
        cz = z0 + max(0, (d - size) // 2)
        return [(x0, cz), (x0 + w - size, cz)]
    cx = x0 + max(0, (w - size) // 2)
    return [(cx, z0), (cx, z0 + d - size)]


def collect_farm_fields(zone_map, farms, roles) -> list[dict]:
    """Farm cells belonging to the farm-ROLE district — rendered as crop fields.

    The generator no longer lays farm fields (`BUILD_FARM_FIELDS = False`), so the
    narrative layer renders the farm district's cells itself (mood-scaled, via
    `farm_field.place_farm_field`). Non-farm districts' farm cells become premade
    builds instead (see `plan_placements`); those are excluded here. Returns one
    `{cell_id, zone, cells}` per farm-district cell (cells are local (x, z)).
    """
    fields: list[dict] = []
    for cell_id, cells in sorted(farms.items(), key=lambda kv: int(kv[0])):
        zone = _zone_of_cell(cells, zone_map)
        if zone is None or roles.get(zone) != "farm":
            continue
        fields.append({
            "cell_id": int(cell_id),
            "zone": zone,
            "cells": [(int(x), int(z)) for x, z in cells],
        })
    return fields


def plan_placements(zone_map, origin, heightmap, farms, roles, path_mask=None,
                    rotation_seed: str = "", rotation_override: int | None = None):
    """Plan premade build(s) per non-farm farm cell. Returns (placements, skipped).

    Each placement is one cell carrying a list of `builds` (one normally; two for
    a large cell -> cluster) plus the `occupied` footprint set for the yard pass.
    A single build is anchored to the cell's road edge (from `path_mask`) when
    one is found, else centered; clusters sit at the rect's ends. Planning is
    tier-independent — mood is applied only at placement.
    """
    ox, oz = int(origin[0]), int(origin[2])
    cache: dict = {}
    placements: list[dict] = []
    skipped = Counter()

    for cell_id, cells in sorted(farms.items(), key=lambda kv: int(kv[0])):
        zone = _zone_of_cell(cells, zone_map)
        if zone is None or zone not in roles:
            skipped["no_zone"] += 1
            continue
        role = roles[zone]
        if role == "farm":
            skipped["farm_district"] += 1
            continue

        fit_rect = largest_rect_with_origin(cells)
        x0, z0, w, d = fit_rect
        fit = min(w, d)

        builds: list[dict] = []
        if (fit >= CLUSTER_FIT and max(w, d) >= CLUSTER_AXIS_MIN
                and _variants_for(role, 7, cache)):
            # Large cell -> two small builds at the rect's ends. Each sub-build
            # picks its own variant (keyed by "<cell>.<i>") so a cluster can mix.
            for i, (ax, az) in enumerate(_cluster_anchors(fit_rect, 7)):
                struct = _pick_variant(_variants_for(role, 7, cache),
                                       rotation_seed, f"{cell_id}.{i}")
                rot = rotation_override if rotation_override is not None else \
                    _rotation_for(rotation_seed, f"{cell_id}.{i}")
                builds.append({"anchor_local": (ax, az), "size": 7,
                               "structure": struct, "rotation": rot})
        else:
            size = _pick_size(fit, role, cache)
            if size is None:
                skipped["too_small"] += 1
                continue
            struct = _pick_variant(_variants_for(role, size, cache),
                                   rotation_seed, cell_id)
            ax, az = _anchor_to_edge(fit_rect, size, _road_dir(fit_rect, path_mask))
            rot = rotation_override if rotation_override is not None else \
                _rotation_for(rotation_seed, cell_id)
            builds.append({"anchor_local": (ax, az), "size": size,
                           "structure": struct, "rotation": rot})

        occupied: set = set()
        for b in builds:
            ax, az = b["anchor_local"]
            occupied |= _footprint_cells(ax, az, b["size"])
            b["anchor_world"] = (ox + ax, oz + az)

        placements.append({
            "cell_id": int(cell_id),
            "zone": zone,
            "role": role,
            "fit": fit,
            "builds": builds,
            "occupied": occupied,
            "cells": [(int(x), int(z)) for x, z in cells],   # local; for the yard pass
        })

    return placements, skipped


# ---------------------------------------------------------------------------
# narrative items: one diary + tool per district, dropped into a build's chest
# ---------------------------------------------------------------------------

def _narrative_chest_snbt(diary, tool, relic=None) -> str | None:
    """Chest block-entity SNBT holding a diary (book) + a tool + a relic, or None.

    Slots are assigned in order of what's present, so a district with only some
    of the three still gets a valid chest. Reuses the diary book-item builder and
    the relic/tool item builder so rendering matches the standalone diary/tool/
    relic chests elsewhere. The relic is a dict (relics.json schema) and shimmers
    by default, matching the standalone relic chest.
    """
    from place_diary_lectern import build_book_item_nbt, glint_for_rarity
    from place_relic_chest import build_item_nbt

    items: list[str] = []
    if diary is not None:
        items.append(build_book_item_nbt(diary, slot=len(items)))
    if tool is not None:
        tool_dict = {
            "name": tool.name, "item_type": tool.item_type,
            "description": tool.description, "lore": tool.lore, "color": tool.color,
        }
        items.append(build_item_nbt(tool_dict, slot=len(items),
                                    glint=glint_for_rarity(tool.rarity)))
    if relic is not None:
        # generate_relics validates name + item_type; other fields may be absent,
        # so default them the same way place_relic_chest.validate_relic does.
        relic_dict = {
            "name": relic.get("name", "Relic"),
            "item_type": relic.get("item_type", ""),
            "description": relic.get("description", ""),
            "lore": relic.get("lore", ""),
            "color": relic.get("color", "white"),
        }
        items.append(build_item_nbt(relic_dict, slot=len(items), glint=True))
    if not items:
        return None
    return "{Items:[" + ",".join(items) + "]}"


def _spread_indices(total: int, n: int) -> list[int]:
    """`n` indices spread evenly across range(total), endpoints included.

    Picks builds that are far apart (nearest + middle + farthest from center)
    rather than the `n` closest, so a district's items scatter across it instead
    of clustering. `total >= n >= 1` in the else branch guarantees distinct,
    increasing indices.
    """
    if n <= 1:
        return [0]
    if n >= total:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _plot_anchor(cells) -> tuple[int, int]:
    """The actual cell nearest a farm plot's centroid — where its chest sits."""
    xs = [int(x) for x, _ in cells]
    zs = [int(z) for _, z in cells]
    cx = sum(xs) / len(xs)
    cz = sum(zs) / len(zs)
    bx, bz = min(((int(x), int(z)) for x, z in cells),
                 key=lambda c: (c[0] - cx) ** 2 + (c[1] - cz) ** 2)
    return bx, bz


def plan_narrative_items(settlement, biome, placements, roles, zone_seed_points,
                         origin, heightmap, fields=None):
    """Generate one diary + tool + relic per district and route them into chests.

    A district whose premade builds carry chests (every non-farm district) gets
    its three items combined into the ONE build chest nearest the district center
    (`zone_seed_points`) — the original, working behavior.

    The farm district has no premade builds (so no build chests); it instead gets
    ONE chest per FARM PLOT, the items spread across the plots (from `fields`,
    chosen via `_spread_indices`), each chest sitting on a plot's centroid
    (`_plot_anchor`) on top of the rendered field. With no plots known
    (e.g. --no-farm-fields) it falls back to one combined chest at the center.

    The relics are generated as one settlement-wide collection (so the set coheres
    as a whole) and then spread ONE-per-district rather than concentrated in a
    single chest. Relic generation has its own warn-and-recover so a relic failure
    still leaves each chest with its diary + tool.

    Returns `(fallback_chests, summary)` where fallback_chests is a list of
    `(world_pos, payload, zone_id)`. Warn-and-recover: any LLM failure logs and
    returns `([], {...})` so the geometry still places.
    """
    from diary_generator import generate_diaries
    from tool_generator import generate_tools

    zone_specs = [
        (f"district_{zid}", *ROLE_NARRATIVE.get(role, (role.title(), "town")))
        for zid, role in sorted(roles.items())
    ]
    try:
        diaries = generate_diaries(settlement=settlement, zone_specs=zone_specs, biome=biome)
        tools = generate_tools(settlement=settlement, zone_specs=zone_specs, biome=biome)
    except Exception as exc:  # noqa: BLE001 - items are optional; keep geometry working
        print(f"[warn] narrative item generation failed ({exc!r}); "
              f"placing builds without items.")
        return [], {"in_build": 0, "fallback": 0, "skipped": 0, "relics": 0}

    # Relics: generate one per district as a single coherent set, then spread them
    # across districts. Separate try so a relic failure keeps diaries + tools.
    relic_by_zone: dict = {}
    try:
        from relic_generator import generate_relics
        relic_theme = getattr(settlement, "theme", None) or settlement.name
        relics = generate_relics(relic_theme, count=len(zone_specs),
                                 settlement=settlement, biome=biome)
        # zone_specs is in sorted(roles) order, so zip assigns one relic per
        # district in that order; if generate_relics returns fewer, the trailing
        # districts simply get no relic.
        for zid, relic in zip(sorted(roles), relics):
            relic_by_zone[f"district_{zid}"] = relic
    except Exception as exc:  # noqa: BLE001 - relics optional; keep diary + tool
        print(f"[warn] relic generation failed ({exc!r}); "
              f"district chests get diary + tool only.")

    diary_by_zone = {d.zone_id: d for d in diaries}
    tool_by_zone = {t.zone_id: t for t in tools}

    # Farm plots per district (for districts with no chest-bearing build), so the
    # farm district's items can sit ONE PER PLOT instead of floating at the center.
    plots_by_zone: dict[int, list] = {}
    for f in (fields or []):
        plots_by_zone.setdefault(int(f["zone"]), []).append(f)

    ox, oz = int(origin[0]), int(origin[2])
    ground_y = heightmap_ground_y(heightmap, origin)
    seeds = [(float(p[0]), float(p[1])) for p in zone_seed_points]

    fallback_chests: list[tuple] = []
    summary = {"in_build": 0, "fallback": 0, "skipped": 0,
               "relics": len(relic_by_zone)}

    for zid in sorted(roles):
        zone_id = f"district_{zid}"
        diary = diary_by_zone.get(zone_id)
        tool = tool_by_zone.get(zone_id)
        relic = relic_by_zone.get(zone_id)
        if diary is None and tool is None and relic is None:
            summary["skipped"] += 1
            continue

        cx, cz = seeds[zid] if zid < len(seeds) else (0.0, 0.0)
        center = (ox + cx, oz + cz)

        # Builds in this district that own a chest and aren't already assigned,
        # sorted by distance to the district center.
        candidates = []
        for p in placements:
            if p["zone"] != zid:
                continue
            for b in p["builds"]:
                if "chest_payload" not in b and chest_local_pos(b["structure"]) is not None:
                    ax, az = b["anchor_world"]
                    dist = (ax - center[0]) ** 2 + (az - center[1]) ** 2
                    candidates.append((dist, b))
        candidates.sort(key=lambda t: t[0])

        if candidates:
            # Non-farm district: its builds already carry chests — drop the diary
            # + tool + relic into the ONE build nearest the district center.
            candidates[0][1]["chest_payload"] = _narrative_chest_snbt(diary, tool, relic)
            summary["in_build"] += 1
        elif plots_by_zone.get(zid):
            # Farm district (no premade builds, so no build chests): ONE chest per
            # farm plot, the items spread across the plots — each chest sits on a
            # plot's centroid, on top of the just-rendered field.
            items = [(k, o) for k, o in (("diary", diary), ("tool", tool), ("relic", relic))
                     if o is not None]
            plots = plots_by_zone[zid]
            n_groups = min(len(plots), len(items))
            groups: list[dict] = [{} for _ in range(n_groups)]
            for i, (kind, obj) in enumerate(items):
                groups[i % n_groups][kind] = obj
            chosen_plots = [plots[i] for i in _spread_indices(len(plots), n_groups)]
            for plot, group in zip(chosen_plots, groups):
                payload = _narrative_chest_snbt(
                    group.get("diary"), group.get("tool"), group.get("relic"))
                px, pz = _plot_anchor(plot["cells"])
                wx, wz = ox + px, oz + pz
                fallback_chests.append(((wx, ground_y(wx, wz) + 1, wz), payload, zone_id))
                summary["fallback"] += 1
        else:
            # No builds and no farm plots (e.g. --no-farm-fields): one combined
            # chest at the district center.
            wx, wz = int(ox + cx), int(oz + cz)
            fallback_chests.append(((wx, ground_y(wx, wz) + 1, wz),
                                    _narrative_chest_snbt(diary, tool, relic), zone_id))
            summary["fallback"] += 1

    return fallback_chests, summary


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def main(
    theme: str = "Fantasy",
    npz: str | None = None,
    plots: str | None = None,
    dry_run: bool = False,
    max_builds: int | None = None,
    biome_override: str | None = None,
    tier_override: str | None = None,
    rotation_override: int | None = None,
    decay: bool = True,
    settlement=None,
    place_items: bool = True,
    farm_fields: bool = True,
) -> None:
    """Place premade builds on the non-farm districts' farm cells.

    `settlement` (optional): reuse a pre-built Settlement (with its mood_tier
    pre-pass already run) instead of generating one here, so an orchestrator can
    thread ONE shared identity through every narrative feature. When None, a
    Settlement + the 3 pre-passes are generated here (graceful without LM Studio).

    `place_items`: also generate one diary + tool + relic per district and drop
    them into the chest of the build nearest each district center (standalone
    chest at the center as a fallback where no build has a chest). The relics are
    a single coherent set spread one-per-district. Skipped on `--dry-run`.

    `farm_fields`: also render the farm-ROLE district's cells as mood-scaled crop
    fields (the generator no longer places them — BUILD_FARM_FIELDS=False). Skip
    with --no-farm-fields to leave that district as bare ground.
    """
    data_path = Path(npz) if npz else _DEFAULT_DATA_NPZ
    plots_path = Path(plots) if plots else _DEFAULT_PLOTS_NPZ
    for label, p in (("settlement_data.npz", data_path), ("settlement_plots.npz", plots_path)):
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find {label} at {p}. Run the generator first:\n"
                "  python main.py   (from the repo root)"
            )

    data = np.load(data_path, allow_pickle=True)
    for key in ("zone_map", "origin", "heightmap", "zone_seed_points"):
        if key not in data.files:
            raise KeyError(f"settlement_data.npz missing {key!r}; rerun generate_zones().")
    zone_map = data["zone_map"]
    origin = data["origin"]
    heightmap = data["heightmap"]
    zone_seed_points = data["zone_seed_points"]
    path_mask = data["path_mask"] if "path_mask" in data.files else None

    plots_data = np.load(plots_path, allow_pickle=True)
    farms = _items_to_dict(plots_data["farms"]) if "farms" in plots_data.files else {}
    if not farms:
        print("[warn] no farm cells in settlement_plots.npz — nothing to place.")
        return

    if settlement is None:
        biome = biome_override or _detect_biome()
        settlement = _make_settlement(theme, biome)
    roles = assign_district_roles(zone_seed_points, settlement.name)

    tier = tier_override or mood_tier_for(settlement)
    forced = " (forced)" if tier_override else ""
    print(f"Settlement: {settlement.name}  |  mood tier: {tier}{forced}")
    print("District roles:")
    for zid in sorted(roles):
        print(f"  zone {zid}: {roles[zid]}")

    placements, skipped = plan_placements(
        zone_map, origin, heightmap, farms, roles, path_mask=path_mask,
        rotation_seed=str(settlement.name), rotation_override=rotation_override,
    )

    fields = collect_farm_fields(zone_map, farms, roles) if farm_fields else []

    n_builds = sum(len(p["builds"]) for p in placements)
    by_role = Counter()
    for p in placements:
        by_role[p["role"]] += len(p["builds"])
    n_clusters = sum(1 for p in placements if len(p["builds"]) > 1)
    print(f"\nPlanned {n_builds} build(s) in {len(placements)} cell(s) "
          f"({', '.join(f'{r}:{n}' for r, n in by_role.items()) or 'none'}"
          f"{f'; {n_clusters} clustered' if n_clusters else ''}); "
          f"skipped {dict(skipped)} of {len(farms)} farm cells.")
    if fields:
        n_field_cells = sum(len(f["cells"]) for f in fields)
        print(f"Planned crop fields for {len(fields)} farm-district cell(s) "
              f"({n_field_cells} field columns) at mood tier '{tier}'.")
    elif farm_fields:
        print("No farm-district cells to render as crop fields.")

    if dry_run:
        show_decay = decay and tier == "struggling"
        if show_decay:
            from decay import plan_decay
        print(f"\n[dry-run] placements (mood tier '{tier}'"
              f"{', decay on' if show_decay else ', no decay'}):")
        for p in placements[: max_builds or None]:
            tag = "cluster" if len(p["builds"]) > 1 else "single"
            print(f"  cell {p['cell_id']:>4}  zone {p['zone']}  {p['role']:<11} "
                  f"fit={p['fit']:>2}  {tag}:")
            for b in p["builds"]:
                ax, az = b["anchor_world"]
                decay_str = ""
                if show_decay:
                    dp = plan_decay(b["structure"],
                                    seed=f"{settlement.name}:{p['cell_id']}:{ax},{az}")
                    decay_str = f"  decay=[{len(dp.remove)} holes, {len(dp.cobwebs)} webs]"
                name = b["structure"].name or f"{ROLE_PREFIX[p['role']]}_{b['size']}"
                print(f"        {name}  "
                      f"rot={b['rotation']}  anchor=({ax}, {az}){decay_str}")
        return

    from gdpc import Editor  # lazy: only real placement needs a world
    editor = Editor(buffering=True)
    ground_y = heightmap_ground_y(heightmap, origin)

    chosen = placements[: max_builds or None]
    seed_name = str(settlement.name)
    totals = Counter()

    # Narrative items: one diary + tool + relic per district, routed into a
    # build's chest. Planned over `chosen` so a payload always lands on a build we
    # actually place; districts without an available chest get a fallback chest.
    fallback_chests: list[tuple] = []
    if place_items:
        print("\nGenerating diaries + tools + relics and routing them into district chests...")
        fallback_chests, item_summary = plan_narrative_items(
            settlement, biome_override, chosen, roles, zone_seed_points,
            origin, heightmap, fields=fields,
        )
        if item_summary:
            print(f"  district items: {item_summary['in_build']} into build chests, "
                  f"{item_summary['fallback']} on farm plots / standalone, "
                  f"{item_summary['skipped']} skipped (no content); "
                  f"{item_summary['relics']} relic(s) spread across districts.")

    decay_note = "" if (decay and tier == "struggling") else " (decay off)" if not decay else ""
    print(f"\nPlacing builds in {len(chosen)} cell(s) at mood tier '{tier}'{decay_note}...")
    for p in chosen:
        # Yard first (decorates the cell around the build(s)), then the build(s).
        place_yard(editor, p["cells"], p["occupied"], origin, ground_y,
                   role=p["role"], mood=tier, seed_name=seed_name)
        foundation = ROLE_FOUNDATION.get(p["role"], DEFAULT_FOUNDATION)
        for b in p["builds"]:
            ax, az = b["anchor_world"]
            stats = build_premade(
                editor, b["structure"], (ax, az), ground_y,
                tier=tier, rotation=b["rotation"], foundation_block=foundation,
                decay=decay, decay_seed=f"{seed_name}:{p['cell_id']}:{ax},{az}",
                chest_payload=b.get("chest_payload"),
            )
            totals["placed"] += stats["placed"]
            totals["removed"] += stats.get("removed", 0)
            totals["cobwebs"] += stats.get("cobwebs", 0)
            totals["chests_filled"] += stats.get("chests_filled", 0)
            totals["compat_remapped"] += stats.get("compat_remapped", 0)
            totals["builds"] += 1

    # Farm-ROLE district: render its cells as mood-scaled crop fields (the
    # generator no longer does — BUILD_FARM_FIELDS=False). Done before the
    # fallback chests so a farm-district chest lands on top of the field.
    if fields:
        from farm_field import place_farm_field
        print(f"Rendering crop fields for {len(fields)} farm-district cell(s)...")
        for f in fields:
            fstats = place_farm_field(editor, f["cells"], origin, ground_y,
                                      mood=tier, seed_name=seed_name)
            totals["crops"] += fstats["crops"]
            totals["field_cols"] += (fstats["border"] + fstats["water"]
                                     + fstats["crops"] + fstats["empty"])

    # Standalone chests for districts whose builds had no chest: the farm
    # district gets one chest per farm plot (placed on top of the just-rendered
    # field); a district with no plots gets center-offset chests.
    if fallback_chests:
        from place_relic_chest import place_chest
        for pos, payload, _zone_id in fallback_chests:
            place_chest(editor, pos, payload)
            totals["chests_filled"] += 1

    editor.flushBuffer()
    decay_summary = ""
    if totals["removed"] or totals["cobwebs"]:
        decay_summary = (f", {totals['removed']} block(s) knocked out, "
                         f"{totals['cobwebs']} cobweb(s) strung")
    item_note = f", {totals['chests_filled']} chest(s) filled" if totals["chests_filled"] else ""
    compat_note = (
        f", {totals['compat_remapped']} server-compatible block remap(s)"
        if totals["compat_remapped"]
        else ""
    )
    field_note = (f", {totals['field_cols']} field column(s) "
                  f"({totals['crops']} crops)") if totals["field_cols"] else ""
    print(f"Done: {totals['builds']} build(s) in {len(chosen)} cell(s), "
          f"{totals['placed']} blocks placed{decay_summary}{item_note}"
          f"{compat_note}{field_note}.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Place premade builds on farm cells (#4).")
    ap.add_argument("--theme", default="Fantasy", help="Settlement theme (default: Fantasy).")
    ap.add_argument("--npz", default=None, help="Path to settlement_data.npz.")
    ap.add_argument("--plots", default=None, help="Path to settlement_plots.npz.")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no world needed.")
    ap.add_argument("--max", type=int, default=None, dest="max_builds",
                    help="Cap the number of builds placed (testing).")
    ap.add_argument("--biome", default=None, dest="biome",
                    help="Override the auto-detected biome.")
    ap.add_argument("--tier", default=None, choices=TIERS, dest="tier",
                    help="Force the mood tier for the whole settlement (testing); "
                         "default uses the LLM-decided settlement.mood_tier.")
    ap.add_argument("--rotation", type=int, default=None, choices=[0, 1, 2, 3],
                    help="Force one rotation for all builds (testing); default is a "
                         "deterministic per-cell random rotation seeded by the settlement.")
    ap.add_argument("--no-decay", action="store_false", dest="decay",
                    help="Disable the struggling-tier decay pass (knocked-out blocks + "
                         "cobwebs). Decay only applies at mood tier 'struggling' anyway.")
    ap.add_argument("--no-items", action="store_false", dest="place_items",
                    help="Skip generating + placing per-district diaries, tools, and "
                         "relics (builds still place; their chests stay empty).")
    ap.add_argument("--no-farm-fields", action="store_false", dest="farm_fields",
                    help="Skip rendering the farm-role district's crop fields "
                         "(the generator no longer places them either).")
    args = ap.parse_args()
    main(
        theme=args.theme, npz=args.npz, plots=args.plots,
        dry_run=args.dry_run, max_builds=args.max_builds, biome_override=args.biome,
        tier_override=args.tier, rotation_override=args.rotation, decay=args.decay,
        place_items=args.place_items, farm_fields=args.farm_fields,
    )
