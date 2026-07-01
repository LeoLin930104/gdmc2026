"""Grammar-driven cached room interior planning.

Each occupied cell remains a distinct room/container. The interior system does
not merge cells; instead it derives a formal room signature from the cell's
spatial constraints, resolves a room plan from hardcoded grammar rules, and
then materialises placeholder composites for the plan's keywords.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from prefab_housing.interior_designs import RoomStyleVariant, load_room_style_variants
from prefab_housing.stairwell import build_stair_stack_plan, emit_stairwell_blocks, stairwell_layout_placements
from prefab_housing.types import (
    RoomComponentPlacement,
    RoomComponentSpec,
    RoomInterior,
    RoomLayoutPlan,
    RoomPlan,
    RoomRequest,
    RoomSignature,
    RoomSpatialConstraints,
    SemanticBlockDict,
    SemanticCell,
)


def _block(
    x: int,
    y: int,
    z: int,
    block_id: str,
    properties: dict[str, str] | None = None,
) -> SemanticBlockDict:
    block: SemanticBlockDict = {"x": x, "y": y, "z": z, "id": block_id}
    if properties:
        block["properties"] = dict(properties)
    return block


def _translate_blocks(
    local_blocks: list[SemanticBlockDict],
    origin: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    ox, oy, oz = origin
    out: list[SemanticBlockDict] = []
    for block in local_blocks:
        translated: SemanticBlockDict = {
            "x": block["x"] + ox,
            "y": block["y"] + oy,
            "z": block["z"] + oz,
            "id": block["id"],
        }
        properties = block.get("properties")
        if properties:
            translated["properties"] = dict(properties)
        out.append(translated)
    return out


COMPONENT_LIBRARY: dict[str, RoomComponentSpec] = {
    "bed_core": RoomComponentSpec("bed_core", "minecraft:red_bed", (2, 2), "corner", "core"),
    "bedside": RoomComponentSpec("bedside", "minecraft:oak_planks", (1, 1), "edge", "core"),
    "wardrobe": RoomComponentSpec("wardrobe", "minecraft:barrel", (1, 1), "wall", "supplementary"),
    "desk": RoomComponentSpec("desk", "minecraft:birch_planks", (2, 1), "window", "supplementary"),
    "chair": RoomComponentSpec("chair", "minecraft:oak_stairs", (1, 1), "interior", "supplementary"),
    "rug": RoomComponentSpec("rug", "minecraft:light_gray_carpet", (2, 2), "centre", "supplementary"),
    "sofa": RoomComponentSpec("sofa", "minecraft:gray_wool", (2, 1), "wall", "core"),
    "coffee_table": RoomComponentSpec("coffee_table", "minecraft:oak_slab", (1, 1), "centre", "core"),
    "bookshelf": RoomComponentSpec("bookshelf", "minecraft:bookshelf", (1, 1), "wall", "supplementary"),
    "plant": RoomComponentSpec("plant", "minecraft:moss_block", (1, 1), "corner", "supplementary"),
    "entry_console": RoomComponentSpec("entry_console", "minecraft:spruce_planks", (2, 1), "door", "core"),
    "coat_storage": RoomComponentSpec("coat_storage", "minecraft:barrel", (1, 1), "door", "supplementary"),
    "shoe_storage": RoomComponentSpec("shoe_storage", "minecraft:oak_slab", (1, 1), "door", "supplementary"),
    "kitchen_counter": RoomComponentSpec("kitchen_counter", "minecraft:smooth_stone", (2, 1), "wall", "core"),
    "kitchen_sink": RoomComponentSpec("kitchen_sink", "minecraft:iron_block", (1, 1), "wall", "core"),
    "stove": RoomComponentSpec("stove", "minecraft:furnace", (1, 1), "wall", "core"),
    "fridge": RoomComponentSpec("fridge", "minecraft:white_concrete", (1, 1), "wall", "supplementary"),
    "pantry": RoomComponentSpec("pantry", "minecraft:barrel", (1, 1), "wall", "supplementary"),
    "dining_nook": RoomComponentSpec("dining_nook", "minecraft:oak_planks", (2, 2), "window", "supplementary"),
    "toilet": RoomComponentSpec("toilet", "minecraft:quartz_stairs", (1, 1), "wall", "core"),
    "shower": RoomComponentSpec("shower", "minecraft:light_blue_stained_glass", (2, 1), "corner", "core"),
    "sink": RoomComponentSpec("sink", "minecraft:quartz_block", (1, 1), "wall", "core"),
    "mirror": RoomComponentSpec("mirror", "minecraft:glass", (1, 1), "wall", "supplementary"),
    "towel_storage": RoomComponentSpec("towel_storage", "minecraft:white_wool", (1, 1), "wall", "supplementary"),
    "hall_runner": RoomComponentSpec("hall_runner", "minecraft:gray_carpet", (1, 3), "centre", "core"),
    "storage_niche": RoomComponentSpec("storage_niche", "minecraft:barrel", (1, 1), "wall", "supplementary"),
    "landing_light": RoomComponentSpec("landing_light", "minecraft:lantern", (1, 1), "centre", "supplementary"),
    "stair_marker": RoomComponentSpec("stair_marker", "minecraft:lime_concrete", (1, 2), "centre", "core"),
    "service_bank": RoomComponentSpec("service_bank", "minecraft:stone_bricks", (2, 1), "wall", "core"),
    "utility_lockers": RoomComponentSpec("utility_lockers", "minecraft:iron_block", (1, 1), "wall", "supplementary"),
    "bunk": RoomComponentSpec("bunk", "minecraft:blue_bed", (2, 1), "wall", "supplementary"),
    "task_table": RoomComponentSpec("task_table", "minecraft:oak_planks", (2, 1), "wall", "core"),
    "ceiling_light": RoomComponentSpec("ceiling_light", "minecraft:lantern", (1, 1), "ceiling_centre", "lighting"),
    "wall_light": RoomComponentSpec("wall_light", "minecraft:torch", (1, 1), "wall", "lighting"),
    "extra_light": RoomComponentSpec("extra_light", "minecraft:sea_lantern", (1, 1), "ceiling_edge", "lighting"),
}


@dataclass(frozen=True, slots=True)
class InteriorCacheStats:
    hits: int = 0
    misses: int = 0


@dataclass(frozen=True, slots=True)
class InteriorStyleProfile:
    id: str
    label: str
    bed_block: str
    wardrobe_block: str
    desk_top_block: str
    table_top_block: str
    sofa_base_block: str
    sofa_top_block: str
    rug_primary_block: str
    rug_secondary_block: str
    kitchen_counter_block: str
    kitchen_top_block: str
    bathroom_wall_block: str
    shower_glass_block: str
    storage_block: str
    accent_light_block: str


@dataclass(frozen=True, slots=True)
class _LayoutContext:
    request: RoomRequest
    interior_size: tuple[int, int, int]
    centre: tuple[float, float]
    door_positions: tuple[tuple[int, int], ...]
    window_faces: frozenset[str]
    door_faces: frozenset[str]
    hard_reserved_cells: frozenset[tuple[int, int]]
    circulation_cells: frozenset[tuple[int, int]]
    floor_y: int
    wall_y: int
    ceiling_y: int


INTERIOR_STYLE_PROFILES: tuple[InteriorStyleProfile, ...] = (
    InteriorStyleProfile(
        id="classic_modular",
        label="Classic Modular",
        bed_block="minecraft:red_bed",
        wardrobe_block="minecraft:barrel",
        desk_top_block="minecraft:birch_slab",
        table_top_block="minecraft:oak_slab",
        sofa_base_block="minecraft:gray_wool",
        sofa_top_block="minecraft:gray_carpet",
        rug_primary_block="minecraft:light_gray_carpet",
        rug_secondary_block="minecraft:white_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:smooth_stone_slab",
        bathroom_wall_block="minecraft:light_blue_concrete",
        shower_glass_block="minecraft:light_blue_stained_glass",
        storage_block="minecraft:barrel",
        accent_light_block="minecraft:lantern",
    ),
    InteriorStyleProfile(
        id="rustic_cabin",
        label="Rustic Cabin",
        bed_block="minecraft:green_bed",
        wardrobe_block="minecraft:dark_oak_planks",
        desk_top_block="minecraft:spruce_slab",
        table_top_block="minecraft:spruce_slab",
        sofa_base_block="minecraft:spruce_planks",
        sofa_top_block="minecraft:green_carpet",
        rug_primary_block="minecraft:green_carpet",
        rug_secondary_block="minecraft:gray_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:spruce_slab",
        bathroom_wall_block="minecraft:spruce_planks",
        shower_glass_block="minecraft:glass_pane",
        storage_block="minecraft:barrel",
        accent_light_block="minecraft:lantern",
    ),
    InteriorStyleProfile(
        id="modern_minimalist",
        label="Modern Minimalist",
        bed_block="minecraft:white_bed",
        wardrobe_block="minecraft:white_concrete",
        desk_top_block="minecraft:quartz_block",
        table_top_block="minecraft:quartz_slab",
        sofa_base_block="minecraft:white_concrete",
        sofa_top_block="minecraft:light_gray_carpet",
        rug_primary_block="minecraft:white_carpet",
        rug_secondary_block="minecraft:light_gray_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:quartz_slab",
        bathroom_wall_block="minecraft:quartz_block",
        shower_glass_block="minecraft:glass_pane",
        storage_block="minecraft:barrel",
        accent_light_block="minecraft:sea_lantern",
    ),
    InteriorStyleProfile(
        id="industrial_loft",
        label="Industrial Loft",
        bed_block="minecraft:gray_bed",
        wardrobe_block="minecraft:smooth_stone",
        desk_top_block="minecraft:dark_oak_planks",
        table_top_block="minecraft:dark_oak_planks",
        sofa_base_block="minecraft:gray_concrete",
        sofa_top_block="minecraft:gray_carpet",
        rug_primary_block="minecraft:gray_carpet",
        rug_secondary_block="minecraft:light_gray_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:smooth_stone_slab",
        bathroom_wall_block="minecraft:smooth_stone",
        shower_glass_block="minecraft:glass_pane",
        storage_block="minecraft:barrel",
        accent_light_block="minecraft:sea_lantern",
    ),
    InteriorStyleProfile(
        id="tropical_breeze",
        label="Tropical Breeze",
        bed_block="minecraft:blue_bed",
        wardrobe_block="minecraft:bamboo_planks",
        desk_top_block="minecraft:bamboo_slab",
        table_top_block="minecraft:bamboo_slab",
        sofa_base_block="minecraft:bamboo_planks",
        sofa_top_block="minecraft:light_blue_carpet",
        rug_primary_block="minecraft:white_carpet",
        rug_secondary_block="minecraft:light_blue_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:bamboo_slab",
        bathroom_wall_block="minecraft:bamboo_planks",
        shower_glass_block="minecraft:light_blue_stained_glass",
        storage_block="minecraft:bookshelf",
        accent_light_block="minecraft:lantern",
    ),
    InteriorStyleProfile(
        id="cozy_scandinavian",
        label="Cosy Scandinavian",
        bed_block="minecraft:yellow_bed",
        wardrobe_block="minecraft:birch_planks",
        desk_top_block="minecraft:birch_slab",
        table_top_block="minecraft:birch_slab",
        sofa_base_block="minecraft:birch_planks",
        sofa_top_block="minecraft:white_carpet",
        rug_primary_block="minecraft:white_carpet",
        rug_secondary_block="minecraft:yellow_carpet",
        kitchen_counter_block="minecraft:smooth_stone",
        kitchen_top_block="minecraft:birch_slab",
        bathroom_wall_block="minecraft:white_wool",
        shower_glass_block="minecraft:glass_pane",
        storage_block="minecraft:bookshelf",
        accent_light_block="minecraft:lantern",
    ),
)


def interior_style_profile(variant_seed: int = 0) -> InteriorStyleProfile:
    return INTERIOR_STYLE_PROFILES[abs(int(variant_seed)) % len(INTERIOR_STYLE_PROFILES)]


ROOM_STYLE_VARIANTS: dict[str, tuple[RoomStyleVariant, ...]] = load_room_style_variants()


def room_style_variant(
    room_type: str,
    variant_index: int,
) -> RoomStyleVariant | None:
    variants = ROOM_STYLE_VARIANTS.get(room_type)
    if not variants:
        return None
    return variants[abs(int(variant_index)) % len(variants)]


def room_interior_style_profile(
    room_type: str,
    *,
    variant_seed: int = 0,
    variant_index: int = 0,
) -> InteriorStyleProfile:
    base = interior_style_profile(variant_seed)
    variant = room_style_variant(room_type, variant_index)
    if variant is None:
        return base
    return replace(
        base,
        id=f"{base.id}:{room_type}:{variant.id}",
        label=f"{base.label} / {variant.label}",
        **variant.overrides,
    )


def _storage_props(block_id: str, facing: str = "up") -> dict[str, str] | None:
    if block_id == "minecraft:barrel":
        return {"facing": facing}
    return None


def _light_props(block_id: str, *, hanging: bool) -> dict[str, str] | None:
    if block_id == "minecraft:lantern":
        return {"hanging": str(hanging).lower()}
    return None


def _top_props(block_id: str) -> dict[str, str] | None:
    if block_id.endswith("_slab"):
        return {"type": "top"}
    return None


def room_cache_key(request: RoomRequest, *, variant_seed: int = 0) -> str:
    constraints = request.constraints
    signature = request.signature or derive_room_signature(request)
    variant_index = room_variant_index(request, variant_seed=variant_seed)
    style = room_interior_style_profile(
        request.room_type,
        variant_seed=variant_seed,
        variant_index=variant_index,
    )
    return "|".join(
        (
            request.utility_type,
            request.room_type,
            request.role,
            f"size={constraints.voxel_size[0]}x{constraints.voxel_size[1]}x{constraints.voxel_size[2]}",
            f"doors={','.join(constraints.door_faces) if constraints.door_faces else '-'}",
            f"windows={','.join(constraints.window_faces) if constraints.window_faces else '-'}",
            f"privacy={signature.privacy_band}",
            f"exposure={signature.exposure}",
            f"occupancy={signature.occupancy_band}",
            f"class={signature.size_class}",
            f"light={signature.lighting_tier}",
            f"style={style.id}",
            f"variant={variant_index}",
        )
    )


def room_variant_index(
    request: RoomRequest,
    *,
    variant_count: int = 3,
    variant_seed: int = 0,
) -> int:
    """Stable low-cardinality variant bucket for repeated room cell types."""
    if variant_count < 1:
        raise ValueError("variant_count must be >= 1")
    cell_index = request.constraints.cell_index
    if cell_index is None:
        return 0
    ix, iy, iz = cell_index
    seed = (ix * 73_856_093) ^ (iy * 19_349_663) ^ (iz * 83_492_791)
    seed += sum(ord(char) for char in request.room_type) * 2_654_435_761
    seed ^= int(variant_seed) * 97_531
    return abs(seed) % variant_count


def layout_variant_index(
    layout: RoomLayoutPlan,
    *,
    variant_count: int = 3,
    variant_seed: int = 0,
) -> int:
    if variant_count < 1:
        raise ValueError("variant_count must be >= 1")
    if layout.cell_index is None:
        return 0
    ix, iy, iz = layout.cell_index
    seed = (ix * 73_856_093) ^ (iy * 19_349_663) ^ (iz * 83_492_791)
    seed += sum(ord(char) for char in layout.plan.signature.room_type) * 2_654_435_761
    seed ^= int(variant_seed) * 97_531
    return abs(seed) % variant_count


def _room_constraints_from_semantic_cell(cell: SemanticCell) -> RoomSpatialConstraints:
    (x0, y0, z0), (x1, y1, z1) = cell.voxel_bbox
    return RoomSpatialConstraints(
        voxel_size=(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1),
        cell_index=cell.cell_index,
        door_faces=cell.door_faces,
        window_faces=cell.window_faces,
        open_faces=cell.open_faces,
        opening_pattern=cell.opening_pattern,
        privacy_depth=cell.privacy_depth,
        occupancy_capacity=cell.occupancy_capacity,
    )


def derive_room_signature(request: RoomRequest) -> RoomSignature:
    vx, vy, vz = request.constraints.voxel_size
    inner_x = max(1, vx - 2)
    inner_y = max(1, vy - 2)
    inner_z = max(1, vz - 2)
    floor_area = inner_x * inner_z
    doorway_count = len(request.constraints.door_faces)
    window_count = len(request.constraints.window_faces)

    if request.constraints.privacy_depth >= 3:
        privacy_band = "deep_private"
    elif request.constraints.privacy_depth >= 1:
        privacy_band = "semi_private"
    else:
        privacy_band = "public"

    if window_count >= 2:
        exposure = "broad"
    elif window_count == 1:
        exposure = "single"
    else:
        exposure = "internal"

    occ = request.constraints.occupancy_capacity
    if occ >= 3:
        occupancy_band = "high"
    elif occ == 2:
        occupancy_band = "medium"
    else:
        occupancy_band = "low"

    if floor_area >= 40:
        size_class = "spacious"
    elif floor_area >= 16:
        size_class = "standard"
    else:
        size_class = "compact"

    lighting_tier = "single_central" if floor_area <= 24 else "central_plus_edges"

    return RoomSignature(
        room_type=request.room_type,
        utility_type=request.utility_type,
        role=request.role,
        voxel_size=request.constraints.voxel_size,
        interior_size=(inner_x, inner_y, inner_z),
        floor_area=floor_area,
        doorway_count=doorway_count,
        window_count=window_count,
        privacy_band=privacy_band,
        exposure=exposure,
        occupancy_band=occupancy_band,
        size_class=size_class,
        lighting_tier=lighting_tier,
    )


def make_room_request(
    cell: SemanticCell,
    *,
    utility_type: str,
) -> RoomRequest:
    constraints = _room_constraints_from_semantic_cell(cell)
    request = RoomRequest(
        room_type=cell.label,
        utility_type=utility_type,  # type: ignore[arg-type]
        role=cell.role,
        constraints=constraints,
    )
    signature = derive_room_signature(request)
    return RoomRequest(
        room_type=request.room_type,
        utility_type=request.utility_type,
        role=request.role,
        constraints=request.constraints,
        signature=signature,
    )


def plan_room(request: RoomRequest) -> RoomPlan:
    signature = request.signature or derive_room_signature(request)
    core: list[str] = []
    supplementary: list[str] = []
    lighting: list[str] = ["ceiling_light"]
    room_type = request.room_type

    if signature.lighting_tier == "central_plus_edges":
        lighting.append("extra_light")
    elif signature.exposure == "internal":
        lighting.append("wall_light")

    if room_type == "bedroom":
        core.append("bed_core")
        core.append("bedside")
        supplementary.append("wardrobe")
        if signature.size_class != "compact":
            supplementary.append("desk")
            supplementary.append("chair")
        if signature.exposure != "internal":
            supplementary.append("rug")
    elif room_type == "living":
        core.extend(("sofa", "coffee_table"))
        supplementary.append("bookshelf")
        if signature.size_class != "compact":
            supplementary.append("plant")
    elif room_type == "kitchen":
        core.extend(("kitchen_counter", "kitchen_sink", "stove"))
        supplementary.append("fridge")
        if signature.size_class != "compact":
            supplementary.extend(("pantry", "dining_nook"))
    elif room_type == "bathroom":
        core.extend(("toilet", "sink", "shower"))
        supplementary.extend(("mirror", "towel_storage"))
        lighting.append("wall_light")
    elif room_type == "entry":
        core.append("entry_console")
        supplementary.extend(("coat_storage", "shoe_storage"))
    elif room_type == "corridor":
        core.append("hall_runner")
        supplementary.append("storage_niche")
        lighting.append("wall_light")
    elif room_type == "stairwell":
        core.append("stair_marker")
        supplementary.append("landing_light")
        lighting.append("wall_light")
    else:
        if request.role == "service":
            core.append("service_bank")
            supplementary.append("utility_lockers")
        elif request.role == "habitable":
            core.append("bunk" if signature.occupancy_band == "high" else "task_table")
        else:
            core.append("hall_runner")

    if signature.occupancy_band == "high" and room_type in {"bedroom", "service_building"}:
        supplementary.append("bunk")

    return RoomPlan(
        signature=signature,
        core_keywords=tuple(dict.fromkeys(core)),
        supplementary_keywords=tuple(dict.fromkeys(supplementary)),
        lighting_keywords=tuple(dict.fromkeys(lighting)),
    )


_RELATED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bedside": ("bed_core",),
    "bookshelf": ("sofa",),
    "chair": ("desk", "dining_nook", "task_table"),
    "coat_storage": ("entry_console",),
    "coffee_table": ("sofa",),
    "fridge": ("kitchen_counter",),
    "kitchen_sink": ("kitchen_counter",),
    "landing_light": ("stair_marker",),
    "mirror": ("sink",),
    "pantry": ("kitchen_counter",),
    "shoe_storage": ("entry_console",),
    "stove": ("kitchen_counter",),
    "towel_storage": ("sink", "shower"),
    "wall_light": ("mirror", "sink"),
}


def _is_floor_cover(spec: RoomComponentSpec) -> bool:
    return "carpet" in spec.block_id


def _is_overlay_component(spec: RoomComponentSpec, keyword: str) -> bool:
    return spec.category == "lighting" or keyword in {"landing_light", "mirror"}


def _footprint_variants(footprint: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    fx, fz = footprint
    if fx == fz:
        return (footprint,)
    return (footprint, (fz, fx))


def _footprint_cells(x: int, z: int, footprint: tuple[int, int]) -> set[tuple[int, int]]:
    fx, fz = footprint
    return {(ix, iz) for ix in range(x, x + fx) for iz in range(z, z + fz)}


def _placement_cells(placement: RoomComponentPlacement) -> set[tuple[int, int]]:
    x, _, z = placement.origin
    return _footprint_cells(x, z, placement.footprint)


def _touching_faces(
    x: int,
    z: int,
    footprint: tuple[int, int],
    interior_size: tuple[int, int, int],
) -> set[str]:
    ix, _, iz = interior_size
    fx, fz = footprint
    faces: set[str] = set()
    if z == 1:
        faces.add("north")
    if z + fz - 1 == iz:
        faces.add("south")
    if x == 1:
        faces.add("west")
    if x + fx - 1 == ix:
        faces.add("east")
    return faces


def _centre_distance(
    x: int,
    z: int,
    footprint: tuple[int, int],
    centre: tuple[float, float],
) -> float:
    fx, fz = footprint
    rect_centre = (x + fx / 2 - 0.5, z + fz / 2 - 0.5)
    return abs(rect_centre[0] - centre[0]) + abs(rect_centre[1] - centre[1])


def _nearest_distance(cells: set[tuple[int, int]], targets: tuple[tuple[int, int], ...]) -> int:
    if not targets:
        return 0
    return min(abs(x - tx) + abs(z - tz) for x, z in cells for tx, tz in targets)


def _shares_edge(cells: set[tuple[int, int]], other_cells: set[tuple[int, int]]) -> bool:
    for x, z in cells:
        if (
            (x + 1, z) in other_cells
            or (x - 1, z) in other_cells
            or (x, z + 1) in other_cells
            or (x, z - 1) in other_cells
        ):
            return True
    return False


def _related_placements(
    keyword: str,
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
) -> list[RoomComponentPlacement]:
    related: list[RoomComponentPlacement] = []
    for target in _RELATED_KEYWORDS.get(keyword, ()): 
        related.extend(placements_by_keyword.get(target, ()))
    return related


def _opening_position(face: str, interior_size: tuple[int, int, int]) -> tuple[int, int]:
    ix, _, iz = interior_size
    centre_x = 1 + (ix - 1) // 2
    centre_z = 1 + (iz - 1) // 2
    if face == "north":
        return (centre_x, 1)
    if face == "south":
        return (centre_x, iz)
    if face == "west":
        return (1, centre_z)
    return (ix, centre_z)


def _door_path(face: str, start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, z = start
    target_x, target_z = end
    path = [(x, z)]
    if face in {"north", "south"}:
        primary = "z"
    else:
        primary = "x"

    while (x, z) != (target_x, target_z):
        if primary == "z" and z != target_z:
            z += 1 if target_z > z else -1
        elif primary == "x" and x != target_x:
            x += 1 if target_x > x else -1
        elif x != target_x:
            x += 1 if target_x > x else -1
        elif z != target_z:
            z += 1 if target_z > z else -1
        path.append((x, z))
    return tuple(path)


def _make_layout_request(plan: RoomPlan, request: RoomRequest | None) -> RoomRequest:
    if request is None:
        return RoomRequest(
            room_type=plan.signature.room_type,
            utility_type=plan.signature.utility_type,
            role=plan.signature.role,
            constraints=RoomSpatialConstraints(voxel_size=plan.signature.voxel_size),
            signature=plan.signature,
        )
    return RoomRequest(
        room_type=request.room_type,
        utility_type=request.utility_type,
        role=request.role,
        constraints=request.constraints,
        signature=plan.signature,
    )


def _make_layout_context(request: RoomRequest, plan: RoomPlan) -> _LayoutContext:
    ix, iy, iz = plan.signature.interior_size
    centre_cell = (1 + (ix - 1) // 2, 1 + (iz - 1) // 2)
    door_positions = tuple(_opening_position(face, plan.signature.interior_size) for face in request.constraints.door_faces)
    hard_reserved: set[tuple[int, int]] = set()
    circulation: set[tuple[int, int]] = set()
    for face, start in zip(request.constraints.door_faces, door_positions):
        path = _door_path(face, start, centre_cell)
        circulation.update(path)
        hard_reserved.update(path[: min(2, len(path))])

    floor_y = 1
    ceiling_y = max(floor_y, iy)
    wall_y = max(floor_y, ceiling_y - 1) if ceiling_y > floor_y else floor_y
    return _LayoutContext(
        request=request,
        interior_size=plan.signature.interior_size,
        centre=((ix + 1) / 2, (iz + 1) / 2),
        door_positions=door_positions,
        window_faces=frozenset(request.constraints.window_faces),
        door_faces=frozenset(request.constraints.door_faces),
        hard_reserved_cells=frozenset(hard_reserved),
        circulation_cells=frozenset(circulation),
        floor_y=floor_y,
        wall_y=wall_y,
        ceiling_y=ceiling_y,
    )


def _iter_candidates(
    interior_size: tuple[int, int, int],
    footprint: tuple[int, int],
) -> list[tuple[int, int, tuple[int, int], set[tuple[int, int]]]]:
    ix, _, iz = interior_size
    candidates: list[tuple[int, int, tuple[int, int], set[tuple[int, int]]]] = []
    for dims in _footprint_variants(footprint):
        fx, fz = dims
        for x in range(1, ix - fx + 2):
            for z in range(1, iz - fz + 2):
                candidates.append((x, z, dims, _footprint_cells(x, z, dims)))
    return candidates


def _score_related_distance(
    keyword: str,
    cells: set[tuple[int, int]],
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
) -> float:
    related = _related_placements(keyword, placements_by_keyword)
    if not related:
        return 0.0
    related_cells = [_placement_cells(item) for item in related]
    shares_edge = any(_shares_edge(cells, item_cells) for item_cells in related_cells)
    overlaps = any(cells & item_cells for item_cells in related_cells)
    min_distance = min(
        abs(x - tx) + abs(z - tz)
        for x, z in cells
        for item_cells in related_cells
        for tx, tz in item_cells
    )
    if keyword == "mirror":
        if overlaps:
            return 7.0
        if shares_edge:
            return 3.0
        return -0.8 * min_distance
    if shares_edge:
        return 5.0
    return 1.5 / (1.0 + min_distance) - 0.7 * min_distance


def _score_blocking_candidate(
    keyword: str,
    spec: RoomComponentSpec,
    x: int,
    z: int,
    footprint: tuple[int, int],
    cells: set[tuple[int, int]],
    context: _LayoutContext,
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
) -> float:
    touching_faces = _touching_faces(x, z, footprint, context.interior_size)
    wall_contacts = len(touching_faces)
    window_contacts = len(touching_faces & context.window_faces)
    door_wall_contacts = len(touching_faces & context.door_faces)
    centre_distance = _centre_distance(x, z, footprint, context.centre)
    door_distance = _nearest_distance(cells, context.door_positions)
    circulation_overlap = len(cells & context.circulation_cells)
    score = -0.35 * centre_distance - 0.9 * circulation_overlap

    if spec.anchor == "corner":
        score += 4.5 if wall_contacts >= 2 else -5.0
        score += 0.7 * door_distance
    elif spec.anchor == "wall":
        score += 3.0 if wall_contacts >= 1 else -4.0
        score += 0.2 * door_distance
    elif spec.anchor == "window":
        score += 5.0 * window_contacts - (4.0 if window_contacts == 0 else 0.0)
        score += 1.0 if wall_contacts >= 1 else -1.5
    elif spec.anchor == "door":
        score += 4.0 * door_wall_contacts
        score += 3.0 / (1.0 + door_distance)
    elif spec.anchor == "centre":
        score -= 1.2 * centre_distance
    elif spec.anchor == "interior":
        score -= 0.7 * centre_distance
        score -= 1.5 * wall_contacts
    elif spec.anchor == "edge":
        score -= 0.5 * centre_distance

    if keyword == "bed_core":
        score += 0.8 * door_distance
        score -= 1.2 * window_contacts
    elif keyword in {
        "coat_storage",
        "fridge",
        "pantry",
        "shoe_storage",
        "storage_niche",
        "utility_lockers",
        "wardrobe",
    }:
        score += 0.3 * door_distance
        score -= 2.0 * window_contacts
    elif keyword == "desk":
        score += 2.5 * window_contacts
    elif keyword == "entry_console":
        score += 2.0 * door_wall_contacts
    elif keyword == "kitchen_counter":
        score += 1.5 * wall_contacts
        score -= 1.0 * window_contacts
    elif keyword == "kitchen_sink":
        score += 1.5 * wall_contacts + 0.8 * window_contacts
    elif keyword == "sink":
        score += 1.0 * wall_contacts + 0.5 * window_contacts
    elif keyword in {"shower", "toilet"}:
        score += 0.6 * wall_contacts + 0.4 * door_distance

    score += _score_related_distance(keyword, cells, placements_by_keyword)
    return score


def _score_floor_cover_candidate(
    keyword: str,
    x: int,
    z: int,
    footprint: tuple[int, int],
    cells: set[tuple[int, int]],
    context: _LayoutContext,
    placements: list[RoomComponentPlacement],
) -> float:
    touching_faces = _touching_faces(x, z, footprint, context.interior_size)
    centre_distance = _centre_distance(x, z, footprint, context.centre)
    circulation_overlap = len(cells & context.circulation_cells)
    core_overlap = sum(len(cells & _placement_cells(item)) for item in placements if item.category == "core")
    score = -0.45 * centre_distance - 0.4 * len(touching_faces)
    if keyword == "hall_runner":
        score += 2.5 * circulation_overlap
    else:
        score += 0.8 * min(core_overlap, 3)
        score += 0.4 * circulation_overlap
    return score


def _score_overlay_candidate(
    keyword: str,
    spec: RoomComponentSpec,
    x: int,
    z: int,
    footprint: tuple[int, int],
    cells: set[tuple[int, int]],
    context: _LayoutContext,
    placements: list[RoomComponentPlacement],
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
    overlay_cells: set[tuple[int, int]],
) -> float:
    touching_faces = _touching_faces(x, z, footprint, context.interior_size)
    wall_contacts = len(touching_faces)
    window_contacts = len(touching_faces & context.window_faces)
    door_wall_contacts = len(touching_faces & context.door_faces)
    centre_distance = _centre_distance(x, z, footprint, context.centre)
    score = -0.25 * centre_distance - 2.0 * len(cells & overlay_cells)

    if keyword == "mirror":
        score += 4.0 if wall_contacts else -4.0
        score -= 1.0 * window_contacts
    elif keyword == "landing_light":
        score -= 0.8 * centre_distance
    elif spec.anchor == "ceiling_centre":
        score -= 1.4 * centre_distance
    elif spec.anchor == "ceiling_edge":
        score += 2.0 * wall_contacts + 0.8 * centre_distance
        score -= 1.2 * window_contacts
        score -= 0.8 * door_wall_contacts
    elif spec.anchor == "wall":
        score += 3.0 if wall_contacts else -4.0
        score -= 1.0 * window_contacts

    score += _score_related_distance(keyword, cells, placements_by_keyword)

    if spec.category == "lighting":
        existing_lights = [item for item in placements if item.category == "lighting"]
        if existing_lights:
            light_distance = min(
                abs(x - lx) + abs(z - lz)
                for item in existing_lights
                for lx, lz in _placement_cells(item)
                for x, z in cells
            )
            score += 1.2 * light_distance
    return score


def _choose_candidate(
    spec: RoomComponentSpec,
    keyword: str,
    context: _LayoutContext,
    placements: list[RoomComponentPlacement],
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
    occupied_floor: set[tuple[int, int]],
    overlay_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates = _iter_candidates(context.interior_size, spec.footprint)
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_score = float("-inf")

    if _is_floor_cover(spec):
        for x, z, footprint, cells in candidates:
            score = _score_floor_cover_candidate(keyword, x, z, footprint, cells, context, placements)
            if score > best_score:
                best = ((x, z), footprint)
                best_score = score
        return best or ((1, 1), spec.footprint)

    if _is_overlay_component(spec, keyword):
        for x, z, footprint, cells in candidates:
            score = _score_overlay_candidate(
                keyword,
                spec,
                x,
                z,
                footprint,
                cells,
                context,
                placements,
                placements_by_keyword,
                overlay_cells,
            )
            if score > best_score:
                best = ((x, z), footprint)
                best_score = score
        return best or ((1, 1), spec.footprint)

    for allow_hard_reserved in (False, True):
        for x, z, footprint, cells in candidates:
            overlap = len(cells & occupied_floor)
            if overlap:
                continue
            if not allow_hard_reserved and cells & context.hard_reserved_cells:
                continue
            score = _score_blocking_candidate(keyword, spec, x, z, footprint, cells, context, placements_by_keyword)
            if allow_hard_reserved:
                score -= 3.0 * len(cells & context.hard_reserved_cells)
            if score > best_score:
                best = ((x, z), footprint)
                best_score = score
        if best is not None:
            return best

    for x, z, footprint, cells in candidates:
        overlap = len(cells & occupied_floor)
        score = _score_blocking_candidate(keyword, spec, x, z, footprint, cells, context, placements_by_keyword)
        score -= 12.0 * overlap
        score -= 4.0 * len(cells & context.hard_reserved_cells)
        if score > best_score:
            best = ((x, z), footprint)
            best_score = score
    return best or ((1, 1), spec.footprint)


def _placement_y(spec: RoomComponentSpec, keyword: str, context: _LayoutContext) -> int:
    if keyword == "mirror":
        return context.wall_y
    if keyword == "landing_light":
        return context.ceiling_y
    if spec.category == "lighting":
        return context.ceiling_y if spec.anchor.startswith("ceiling") else context.wall_y
    return context.floor_y


def _ordered_layout_keywords(plan: RoomPlan) -> list[str]:
    furniture: list[str] = []
    overlays: list[str] = []
    floor_covers: list[str] = []
    for keyword in list(plan.core_keywords) + list(plan.supplementary_keywords):
        spec = COMPONENT_LIBRARY[keyword]
        if _is_floor_cover(spec):
            floor_covers.append(keyword)
        elif _is_overlay_component(spec, keyword):
            overlays.append(keyword)
        else:
            furniture.append(keyword)
    return furniture + overlays + floor_covers + list(plan.lighting_keywords)


def plan_room_layout(plan: RoomPlan, request: RoomRequest | None = None) -> RoomLayoutPlan:
    layout_request = _make_layout_request(plan, request)
    if plan.signature.room_type == "stairwell":
        placements = stairwell_layout_placements(
            plan.signature.interior_size,
            layout_request.constraints.cell_index,
            has_up="up" in layout_request.constraints.open_faces,
            has_down="down" in layout_request.constraints.open_faces,
        )
        return RoomLayoutPlan(
            plan=plan,
            interior_size=plan.signature.interior_size,
            cell_index=layout_request.constraints.cell_index,
            placements=placements,
            door_faces=layout_request.constraints.door_faces,
            window_faces=layout_request.constraints.window_faces,
            open_faces=layout_request.constraints.open_faces,
            opening_pattern=layout_request.constraints.opening_pattern,
        )

    context = _make_layout_context(layout_request, plan)
    placements: list[RoomComponentPlacement] = []
    placements_by_keyword: dict[str, list[RoomComponentPlacement]] = {}
    occupied_floor: set[tuple[int, int]] = set()
    overlay_cells: set[tuple[int, int]] = set()

    for keyword in _ordered_layout_keywords(plan):
        spec = COMPONENT_LIBRARY[keyword]
        (x, z), footprint = _choose_candidate(
            spec,
            keyword,
            context,
            placements,
            placements_by_keyword,
            occupied_floor,
            overlay_cells,
        )
        placement = RoomComponentPlacement(
            keyword=keyword,
            block_id=spec.block_id,
            category=spec.category,
            origin=(x, _placement_y(spec, keyword, context), z),
            footprint=footprint,
            anchor=spec.anchor,
        )
        placements.append(placement)
        placements_by_keyword.setdefault(keyword, []).append(placement)
        cells = _placement_cells(placement)
        if _is_overlay_component(spec, keyword):
            overlay_cells.update(cells)
        elif not _is_floor_cover(spec):
            occupied_floor.update(cells)

    return RoomLayoutPlan(
        plan=plan,
        interior_size=plan.signature.interior_size,
        cell_index=layout_request.constraints.cell_index,
        placements=tuple(placements),
        door_faces=layout_request.constraints.door_faces,
        window_faces=layout_request.constraints.window_faces,
        open_faces=layout_request.constraints.open_faces,
        opening_pattern=layout_request.constraints.opening_pattern,
    )


_FACE_OPPOSITES: dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

_RUG_VARIANTS: tuple[tuple[str, str], ...] = (
    ("minecraft:light_gray_carpet", "minecraft:white_carpet"),
    ("minecraft:blue_carpet", "minecraft:light_blue_carpet"),
    ("minecraft:green_carpet", "minecraft:lime_carpet"),
)

_SOFA_TOP_VARIANTS: tuple[str, ...] = (
    "minecraft:gray_carpet",
    "minecraft:light_blue_carpet",
    "minecraft:green_carpet",
)


def _simple_component_blocks(placement: RoomComponentPlacement) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    fx, fz = placement.footprint
    return [
        _block(x0 + dx, y0, z0 + dz, placement.block_id)
        for dx in range(fx)
        for dz in range(fz)
    ]


def _placement_positions(placement: RoomComponentPlacement) -> tuple[tuple[int, int], ...]:
    x0, _, z0 = placement.origin
    fx, fz = placement.footprint
    return tuple((x0 + dx, z0 + dz) for dx in range(fx) for dz in range(fz))


def _placement_centre(placement: RoomComponentPlacement) -> tuple[float, float]:
    x0, _, z0 = placement.origin
    fx, fz = placement.footprint
    return (x0 + (fx - 1) / 2, z0 + (fz - 1) / 2)


def _face_from_delta(dx: float, dz: float) -> str:
    if abs(dx) >= abs(dz) and dx != 0:
        return "east" if dx > 0 else "west"
    if dz != 0:
        return "south" if dz > 0 else "north"
    return "south"


def _first_wall_face(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> str | None:
    x0, _, z0 = placement.origin
    faces = _touching_faces(x0, z0, placement.footprint, interior_size)
    for face in ("north", "south", "west", "east"):
        if face in faces:
            return face
    return None


def _inward_wall_face(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> str:
    wall_face = _first_wall_face(placement, interior_size)
    if wall_face is None:
        return "south"
    return _FACE_OPPOSITES[wall_face]


def _nearest_related_placement(
    placement: RoomComponentPlacement,
    candidates: list[RoomComponentPlacement],
) -> RoomComponentPlacement | None:
    if not candidates:
        return None
    px, pz = _placement_centre(placement)
    return min(
        candidates,
        key=lambda item: abs(_placement_centre(item)[0] - px)
        + abs(_placement_centre(item)[1] - pz),
    )


def _bed_facing_and_head_cells(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> tuple[str, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    x0, _, z0 = placement.origin
    fx, fz = placement.footprint
    ix, _, iz = interior_size
    touching = _touching_faces(x0, z0, placement.footprint, interior_size)
    use_z_axis = fz >= 2 and ("north" in touching or "south" in touching or fx == fz)

    if use_z_axis:
        if "south" in touching:
            facing = "south"
            head_z = z0 + fz - 1
            foot_z = head_z - 1
        else:
            facing = "north"
            head_z = z0
            foot_z = head_z + 1
        width_cells = tuple(x for x in range(x0, x0 + fx) if 1 <= x <= ix)
        head_cells = tuple((x, head_z) for x in width_cells if 1 <= head_z <= iz)
        foot_cells = tuple((x, foot_z) for x in width_cells if 1 <= foot_z <= iz)
        return facing, head_cells, foot_cells

    if "east" in touching:
        facing = "east"
        head_x = x0 + fx - 1
        foot_x = head_x - 1
    else:
        facing = "west"
        head_x = x0
        foot_x = head_x + 1
    width_cells = tuple(z for z in range(z0, z0 + fz) if 1 <= z <= iz)
    head_cells = tuple((head_x, z) for z in width_cells if 1 <= head_x <= ix)
    foot_cells = tuple((foot_x, z) for z in width_cells if 1 <= foot_x <= ix)
    return facing, head_cells, foot_cells


def _emit_bed_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    *,
    block_id: str,
) -> list[SemanticBlockDict]:
    _, y0, _ = placement.origin
    _, iy, _ = interior_size
    facing, head_cells, foot_cells = _bed_facing_and_head_cells(placement, interior_size)
    blocks: list[SemanticBlockDict] = []
    for x, z in head_cells:
        blocks.append(_block(x, y0, z, block_id, {"part": "head", "facing": facing}))
        if y0 + 1 <= iy:
            blocks.append(
                _block(
                    x,
                    y0 + 1,
                    z,
                    "minecraft:spruce_trapdoor",
                    {"facing": facing, "half": "top"},
                )
            )
    for x, z in foot_cells:
        blocks.append(_block(x, y0, z, block_id, {"part": "foot", "facing": facing}))
    return blocks


def _emit_bedside_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    _, iy, _ = interior_size
    blocks = [_block(x0, y0, z0, style.storage_block, _storage_props(style.storage_block))]
    if y0 + 1 <= iy:
        blocks.append(
            _block(
                x0,
                y0 + 1,
                z0,
                style.accent_light_block,
                _light_props(style.accent_light_block, hanging=False),
            )
        )
    return blocks


def _emit_wardrobe_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    _, iy, _ = interior_size
    facing = _inward_wall_face(placement, interior_size)
    blocks: list[SemanticBlockDict] = []
    for x, z in _placement_positions(placement):
        height = max(1, min(3, iy - placement.origin[1] + 1))
        for dy in range(height):
            y = placement.origin[1] + dy
            if dy == height - 1 and height >= 3:
                blocks.append(_block(x, y, z, style.wardrobe_block))
            elif (props := _storage_props(style.wardrobe_block, facing)) is not None:
                blocks.append(_block(x, y, z, style.wardrobe_block, props))
            else:
                blocks.append(_block(x, y, z, style.wardrobe_block))
    return blocks


def _emit_desk_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    _, iy, _ = interior_size
    y0 = placement.origin[1]
    blocks: list[SemanticBlockDict] = []
    cells = _placement_positions(placement)
    for x, z in cells:
        blocks.append(_block(x, y0, z, "minecraft:oak_fence"))
        if y0 + 1 <= iy:
            blocks.append(_block(x, y0 + 1, z, style.desk_top_block, _top_props(style.desk_top_block)))
    if cells and y0 + 2 <= iy:
        x, z = cells[0]
        blocks.append(
            _block(
                x,
                y0 + 2,
                z,
                style.accent_light_block,
                _light_props(style.accent_light_block, hanging=False),
            )
        )
    return blocks


def _emit_chair_blocks(
    placement: RoomComponentPlacement,
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
) -> list[SemanticBlockDict]:
    related = _nearest_related_placement(placement, placements_by_keyword.get("desk", []))
    if related is None:
        facing = "south"
    else:
        px, pz = _placement_centre(placement)
        rx, rz = _placement_centre(related)
        facing = _face_from_delta(rx - px, rz - pz)
    x0, y0, z0 = placement.origin
    return [
        _block(
            x0,
            y0,
            z0,
            "minecraft:oak_stairs",
            {"facing": facing, "half": "bottom", "shape": "straight"},
        )
    ]


def _emit_rug_blocks(
    placement: RoomComponentPlacement,
    *,
    variant_index: int,
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    if style.id == "classic_modular":
        primary, secondary = _RUG_VARIANTS[variant_index % len(_RUG_VARIANTS)]
    else:
        primary, secondary = (
            style.rug_primary_block,
            style.rug_secondary_block,
        )
        if variant_index % 2 == 1:
            primary, secondary = secondary, primary
    for index, (x, z) in enumerate(_placement_positions(placement)):
        block_id = primary if index % 2 == 0 else secondary
        blocks.append(_block(x, placement.origin[1], z, block_id))
    return blocks


def _emit_ceiling_light_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    x0, _, z0 = placement.origin
    _, iy, _ = interior_size
    return [
        _block(
            x0,
            iy,
            z0,
            style.accent_light_block,
            _light_props(style.accent_light_block, hanging=True),
        )
    ]


def _emit_floor_lamp_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    x0, _, z0 = placement.origin
    _, iy, _ = interior_size
    if iy <= 1:
        return [_block(x0, 1, z0, "minecraft:sea_lantern")]
    lamp_y = min(iy, 3)
    blocks = [_block(x0, y, z0, "minecraft:oak_fence") for y in range(1, lamp_y)]
    blocks.append(
        _block(
            x0,
            lamp_y,
            z0,
            style.accent_light_block,
            _light_props(style.accent_light_block, hanging=False),
        )
    )
    return blocks


def _emit_wall_light_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    y = min(interior_size[1], max(2, y0))
    return [
        _block(
            x0,
            y,
            z0,
            "minecraft:wall_torch",
            {"facing": _inward_wall_face(placement, interior_size)},
        )
    ]


def _emit_table_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    *,
    top_block: str,
    leg_block: str = "minecraft:oak_fence",
) -> list[SemanticBlockDict]:
    _, iy, _ = interior_size
    y0 = placement.origin[1]
    blocks: list[SemanticBlockDict] = []
    for x, z in _placement_positions(placement):
        blocks.append(_block(x, y0, z, leg_block))
        if y0 + 1 <= iy:
            blocks.append(_block(x, y0 + 1, z, top_block, _top_props(top_block)))
    return blocks


def _emit_stack_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    *,
    block_id: str,
    height: int = 2,
    properties: dict[str, str] | None = None,
) -> list[SemanticBlockDict]:
    _, iy, _ = interior_size
    blocks: list[SemanticBlockDict] = []
    for x, z in _placement_positions(placement):
        for y in range(placement.origin[1], min(iy, placement.origin[1] + height - 1) + 1):
            blocks.append(_block(x, y, z, block_id, properties))
    return blocks


def _emit_sofa_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    *,
    variant_index: int,
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    facing = _inward_wall_face(placement, interior_size)
    top_block = style.sofa_top_block
    if variant_index % 3 == 1 and style.id == "classic_modular":
        top_block = _SOFA_TOP_VARIANTS[variant_index % len(_SOFA_TOP_VARIANTS)]
    blocks: list[SemanticBlockDict] = []
    for x, z in _placement_positions(placement):
        blocks.append(
            _block(
                x,
                placement.origin[1],
                z,
                style.sofa_base_block,
            )
        )
        if placement.origin[1] + 1 <= interior_size[1]:
            blocks.append(
                _block(
                    x,
                    placement.origin[1] + 1,
                    z,
                    top_block,
                )
            )
    edge = _first_wall_face(placement, interior_size)
    if edge is not None and placement.origin[1] + 1 <= interior_size[1]:
        for x, z in _placement_positions(placement):
            if edge == "north" and z == placement.origin[2]:
                blocks.append(_block(x, placement.origin[1] + 1, z, "minecraft:oak_stairs", {"facing": facing}))
            elif edge == "south" and z == placement.origin[2] + placement.footprint[1] - 1:
                blocks.append(_block(x, placement.origin[1] + 1, z, "minecraft:oak_stairs", {"facing": facing}))
            elif edge == "west" and x == placement.origin[0]:
                blocks.append(_block(x, placement.origin[1] + 1, z, "minecraft:oak_stairs", {"facing": facing}))
            elif edge == "east" and x == placement.origin[0] + placement.footprint[0] - 1:
                blocks.append(_block(x, placement.origin[1] + 1, z, "minecraft:oak_stairs", {"facing": facing}))
    return blocks


def _emit_bookshelf_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    return _emit_stack_blocks(placement, interior_size, block_id="minecraft:bookshelf", height=3)


def _emit_plant_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    blocks = [_block(x0, y0, z0, "minecraft:flower_pot")]
    if y0 + 1 <= interior_size[1]:
        blocks.append(_block(x0, y0 + 1, z0, "minecraft:azalea"))
    return blocks


def _emit_console_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    top_block = (
        "minecraft:spruce_slab"
        if style.id == "classic_modular"
        else style.table_top_block
    )
    blocks = _emit_table_blocks(
        placement,
        interior_size,
        top_block=top_block,
    )
    if placement.origin[1] + 2 <= interior_size[1]:
        x, z = _placement_positions(placement)[0]
        blocks.append(
            _block(
                x,
                placement.origin[1] + 2,
                z,
                style.accent_light_block,
                _light_props(style.accent_light_block, hanging=False),
            )
        )
    return blocks


def _emit_kitchen_counter_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    for x, z in _placement_positions(placement):
        blocks.append(_block(x, placement.origin[1], z, style.kitchen_counter_block))
        if placement.origin[1] + 1 <= interior_size[1]:
            blocks.append(
                _block(
                    x,
                    placement.origin[1] + 1,
                    z,
                    style.kitchen_top_block,
                    _top_props(style.kitchen_top_block),
                )
            )
    return blocks


def _emit_sink_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    blocks = [_block(x0, y0, z0, "minecraft:cauldron")]
    if y0 + 1 <= interior_size[1]:
        blocks.append(_block(x0, y0 + 1, z0, "minecraft:iron_trapdoor", {"half": "top"}))
    return blocks


def _emit_stove_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    blocks = [_block(x0, y0, z0, "minecraft:furnace", {"facing": _inward_wall_face(placement, interior_size)})]
    if y0 + 1 <= interior_size[1]:
        blocks.append(_block(x0, y0 + 1, z0, "minecraft:iron_trapdoor", {"half": "top"}))
    return blocks


def _emit_fridge_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    blocks = _emit_stack_blocks(
        placement,
        interior_size,
        block_id=style.wardrobe_block
        if style.wardrobe_block != "minecraft:barrel"
        else "minecraft:white_concrete",
        height=2,
    )
    if placement.origin[1] + 2 <= interior_size[1]:
        x, z = _placement_positions(placement)[0]
        blocks.append(_block(x, placement.origin[1] + 2, z, "minecraft:iron_block"))
    return blocks


def _emit_dining_nook_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    positions = _placement_positions(placement)
    table_cells = positions[: max(1, len(positions) // 2)]
    chair_cells = positions[max(1, len(positions) // 2):]
    for x, z in table_cells:
        blocks.append(_block(x, placement.origin[1], z, "minecraft:oak_fence"))
        if placement.origin[1] + 1 <= interior_size[1]:
            blocks.append(
                _block(
                    x,
                    placement.origin[1] + 1,
                    z,
                    style.table_top_block,
                    _top_props(style.table_top_block),
                )
            )
    table_centre_x = sum(x for x, _ in table_cells) / len(table_cells)
    table_centre_z = sum(z for _, z in table_cells) / len(table_cells)
    for x, z in chair_cells:
        facing = _face_from_delta(table_centre_x - x, table_centre_z - z)
        blocks.append(_block(x, placement.origin[1], z, "minecraft:oak_stairs", {"facing": facing}))
    return blocks


def _emit_toilet_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    facing = _inward_wall_face(placement, interior_size)
    blocks = [_block(x0, y0, z0, "minecraft:quartz_stairs", {"facing": facing})]
    if y0 + 1 <= interior_size[1]:
        blocks.append(_block(x0, y0 + 1, z0, "minecraft:quartz_block"))
    return blocks


def _emit_shower_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    _, iy, _ = interior_size
    blocks: list[SemanticBlockDict] = []
    positions = _placement_positions(placement)
    for x, z in positions:
        blocks.append(_block(x, placement.origin[1], z, style.bathroom_wall_block))
        if placement.origin[1] + 1 <= iy:
            blocks.append(_block(x, placement.origin[1] + 1, z, style.shower_glass_block))
    if positions and placement.origin[1] + 2 <= iy:
        x, z = positions[0]
        blocks.append(
            _block(
                x,
                placement.origin[1] + 2,
                z,
                style.accent_light_block,
                _light_props(style.accent_light_block, hanging=False),
            )
        )
    return blocks


def _emit_mirror_blocks(
    placement: RoomComponentPlacement,
    interior_size: tuple[int, int, int],
) -> list[SemanticBlockDict]:
    x0, y0, z0 = placement.origin
    y = min(interior_size[1], max(2, y0))
    return [_block(x0, y, z0, "minecraft:glass_pane")]


def _emit_hall_runner_blocks(
    placement: RoomComponentPlacement,
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    for index, (x, z) in enumerate(_placement_positions(placement)):
        if style.id == "classic_modular":
            block_id = (
                "minecraft:gray_carpet"
                if index % 2 == 0
                else "minecraft:light_gray_carpet"
            )
        else:
            block_id = (
                style.rug_primary_block
                if index % 2 == 0
                else style.rug_secondary_block
            )
        blocks.append(_block(x, placement.origin[1], z, block_id))
    return blocks


def _emit_furnished_placement(
    placement: RoomComponentPlacement,
    layout: RoomLayoutPlan,
    placements_by_keyword: dict[str, list[RoomComponentPlacement]],
    *,
    variant_index: int,
    style: InteriorStyleProfile,
) -> list[SemanticBlockDict]:
    if placement.keyword == "bed_core":
        return _emit_bed_blocks(placement, layout.interior_size, block_id=style.bed_block)
    if placement.keyword == "bunk":
        return _emit_bed_blocks(placement, layout.interior_size, block_id="minecraft:blue_bed")
    if placement.keyword == "bedside":
        return _emit_bedside_blocks(placement, layout.interior_size, style)
    if placement.keyword == "wardrobe":
        return _emit_wardrobe_blocks(placement, layout.interior_size, style)
    if placement.keyword == "desk":
        return _emit_desk_blocks(placement, layout.interior_size, style)
    if placement.keyword == "chair":
        return _emit_chair_blocks(placement, placements_by_keyword)
    if placement.keyword == "rug":
        return _emit_rug_blocks(
            placement,
            variant_index=variant_index,
            style=style,
        )
    if placement.keyword == "ceiling_light":
        return _emit_ceiling_light_blocks(placement, layout.interior_size, style)
    if placement.keyword == "extra_light":
        return _emit_floor_lamp_blocks(placement, layout.interior_size, style)
    if placement.keyword == "wall_light":
        return _emit_wall_light_blocks(placement, layout.interior_size)
    if placement.keyword == "sofa":
        return _emit_sofa_blocks(
            placement,
            layout.interior_size,
            variant_index=variant_index,
            style=style,
        )
    if placement.keyword == "coffee_table":
        return _emit_table_blocks(
            placement,
            layout.interior_size,
            top_block=style.table_top_block,
        )
    if placement.keyword == "bookshelf":
        return _emit_bookshelf_blocks(placement, layout.interior_size)
    if placement.keyword == "plant":
        return _emit_plant_blocks(placement, layout.interior_size)
    if placement.keyword == "entry_console":
        return _emit_console_blocks(placement, layout.interior_size, style)
    if placement.keyword in {"coat_storage", "storage_niche"}:
        return _emit_stack_blocks(
            placement,
            layout.interior_size,
            block_id=style.storage_block,
            height=2,
            properties=_storage_props(
                style.storage_block,
                _inward_wall_face(placement, layout.interior_size),
            ),
        )
    if placement.keyword == "shoe_storage":
        return _emit_table_blocks(
            placement,
            layout.interior_size,
            top_block=style.table_top_block,
        )
    if placement.keyword == "kitchen_counter":
        return _emit_kitchen_counter_blocks(placement, layout.interior_size, style)
    if placement.keyword in {"kitchen_sink", "sink"}:
        return _emit_sink_blocks(placement, layout.interior_size)
    if placement.keyword == "stove":
        return _emit_stove_blocks(placement, layout.interior_size)
    if placement.keyword == "fridge":
        return _emit_fridge_blocks(placement, layout.interior_size, style)
    if placement.keyword == "pantry":
        return _emit_stack_blocks(
            placement,
            layout.interior_size,
            block_id=style.storage_block,
            height=3,
            properties=_storage_props(
                style.storage_block,
                _inward_wall_face(placement, layout.interior_size),
            ),
        )
    if placement.keyword == "dining_nook":
        return _emit_dining_nook_blocks(placement, layout.interior_size, style)
    if placement.keyword == "toilet":
        return _emit_toilet_blocks(placement, layout.interior_size)
    if placement.keyword == "shower":
        return _emit_shower_blocks(placement, layout.interior_size, style)
    if placement.keyword == "mirror":
        return _emit_mirror_blocks(placement, layout.interior_size)
    if placement.keyword == "towel_storage":
        return _emit_stack_blocks(
            placement,
            layout.interior_size,
            block_id=style.bathroom_wall_block,
            height=2,
        )
    if placement.keyword == "hall_runner":
        return _emit_hall_runner_blocks(placement, style)
    if placement.keyword == "service_bank":
        return _emit_stack_blocks(placement, layout.interior_size, block_id="minecraft:stone_bricks", height=2)
    if placement.keyword == "utility_lockers":
        return _emit_stack_blocks(placement, layout.interior_size, block_id="minecraft:iron_block", height=2)
    if placement.keyword == "task_table":
        return _emit_table_blocks(
            placement,
            layout.interior_size,
            top_block=style.table_top_block,
        )
    return _simple_component_blocks(placement)


def _within_interior(block: SemanticBlockDict, interior_size: tuple[int, int, int]) -> bool:
    ix, iy, iz = interior_size
    return (
        1 <= int(block["x"]) <= ix
        and 1 <= int(block["y"]) <= iy
        and 1 <= int(block["z"]) <= iz
    )


def _append_unique_block(
    blocks: list[SemanticBlockDict],
    occupied_positions: set[tuple[int, int, int]],
    block: SemanticBlockDict,
) -> None:
    key = (int(block["x"]), int(block["y"]), int(block["z"]))
    if key in occupied_positions:
        return
    occupied_positions.add(key)
    blocks.append(block)


def _furnished_component_blocks(
    layout: RoomLayoutPlan,
    *,
    variant_seed: int = 0,
) -> list[SemanticBlockDict]:
    blocks: list[SemanticBlockDict] = []
    occupied_positions: set[tuple[int, int, int]] = set()
    placements_by_keyword: dict[str, list[RoomComponentPlacement]] = {}
    for placement in layout.placements:
        placements_by_keyword.setdefault(placement.keyword, []).append(placement)

    variant_index = layout_variant_index(layout, variant_seed=variant_seed)
    style = room_interior_style_profile(
        layout.plan.signature.room_type,
        variant_seed=variant_seed,
        variant_index=variant_index,
    )
    for placement in layout.placements:
        for block in _emit_furnished_placement(
            placement,
            layout,
            placements_by_keyword,
            variant_index=variant_index,
            style=style,
        ):
            if _within_interior(block, layout.interior_size):
                _append_unique_block(blocks, occupied_positions, block)
    return blocks


def _component_blocks(
    layout: RoomLayoutPlan,
    *,
    variant_seed: int = 0,
) -> list[SemanticBlockDict]:
    if layout.plan.signature.room_type == "stairwell":
        return emit_stairwell_blocks(layout)
    return _furnished_component_blocks(layout, variant_seed=variant_seed)


class RoomInteriorCache:
    def __init__(self, *, variant_seed: int = 0) -> None:
        self._cache: dict[str, tuple[str, RoomPlan, RoomLayoutPlan, list[SemanticBlockDict]]] = {}
        self._hits = 0
        self._misses = 0
        self._variant_seed = int(variant_seed)

    def build_for_cell(self, cell: SemanticCell, *, utility_type: str) -> RoomInterior:
        request = make_room_request(cell, utility_type=utility_type)
        cache_key = room_cache_key(request, variant_seed=self._variant_seed)
        cached = self._cache.get(cache_key)
        if cached is None:
            plan = plan_room(request)
            layout = plan_room_layout(plan, request)
            variant_index = layout_variant_index(
                layout,
                variant_seed=self._variant_seed,
            )
            style = room_interior_style_profile(
                request.room_type,
                variant_seed=self._variant_seed,
                variant_index=variant_index,
            )
            variant_id = (
                f"grammar:{request.room_type}:{style.id}:v{variant_index}:"
                f"{len(self._cache)}"
            )
            local_blocks = _component_blocks(layout, variant_seed=self._variant_seed)
            self._cache[cache_key] = (variant_id, plan, layout, local_blocks)
            self._misses += 1
        else:
            variant_id, plan, layout, local_blocks = cached
            self._hits += 1

        origin, _ = cell.voxel_bbox
        world_blocks = _translate_blocks(local_blocks, origin)
        return RoomInterior(
            cell_index=cell.cell_index,
            room_type=cell.label,
            variant_id=variant_id,
            cache_key=cache_key,
            voxel_bbox=cell.voxel_bbox,
            blocks=world_blocks,
            signature=request.signature,
            plan=plan,
            layout=layout,
        )

    def stats(self) -> InteriorCacheStats:
        return InteriorCacheStats(hits=self._hits, misses=self._misses)


def _stair_stack_groups(semantic_cells: list[SemanticCell]) -> tuple[tuple[SemanticCell, ...], ...]:
    by_index = {cell.cell_index: cell for cell in semantic_cells if cell.label == "stairwell"}
    groups: list[tuple[SemanticCell, ...]] = []
    seen: set[tuple[int, int, int]] = set()

    for cell in sorted(by_index.values(), key=lambda item: item.cell_index):
        if cell.cell_index in seen:
            continue
        stack: list[SemanticCell] = [cell]
        seen.add(cell.cell_index)

        iy = cell.cell_index[1] + 1
        while True:
            next_index = (cell.cell_index[0], iy, cell.cell_index[2])
            next_cell = by_index.get(next_index)
            if next_cell is None:
                break
            stack.append(next_cell)
            seen.add(next_index)
            iy += 1

        groups.append(tuple(stack))

    return tuple(groups)


def _build_stair_stack_interiors(
    stack: tuple[SemanticCell, ...],
    *,
    utility_type: str,
) -> list[RoomInterior]:
    stack_plan = build_stair_stack_plan(stack)
    by_index = {cell.cell_index: cell for cell in stack}
    interiors: list[RoomInterior] = []

    for cell in stack:
        request = make_room_request(cell, utility_type=utility_type)
        plan = plan_room(request)
        layout = plan_room_layout(plan, request)
        cell_plan = stack_plan.for_cell(cell.cell_index)
        if cell_plan is None:
            raise ValueError(f"missing stair stack plan for {cell.cell_index}")
        origin, _ = cell.voxel_bbox
        interiors.append(
            RoomInterior(
                cell_index=cell.cell_index,
                room_type=cell.label,
                variant_id=f"stair_stack:{stack[0].cell_index[0]}:{stack[0].cell_index[1]}:{stack[0].cell_index[2]}",
                cache_key="stair_stack",
                voxel_bbox=cell.voxel_bbox,
                blocks=_translate_blocks(list(cell_plan.local_blocks), origin),
                signature=request.signature,
                plan=plan,
                layout=layout,
            )
        )

    return interiors


def generate_room_interiors(
    semantic_cells: list[SemanticCell],
    *,
    utility_type: str,
    cache: RoomInteriorCache | None = None,
    variant_seed: int = 0,
) -> tuple[list[RoomInterior], InteriorCacheStats]:
    working_cache = cache or RoomInteriorCache(variant_seed=variant_seed)
    stairwell_indices = {cell.cell_index for cell in semantic_cells if cell.label == "stairwell"}
    interiors: list[RoomInterior] = []
    for stack in _stair_stack_groups(semantic_cells):
        interiors.extend(_build_stair_stack_interiors(stack, utility_type=utility_type))
    for cell in semantic_cells:
        if cell.cell_index in stairwell_indices:
            continue
        interiors.append(working_cache.build_for_cell(cell, utility_type=utility_type))
    interiors.sort(key=lambda room: room.cell_index)
    return interiors, working_cache.stats()


__all__ = [
    "InteriorCacheStats",
    "InteriorStyleProfile",
    "INTERIOR_STYLE_PROFILES",
    "COMPONENT_LIBRARY",
    "ROOM_STYLE_VARIANTS",
    "RoomInteriorCache",
    "derive_room_signature",
    "generate_room_interiors",
    "interior_style_profile",
    "layout_variant_index",
    "make_room_request",
    "plan_room_layout",
    "plan_room",
    "room_interior_style_profile",
    "room_cache_key",
    "room_style_variant",
    "room_variant_index",
]
