"""Pod-type registry and face-category compatibility table.

Coarse face categories used across the WFC face-signature lookup. The actual
voxel-level face signature from ``voxel_renderer.prefab.face_signature`` is
*not* used at solve time in v1 — the categorical equivalence here is what
WFC operates on. Voxel-level signatures are reserved for materialisation
sanity checks.

Pod-type face profile
---------------------
Each pod-type carries a tuple of 6 face categories (north, east, south, west,
up, down) at canonical rotation 0. Rotated tiles are a derived view: rotation
``r`` permutes the horizontal four cyclically by ``r`` steps.

Face categories
---------------
- ``WALL``   — solid wall
- ``DOOR``   — door opening (interior connection)
- ``WINDOW`` — window opening (exterior contact preferred)
- ``OPEN``   — fully open face (corridors, balconies)
- ``FLOOR``  — bottom face of a habitable cell (rests on something below)
- ``CEILING``— top face of a habitable cell (roof or floor of cell above)
- ``EXTERIOR``— sentinel for the boundary face of the cell grid
- ``EMPTY``  — sentinel for empty cells (no pod)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# --- Face categories (small int domain → fits int8) --------------------------

WALL: Final[int] = 0
DOOR: Final[int] = 1
WINDOW: Final[int] = 2
OPEN: Final[int] = 3
FLOOR: Final[int] = 4
CEILING: Final[int] = 5
EXTERIOR: Final[int] = 6
EMPTY: Final[int] = 7

NUM_FACE_CATEGORIES: Final[int] = 8

FACE_CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "WALL", "DOOR", "WINDOW", "OPEN", "FLOOR", "CEILING", "EXTERIOR", "EMPTY",
)

# --- Pod-type registry (M1) --------------------------------------------------

POD_ENTRY: Final[str] = "entry"
POD_LIVING: Final[str] = "living"
POD_KITCHEN: Final[str] = "kitchen"
POD_BATHROOM: Final[str] = "bathroom"
POD_BEDROOM: Final[str] = "bedroom"
POD_CORRIDOR: Final[str] = "corridor"
POD_STAIRWELL: Final[str] = "stairwell"

# Special non-occupancy sentinels (not real pods, but represented as tile IDs).
POD_STRUCTURAL_VOID: Final[str] = "_structural_void"
POD_TERRACE_VOID: Final[str] = "_terrace_void"

POD_LABELS: Final[tuple[str, ...]] = (
    POD_STRUCTURAL_VOID,  # 0
    POD_TERRACE_VOID,     # 1
    POD_ENTRY,            # 2
    POD_LIVING,           # 3
    POD_KITCHEN,          # 4
    POD_BATHROOM,         # 5
    POD_BEDROOM,          # 6
    POD_CORRIDOR,         # 7
    POD_STAIRWELL,        # 8
)
POD_INDEX: Final[dict[str, int]] = {n: i for i, n in enumerate(POD_LABELS)}

# Roles inform scoring (privacy depth, daylight target set, occupancy capacity).
ROLE_HABITABLE: Final[str] = "habitable"
ROLE_SERVICE: Final[str] = "service"
ROLE_CIRCULATION: Final[str] = "circulation"
ROLE_EXTERIOR: Final[str] = "exterior"

POD_ROLE: Final[tuple[str, ...]] = (
    ROLE_EXTERIOR,    # _structural_void
    ROLE_EXTERIOR,    # _terrace_void
    ROLE_CIRCULATION, # entry
    ROLE_HABITABLE,   # living
    ROLE_HABITABLE,   # kitchen
    ROLE_SERVICE,     # bathroom
    ROLE_HABITABLE,   # bedroom
    ROLE_CIRCULATION, # corridor
    ROLE_CIRCULATION, # stairwell
)

# Default occupancy capacity per pod (used by functional-adequacy scoring).
POD_OCCUPANCY: Final[tuple[int, ...]] = (
    0,  # _structural_void
    0,  # _terrace_void
    0,  # entry
    0,  # living  (gathering, not sleeping)
    0,  # kitchen
    0,  # bathroom
    2,  # bedroom (one bed-budget cell sleeps up to 2)
    0,  # corridor
    0,  # stairwell
)


# Per-pod horizontal voxel-size multiplier, keyed by ``POD_INDEX``. Applied
# by the spatial-layout factory to base cell width/depth (xz only); storey
# height (y) is invariant in v1 to keep stair-block geometry simple.
#
# A multiplier of 1.0 reproduces the uniform-layout dimensions. Values >1
# inflate cells of that pod-type along the horizontal axes; the banded
# layout factory aggregates per-column maxima so neighbouring cells in
# the same ix-column or iz-row keep matching face areas (otherwise shared
# walls would not align).
#
# v1 defaults: all 1.0 — mechanism is wired but produces parity output
# until callers override. Tunable per pod in subsequent commits.
POD_SIZE_MULTIPLIER: Final[tuple[float, ...]] = (
    1.0,  # _structural_void
    1.0,  # _terrace_void
    1.0,  # entry
    1.0,  # living
    1.0,  # kitchen
    1.0,  # bathroom
    1.0,  # bedroom
    1.0,  # corridor
    1.0,  # stairwell
)


@dataclass(frozen=True, slots=True)
class PodFaceProfile:
    """Face categories at canonical rotation 0, ordered (N, E, S, W, U, D)."""

    pod_index: int
    faces: tuple[int, int, int, int, int, int]
    has_window: bool
    has_door: bool
    needs_floor_support: bool   # cell below must be habitable/circulation/empty-with-ground
    is_top_capable: bool        # may sit at top storey (its UP face is roof-like)


def _profile(
    pod: str,
    n: int, e: int, s: int, w: int, u: int, d: int,
    needs_floor_support: bool = True,
    is_top_capable: bool = True,
) -> PodFaceProfile:
    faces = (n, e, s, w, u, d)
    return PodFaceProfile(
        pod_index=POD_INDEX[pod],
        faces=faces,
        has_window=any(f == WINDOW for f in faces),
        has_door=any(f == DOOR for f in faces),
        needs_floor_support=needs_floor_support,
        is_top_capable=is_top_capable,
    )


# Canonical face profiles. Rotational variants are *derived* by rotating the
# horizontal four faces cyclically — see :func:`profile_at_rotation`.
POD_PROFILES: Final[dict[str, PodFaceProfile]] = {
    POD_STRUCTURAL_VOID: _profile(
        POD_STRUCTURAL_VOID,
        EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
        needs_floor_support=False,
    ),
    POD_TERRACE_VOID: _profile(
        POD_TERRACE_VOID,
        WINDOW, WINDOW, WINDOW, WINDOW, EMPTY, EMPTY,
        needs_floor_support=False,
    ),
    # Entry is north-facing by default (front door towards -z).
    POD_ENTRY: _profile(
        POD_ENTRY,
        DOOR, WALL, DOOR, WALL, CEILING, FLOOR,
    ),
    POD_LIVING: _profile(
        POD_LIVING,
        WINDOW, WALL, DOOR, WINDOW, CEILING, FLOOR,
    ),
    POD_KITCHEN: _profile(
        POD_KITCHEN,
        WINDOW, WALL, DOOR, WALL, CEILING, FLOOR,
    ),
    POD_BATHROOM: _profile(
        POD_BATHROOM,
        WALL, WALL, DOOR, WINDOW, CEILING, FLOOR,
    ),
    POD_BEDROOM: _profile(
        POD_BEDROOM,
        WINDOW, WALL, DOOR, WALL, CEILING, FLOOR,
    ),
    POD_CORRIDOR: _profile(
        POD_CORRIDOR,
        DOOR, OPEN, DOOR, OPEN, CEILING, FLOOR,
    ),
    POD_STAIRWELL: _profile(
        POD_STAIRWELL,
        WALL, DOOR, WALL, WALL, OPEN, OPEN,  # vertical pass-through
        is_top_capable=True,
    ),
}


def profile_at_rotation(pod: str, rotation_steps: int) -> tuple[int, int, int, int, int, int]:
    """Rotate horizontal faces cyclically; vertical faces are invariant."""
    base = POD_PROFILES[pod].faces
    r = rotation_steps % 4
    if r == 0:
        return base
    n, e, s, w, u, d = base
    horiz = (n, e, s, w)
    rotated = horiz[-r:] + horiz[:-r]   # rotate clockwise (north -> east)
    return (rotated[0], rotated[1], rotated[2], rotated[3], u, d)


# --- Face-category compatibility table ---------------------------------------
# Symmetric. ``compat[a, b] == True`` iff face A on cell P can sit opposite
# face B on cell Q across a shared cell-grid face.

def _build_compat_table(*, allow_floor_empty: bool = False) -> np.ndarray:
    n = NUM_FACE_CATEGORIES
    t = np.zeros((n, n), dtype=bool)

    def allow(a: int, b: int) -> None:
        t[a, b] = True
        t[b, a] = True

    # Empty ↔ Empty is fine (gap inside grid).
    allow(EMPTY, EMPTY)
    # Empty ↔ EXTERIOR boundary — we don't care; EMPTY at boundary is fine.
    allow(EMPTY, EXTERIOR)
    # Empty next to a real face: only WALL or EXTERIOR-flavour faces tolerate
    # an empty neighbour. Specifically, a WALL meeting EMPTY is just an
    # exterior wall facing nothing — acceptable.
    allow(EMPTY, WALL)
    allow(EMPTY, WINDOW)        # window into the void = legitimate exterior
    # Doors and OPEN faces need a real adjacent pod to function.
    # → DOOR/OPEN ↔ EMPTY = forbidden (left as default False).
    # FLOOR ↔ EMPTY: normally forbidden — a habitable pod must rest on a
    # non-EMPTY cell below or on the ground (FLOOR ↔ EXTERIOR at storey
    # 0). Quirky profiles may opt in so laterally-supported cantilevers
    # can be searched, with the structural score still penalising truly
    # unsupported islands.
    if allow_floor_empty:
        allow(EMPTY, FLOOR)
    # CEILING ↔ EMPTY remains allowed: a habitable pod may have an
    # EMPTY cell above (top-storey roof or rooftop terrace).
    allow(EMPTY, CEILING)

    # WALL ↔ WALL — abuts another wall (interior or shared between pods).
    allow(WALL, WALL)
    # WALL ↔ EXTERIOR — exterior face of a wall.
    allow(WALL, EXTERIOR)

    # DOOR ↔ DOOR — two pods linked by a doorway.
    allow(DOOR, DOOR)
    # DOOR ↔ OPEN — door opens into a corridor.
    allow(DOOR, OPEN)
    # DOOR ↔ EXTERIOR — front door (entry pod). Permitted at category level;
    # the programme resolver enforces "only entry has an exterior door" via
    # tile-domain pruning at boundary cells.
    allow(DOOR, EXTERIOR)

    # WINDOW ↔ EXTERIOR — windows look outside.
    allow(WINDOW, EXTERIOR)
    # WINDOW ↔ EMPTY already allowed.
    # WINDOW ↔ WINDOW — two windows back-to-back is acceptable for shared
    # exterior pockets like courtyards (rare in v1 but harmless to allow).
    allow(WINDOW, WINDOW)

    # OPEN ↔ OPEN — corridor → corridor or balcony → walkway.
    allow(OPEN, OPEN)
    # OPEN ↔ EXTERIOR — balcony/corridor opening to outside is legitimate.
    allow(OPEN, EXTERIOR)
    # OPEN ↔ DOOR already allowed.

    # FLOOR ↔ CEILING — vertical stacking of pods.
    allow(FLOOR, CEILING)
    # FLOOR ↔ EXTERIOR (downward at storey 0) — ground.
    allow(FLOOR, EXTERIOR)
    # CEILING ↔ EXTERIOR (upward at top storey) — roof.
    allow(CEILING, EXTERIOR)
    # CEILING ↔ CEILING — two cells staring at each other vertically: forbid
    # (would imply a missing storey between, structurally implausible).
    # FLOOR ↔ FLOOR — same: forbidden.
    # FLOOR ↔ CEILING handled above.

    return t


FACE_CATEGORY_COMPAT: Final[np.ndarray] = _build_compat_table()


VOID_POD_INDICES: Final[tuple[int, int]] = (
    POD_INDEX[POD_STRUCTURAL_VOID],
    POD_INDEX[POD_TERRACE_VOID],
)


def is_void_pod_index(pod_index: int) -> bool:
    return int(pod_index) in VOID_POD_INDICES


def build_face_category_compat(*, allow_floor_empty: bool = False) -> np.ndarray:
    """Return a compatibility table for the requested support strictness."""
    return _build_compat_table(allow_floor_empty=allow_floor_empty)


def categories_compatible(a: int, b: int) -> bool:
    return bool(FACE_CATEGORY_COMPAT[a, b])


__all__ = [
    "CEILING",
    "DOOR",
    "EMPTY",
    "EXTERIOR",
    "FACE_CATEGORY_COMPAT",
    "FACE_CATEGORY_NAMES",
    "FLOOR",
    "NUM_FACE_CATEGORIES",
    "OPEN",
    "POD_BATHROOM",
    "POD_BEDROOM",
    "POD_CORRIDOR",
    "POD_STRUCTURAL_VOID",
    "POD_TERRACE_VOID",
    "POD_ENTRY",
    "POD_INDEX",
    "POD_KITCHEN",
    "POD_LABELS",
    "POD_LIVING",
    "POD_OCCUPANCY",
    "POD_PROFILES",
    "POD_ROLE",
    "POD_SIZE_MULTIPLIER",
    "POD_STAIRWELL",
    "PodFaceProfile",
    "ROLE_CIRCULATION",
    "ROLE_EXTERIOR",
    "ROLE_HABITABLE",
    "ROLE_SERVICE",
    "VOID_POD_INDICES",
    "WALL",
    "WINDOW",
    "build_face_category_compat",
    "categories_compatible",
    "is_void_pod_index",
    "profile_at_rotation",
]
