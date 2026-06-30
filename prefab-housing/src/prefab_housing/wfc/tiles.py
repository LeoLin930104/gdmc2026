"""Tile set construction and face-compatibility precompute.

A *tile* is a pair ``(pod_index, rotation_steps)`` packed into an integer
``tile_id``. Hot loops index packed numpy arrays; the named constants in
:mod:`prefab_housing.catalogue.pod_types` exist only for boundary readability.

Layout (DOD)
------------
``TileSet`` holds parallel arrays of length ``T`` (number of tiles):

- ``pod_index : int8[T]``
- ``rotation  : int8[T]``               (0..3)
- ``faces     : int8[T, 6]``             face category per face index
- ``has_window : bool[T]``
- ``has_door   : bool[T]``
- ``occupancy  : int8[T]``
- ``role       : int8[T]``               role enum index

Plus the precomputed compatibility table:

- ``compat : bool[T, 6, T+1]``  where index ``T`` denotes the EXTERIOR sentinel.

Indexing rule: ``compat[a, f, b]`` means "tile ``a``'s face ``f`` may sit
opposite tile ``b``'s opposite face". The EXTERIOR slot is queried only at
grid boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from prefab_housing.catalogue import pod_types as pt
from prefab_housing.grid import NUM_FACES, OPPOSITE_FACE

# Role enum packed as int8.
ROLE_INDEX: Final[dict[str, int]] = {
    pt.ROLE_EXTERIOR: 0,
    pt.ROLE_HABITABLE: 1,
    pt.ROLE_SERVICE: 2,
    pt.ROLE_CIRCULATION: 3,
}


@dataclass(frozen=True, slots=True)
class TileSet:
    pod_index: np.ndarray         # int8[T]
    rotation: np.ndarray          # int8[T]
    faces: np.ndarray             # int8[T, 6]
    has_window: np.ndarray        # bool[T]
    has_door: np.ndarray          # bool[T]
    occupancy: np.ndarray         # int8[T]
    role: np.ndarray              # int8[T]
    compat: np.ndarray            # bool[T, 6, T+1]
    structural_void_tile_id: int  # the structural void pod, rotation 0
    terrace_void_tile_id: int     # the terrace void pod, rotation 0
    pod_to_tiles: dict[int, tuple[int, ...]]  # pod_index → tuple of tile_ids
    tile_label: tuple[str, ...]   # human-readable label per tile

    @property
    def num_tiles(self) -> int:
        return int(self.pod_index.shape[0])


def build_tile_set(*, allow_floor_empty: bool = False) -> TileSet:
    """Construct the v1 tile set: every (pod, rotation) tuple plus EMPTY.

    Each pod gets 4 rotation variants, even when its profile is rotation-
    symmetric. Eliminating equivalent tiles is a v2 optimisation; the
    redundancy here is small (~32 tiles vs ~12 minimal) and keeps code paths
    simple.
    """
    pods = pt.POD_LABELS
    tiles_pod: list[int] = []
    tiles_rot: list[int] = []
    tile_labels: list[str] = []

    # Void tiles appear once each (rotation-invariant).
    tiles_pod.append(pt.POD_INDEX[pt.POD_STRUCTURAL_VOID])
    tiles_rot.append(0)
    tile_labels.append("_structural_void@0")
    tiles_pod.append(pt.POD_INDEX[pt.POD_TERRACE_VOID])
    tiles_rot.append(0)
    tile_labels.append("_terrace_void@0")

    for pod in pods:
        if pod in (pt.POD_STRUCTURAL_VOID, pt.POD_TERRACE_VOID):
            continue
        for r in range(4):
            tiles_pod.append(pt.POD_INDEX[pod])
            tiles_rot.append(r)
            tile_labels.append(f"{pod}@{r * 90}")

    T = len(tiles_pod)
    pod_arr = np.array(tiles_pod, dtype=np.int8)
    rot_arr = np.array(tiles_rot, dtype=np.int8)

    faces = np.zeros((T, NUM_FACES), dtype=np.int8)
    has_window = np.zeros(T, dtype=bool)
    has_door = np.zeros(T, dtype=bool)
    occupancy = np.zeros(T, dtype=np.int8)
    role = np.zeros(T, dtype=np.int8)

    for t in range(T):
        pod_label = pt.POD_LABELS[int(pod_arr[t])]
        prof = pt.profile_at_rotation(pod_label, int(rot_arr[t]))
        faces[t] = prof
        has_window[t] = pt.WINDOW in prof
        has_door[t] = pt.DOOR in prof
        occupancy[t] = pt.POD_OCCUPANCY[int(pod_arr[t])]
        role[t] = ROLE_INDEX[pt.POD_ROLE[int(pod_arr[t])]]

    pod_to_tiles: dict[int, list[int]] = {}
    for t, p in enumerate(pod_arr.tolist()):
        pod_to_tiles.setdefault(int(p), []).append(t)

    structural_void_tile_id = pod_to_tiles[pt.POD_INDEX[pt.POD_STRUCTURAL_VOID]][0]
    terrace_void_tile_id = pod_to_tiles[pt.POD_INDEX[pt.POD_TERRACE_VOID]][0]

    compat = _build_compat_table(faces, allow_floor_empty=allow_floor_empty)

    return TileSet(
        pod_index=pod_arr,
        rotation=rot_arr,
        faces=faces,
        has_window=has_window,
        has_door=has_door,
        occupancy=occupancy,
        role=role,
        compat=compat,
        structural_void_tile_id=structural_void_tile_id,
        terrace_void_tile_id=terrace_void_tile_id,
        pod_to_tiles={k: tuple(v) for k, v in pod_to_tiles.items()},
        tile_label=tuple(tile_labels),
    )


# Indexing convention: compat[a, f, T] = compatibility against EXTERIOR.
# We allocate one extra column on axis 2 for that sentinel.
EXTERIOR_TILE_INDEX_OFFSET: Final[int] = 0  # placeholder; resolved at runtime as T


def _build_compat_table(faces: np.ndarray, *, allow_floor_empty: bool = False) -> np.ndarray:
    """Precompute boolean table compat[a, f, b] including EXTERIOR sentinel.

    ``compat[a, f, T] == True`` iff tile ``a``'s face ``f`` can sit at the grid
    boundary (i.e. its category is compatible with EXTERIOR).
    """
    T = faces.shape[0]
    compat = np.zeros((T, NUM_FACES, T + 1), dtype=bool)
    cat_compat = pt.build_face_category_compat(allow_floor_empty=allow_floor_empty)

    for a in range(T):
        for f in range(NUM_FACES):
            ca = int(faces[a, f])
            opp_f = OPPOSITE_FACE[f]
            # Tile-vs-tile compat
            for b in range(T):
                cb = int(faces[b, opp_f])
                compat[a, f, b] = cat_compat[ca, cb]
            # Tile-vs-EXTERIOR
            compat[a, f, T] = cat_compat[ca, pt.EXTERIOR]

    return compat


__all__ = [
    "ROLE_INDEX",
    "TileSet",
    "build_tile_set",
]
