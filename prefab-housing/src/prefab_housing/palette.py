"""Material palette registry, keyed by ``material_theme``.

Slot names are referenced by catalogue templates; resolution happens at
materialisation time. v1 ships ``sci_fi_modular`` only.

Slot taxonomy
-------------
- Shell slots: ``wall_exterior`` / ``wall_interior`` / ``floor`` / ``ceiling``
  / ``window_*`` / ``door_frame`` / ``structural_strut``.
- Pod-utility accents: ``pod_accent_<pod_name>`` — legacy planning/preview
  colours. The exterior shell no longer uses these to label room types.
- Whole-house decoration: ``roof_block`` / ``roof_stair`` (top-storey cap),
  ``trim_band`` (inter-storey horizontal band), ``foundation`` (storey-0
  ground course).

Adding a slot is non-breaking iff every theme provides a value or the
caller resolves with ``palette.get(slot)`` rather than indexing.
"""

from __future__ import annotations

from typing import Final

# --- Shell slots ------------------------------------------------------------
SLOT_WALL_EXTERIOR: Final[str] = "wall_exterior"
SLOT_WALL_INTERIOR: Final[str] = "wall_interior"
SLOT_FLOOR: Final[str] = "floor"
SLOT_CEILING: Final[str] = "ceiling"
SLOT_WINDOW_FRAME: Final[str] = "window_frame"
SLOT_WINDOW_GLASS: Final[str] = "window_glass"
SLOT_DOOR_FRAME: Final[str] = "door_frame"
SLOT_STRUT: Final[str] = "structural_strut"
SLOT_FRAME_STAIR: Final[str] = "frame_stair"
SLOT_FRAME_BLOCK: Final[str] = "frame_block"

# --- Whole-house decoration slots -------------------------------------------
SLOT_ROOF_BLOCK: Final[str] = "roof_block"
SLOT_ROOF_STAIR: Final[str] = "roof_stair"
SLOT_TRIM_BAND: Final[str] = "trim_band"
SLOT_FOUNDATION: Final[str] = "foundation"

# --- Pod-utility accent slots -----------------------------------------------
# Keyed by pod name (see ``catalogue.pod_types.POD_LABELS``).
def pod_accent_slot(pod_name: str) -> str:
    return f"pod_accent_{pod_name}"


DEFAULT_THEME: Final[str] = "sci_fi_modular"

PALETTE_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "sci_fi_modular": {
        # Active surface: white-concrete pod walls + black-concrete outline
        # ring + neutral glass inset + dark-oak crown roof.
        SLOT_WALL_EXTERIOR: "minecraft:white_concrete",
        SLOT_ROOF_BLOCK:    "minecraft:dark_oak_planks",
        SLOT_ROOF_STAIR:    "minecraft:dark_oak_stairs",
        # Pod outline frame: black concrete traces a proud-out 1-voxel
        # ring around every wall face.
        SLOT_FRAME_BLOCK:   "minecraft:black_concrete",
        # Legacy pod accents retained for topology previews and compatibility.
        pod_accent_slot("entry"):     "minecraft:orange_concrete",
        pod_accent_slot("living"):    "minecraft:red_concrete",
        pod_accent_slot("kitchen"):   "minecraft:yellow_concrete",
        pod_accent_slot("bathroom"):  "minecraft:light_blue_concrete",
        pod_accent_slot("bedroom"):   "minecraft:purple_concrete",
        pod_accent_slot("corridor"):  "minecraft:gray_concrete",
        pod_accent_slot("stairwell"): "minecraft:lime_concrete",
        # Slots intentionally absent from v1 (resolved via palette.get,
        # so consumers gracefully no-op): wall_interior, floor, ceiling,
        # window_frame, window_glass, door_frame, structural_strut,
        # frame_block, frame_stair, trim_band, foundation. Re-introduce
        # alongside the v2 interior / aperture work.
    },
}


def resolve_palette(theme: str | None) -> dict[str, str]:
    """Return the palette for ``theme``, falling back to the default."""
    key = theme or DEFAULT_THEME
    if key not in PALETTE_REGISTRY:
        raise KeyError(
            f"unknown material_theme {key!r}; available: {sorted(PALETTE_REGISTRY.keys())}"
        )
    return PALETTE_REGISTRY[key]


__all__ = [
    "DEFAULT_THEME",
    "PALETTE_REGISTRY",
    "SLOT_CEILING",
    "SLOT_DOOR_FRAME",
    "SLOT_FLOOR",
    "SLOT_FOUNDATION",
    "SLOT_FRAME_BLOCK",
    "SLOT_FRAME_STAIR",
    "SLOT_ROOF_BLOCK",
    "SLOT_ROOF_STAIR",
    "SLOT_STRUT",
    "SLOT_TRIM_BAND",
    "SLOT_WALL_EXTERIOR",
    "SLOT_WALL_INTERIOR",
    "SLOT_WINDOW_FRAME",
    "SLOT_WINDOW_GLASS",
    "pod_accent_slot",
    "resolve_palette",
]
