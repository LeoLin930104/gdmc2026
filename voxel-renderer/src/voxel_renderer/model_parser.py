"""
Minecraft Block Model Parser — Phase 2

Parses Minecraft JSON block models (1.21+) into trimesh geometry with
texture-mapped faces.  Handles:
  - Parent model inheritance (recursive resolution).
  - Element cuboids with per-face UV mapping.
  - Y-axis rotation (from blockstate apply entries).
  - Texture variable resolution (#texture → actual path).

All geometry is produced in a unit-cube coordinate space: Minecraft's
[0, 16] pixel grid maps to [0, 1] world units.

Blockstate resolution (multipart + variants) is handled by
``blockstate_resolver.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

from voxel_renderer.assets import get_asset_root

_ASSETS_ROOT = str(get_asset_root())
_MODELS_DIR = os.path.join(_ASSETS_ROOT, "models", "block")
_TEXTURES_DIR = os.path.join(_ASSETS_ROOT, "textures", "block")

_TEXTURES_ENABLED: bool = True
try:
    from PIL import Image as PILImage
except ImportError:
    _TEXTURES_ENABLED = False

# ---------------------------------------------------------------------------
# Model JSON loading + parent resolution
# ---------------------------------------------------------------------------

# Face direction → axis-aligned normal (Minecraft Y-up, right-handed)
_FACE_NORMALS: dict[str, np.ndarray] = {
    "down": np.array([0, -1, 0], dtype=float),
    "up": np.array([0, 1, 0], dtype=float),
    "north": np.array([0, 0, -1], dtype=float),
    "south": np.array([0, 0, 1], dtype=float),
    "west": np.array([-1, 0, 0], dtype=float),
    "east": np.array([1, 0, 0], dtype=float),
}


@lru_cache(maxsize=2048)
def _load_model_json(name: str) -> dict[str, Any] | None:
    """
    Load a model JSON by Minecraft resource name.

    Accepts formats:
      - "minecraft:block/oak_fence_post" → models/block/oak_fence_post.json
      - "block/oak_fence_post" → models/block/oak_fence_post.json
      - "oak_fence_post" → models/block/oak_fence_post.json
    """
    # Normalise name
    name = name.removeprefix("minecraft:")
    name = name.removeprefix("block/")

    path = os.path.join(_MODELS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to load model '%s': %s", name, e)
        return None


def _resolve_model(name: str, _depth: int = 0) -> dict[str, Any] | None:
    """
    Recursively resolve a model, merging parent fields.

    Returns a merged dict with at minimum 'textures' and 'elements'.
    Inheritance rules:
      - 'elements' from the most-derived model that defines them.
      - 'textures' are merged (child overrides parent).
    """
    if _depth > 10:
        logger.warning("Model parent chain exceeded 10 levels for '%s'", name)
        return None

    raw = _load_model_json(name)
    if raw is None:
        return None

    parent_name = raw.get("parent")
    if parent_name:
        parent = _resolve_model(parent_name, _depth + 1)
        if parent:
            merged: dict[str, Any] = {}
            # Textures: parent first, child overrides
            merged["textures"] = {
                **parent.get("textures", {}),
                **raw.get("textures", {}),
            }
            # Elements: use child's if present, else parent's
            if "elements" in raw:
                merged["elements"] = raw["elements"]
            elif "elements" in parent:
                merged["elements"] = parent["elements"]
            return merged

    # No parent or parent not found — return as-is
    return {
        "textures": raw.get("textures", {}),
        "elements": raw.get("elements", []),
    }


# ---------------------------------------------------------------------------
# Texture resolution
# ---------------------------------------------------------------------------


def _resolve_texture_ref(ref: str, textures: dict[str, str], _depth: int = 0) -> str | None:
    """
    Resolve a texture reference like '#texture' through the texture map.

    Returns a plain texture name (e.g. 'oak_planks') or None.
    """
    if _depth > 10:
        return None

    if ref.startswith("#"):
        key = ref[1:]
        if key in textures:
            return _resolve_texture_ref(textures[key], textures, _depth + 1)
        return None

    # It's a direct path like "minecraft:block/oak_planks"
    ref = ref.removeprefix("minecraft:")
    ref = ref.removeprefix("block/")
    return ref


@lru_cache(maxsize=1024)
def _load_texture_image(name: str) -> "PILImage.Image | None":
    """Load a texture PNG, returning a 16x16 RGBA image or None."""
    if not _TEXTURES_ENABLED:
        return None
    path = os.path.join(_TEXTURES_DIR, f"{name}.png")
    if not os.path.exists(path):
        return None
    try:
        img = PILImage.open(path).convert("RGBA")
        return img.resize((16, 16), PILImage.Resampling.NEAREST)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cuboid → trimesh conversion
# ---------------------------------------------------------------------------

# Face vertex ordering for a cuboid defined by (x0,y0,z0)→(x1,y1,z1)
# Each face has 4 vertices, wound CCW when viewed from outside.
# Coords are in Minecraft pixel space [0,16] — caller scales to [0,1].


def _cuboid_face_verts(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    face: str,
) -> np.ndarray:
    """Return 4 vertices (CCW from outside) for one face of a cuboid."""
    V = {
        "down": [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        "up": [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        "north": [[x1, y1, z0], [x0, y1, z0], [x0, y0, z0], [x1, y0, z0]],
        "south": [[x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [x0, y0, z1]],
        "west": [[x0, y1, z0], [x0, y1, z1], [x0, y0, z1], [x0, y0, z0]],
        "east": [[x1, y1, z1], [x1, y1, z0], [x1, y0, z0], [x1, y0, z1]],
    }
    return np.array(V[face], dtype=float)


def _face_uv_coords(
    uv: list[float] | None,
    face: str,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
) -> np.ndarray:
    """
    Compute UV coordinates for a face.

    If explicit UV is given [u0, v0, u1, v1] (in pixel space 0-16),
    normalise to [0,1].  Otherwise auto-derive from cuboid extents.
    """
    if uv is not None and len(uv) == 4:
        u0, v0, u1, v1 = uv[0] / 16.0, uv[1] / 16.0, uv[2] / 16.0, uv[3] / 16.0
    else:
        # Auto-derive UV from face extents
        auto = {
            "down": [x0, z0, x1, z1],
            "up": [x0, z0, x1, z1],
            "north": [x0, y0, x1, y1],
            "south": [x0, y0, x1, y1],
            "west": [z0, y0, z1, y1],
            "east": [z0, y0, z1, y1],
        }
        coords = auto.get(face, [0, 0, 16, 16])
        u0, v0, u1, v1 = (
            coords[0] / 16.0,
            coords[1] / 16.0,
            coords[2] / 16.0,
            coords[3] / 16.0,
        )

    # 4 vertices: TL, TR, BR, BL → standard quad UV mapping
    # Matching the CCW vertex order from _cuboid_face_verts.
    # V is flipped (1 - v) because Minecraft has V=0 at top, OpenGL at bottom.
    return np.array(
        [
            [u0, 1 - v0],  # 0: top-left
            [u1, 1 - v0],  # 1: top-right
            [u1, 1 - v1],  # 2: bottom-right
            [u0, 1 - v1],  # 3: bottom-left
        ],
        dtype=float,
    )


def _build_element_mesh(
    element: dict[str, Any],
    textures: dict[str, str],
) -> trimesh.Trimesh | None:
    """
    Convert a single Minecraft model element into a trimesh.

    Returns None if the element has no renderable faces.
    """
    from_xyz = element.get("from", [0, 0, 0])
    to_xyz = element.get("to", [16, 16, 16])
    faces_def = element.get("faces", {})

    if not faces_def:
        return None

    x0, y0, z0 = float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2])
    x1, y1, z1 = float(to_xyz[0]), float(to_xyz[1]), float(to_xyz[2])

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_uvs: list[np.ndarray] = []
    all_tex_images: list[Any] = []
    vertex_offset = 0

    for face_name, face_data in faces_def.items():
        if face_name not in _FACE_NORMALS:
            continue

        tex_ref = face_data.get("texture", "")
        tex_name = _resolve_texture_ref(tex_ref, textures)
        tex_img = _load_texture_image(tex_name) if tex_name else None

        verts = _cuboid_face_verts(x0, y0, z0, x1, y1, z1, face_name)
        # Scale from pixel space [0, 16] to unit space [0, 1]
        verts /= 16.0

        uv_raw = face_data.get("uv")
        uvs = _face_uv_coords(uv_raw, face_name, x0, y0, z0, x1, y1, z1)

        # Two triangles per quad.
        # down/up vertices are CCW from outside → standard [0,1,2],[0,2,3].
        # Side faces (north/south/west/east) are CW → reverse triangulation.
        if face_name in ("down", "up"):
            f = np.array(
                [
                    [vertex_offset, vertex_offset + 1, vertex_offset + 2],
                    [vertex_offset, vertex_offset + 2, vertex_offset + 3],
                ]
            )
        else:
            f = np.array(
                [
                    [vertex_offset, vertex_offset + 2, vertex_offset + 1],
                    [vertex_offset, vertex_offset + 3, vertex_offset + 2],
                ]
            )

        all_verts.append(verts)
        all_faces.append(f)
        all_uvs.append(uvs)
        all_tex_images.append(tex_img)
        vertex_offset += 4

    if not all_verts:
        return None

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    uvs = np.vstack(all_uvs)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Build a combined texture atlas if we have textures
    # Each face gets a 16x16 tile in a horizontal atlas
    unique_textures = []
    for img in all_tex_images:
        if img is not None and img not in unique_textures:
            unique_textures.append(img)

    if unique_textures and _TEXTURES_ENABLED:
        # Build atlas: one 16x16 tile per unique texture
        n_tex = len(unique_textures)
        atlas_w = 16 * n_tex
        atlas = PILImage.new("RGBA", (atlas_w, 16))
        tex_index_map: dict[int, int] = {}  # id(img) → atlas index

        for idx, img in enumerate(unique_textures):
            atlas.paste(img, (idx * 16, 0))
            tex_index_map[id(img)] = idx

        # Remap UVs to atlas space
        remapped_uvs = np.copy(uvs)
        face_idx = 0
        for img in all_tex_images:
            if img is not None and id(img) in tex_index_map:
                atlas_idx = tex_index_map[id(img)]
                u_offset = atlas_idx / n_tex
                u_scale = 1.0 / n_tex
                for v in range(4):
                    remapped_uvs[face_idx * 4 + v, 0] = (
                        u_offset + remapped_uvs[face_idx * 4 + v, 0] * u_scale
                    )
            else:
                # No texture — map to a fallback region (first tile)
                u_scale = 1.0 / n_tex
                for v in range(4):
                    remapped_uvs[face_idx * 4 + v, 0] *= u_scale
            face_idx += 1

        material = trimesh.visual.material.SimpleMaterial(image=atlas)
        mesh.visual = trimesh.visual.TextureVisuals(uv=remapped_uvs, material=material)
    else:
        # Flat colour fallback — use average colour from first available texture
        colour = _average_texture_colour(all_tex_images)
        rgba = np.array([colour[0], colour[1], colour[2], 255], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))

    # Handle element rotation (around an axis through an origin point)
    rotation = element.get("rotation")
    if rotation:
        _apply_element_rotation(mesh, rotation)

    return mesh


def _average_texture_colour(
    images: list[Any],
) -> tuple[int, int, int]:
    """Extract average colour from available texture images, or return grey."""
    if not _TEXTURES_ENABLED:
        return (128, 128, 128)

    for img in images:
        if img is not None:
            arr = np.array(img)
            # Only average non-transparent pixels
            mask = arr[:, :, 3] > 128
            if mask.any():
                avg = arr[mask][:, :3].mean(axis=0).astype(int)
                return (int(avg[0]), int(avg[1]), int(avg[2]))

    return (128, 128, 128)


def _apply_element_rotation(mesh: trimesh.Trimesh, rotation: dict[str, Any]) -> None:
    """Apply Minecraft element rotation (axis, angle, origin)."""
    axis_name = rotation.get("axis", "y")
    angle = rotation.get("angle", 0)
    origin = rotation.get("origin", [8, 8, 8])

    if angle == 0:
        return

    # Convert origin from pixel space to unit space
    ox, oy, oz = origin[0] / 16.0, origin[1] / 16.0, origin[2] / 16.0

    axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(axis_name, [0, 1, 0])

    rad = math.radians(angle)
    rot_matrix = trimesh.transformations.rotation_matrix(rad, axis_vec, point=[ox, oy, oz])
    mesh.apply_transform(rot_matrix)


# ---------------------------------------------------------------------------
# Y-rotation for blockstate apply entries
# ---------------------------------------------------------------------------


def apply_y_rotation(mesh: trimesh.Trimesh, y_degrees: float) -> None:
    """
    Rotate a mesh around the Y axis through the block centre (0.5, 0.5, 0.5).

    Used for blockstate model applications that specify y rotation.
    """
    if y_degrees == 0:
        return
    rad = math.radians(y_degrees)
    rot = trimesh.transformations.rotation_matrix(rad, [0, 1, 0], point=[0.5, 0.5, 0.5])
    mesh.apply_transform(rot)


def apply_x_rotation(mesh: trimesh.Trimesh, x_degrees: float) -> None:
    """Rotate a mesh around the X axis through the block centre."""
    if x_degrees == 0:
        return
    rad = math.radians(x_degrees)
    rot = trimesh.transformations.rotation_matrix(rad, [1, 0, 0], point=[0.5, 0.5, 0.5])
    mesh.apply_transform(rot)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def build_model_mesh(model_name: str, y_rot: float = 0, x_rot: float = 0) -> trimesh.Trimesh | None:
    """
    Build a trimesh from a Minecraft model name.

    Parameters
    ----------
    model_name : str
        e.g. "minecraft:block/oak_fence_post" or "oak_planks"
    y_rot : float
        Y-axis rotation in degrees (from blockstate).
    x_rot : float
        X-axis rotation in degrees (from blockstate).

    Returns
    -------
    trimesh.Trimesh or None
        The assembled mesh, or None if the model cannot be resolved.
    """
    resolved = _resolve_model(model_name)
    if resolved is None:
        return None

    elements = resolved.get("elements", [])
    textures = resolved.get("textures", {})

    if not elements:
        return None

    meshes: list[trimesh.Trimesh] = []
    for elem in elements:
        m = _build_element_mesh(elem, textures)
        if m is not None:
            meshes.append(m)

    if not meshes:
        return None

    if len(meshes) == 1:
        combined = meshes[0]
    else:
        combined = trimesh.util.concatenate(meshes)

    # Apply blockstate rotations BEFORE recentring.  The rotation functions
    # pivot around [0.5, 0.5, 0.5] — the block centre in Minecraft's [0, 1]
    # unit space.  Translating first would move the centre to [0, 0, 0],
    # making the [0.5, 0.5, 0.5] pivot incorrect and flinging rotated
    # sub-model parts (e.g. fence side bars) to wrong positions.
    apply_y_rotation(combined, y_rot)
    apply_x_rotation(combined, x_rot)

    # Now shift from [0, 1] Minecraft space to [-0.5, 0.5] centred space.
    # Our scene places blocks at integer grid coords, and procedural
    # generators (box()) produce centred-at-origin meshes, so Phase 2
    # meshes must match: [0, 1] → [-0.5, 0.5].
    combined.apply_translation([-0.5, -0.5, -0.5])

    return combined
