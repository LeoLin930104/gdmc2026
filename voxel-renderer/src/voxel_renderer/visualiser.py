"""
Visualiser Middleware — Stateless rasteriser for voxel prefabs.

Converts a semantic block array into four flat-shaded orthographic/isometric
images against a #FF00FF background, encoded as Base64 PNG strings.

Rendering backend: trimesh + pyrender (offscreen via OSMesa/EGL).
Falls back to trimesh's built-in rasteriser if pyrender is unavailable.

This module has ZERO dependency on a running Minecraft server or the block
client.  It operates purely on the spec's semantic payload format.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
from typing import Any

# ---------------------------------------------------------------------------
# OpenGL platform selection for headless rendering.
# Must happen BEFORE any OpenGL import (pyrender triggers one at import time).
# Probe order: EGL (Nvidia/Mesa) -> OSMesa (software).
#
# The env var must be set before OpenGL is imported for the first time,
# because OpenGL.platform caches the selection on first module load.
# We test availability by attempting a ctypes load of the native library
# rather than importing any OpenGL module.
# ---------------------------------------------------------------------------
if "PYOPENGL_PLATFORM" not in os.environ:
    import ctypes
    import ctypes.util

    for _platform, _lib_names in (("egl", ("EGL",)), ("osmesa", ("OSMesa",))):
        _found = False
        for _lib in _lib_names:
            _path = ctypes.util.find_library(_lib)
            if _path:
                _found = True
                break
            # Try direct load as a fallback (find_library may miss it)
            try:
                ctypes.CDLL(f"lib{_lib}.so.1")
                _found = True
                break
            except OSError:
                try:
                    ctypes.CDLL(f"lib{_lib}.so")
                    _found = True
                    break
                except OSError:
                    pass
        if _found:
            os.environ["PYOPENGL_PLATFORM"] = _platform
            break
    del ctypes

import numpy as np
import trimesh

from voxel_renderer.block_registry import create_coloured_block_mesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Background colour — neutral dark grey for VLM readability.
# Magenta (#FF00FF) was previously used as a chroma key, but VLMs struggle
# to parse structure details against a saturated background.  Dark grey
# provides high contrast with most block textures without visual distraction.
BG_COLOUR = np.array([64, 64, 64, 255], dtype=np.uint8)
DEFAULT_BG_RGB = (64, 64, 64)

# Default render resolution per view
RENDER_WIDTH = 512
RENDER_HEIGHT = 512

# Camera view definitions.
# Each entry: (name, elevation_deg, azimuth_deg)
# Elevation: 0 = horizon, 90 = straight down.
# Azimuth: 0 = +Z axis (south), measured clockwise from above.
CAMERA_VIEWS = [
    ("top", 90.0, 0.0),
    ("profile", 0.0, 0.0),
    ("iso_right", 45.0, 45.0),
    ("iso_left", 45.0, 315.0),
]

CAMERA_VIEW_MAP: dict[str, tuple[float, float]] = {
    name: (elev, azim) for name, elev, azim in CAMERA_VIEWS
}


# ---------------------------------------------------------------------------
# Scene Assembly
# ---------------------------------------------------------------------------


def assemble_block_scene(
    block_array: list[dict[str, Any]],
    resolve_adjacency: bool = True,
) -> trimesh.Scene:
    """
    Assemble a trimesh Scene from a semantic block array.

    Each block is instantiated as a coloured mesh, translated to its
    (x, y, z) grid position.  Air blocks are skipped.

    When ``resolve_adjacency`` is True (default), connection properties
    for fences, walls, and panes are derived from spatial neighbours,
    overriding any stored properties.

    Parameters
    ----------
    block_array : list[dict]
        Spec-format block entries: [{x, y, z, id, properties?}, ...]
    resolve_adjacency : bool
        Whether to derive connection properties from neighbours.

    Returns
    -------
    trimesh.Scene
    """
    scene = trimesh.Scene()

    # Build spatial index for adjacency resolution
    if resolve_adjacency:
        from voxel_renderer.adjacency import BlockGrid, resolve_all_connections

        grid = BlockGrid()
        for block in block_array:
            block_id = block["id"]
            if block_id == "minecraft:air":
                continue
            x, y, z = int(block["x"]), int(block["y"]), int(block["z"])
            props = block.get("properties") or {}
            grid.set(x, y, z, block_id, props)

        # Resolve all connections in one pass
        resolve_all_connections(grid)

    for i, block in enumerate(block_array):
        block_id = block["id"]
        if block_id == "minecraft:air":
            continue

        x, y, z = int(block["x"]), int(block["y"]), int(block["z"])

        if resolve_adjacency:
            entry = grid.get(x, y, z)
            props = entry["props"] if entry else (block.get("properties") or {})
        else:
            props = block.get("properties") or {}

        mesh = create_coloured_block_mesh(block_id, props)

        translation = np.array([float(x), float(y), float(z)], dtype=float)
        mesh.apply_translation(translation)

        scene.add_geometry(mesh, node_name=f"block_{i}")

    return scene


# ---------------------------------------------------------------------------
# Camera Mathematics
# ---------------------------------------------------------------------------


def compute_orbital_camera_transform(
    centre: np.ndarray,
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    """
    Compute a 4x4 camera-to-world matrix for an orbital camera.

    The camera looks at `centre` from a spherical position defined by
    elevation and azimuth at the given distance.

    Uses OpenGL convention: camera looks down -Z, Y is up.
    """
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)

    # Spherical to Cartesian (Y-up)
    # azimuth 0 = looking from +Z towards centre
    cam_x = distance * math.cos(elev) * math.sin(azim)
    cam_y = distance * math.sin(elev)
    cam_z = distance * math.cos(elev) * math.cos(azim)

    eye = centre + np.array([cam_x, cam_y, cam_z])
    target = centre

    # look-at matrix construction
    forward = target - eye
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-8:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / forward_len

    world_up = np.array([0.0, 1.0, 0.0])

    # Handle degenerate case where forward is parallel to world_up (top-down view)
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, -1.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Camera-to-world: columns are right, up, -forward (OpenGL convention)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    return pose


# ---------------------------------------------------------------------------
# Rendering (pyrender backend)
# ---------------------------------------------------------------------------


def _patch_nearest_sampler(pr_mesh: Any) -> None:
    """Force GL_NEAREST on all texture samplers for pixel-art fidelity.

    Pyrender defaults to GL_LINEAR (bilinear interpolation) when the
    sampler's magFilter/minFilter are None.  Minecraft's 16×16 textures
    must be sampled with nearest-neighbour to preserve sharp pixel edges.
    """
    import pyrender
    from pyrender.constants import GLTF

    nearest = pyrender.Sampler(
        magFilter=GLTF.NEAREST,
        minFilter=GLTF.NEAREST,
    )
    for prim in pr_mesh.primitives:
        mat = prim.material
        if mat is None:
            continue
        for tex in mat.textures:
            if tex is not None:
                tex.sampler = nearest


def render_all_views_via_pyrender(
    scene: trimesh.Scene,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> dict[str, str]:
    """
    Render four views using pyrender offscreen renderer.

    Returns a dict mapping view name -> Base64-encoded PNG string.
    """
    import pyrender

    # Convert trimesh scene to pyrender scene
    # Prefer grid-cell bounds (stable for thin geometry) over mesh vertex bounds.
    grid_bounds = scene.metadata.get("grid_bounds")
    if grid_bounds is not None:
        bounds = grid_bounds
    else:
        bounds = scene.bounds  # (2, 3) array: [min_corner, max_corner]
    centre = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))

    if max_extent < 1e-6:
        max_extent = 1.0

    # Orthographic camera sized to fit the scene with padding
    xmag = max_extent * 0.7
    ymag = max_extent * 0.7
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.01, zfar=max_extent * 10)

    # Distance from centre — far enough to avoid clipping
    distance = max_extent * 2.5

    # Two-light rig for face-normal contrast on small furniture geometry.
    # Key light: upper-front-right (bright, primary shadow caster)
    # Fill light: lower-back-left (soft, reduces harsh shadows)
    # Reduced ambient prevents flat washing while retaining base visibility.
    ambient_light = np.array([0.35, 0.35, 0.35])
    KEY_LIGHT_INTENSITY = 4.0
    KEY_LIGHT_ELEVATION = 60.0
    KEY_LIGHT_AZIMUTH = 30.0
    FILL_LIGHT_INTENSITY = 1.5
    FILL_LIGHT_ELEVATION = 20.0
    FILL_LIGHT_AZIMUTH = 210.0
    # Distance for light pose — only direction matters for DirectionalLight,
    # but we need a valid transform so place it at the scene's bounding radius.
    light_distance = max_extent * 3.0

    renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
    )

    results: dict[str, str] = {}

    try:
        for view_name, elev, azim in CAMERA_VIEWS:
            pr_scene = pyrender.Scene(
                bg_color=BG_COLOUR.astype(float) / 255.0,
                ambient_light=ambient_light,
            )

            # Add all meshes from the trimesh scene
            for node_name, geom in scene.geometry.items():
                # Get the transform for this geometry from the scene graph
                node_names = [
                    n
                    for n in scene.graph.nodes
                    if n in scene.graph.geometry_nodes
                    and scene.graph.geometry_nodes[n] == node_name
                ]
                if node_names:
                    transform = scene.graph.get(node_names[0])[0]
                else:
                    transform = np.eye(4)

                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                _patch_nearest_sampler(pr_mesh)
                pr_scene.add(pr_mesh, pose=transform)

            # Add camera
            cam_pose = compute_orbital_camera_transform(centre, distance, elev, azim)
            pr_scene.add(camera, pose=cam_pose)

            # Add directional lights.
            # DirectionalLight shines along -Z of its pose (same as camera look-at),
            # so we reuse compute_orbital_camera_transform to aim each light at the
            # scene centre from the specified spherical coordinates.
            key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=KEY_LIGHT_INTENSITY)
            key_pose = compute_orbital_camera_transform(
                centre, light_distance, KEY_LIGHT_ELEVATION, KEY_LIGHT_AZIMUTH
            )
            pr_scene.add(key_light, pose=key_pose)

            fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=FILL_LIGHT_INTENSITY)
            fill_pose = compute_orbital_camera_transform(
                centre, light_distance, FILL_LIGHT_ELEVATION, FILL_LIGHT_AZIMUTH
            )
            pr_scene.add(fill_light, pose=fill_pose)

            colour, _ = renderer.render(pr_scene)

            # Encode to PNG via Pillow
            from PIL import Image

            img = Image.fromarray(colour)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            results[view_name] = base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        renderer.delete()

    return results


# ---------------------------------------------------------------------------
# Rendering (trimesh fallback)
# ---------------------------------------------------------------------------


def render_all_views_via_trimesh(
    scene: trimesh.Scene,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> dict[str, str]:
    """
    Fallback renderer using trimesh's built-in scene rendering.

    This requires a display (real or virtual via xvfb).  Less reliable
    than pyrender but has no OpenGL dependency beyond pyglet.
    """
    from PIL import Image

    grid_bounds = scene.metadata.get("grid_bounds")
    if grid_bounds is not None:
        bounds = grid_bounds
    else:
        bounds = scene.bounds
    centre = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))

    if max_extent < 1e-6:
        max_extent = 1.0

    distance = max_extent * 2.5

    results: dict[str, str] = {}

    for view_name, elev, azim in CAMERA_VIEWS:
        cam_pose = compute_orbital_camera_transform(centre, distance, elev, azim)

        # trimesh scene rendering
        scene.camera_transform = cam_pose

        try:
            png_data = scene.save_image(resolution=(width, height), visible=False)
            # Composite onto background
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            bg = Image.new("RGBA", (width, height), (*DEFAULT_BG_RGB, 255))
            bg.paste(img, (0, 0), img)
            buf = io.BytesIO()
            bg.convert("RGB").save(buf, format="PNG")
            results[view_name] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.warning("trimesh render failed for view '%s': %s", view_name, e)
            # Return a solid background image as fallback
            bg = Image.new("RGB", (width, height), DEFAULT_BG_RGB)
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            results[view_name] = base64.b64encode(buf.getvalue()).decode("ascii")

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_orthographic_views(
    block_array: list[dict[str, Any]],
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
    backend: str = "auto",
) -> dict[str, str]:
    """
    Render four orthographic/isometric views of a block array.

    Parameters
    ----------
    block_array : list[dict]
        Spec-format block entries.
    width, height : int
        Pixel dimensions per view.
    backend : "auto" | "pyrender" | "trimesh"
        Rendering backend.  "auto" tries pyrender first, falls back to trimesh.

    Returns
    -------
    dict[str, str]
        Maps view name ("top", "profile", "iso_right", "iso_left") to
        Base64-encoded PNG string.
    """
    if not block_array:
        from PIL import Image

        # No blocks — return four background-colour images
        results: dict[str, str] = {}
        for view_name, _, _ in CAMERA_VIEWS:
            bg = Image.new("RGB", (width, height), DEFAULT_BG_RGB)
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            results[view_name] = base64.b64encode(buf.getvalue()).decode("ascii")
        return results

    scene = assemble_block_scene(block_array)

    if not scene.geometry:
        # All blocks were air
        return render_orthographic_views([], width=width, height=height)

    # Compute grid-cell bounds (each block occupies a full 1×1×1 cell)
    # to ensure camera framing is stable regardless of mesh narrowness.
    non_air = [b for b in block_array if b["id"] != "minecraft:air"]
    if non_air:
        xs = [int(b["x"]) for b in non_air]
        ys = [int(b["y"]) for b in non_air]
        zs = [int(b["z"]) for b in non_air]
        grid_min = np.array([min(xs) - 0.5, min(ys) - 0.5, min(zs) - 0.5])
        grid_max = np.array([max(xs) + 0.5, max(ys) + 0.5, max(zs) + 0.5])
        scene.metadata["grid_bounds"] = np.array([grid_min, grid_max])

    if backend == "pyrender" or backend == "auto":
        try:
            return render_all_views_via_pyrender(scene, width, height)
        except Exception as e:
            if backend == "pyrender":
                raise
            logger.info("pyrender unavailable (%s), falling back to trimesh renderer.", e)

    return render_all_views_via_trimesh(scene, width, height)


# ---------------------------------------------------------------------------
# Single-view render (for animation preview)
# ---------------------------------------------------------------------------


def render_single_view(
    scene: trimesh.Scene,
    view_name: str = "iso_right",
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
    bg_colour: tuple[int, int, int] = DEFAULT_BG_RGB,
    backend: str = "auto",
) -> str:
    """
    Render a single named view of a pre-assembled scene.

    Parameters
    ----------
    scene : trimesh.Scene
        Pre-built scene (from assemble_block_scene).
    view_name : str
        One of "top", "profile", "iso_right", "iso_left".
    width, height : int
        Pixel dimensions.
    bg_colour : tuple[int, int, int]
        RGB background colour.
    backend : "auto" | "pyrender" | "trimesh"

    Returns
    -------
    str
        Base64-encoded PNG string.
    """
    if view_name not in CAMERA_VIEW_MAP:
        raise ValueError(f"Unknown view '{view_name}'. Available: {sorted(CAMERA_VIEW_MAP.keys())}")

    elev, azim = CAMERA_VIEW_MAP[view_name]
    grid_bounds = scene.metadata.get("grid_bounds")
    if grid_bounds is not None:
        bounds = grid_bounds
    else:
        bounds = scene.bounds
    centre = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent < 1e-6:
        max_extent = 1.0
    distance = max_extent * 2.5

    bg_rgba_float = np.array([bg_colour[0], bg_colour[1], bg_colour[2], 255], dtype=float) / 255.0

    # Try pyrender first
    if backend in ("pyrender", "auto"):
        try:
            return _render_single_pyrender(
                scene,
                elev,
                azim,
                centre,
                max_extent,
                distance,
                bg_rgba_float,
                width,
                height,
            )
        except Exception as e:
            if backend == "pyrender":
                raise
            logger.info("pyrender unavailable for single view (%s), falling back.", e)

    # Trimesh fallback
    return _render_single_trimesh(
        scene,
        elev,
        azim,
        centre,
        distance,
        bg_colour,
        width,
        height,
    )


def _render_single_pyrender(
    scene: trimesh.Scene,
    elev: float,
    azim: float,
    centre: np.ndarray,
    max_extent: float,
    distance: float,
    bg_rgba_float: np.ndarray,
    width: int,
    height: int,
) -> str:
    """Pyrender backend for single-view render."""
    import pyrender
    from PIL import Image

    xmag = max_extent * 0.7
    ymag = max_extent * 0.7
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.01, zfar=max_extent * 10)

    pr_scene = pyrender.Scene(
        bg_color=bg_rgba_float,
        ambient_light=np.array([0.35, 0.35, 0.35]),
    )

    for node_name, geom in scene.geometry.items():
        node_names = [
            n
            for n in scene.graph.nodes
            if n in scene.graph.geometry_nodes and scene.graph.geometry_nodes[n] == node_name
        ]
        transform = scene.graph.get(node_names[0])[0] if node_names else np.eye(4)
        pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        _patch_nearest_sampler(pr_mesh)
        pr_scene.add(pr_mesh, pose=transform)

    cam_pose = compute_orbital_camera_transform(centre, distance, elev, azim)
    pr_scene.add(camera, pose=cam_pose)

    light_distance = max_extent * 3.0
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    key_pose = compute_orbital_camera_transform(centre, light_distance, 60.0, 30.0)
    pr_scene.add(key_light, pose=key_pose)

    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    fill_pose = compute_orbital_camera_transform(centre, light_distance, 20.0, 210.0)
    pr_scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        colour, _ = renderer.render(pr_scene)
    finally:
        renderer.delete()

    img = Image.fromarray(colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_single_trimesh(
    scene: trimesh.Scene,
    elev: float,
    azim: float,
    centre: np.ndarray,
    distance: float,
    bg_colour: tuple[int, int, int],
    width: int,
    height: int,
) -> str:
    """Trimesh fallback for single-view render."""
    from PIL import Image

    cam_pose = compute_orbital_camera_transform(centre, distance, elev, azim)
    scene.camera_transform = cam_pose

    try:
        png_data = scene.save_image(resolution=(width, height), visible=False)
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGBA", (width, height), (*bg_colour, 255))
        bg.paste(img, (0, 0), img)
        buf = io.BytesIO()
        bg.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning("trimesh single-view render failed: %s", e)
        bg = Image.new("RGB", (width, height), bg_colour)
        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Composite grid (diagnostic & agent-visible artefact)
# ---------------------------------------------------------------------------

# View order for the 2×2 composite grid.
_COMPOSITE_VIEW_ORDER = ("top", "profile", "iso_right", "iso_left")

# Human-readable labels shown in the top-left corner of each quadrant.
_COMPOSITE_VIEW_LABELS = {
    "top": "Top",
    "profile": "Front",
    "iso_right": "Iso-Right",
    "iso_left": "Iso-Left",
}


def compose_view_grid(
    views: dict[str, str],
    *,
    order: tuple[str, ...] = _COMPOSITE_VIEW_ORDER,
    label: bool = True,
    label_colour: tuple[int, int, int] = (255, 255, 255),
    label_bg_colour: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    """Arrange the four orthographic views into a labelled 2×2 composite PNG.

    Parameters
    ----------
    views:
        ``dict[str, str]`` as returned by ``render_orthographic_views()`` —
        keys are view names, values are base64-encoded PNG strings.
    order:
        View placement order: top-left, top-right, bottom-left, bottom-right.
    label:
        If ``True``, draw a text label in the top-left corner of each cell.
    label_colour:
        RGB colour for label text.
    label_bg_colour:
        RGB colour for the label background strip.

    Returns
    -------
    bytes:
        PNG-encoded bytes of the 2×2 composite image.
    """
    from PIL import Image, ImageDraw, ImageFont

    images: list[Image.Image] = []
    for name in order:
        b64 = views.get(name)
        if b64 is None:
            # Missing view — create a placeholder
            images.append(Image.new("RGB", (RENDER_WIDTH, RENDER_HEIGHT), DEFAULT_BG_RGB))
        else:
            images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

    cell_w, cell_h = images[0].size
    grid = Image.new("RGB", (cell_w * 2, cell_h * 2), DEFAULT_BG_RGB)
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]

    for i, (img, pos) in enumerate(zip(images, positions)):
        grid.paste(img, pos)

    if label:
        draw = ImageDraw.Draw(grid)
        # Use default bitmap font — always available, no .ttf dependency.
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for i, (name, pos) in enumerate(zip(order, positions)):
            text = _COMPOSITE_VIEW_LABELS.get(name, name)
            # Text metrics
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 4
            # Background rectangle
            rx, ry = pos[0] + pad, pos[1] + pad
            draw.rectangle(
                [rx - 1, ry - 1, rx + tw + pad * 2, ry + th + pad],
                fill=label_bg_colour,
            )
            draw.text((rx + pad, ry), text, fill=label_colour, font=font)

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


def compose_comparison_strip(
    labelled_images: list[tuple[str, str]],
    *,
    label_colour: tuple[int, int, int] = (255, 255, 255),
    label_bg_colour: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    """Arrange labelled base64 PNGs into a horizontal comparison strip.

    Intended for candidate selection prompts where the model should compare
    several alternatives in one shared visual context.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not labelled_images:
        raise ValueError("compose_comparison_strip requires at least one image")

    decoded: list[tuple[str, Image.Image]] = []
    for label, b64 in labelled_images:
        decoded.append((label, Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")))

    cell_w, cell_h = decoded[0][1].size
    strip = Image.new("RGB", (cell_w * len(decoded), cell_h), DEFAULT_BG_RGB)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(decoded):
        x = idx * cell_w
        strip.paste(img, (x, 0))
        draw = ImageDraw.Draw(strip)
        draw.rectangle((x + 4, 4, x + 120, 28), fill=label_bg_colour)
        draw.text((x + 8, 7), label, fill=label_colour, font=font)

    buf = io.BytesIO()
    strip.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gallery grid — N completed modules tiled as an r×c PNG mosaic.
# ---------------------------------------------------------------------------


def compose_gallery_grid(
    tiles: list[tuple[str, str | None]],
    *,
    columns: int,
    label_colour: tuple[int, int, int] = (255, 255, 255),
    label_bg_colour: tuple[int, int, int] = (0, 0, 0),
    placeholder_text: str = "(failed)",
    placeholder_bg_rgb: tuple[int, int, int] = (96, 32, 32),
) -> bytes:
    """Tile many single-view renders into an r×c PNG wall.

    Unlike :func:`compose_view_grid` (which arranges the four views of
    ONE build in a fixed 2×2) and :func:`compose_comparison_strip`
    (which lays out a 1×N strip of candidates), this helper arranges
    N independent build renders into a general r×c grid for gallery
    display.

    Parameters
    ----------
    tiles:
        List of ``(label, b64_png_or_None)`` pairs.  ``None`` on the
        second element renders a labelled red placeholder cell — used
        when the gallery runner marked that slot dead (Expander bounds
        exhaustion, draftsman crash, refinement failure).
    columns:
        Target column count.  Row count is derived as
        ``ceil(len(tiles) / columns)``.  The last row is padded with
        empty cells if necessary so the image is rectangular.
    label_colour, label_bg_colour:
        Colours for the per-tile caption strip.
    placeholder_text:
        Text drawn in dead-tile cells.
    placeholder_bg_rgb:
        Background colour for dead-tile cells (red-tinted by default
        so failures are impossible to miss at a glance).

    Returns
    -------
    bytes:
        PNG-encoded bytes of the gallery image.

    Raises
    ------
    ValueError:
        When ``tiles`` is empty or ``columns`` < 1.

    Vulnerabilities
    ---------------
    - All tiles are assumed to share the first tile's dimensions; if
      the caller supplies mismatched renders the later cells will be
      resampled to match the first.  The gallery driver always calls
      ``render_orthographic_views`` with a single ``render_size`` so
      this is uniform in practice.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not tiles:
        raise ValueError("compose_gallery_grid requires at least one tile")
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")

    rows = (len(tiles) + columns - 1) // columns

    # Pick cell dimensions: prefer the first non-None tile's size, fall
    # back to the default render size.  This keeps placeholder-only
    # galleries renderable (unusual, but not invalid).
    first_img: Image.Image | None = None
    for _label, b64 in tiles:
        if b64 is not None:
            first_img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            break
    if first_img is not None:
        cell_w, cell_h = first_img.size
    else:
        cell_w, cell_h = RENDER_WIDTH, RENDER_HEIGHT

    grid = Image.new("RGB", (cell_w * columns, cell_h * rows), DEFAULT_BG_RGB)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for idx, (label, b64) in enumerate(tiles):
        r, c = divmod(idx, columns)
        x, y = c * cell_w, r * cell_h

        if b64 is None:
            # Dead tile — red placeholder with centred failure text.
            placeholder = Image.new("RGB", (cell_w, cell_h), placeholder_bg_rgb)
            grid.paste(placeholder, (x, y))
            ptxt = placeholder_text
            bbox = draw.textbbox((0, 0), ptxt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x + (cell_w - tw) // 2, y + (cell_h - th) // 2),
                ptxt,
                fill=label_colour,
                font=font,
            )
        else:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h))
            grid.paste(img, (x, y))

        # Caption strip in the bottom-left of every cell (present or
        # dead) so the viewer can always read the slug.
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        lx = x + pad
        ly = y + cell_h - th - pad * 2
        draw.rectangle(
            [lx - 1, ly - 1, lx + tw + pad * 2, ly + th + pad],
            fill=label_bg_colour,
        )
        draw.text((lx + pad, ly), label, fill=label_colour, font=font)

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()
