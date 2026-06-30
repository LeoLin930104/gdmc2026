"""Renderer-only MCP server.

This module is the Phase-B extraction boundary for using the visualiser as a
drop-in MCP renderer in projects that do not use the VPS agentic harness or a
Minecraft server.  The primary tool is stateless: callers pass a semantic block
array and receive rendered PNG views.  A small optional in-memory session layer
exists for agents that prefer commit/render/dump workflows, but it is backed by
``VoxelStore`` rather than the GDPC-capable ``vps.block_client.BlockClient``.

Unlike ``vps.mcp_server``, this server does **not** enforce the Minecraft
placement palette.  Unknown block IDs remain renderable via the renderer's
fallback material path.  Placement legality belongs to a separate Minecraft
adapter layer.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from voxel_renderer.api import render_orthographic_views
from voxel_renderer.state import VoxelStore

logger = logging.getLogger(__name__)


mcp = FastMCP(
    name="voxel-renderer",
    instructions=(
        "Standalone voxel renderer. Render semantic block arrays to base64 PNG "
        "orthographic/isometric views. Unknown block IDs are allowed and render "
        "with fallback materials; Minecraft placement legality is out of scope."
    ),
)


_sessions: dict[str, VoxelStore] = {}


def _validate_block_array(blocks: list[dict[str, Any]]) -> str | None:
    """Return an error string when *blocks* is not a semantic block array."""
    if not isinstance(blocks, list):
        return "blocks must be a JSON array"
    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            return f"block at index {i} must be an object"
        missing = {"x", "y", "z", "id"} - set(block)
        if missing:
            return f"block at index {i} missing required field(s): {sorted(missing)}"
        try:
            int(block["x"])
            int(block["y"])
            int(block["z"])
        except (TypeError, ValueError):
            return f"block at index {i} has non-integer coordinate(s)"
        if not isinstance(block["id"], str) or not block["id"].strip():
            return f"block at index {i} has invalid id"
    return None


@mcp.tool()
def renderer_render_blocks(
    blocks: list[dict[str, Any]],
    width: int = 512,
    height: int = 512,
    backend: str = "auto",
) -> str:
    """Render a semantic block array without mutating server state.

    Returns JSON:
        {"views": {view_name: base64_png}, "block_count": int}
    """
    error = _validate_block_array(blocks)
    if error is not None:
        return json.dumps({"error": error})
    try:
        views = render_orthographic_views(blocks, width=width, height=height, backend=backend)
    except Exception as exc:  # pragma: no cover - backend/environment dependent
        logger.error("Renderer stateless render failed: %s", exc, exc_info=True)
        return json.dumps({"error": f"Render failed: {exc}"})
    non_air_count = sum(1 for block in blocks if block.get("id") != "minecraft:air")
    return json.dumps({"views": views, "block_count": non_air_count})


@mcp.tool()
def renderer_create_session() -> str:
    """Create an isolated in-memory render session and return its id."""
    session_id = uuid.uuid4().hex
    _sessions[session_id] = VoxelStore()
    return json.dumps({"session_id": session_id})


def _get_session(session_id: str) -> VoxelStore | None:
    return _sessions.get(session_id)


@mcp.tool()
def renderer_commit_blocks(session_id: str, blocks: list[dict[str, Any]]) -> str:
    """Apply block additions/deletions to a renderer session.

    ``minecraft:air`` deletes a coordinate.  No Minecraft placement palette is
    enforced; malformed semantic payloads are rejected before mutation.
    """
    store = _get_session(session_id)
    if store is None:
        return json.dumps({"error": f"Unknown session_id: {session_id}"})
    error = _validate_block_array(blocks)
    if error is not None:
        return json.dumps({"error": error})
    count = store.commit(blocks)
    return json.dumps({"applied": count, "total_blocks": store.get_count()})


@mcp.tool()
def renderer_get_session_render(
    session_id: str,
    width: int = 512,
    height: int = 512,
    backend: str = "auto",
) -> str:
    """Render the current block state of an in-memory session."""
    store = _get_session(session_id)
    if store is None:
        return json.dumps({"error": f"Unknown session_id: {session_id}"})
    return renderer_render_blocks(store.get_all(), width=width, height=height, backend=backend)


@mcp.tool()
def renderer_get_session_dump(session_id: str) -> str:
    """Return all non-air blocks currently stored in a renderer session."""
    store = _get_session(session_id)
    if store is None:
        return json.dumps({"error": f"Unknown session_id: {session_id}"})
    return json.dumps(store.get_all())


@mcp.tool()
def renderer_get_session_bounds(session_id: str) -> str:
    """Return the axis-aligned bounding box for a renderer session."""
    store = _get_session(session_id)
    if store is None:
        return json.dumps({"error": f"Unknown session_id: {session_id}"})
    bounds = store.get_bounding_box()
    if bounds is None:
        return json.dumps({"empty": True})
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return json.dumps(
        {
            "x_min": x_min,
            "y_min": y_min,
            "z_min": z_min,
            "x_max": x_max,
            "y_max": y_max,
            "z_max": z_max,
            "block_count": store.get_count(),
        }
    )


@mcp.tool()
def renderer_clear_session(session_id: str) -> str:
    """Clear an existing renderer session."""
    store = _get_session(session_id)
    if store is None:
        return json.dumps({"error": f"Unknown session_id: {session_id}"})
    cleared = store.get_count()
    store.clear()
    return json.dumps({"cleared": cleared, "total_blocks": 0})


@mcp.tool()
def renderer_delete_session(session_id: str) -> str:
    """Delete an in-memory renderer session."""
    existed = _sessions.pop(session_id, None) is not None
    return json.dumps({"deleted": existed})


def main() -> None:
    """Run the renderer MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
