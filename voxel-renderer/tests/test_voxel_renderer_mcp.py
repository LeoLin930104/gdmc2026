"""Tests for the renderer-only MCP tool functions.

These invoke the functions directly and bypass MCP transport, mirroring the
legacy ``tests/test_mcp_tools.py`` pattern while proving the new package does
not depend on ``vps.block_client`` or Minecraft placement validation.
"""

from __future__ import annotations

import json

from voxel_renderer.mcp_server import (
    renderer_clear_session,
    renderer_commit_blocks,
    renderer_create_session,
    renderer_delete_session,
    renderer_get_session_bounds,
    renderer_get_session_dump,
    renderer_get_session_render,
    renderer_render_blocks,
)


def test_stateless_render_accepts_unknown_block_id() -> None:
    result = json.loads(
        renderer_render_blocks(
            [{"x": 0, "y": 0, "z": 0, "id": "custom:titanium_block"}],
            width=64,
            height=64,
        )
    )
    assert "error" not in result
    assert result["block_count"] == 1
    assert set(result["views"]) == {"top", "profile", "iso_right", "iso_left"}


def test_stateless_render_rejects_malformed_blocks() -> None:
    result = json.loads(renderer_render_blocks([{"x": 0, "z": 0, "id": "custom:block"}]))
    assert "error" in result
    assert "missing" in result["error"]


def test_session_lifecycle_and_air_deletion() -> None:
    session = json.loads(renderer_create_session())["session_id"]
    try:
        commit = json.loads(
            renderer_commit_blocks(
                session,
                [
                    {"x": 0, "y": 0, "z": 0, "id": "custom:a"},
                    {"x": 2, "y": 1, "z": -1, "id": "custom:b"},
                ],
            )
        )
        assert commit == {"applied": 2, "total_blocks": 2}

        bounds = json.loads(renderer_get_session_bounds(session))
        assert bounds["x_min"] == 0
        assert bounds["x_max"] == 2
        assert bounds["z_min"] == -1
        assert bounds["block_count"] == 2

        dump = json.loads(renderer_get_session_dump(session))
        assert {b["id"] for b in dump} == {"custom:a", "custom:b"}

        delete = json.loads(
            renderer_commit_blocks(session, [{"x": 0, "y": 0, "z": 0, "id": "minecraft:air"}])
        )
        assert delete["total_blocks"] == 1

        render = json.loads(renderer_get_session_render(session, width=32, height=32))
        assert render["block_count"] == 1

        cleared = json.loads(renderer_clear_session(session))
        assert cleared == {"cleared": 1, "total_blocks": 0}
    finally:
        deleted = json.loads(renderer_delete_session(session))
        assert deleted["deleted"] is True


def test_unknown_session_returns_error() -> None:
    result = json.loads(renderer_get_session_dump("missing"))
    assert "error" in result
