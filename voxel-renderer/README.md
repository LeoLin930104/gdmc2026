# Standalone renderer extraction

This folder is a standalone renderer-first package.
It is intended to be copied or packaged into other projects without the agentic
VPS harness, local LLM clients, catalogue pipeline, or GDPC placement adapter.

## Boundary

Renderer package:

```text
packages/voxel-renderer/src/voxel_renderer/
  api.py                    # public render/composite API
  visualiser.py             # orthographic/isometric renderer
  block_registry.py         # Minecraft/fallback mesh registry
  blockstate_resolver.py    # Minecraft blockstate JSON resolver
  model_parser.py           # Minecraft model JSON parser
  adjacency.py              # fence/wall/pane connection derivation
  types.py                  # neutral semantic block record
  state.py                  # VoxelStore + canonicalisation
  assets.py                 # asset-root discovery
  palette.py                # Minecraft placement palette helper
  mcp_server.py             # renderer-only MCP server
```

Bundled asset tree for Minecraft-fidelity rendering:

```text
assets/minecraft/
  blockstates/
  models/
  textures/
```

The legacy `vps.visualiser`, `vps.block_registry`, `vps.blockstate_resolver`,
`vps.model_parser`, and `vps.adjacency` modules are compatibility shims.  New
code should import from `voxel_renderer` directly.

## Public Python API

```python
from voxel_renderer import render_orthographic_views, VoxelStore

blocks = [
    {"x": 0, "y": 0, "z": 0, "id": "minecraft:stone"},
    {"x": 1, "y": 0, "z": 0, "id": "custom:titanium_block"},
]

views = render_orthographic_views(blocks, width=512, height=512)
# views: {"top": base64_png, "profile": ..., "iso_right": ..., "iso_left": ...}

store = VoxelStore()
store.commit(blocks)
store.commit([{"x": 0, "y": 0, "z": 0, "id": "minecraft:air"}])  # delete
assert store.get_count() == 1
```

Unknown block IDs are renderable.  The renderer falls back to deterministic
procedural material/colour paths when it cannot resolve a Minecraft asset.  This
is intentional: renderability is not the same property as Minecraft placement
legality.

## Renderer MCP server

Console entry point:

```bash
uv run voxel-renderer-mcp
```

Primary stateless tool:

```text
renderer_render_blocks(blocks, width=512, height=512, backend="auto")
```

It returns JSON:

```json
{
  "views": {
    "top": "<base64 png>",
    "profile": "<base64 png>",
    "iso_right": "<base64 png>",
    "iso_left": "<base64 png>"
  },
  "block_count": 2
}
```

Optional session tools exist for agents that prefer a working-volume workflow:

- `renderer_create_session()`
- `renderer_commit_blocks(session_id, blocks)`
- `renderer_get_session_render(session_id, width, height, backend)`
- `renderer_get_session_dump(session_id)`
- `renderer_get_session_bounds(session_id)`
- `renderer_clear_session(session_id)`
- `renderer_delete_session(session_id)`

The session layer uses `voxel_renderer.VoxelStore`, not `vps.block_client`, so it
has no GDPC or Minecraft placement dependency.

## Asset discovery

Default discovery first checks the package-local `assets/minecraft/` directory,
then falls back to `VOXEL_RENDERER_ASSET_ROOT`, then walks upward for a legacy
`assets/` directory.  For external asset packs, set:

```bash
export VOXEL_RENDERER_ASSET_ROOT=/path/to/assets
```

The directory supplied by `VOXEL_RENDERER_ASSET_ROOT` must directly contain the
`blockstates/`, `models/`, and `textures/` subdirectories if Minecraft asset
fidelity is required.  Without assets, unknown/fallback rendering still works,
but Minecraft-specific model fidelity will degrade.

## Prefab/WFC utilities

Container-housing and wave-function-collapse experiments usually need simple
deterministic geometry operations before rendering.  `voxel_renderer.prefab`
provides:

- `get_bounds(blocks)`
- `normalise_to_origin(blocks)`
- `translate_blocks(blocks, dx, dy, dz)`
- `rotate_y(blocks, 0|90|180|270)`
- `merge_prefabs(*prefabs)` using last-write-wins semantics
- `face_signature(blocks, face)` for boundary compatibility checks
- `opposite_face(face)`

These helpers are intentionally block-array level operations.  They do not yet
rotate Minecraft orientation properties such as stair `facing`; WFC adjacency
should treat them as coarse module transforms until an orientation transformer is
added.

## Palette and placement policy

`voxel_renderer.palette` exposes a Minecraft placement palette derived from
`assets/blockstates/`, but the renderer and renderer-only MCP do not enforce it.

Policy split:

- **Renderer policy:** accept any well-formed block ID and render fallback
  geometry/materials when assets are unknown.
- **Minecraft placement policy:** `vps.block_client.BlockClient` enforces the
  Minecraft allowlist before GDPC placement.
- **Agent generation policy:** the harness can still constrain LLM outputs using
  curated prompts and post-generation gates.

This prevents the old allowlist from backfiring when the renderer is used in
non-Minecraft-exclusive projects.

## Current dependency state

This package's `pyproject.toml` exposes `voxel-renderer-mcp`.  The dependency
layout is intentionally narrow:

- `renderer`
- `mcp-server`
- `gdpc`
- `harness`

The base dependency set is intentionally still compatible with the historical
monorepo during this extraction checkpoint.  A later packaging pass can harden
extras by moving GDPC/MCP/LLM dependencies out of the default install.

## Known vulnerabilities / limits

1. **Asset redistribution:** Minecraft textures/models may have licensing
   constraints.  Downstream packages should prefer external asset-pack paths
   until redistribution rights are explicit.
2. **OpenGL portability:** `pyrender` may require EGL/OSMesa availability on
   headless hosts.  The renderer falls back to trimesh where possible, but image
   fidelity may differ.
3. **Compatibility shims remain:** the VPS harness still imports some legacy
   `vps.*` paths.  This is deliberate to keep the harness stable while the
   renderer boundary hardens.
4. **MCP session store is process-local:** session IDs are not persistent and are
   unsuitable for multi-process shared state.

## Verification commands

```bash
uv run python -c "from voxel_renderer import render_orthographic_views; print(render_orthographic_views([], 8, 8).keys())"
uv run pytest tests/test_visualiser.py tests/test_voxel_renderer_mcp.py tests/test_mcp_tools.py
uv run pytest
```
