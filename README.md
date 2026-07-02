# GDMC 2026 Procedural Settlement Pipeline

This repository builds Minecraft-compatible GDMC settlements from generated
terrain context, narrative identity, prefab housing packages, and final world
decoration passes.

The current main path is orchestration-first: derive identity, bake the matching
wall-face packages, generate the town, add the narrative layer, then run the
final lighting sweep.

## Modules

The project is split around three collaborator-owned workstreams plus shared
integration tools.

---
- **Map and terraformation** (`map_manager.py`, `voronoi.py`, `marker.py`,
  `terraformer.py`, `plotter.py`, `builder.py`) owns the settlement site. It
  captures map data, derives buildable regions, marks paths and plots, reshapes
  terrain, previews volumes, and writes final blocks to the Minecraft world.

---

- **Narration** (`narrative/`) owns the settlement fiction layer. It generates
  identity, biome and mood context, district content, area discovery data,
  relics, premade placements, and wall-face package selection. It can call an
  LLM, but has offline fallback content.

---

- **Housing** (`prefab-housing/`) owns procedural building generation. The
  installable `prefab_housing` package turns housing requests into topology
  plans, semantic cells, staged block output, interiors, wall-face skins,
  upgrade packages, and town-lighting inputs. Housing interior is generated using MCP-backed Vision-LLM Assited Design Loop, then cached into local files for runtime use.
- **Voxel rendering** (`voxel-renderer/`) is shared review tooling. The
  installable `voxel_renderer` package renders neutral block arrays and exposes
  `voxel-renderer-mcp`; This is so that design iterations are isolated from Live Minecraft Client.
- **Orchestration and scripts** (`run_settlement.py`, `scripts/`) connect the
  workstreams. `run_settlement.py` runs identity, wall-face bake, town
  generation, narrative placement, and final lighting in order. `scripts/`
  contains focused preview, sweep, package-generation, live-placement, and
  animation commands.

---

## Quick Start

```bash
uv sync --all-packages
uv run python run_settlement.py
```

## Runtime Notes

- Python 3.12+ and `uv` are expected.
- Hosted narrative generation uses `LLM_API_KEY` when present.
- Missing or blank LLM configuration falls back to authored offline content.
- Generated galleries, caches, world outputs, and local scratch files should
  stay out of source control.
