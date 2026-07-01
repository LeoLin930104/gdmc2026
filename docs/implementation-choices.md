# Implementation Choices

This document records the **architectural and algorithmic decisions** made for the
GDMC procedural prefab-housing pipeline, with **rationale** and **deferred trade-offs**.
It is updated whenever a binding choice is made or revisited.

For a chronological record of *what was changed and when*, see
[`implementation-history.md`](./implementation-history.md).

---

## 1. Scope (v1)

**In scope**
- Exterior shell + macro shape + utility-typed cell labelling for broad utility
  classes: `residential`, `commercial`, `service_building`, `storage_utility`
- Coarse 3D module WFC + MCTS-guided collapse driven by weighted utility score
- Hybrid pod connectivity: face-shared backbone + distinct air-exposed pod types
- Planner-inferred scale, storey distribution, and massing profile
- Multi-storey support with inferred height by default
- Explicit post-plan opening policy and semantic room-opening export
- Cached room-layout planning, interior block population, and fixed procedural stairwell interiors
- Ordered block-generation stage signals for construction animation and cache filtering
- Statistical determinism via `seed`
- Renderer galleries on demand (gitignored)

**Out of scope (v1)**
- Terrain / biome / heightmap adaptation
- GDPC placement (production renderer)
- Settlement-scale relations between houses

---

## 2. Top-Level Architecture

```
Brief, footprint, utility_type
        │
        ▼
[1] Programme Resolver      → required pods (counts, labels)
[2] Catalogue Synth         → PodTemplate set (shell, face signatures)
[3] WFC Setup               → Tile set + face-compatibility lookup
[4] MCTS Collapse           → Cell→Tile assignment
[5] Connection Policy       → explicit openings after topology solve
[6] Exterior Stage Emit     → structural shell, carving, decor, roof, clipping
[7] Semantic Annotation     → list[SemanticCell]
[8] Interior Population     → room interiors as a late block stage
        │
        ▼
HouseResult { blocks, semantic_cells, block_stages, metadata, schema_version }
```

Block output stages are exposed as `BlockGenerationStage` records. Existing
callers still receive a flat `blocks` list; construction animation and future
structure-template baking should consume `block_stages`.

---

## 3. Repository Layout — uv Workspace

Two workspace members:

- `voxel-renderer/` — dev-only headless renderer. Production placement uses GDPC
  via a separate (out-of-scope) sibling project.
- `prefab-housing/` — this project's pipeline.

The root `pyproject.toml` declares the workspace; the root `main.py` is a smoke
entry point that calls `prefab_housing.build_house`.

---

## 4. Performance Posture

Hot paths are written in a **data-oriented design (DOD)** style — structs of
arrays, integer-indexed tables, no deep object graphs — so search and scoring
can be iterated without architectural churn.

Concrete commitments:
- Tile sets are `list[Tile]` indexed by `tile_id: int`; never traversed by attribute lookup in hot loops.
- Face-compatibility table is a `numpy.ndarray[bool]` shape `(T, 6, T, 6)`.
- MCTS state is a `numpy.ndarray[int16]` of shape `(C,)` where `C = cells_total`.
- Score components are decomposable: each returns a scalar in `[0, 1]` from primitive arrays, no mutation of shared state.
- Structural occupancy/support analysis is now JIT-accelerated with `numba`.
- Planning scripts expose per-stage timing logs through `--log-level` so bottlenecks can be observed without profiler-only workflows.

Current bottleneck posture:

- Search dominates runtime by a large margin.
- Planner pre-search stages are currently low-yield to optimise further.
- Structural analysis is cached within scoring so repeated structure passes do not multiply search cost unnecessarily.

---

## 5. Determinism

**Statistical determinism only.** Same `seed` produces statistically identical
distributions; bit-exact output not guaranteed under parallel execution. v1
runs sequentially and *is* bit-exact; the contract relaxes when M3 introduces
parallel rollouts.

All RNG paths derive from `random.Random(seed)`; iteration order over cells uses
sorted coordinate tuples; iteration over tiles uses ascending `tile_id`.

---

## 6. Cell Grid

| Parameter | Default | Rationale |
|---|---|---|
| Cell size (xz) | 8 voxels | Leaves ~6×6 walkable interior after wall thickness |
| Cell size (y)  | 6 voxels | 2-block ceiling clearance + floor + ceiling |
| Storey cap     | 4        | Default inference cap when the request omits an explicit limit |
| Footprint cell grid for 50×50 | 6×6 | `floor(50/8) = 6` |

Cells are uniform in v1. The planner treats footprint as a cap, not a fill
target, and now infers target height internally from utility load and massing
policy rather than expecting `max_storeys` to be present on routine inputs.

---

## 7. Pod Sizing

**M1**: strictly 1×1 cells per pod.
**M2**: multi-cell footprints `{1×2, 2×1, 2×2}` for residential variety.

The "1.5×" intuition from the user is honoured by 1×2 rectangles, which produce
a non-cubic silhouette without leaving the integer grid.

---

## 8. Pod Connectivity

Hybrid:
- Face-shared backbone for habitable pods.
- Distinct air-exposed pod types (`balcony`, `terrace`, `walkway`) in M2 — these
  have face-signature rules that *require* exterior open faces and *forbid* an
  upper neighbour.

---

## 9. WFC Variant

**Coarse 3D tile WFC over the cell grid.** Tiles = (template, rotation∈{0,90,180,270}).

Face signatures are categorised into a coarse equivalence class
`{wall, opening_door, opening_window, exterior_open, floor, ceiling}` before
comparison. Direct `(u,v,id)` equality is too strict for procedural variety.

**AC-3 propagation** prunes domains after each collapse step.

---

## 10. Search Strategy

**MCTS over partial collapse states** with UCB1 selection biased by utility-prior
and topology-native void pressure.

- State: partial cell→tile assignment + per-cell domain.
- Action: pick lowest-entropy cell, then choose a tile from its current domain.
- Rollout: random-weighted-by-prior collapse with AC-3.
- Backup: utility score of terminal layout.
- Budget: configurable; default ~200 rollouts in M1.
- Structural void and terrace void are distinct tile classes.
- Terrace-void weighting is storey-aware and asymmetric so upper setbacks remain reachable without degenerating into symmetric boxes.

Pre-search planning stages now run before WFC/MCTS:

1. request resolution
2. programme resolution
3. scale-class inference
4. storey-distribution planning
5. planning-grid selection
6. massing-profile planning

Score components (weights tunable):

| Component | Weight | Hard floor? |
|---|---|---|
| `functional_adequacy` | 0.25 | yes — score = 0 if any required pod missing |
| `circulation` | 0.15 | no |
| `privacy_gradient` | 0.15 | no |
| `daylight` | 0.15 | no |
| `vertical_service_stack` | 0.10 | no |
| `aesthetic_facade` | 0.10 | no |
| `structural_plausibility` | 0.10 | no |

---

## 11. Orientation Property Handling

The voxel-renderer `rotate_y` does **not** transform Minecraft orientation
properties. v1 adds a **block-orientation transformer** to voxel-renderer:

- New module `voxel_renderer/orientation.py`
- Whitelisted property rotators: `facing`, `axis`, `rotation`
- Unknown properties pass through with a warning

See voxel-renderer's own change record in [`implementation-history.md`](./implementation-history.md).

---

## 12. Materials

Palette **registry keyed by `material_theme`**. v1 ships `sci_fi_modular` only
as the default theme; M2 adds further themes.

Materials are referenced by *slot* (`wall_exterior`, `floor`, `window_glass`, …)
in catalogue templates and resolved at materialisation.

The current `sci_fi_modular` theme resolves to the accepted v1 shell:

- `wall_exterior = minecraft:white_concrete`
- `frame_block = minecraft:black_concrete`
- `roof_block = minecraft:dark_oak_planks`
- `roof_stair = minecraft:dark_oak_stairs`
- utility accents = coloured concrete keyed by pod name

The shell geometry assumes a **sealed modular-cell model**:

- Occupied cells are placed as fully boxed modules first.
- Horizontal occupied-neighbour pairs are drilled open afterward by the staged
  materialiser rather than trusting legacy per-tile door semantics.
- Accent overlays emit only on air-exposed side faces after connection carving.
- Each exposed face uses exactly two exterior rectangles: a proud outer frame and
  a filled inner accent rectangle.
- The outer frame and inner fill keep at least one air block between them on all
  four sides.
- The site AABB is a structural/interior boundary, not a wall-face decoration
  boundary. `wall_face_textures` deliberately ignores the final footprint clip
  so proud multi-layer panels keep their designed depth even on outermost faces.
- UP emits no shell decor; the roof pass owns the top surface.
- Roof generation is a late decor stage after wall-face textures and wins
  same-position collisions during deterministic stage replay.

This keeps the palette high-contrast and mechanically legible at the fixed
8×6×8 cell scale. Apertures and interior-specific slots are handled by the
explicit opening policy and interior population stage.

The current standalone shell review loop is `scripts/preview_pod.py`. It renders
the shared single-face treatment without WFC, MCTS, roof, or house-scale noise.

Whole-building exterior composition now lives in `prefab_housing.exterior` so
shell, decoration, and roof design can evolve behind one module boundary instead
of being reassembled ad hoc by each caller.

Structure-template baking must omit swappable detail stages. The current
template-eligible stages are:

- `structural_shell`
- `connection_openings`

The following are intentionally late/swappable and must be excluded from the
structure template cache: `wall_face_textures`, `foundation`, `trim_bands`,
`roof`, and `populate_interiors`.

---

## 13. Output Contracts

Opening authority is now explicit and sequential:

- topology search decides occupied cells and room labels first
- `ConnectionPolicy` derives permitted `door_faces`, `open_faces`, and
  `opening_pattern` afterwards
- shell carving, semantic export, and interior layout all consume that policy

Adjacency alone is therefore not sufficient to create a doorway.

```python
@dataclass(frozen=True, slots=True)
class SemanticCell:
    cell_index: tuple[int, int, int]
    voxel_bbox: tuple[tuple[int,int,int], tuple[int,int,int]]
    label: str
    role: Literal["habitable","service","circulation","exterior"]
    occupancy_capacity: int
    daylight_score: float
    privacy_depth: int
    door_faces: list[str]
    window_faces: list[str]
    open_faces: list[str]
    opening_pattern: str
    interior_volume_voxels: int
    pod_template_id: str
    properties: dict[str, str]
```

```python
@dataclass(frozen=True, slots=True)
class HouseResult:
    blocks: list[SemanticBlockDict]
    semantic_cells: list[SemanticCell]
    block_stages: tuple[BlockGenerationStage, ...]
    metadata: HouseMetadata
    schema_version: str   # "1.4"
```

`schema_version` is bumped on any breaking change to the interior-team contract.

---

## 14. Testing & Galleries

- pytest under both workspace members
- Gallery PNGs generated on demand into `out/galleries/<seed>/` (gitignored)
- LLM-vision review consumes those PNGs when invoked
- v1 smoke test asserts: required pods present, all habitable cells reachable
  from entry, no floating cells, deterministic output for a fixed seed
- Request-driven planning tests now prefer inferred height over explicit
  `max_storeys`, except where cap behaviour itself is under test.
- `scripts/preview_housing_plan.py --log-level INFO` and
  `scripts/plan_profile_sweep.py --log-level INFO` print stage timings to expose
  search-heavy bottlenecks.
- Room/interior review loops now include:
  - `scripts/preview_room_layout.py`
  - `scripts/preview_stair_stack_three_storey.py`
- Wall-face review loops now include:
  - `scripts/preview_pod.py`
  - `scripts/preview_wallface_design.py`
  - `scripts/wallface_editor.py`
- The project-local `opencode.jsonc` registers the existing `voxel-renderer`
  MCP server for low-latency render iteration inside opencode sessions.

---

## 15. Known Vulnerabilities

1. **Pure-Python MCTS latency.** Mitigated by AC-3 pruning + decomposable score.
2. **Face-signature equivalence is opinionated.** Raw signatures retained behind a flag.
3. **Orientation whitelist is finite.** Unknown properties warned on, not crashed.
4. **Statistical-only determinism with parallel rollouts.** v1 sequential; deferred.
5. **Schema lock-in.** `schema_version` field + deprecation policy.
6. **Privacy/daylight scored without interior** — assumptions published in `SemanticCell.properties`.

---

## 16. Open Items

- Stairwell pod grammar beyond the current fixed compact wall-hugging pattern (variants,
  asymmetric exits, wider shafts)
- Whether `entry` must be on storey-0 perimeter (likely hard constraint)
- AC-3 vs full arc-consistency vs domain restriction only — leaning AC-3
