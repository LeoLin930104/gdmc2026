# Implementation History

Chronological record of significant changes. New entries go on top.

Entries should answer: **what changed**, **where**, **why**.

---

## 2026-06-12 — Residential upgrade Minecraft animation export

**What:**

- Added GDPC as a root dependency so live Minecraft placement can run from the
  current workspace environment.
- Added an ignored `quarantine/` workspace for inspecting the older ICELAB
  animation/GDPC logic without importing its renderer code into this package.
- Added a residential upgrade export/live-placement path that builds level 1,
  then applies only level 1→2 and level 2→3 block diffs.
- Added local structure-cache exports beside each rendered level. These payloads
  replay only stages marked `include_in_structure_template`, so they contain the
  cacheable house body and exclude swappable wall-face decoration, roof, trim,
  foundation, and interior population.
- Added regression coverage for diff reconstruction and structure-cache file
  emission.

**Where:** `.gitignore`, `pyproject.toml`, `uv.lock`,
`prefab-housing/src/prefab_housing/minecraft_animation.py`,
`prefab-housing/tests/test_minecraft_animation.py`,
`scripts/animate_residential_upgrade_minecraft.py`, `README.md`,
`docs/{implementation-history.md,onboarding.md}`.

**Why:** Residential upgrade levels need a reusable data path for both local
cache testing and future construction animation. Keeping GDPC at the script
boundary preserves testable generation/diff logic while still allowing live
client playback when GDMC-HTTP is available.

---

## 2026-06-11 — Wall-face decoration exempt from structural AABB clipping

**What:**

- Preserved wall-face overlays as proud multi-layer decoration on every
  air-exposed side face.
- Scoped `site_footprint_clip` so it ignores `wall_face_textures` when
  computing removed positions.
- Added a regression test proving full-footprint houses keep protruding
  wall-face accent blocks after final stage replay.

**Where:** `prefab-housing/src/prefab_housing/exterior.py`,
`prefab-housing/tests/{test_modular_assembly.py,test_upgrade.py}`, `README.md`,
`docs/{implementation-choices.md,implementation-history.md,onboarding.md}`.

**Why:** The construction AABB is the boundary for the modular house body and
interiors, not for swappable wall-face skins. Flattening boundary faces to obey
the AABB collapsed the intended multi-layer panel shape and produced visibly
broken facades.

---

## 2026-06-11 — Staged block emission and legacy preview cleanup

**What:**

- Reordered exterior block generation into explicit replayable stages:
  `structural_shell`, `connection_openings`, `wall_face_textures`,
  `foundation`, `trim_bands`, `roof`, and site clipping.
- Added `BlockGenerationStage` output on `HouseResult` so construction
  animation systems can observe generation phases directly.
- Marked only `structural_shell` and `connection_openings` as structure-template
  cacheable. Wall-face textures, foundation, trim, roof, and
  `populate_interiors` are late/swappable phases.
- Removed superseded preview scripts:
  - `scripts/preview_stairwell.py`
  - `scripts/preview_stairwell_context.py`
  - `scripts/preview_stair_stack.py`
  - `scripts/render_single_occupant_house.py`
- Removed generated `scripts/__pycache__/` artefacts from the working tree.
- Updated current-facing README/onboarding/implementation-choice docs to remove
  stale review-loop commands and old output-contract text.

**Where:** `prefab-housing/src/prefab_housing/{api.py,boundary.py,decorate.py,exterior.py,materialise.py,types.py,__init__.py}`,
`prefab-housing/tests/test_modular_assembly.py`, `scripts/`, `README.md`,
`docs/{implementation-choices.md,implementation-history.md,onboarding.md}`.

**Why:** Roof generation was effectively too early in the flat block-list
composition and could visually compete with wall-face texture blocks. The new
stage replay makes ordering explicit, gives roof blocks deterministic
late-stage priority, and separates reusable structural templates from swappable
decor/interior population.

Verified:

- `uv run pytest -q prefab-housing/tests/test_modular_assembly.py prefab-housing/tests/test_openings.py prefab-housing/tests/test_wallface.py prefab-housing/tests/test_housing_plan.py` → 36 passed.
- `uv run pytest -q` → 100 passed.

---

## 2026-05-22 — Sequential opening policy, room-layout review, and finalised stairwell ascent

**What:**

- Added explicit per-cell opening policy contracts and threaded them through the
  live pipeline:
  - `types.py` now carries `open_faces`, `opening_pattern`, and `cell_index`
    through semantic-cell, room-constraint, and room-layout records.
  - New `connection_policy.py` derives a post-plan `ConnectionPolicy` from room
    labels plus adjacency, distinguishing explicit `door_faces` from free
    `open_faces`.
  - `annotate.py` and `materialise.py` now consume `ConnectionPolicy` instead of
    inferring passages directly from raw adjacency.
- Added room-layout review tooling:
  - new `room_review.py` report layer with in-figure legend and `D` / `O` / `W`
    boundary annotations
  - new `scripts/preview_room_layout.py`
- Reworked `interior.py` room planning from fixed anchor-slot placement to an
  occupancy-aware layout pass with circulation reservation, relation heuristics,
  rotated footprints, and overlay-safe overlap rules.
- Added a fixed procedural stairwell generator in `stairwell.py` and completed
  the first functionally coherent version:
  - shared central vertical shaft aperture between stacked stairwell cells
  - compact wall-hugging ascent route with a full-block turn buffer and no
    teleport gap into the upper landing
  - explicit supported upper and lower landings
  - stair blocks now emit explicit `shape="straight"`
  - stairwell interiors no longer refill the carved shaft aperture with landing
    slabs
- Preserved blockstate `properties` during interior block translation so emitted
  stair orientation survives into full assembled-house renders.
- Normalised renderer blockstate defaults for stairs in
  `voxel_renderer.blockstate_resolver`: omitted stair `shape` now resolves to
  `straight` instead of falling back to the first blockstate variant.
- Added dedicated stair review scripts:
  - `scripts/preview_stairwell.py`
  - `scripts/preview_stairwell_context.py`
  - `scripts/preview_stair_stack.py`
- Added project-local opencode MCP wiring via `opencode.jsonc` so the existing
  `voxel-renderer` MCP server can be used directly for iterative review.

**Where:** `prefab-housing/src/prefab_housing/{types.py,annotate.py,materialise.py,connection_policy.py,interior.py,room_review.py,stairwell.py,api.py,exterior.py,roof.py,__init__.py}`,
`prefab-housing/tests/{test_openings.py,test_room_review.py,test_modular_assembly.py}`,
`voxel-renderer/src/voxel_renderer/blockstate_resolver.py`,
`voxel-renderer/tests/test_visualiser_smoke.py`,
`scripts/{preview_room_layout.py,preview_stairwell.py,preview_stairwell_context.py,preview_stair_stack.py,render_single_occupant_house.py}`,
`opencode.jsonc`.

**Why:** The previous pipeline conflated adjacency with permission to open,
which made it impossible to suppress passages cleanly or hand room layout the
real opening contract. The first stairwell pass also produced a visually broken
flight because its ascent path teleported across the room and the renderer would
mis-resolve stairs without an explicit `shape`. The current pass closes both
systemic defects: opening authority is now explicit and sequential, and the
stairwell is render-stable enough for further architectural iteration rather
than basic debugging.

Verified:

- `uv run pytest -q prefab-housing/tests/test_openings.py prefab-housing/tests/test_room_review.py` → 12 passed.
- `uv run pytest -q voxel-renderer/tests/test_visualiser_smoke.py voxel-renderer/tests/test_voxel_renderer_mcp.py` → 8 passed.
- `uv run pytest -q` → 88 passed.
- `uv run python scripts/preview_stairwell.py` succeeded.
- `uv run python scripts/preview_stairwell_context.py` succeeded.
- `uv run python scripts/preview_stair_stack.py` succeeded.
- `uv run python scripts/render_single_occupant_house.py` → `grid=(2, 2, 2) score=0.321 blocks=1404`.

**Vulnerability:** The finalised stairwell is mechanically coherent but still a
single compact wall-hugging pattern per storey parity. That is the least
fragile v1 shape, but it will become repetitive in large multi-stack houses
unless a later phase adds controlled stairwell variants without reopening the
shaft/landing consistency problem.

---

## 2026-05-15 — Modular connection drilling, two-rectangle face rule, and script cleanup

**What:**

- `materialise.py` now resolves horizontal neighbour passages by occupied-cell
  adjacency, drilling both sides of each pair in one pass instead of depending
  on legacy one-sided `DOOR` / `OPEN` solved-face semantics.
- `catalogue.shell.build_face_texture_panel(...)` now follows the accepted wall
  rule directly:
  - one proud outer frame rectangle
  - one filled inner accent rectangle
  - at least one air block between the two on every side
- `scripts/preview_pod.py` was rewritten as a thin wrapper over the shared face
  builder so the standalone study cannot drift from live shell geometry.
- Deleted obsolete legacy sweep scripts `scripts/sample_sweep.py` and
  `scripts/sweep_render.py`; the active review loops are now
  `preview_pod.py`, `preview_housing_plan.py`, `plan_profile_sweep.py`, and
  `render_single_occupant_house.py`.

**Where:** `prefab-housing/src/prefab_housing/{materialise.py,catalogue/shell.py}`,
`scripts/{preview_pod.py,render_single_occupant_house.py}`,
`docs/{implementation-choices.md,implementation-history.md,onboarding.md}`,
`README.md`.

**Why:** The shared-face preview had drifted into revoked `narrow_core` variants,
while the live renderer still had two systemic defects: some occupied neighbour
pairs remained sealed, and the accepted exterior overlay rule was not enforced as
an exact two-rectangle composition with a visible air band. Unifying the study
script around the real builder removes preview-only entropy and makes visual
inspection lower-risk.

Verified:

- `uv run pytest -q prefab-housing/tests/test_api_smoke.py prefab-housing/tests/test_housing_plan.py` → 15 passed.
- `uv run python scripts/render_single_occupant_house.py` → `grid=(2, 2, 2) score=0.369 blocks=1384`.
- Mechanical audit on the seed-42 single-occupant render confirmed:
  - every occupied horizontal neighbour pair opened `12/12` passage voxels
  - every exposed filled accent face retained a one-block air gap on all sides

**Vulnerability:** The inner accent now degrades by omission on faces smaller
than `6` voxels on either axis. That is the least fragile failure mode under the
current rule, but it means very small future cell sizes will lose accent fill
rather than scaling smoothly.

---

## 2026-05-15 — Exterior composition module and full profile render pass

**What:**

- Added `prefab_housing.exterior` as the dedicated whole-building exterior
  design boundary.
- New functions:
  - `design_plan_exterior_layout(...)`
  - `render_plan_exterior(...)`
  - `render_plan_exterior_with_layout(...)`
- `api.build_house(...)` now delegates shell + decoration + roof assembly to the
  new exterior module instead of composing those passes inline.
- `scripts/plan_profile_sweep.py` now renders standard presets with the live
  exterior pipeline rather than topology-only coloured cubes.
- `preview_housing_plan.py` remains the fast topology-only loop; the profile
  sweep is now the slower but higher-fidelity whole-exterior comparison pass.

**Where:** `prefab-housing/src/prefab_housing/{api.py,exterior.py,__init__.py}`,
`scripts/plan_profile_sweep.py`, `prefab-housing/tests/test_housing_plan.py`,
`README.md`, `docs/{onboarding.md,implementation-choices.md,implementation-history.md}`.

**Why:** Roof and shell iteration had no stable module boundary; callers were
rebuilding the full exterior stack manually. Extracting a dedicated exterior
module lowers coupling and makes future facade/roof work a single-system edit.
Switching the profile sweep onto that module also gives a real whole-building
render pass across the standard presets instead of a topology-only proxy.

Verified:

- `uv run pytest -q prefab-housing/tests/test_housing_plan.py prefab-housing/tests/test_api_smoke.py` → 16 passed.
- `uv run python scripts/plan_profile_sweep.py --log-level INFO` succeeded.
- Exterior profile outputs written to `out/galleries/plan_profile_sweep/` for:
  - `small_house`
  - `townhouse`
  - `courtyard_family`
  - `quirky_stack`
  - `grand_mansion`
  - `sky_scraper`
- Overview image written to `out/galleries/plan_profile_sweep/overview_iso.png`.

**Vulnerability:** The full exterior sweep is materially slower than the old
topology-cube pass because it now runs modular shell placement, connection
drilling, decoration, roof generation, and rendering for every profile. The
fast planning-only loop therefore still matters operationally.

---

## 2026-05-15 — Inferred-height planner tuning, stage timings, and structural JIT

**What:**

- `housing_plan` now resolves explicit internal planning stages before search:
  scale class, storey distribution, planning grid, and massing profile.
- `HousingRequest` gained `capacity_override`; exact capacity is now supported
  without reintroducing old `residential_single` / `residential_multi` request
  semantics.
- Residential height selection shifted to inferred behaviour by default. Larger
  family footprints now receive stronger vertical pressure, while small houses
  no longer over-verticalise when no explicit storey cap is present.
- Grid ranking now penalises vertical shortfall before favouring wide floor
  plates, which lifted `courtyard_family` and `grand_mansion` out of shallow
  2-storey boxes.
- Added stage-level timing logs to `generate_housing_plan`, surfaced through
  `--log-level` in `scripts/preview_housing_plan.py` and
  `scripts/plan_profile_sweep.py`.
- Added `numba` and JIT-accelerated the structural occupancy/support pass in
  `structure.py`. `search.score` now reuses one structural report across the
  three structure-dependent score components instead of recomputing it.

**Where:** `prefab-housing/src/prefab_housing/housing_plan.py`,
`prefab-housing/src/prefab_housing/search/{mcts,priors,score}.py`,
`prefab-housing/src/prefab_housing/structure.py`,
`prefab-housing/tests/{test_api_smoke,test_empty_perimeter_prior,test_housing_plan}.py`,
`scripts/{preview_housing_plan,plan_profile_sweep}.py`,
`prefab-housing/pyproject.toml`, `uv.lock`.

**Why:** The planner was explicitly asking large-family profiles to remain
shallow, so MCTS was solving the wrong envelope rather than merely choosing poor
tiles. The timing output also made the runtime shape visible: search remains the
dominant cost by a wide margin, so structural analysis was the first safe JIT
target.

Verified:

- `uv run pytest -q` → 52 passed.
- `uv run python scripts/plan_profile_sweep.py --log-level INFO` succeeded.
- `courtyard_family` now selects `(4, 3, 3)` instead of `(4, 2, 3)`.
- `grand_mansion` now selects `(3, 3, 3)` instead of `(3, 2, 3)`.

**Vulnerability:** Search still dominates total runtime by roughly two orders of
magnitude over the planning stages, so structural JIT alone does not materially
change settlement-scale throughput yet. The next low-yield vector is likely
deeper MCTS rollout/score profiling rather than more planner micro-optimisation.

---

## 2026-05-12 — Sealed-pod sci-fi shell: cube-edge outline, support rule, preview sweep

**What:**

- Replaced the transient Tarrytown/chamfer experiment with the accepted v1
  exterior language: **sealed sci-fi prefab pods**.
- `catalogue.shell` rewritten around three emitters:
  - **Inset wall cube.** Horizontal wall planes shift inward by 1 voxel, so
    proud-out detail can sit on the cell boundary without leaking beyond the
    cell footprint.
  - **Cube-edge outline.** `_emit_pod_outline` emits black-concrete corner
    posts plus the floor-perimeter ring. The top course is intentionally
    omitted so the roof generator owns the upper silhouette.
  - **Proud accent panel.** Utility-coloured accent rectangles emit only on
    air-exposed horizontal faces. Internal pod seams remain neutral.
- `materialise._air_exposed_face_mask` now distinguishes true air exposure
  from mere grid-boundary status. A face emits white wall + accent when the
  neighbour cell is absent or EMPTY; otherwise the wall dissolves.
- `palette.sci_fi_modular` now resolves to the final accepted v1 surface set:
  white-concrete walls, black-concrete pod frame, dark-oak roof, coloured
  concrete utility accents.
- `pod_types._build_compat_table` no longer allows `EMPTY ↔ FLOOR`. A
  habitable pod may sit on ground (`FLOOR ↔ EXTERIOR` at storey 0) or on a
  non-EMPTY pod below, but not above void. `EMPTY ↔ CEILING` remains allowed.
- `tests/test_empty_perimeter_prior.py` updated to assert the structural rule
  directly instead of assuming `strength=0` implies a fully occupied grid.
- Added iteration harnesses:
  - `scripts/preview_pod.py` for a deterministic single-pod shell render.
  - `scripts/sample_sweep.py` for a 12-sample visual sweep across storey,
    footprint, occupancy and seed ranges.

**Where:** `prefab-housing/src/prefab_housing/catalogue/shell.py`,
`prefab-housing/src/prefab_housing/materialise.py`,
`prefab-housing/src/prefab_housing/palette.py`,
`prefab-housing/src/prefab_housing/catalogue/pod_types.py`,
`prefab-housing/tests/test_empty_perimeter_prior.py`,
`scripts/preview_pod.py`, `scripts/sample_sweep.py`.

**Why:** The accepted exterior needed to read cleanly at the fixed v1 voxel
budget, preserve identifiable per-pod silhouettes in assembled houses, and
avoid structurally implausible floating rooms. The sealed-pod model achieves
that with a low-complexity rule set: only air-exposed walls render white shell
surfaces, every pod always writes its outline, and support remains a purely
local compatibility constraint.

Verified:

- `uv run pytest -q` → 42 passed.
- `scripts/preview_pod.py` → 348 blocks, top-view confirms closed cube-edge
  outline with no diagonal corner gap.
- Seed-42 house render (`footprint=(24,24)`, `search_iterations=64`) → 4376
  blocks, `score_total=0.7853125`, `structural_plausibility=1.0`.
- `scripts/sample_sweep.py` rendered 12 samples into
  `out/galleries/sample_sweep_v2/`; the shell held across 1-4 storeys,
  16x16-40x40 footprints, and multiple seeds.

**Vulnerability:** the roof clamp still produces a fairly tall central pyramid
on wide single-storey buildings. This is a roof-shaping issue rather than a
shell defect, but it dominates the silhouette in some sweep outputs.

---

## 2026-05-08 — Modular pod aesthetic: chamfered corner posts + rounded-rect utility decor [SUPERSEDED]


**What:**

- `catalogue.shell` exterior WALL faces now emit a bounded Tarrytown-inspired
  skin within the existing 8×6×8 cell envelope:
  - cream/quartz perimeter posts and rails on exterior boundary walls;
  - two side-by-side coloured concrete pod panels retained as utility labels;
  - cream central mullion between the two panels;
  - door/window cuts still take precedence and keep dedicated aperture frames.
- `palette.sci_fi_modular` aperture materials changed from industrial
  iron/grey-concrete framing to `stripped_spruce_wood`, while the inter-storey
  band remains `spruce_planks` and pod accents remain coloured concrete.
- Rendered targeted visual gallery: `out/galleries/tarrytown_seed42_v2/`
  (gitignored). Designer review accepted the result as a v1 approximation under
  current cell-size and two-panel constraints.

**Where:** `prefab-housing/src/prefab_housing/catalogue/shell.py`,
`prefab-housing/src/prefab_housing/palette.py`.

**Why:** The previous exterior read as sci-fi panel modules. The reference set
for Tarrytown/BOTW uses a high-contrast white structural shell, painted vertical
colour panels, warm timber details, and dark stepped roofs. This pass shifts the
material language toward that target without changing macro shape, public API,
search, interiors, terrain, or GDPC placement.

Verified: `uv run pytest -q` → 18 passed. Seed 42 targeted render at 48
iterations produced score 0.821875 with `structural_plausibility=1.0`.

**Vulnerability:** at the fixed v1 cell height, panel infill has only a small
vertical span and remains rectilinear rather than chamfered/octagonal. Closer
Tarrytown fidelity would require either larger cells or extra protruding trim,
which would change the current silhouette contract.

---

## 2026-05-08 — Unified mask-erosion roof per same-height region

**What:**

- `roof.generate_roof` rewritten from per-cell pyramids to a region-based
  mask-erosion stepped roof. New algorithm:
  1. Compute `top_iy[ix, iz]` = highest occupied storey per column, or -1.
  2. 4-connect cells on the `(ix, iz)` plane sharing the same `top_iy`
     value → list of regions.
  3. For each region, build a 2-D `bool` voxel mask over the region's
     bounding `(x, z)` rectangle, True where any region cell's AABB lies.
  4. Course 0 paints `roof_block` over masked voxels at `y = top_y + 1`;
     perimeter voxels become `roof_stair`. Erode mask by 1 voxel on every
     side and repeat at the next course until empty.
- New helpers: `_column_top_storey`, `_partition_regions`,
  `_emit_region_roof`. Numpy used for mask erosion (axis-aligned 4-neighbour
  intersection of shifted slices).

**Where:** `prefab-housing/src/prefab_housing/roof.py` (full rewrite, same
public signature).

**Why:** Per-cell pyramids produced an "egg-carton" of valleys between
adjacent same-height cells. The unified region pyramid reads as one
cohesive roof per silhouette region, while the column-top lookup makes
the roof adapt automatically to EMPTY notches introduced by the Step-4
prior — partial columns now correctly cap at the topmost occupied cell
instead of leaving an exposed dark-oak ceiling.

Verified: 36 tests pass; seed=42 produces an L-shaped house with two
distinct roof units (upper-storey hipped pyramid + stepped-back lower wing
roof). Determinism preserved.

**Vulnerability:** even-sided regions terminate erosion at a 2-voxel-wide
ridge (block-format limitation). Mask erosion is axis-aligned, so non-
rectangular regions show 1-voxel staircase aliasing on diagonals — accepted
in the voxel idiom.

---

## 2026-05-08 — Step 4: EMPTY-perimeter prior

**What:**

- `MCTSConfig.empty_perimeter_strength: float = 10.0` — new sampling-prior
  knob that boosts the unique POD_EMPTY tile's weight on perimeter cells,
  scaled by horizontal-boundary count and storey depth:
  `bias = 1 + strength * (horiz_b / 2) * (iy / max(1, cy-1))`. Ground floor
  receives zero bias (preserves entry/circulation viability); top-storey
  corners get the full `1 + strength` multiplier.
- `_compute_empty_perimeter_bias(grid, strength) -> float64[C]` —
  precomputed once per search at start of `mcts_search`. Threaded as an
  explicit array argument through `_make_node`, `_expand`, `_rollout`, and
  `_tile_weights`.
- `_tile_weights` extended: when a candidate tile is the EMPTY tile its
  weight is multiplied by `empty_bias[flat]`. Required-pod multiplier still
  applies to non-EMPTY tiles.

**Where:** `prefab-housing/src/prefab_housing/search/mcts.py`.

**Why:** Filling every cell produced an "ugly bounding box made of small
boxes" silhouette. A soft sampling bias toward EMPTY at upper-storey
perimeter cells reliably yields step-back / chamfered-corner forms across
seeds (2–5 EMPTY cells at strength=10 on a 3×2×3 grid) without breaking
the functional/circulation hard-floors. Score remains ≥0.5 in all sampled
seeds; bit-exact determinism for fixed `(brief, footprint, seed)` preserved.

Verified: 32 tests pass; seed=42 produces 4 EMPTY cells, score 0.834,
silhouette is L-shaped at top storey with lower-storey roof exposed as
de-facto terraces (proper roof handling on partially-occupied columns
follows in the next step).

---

## 2026-05-08 — Public API, annotation, circulation hard-floor + smoke tests

**What:**

- `api.build_house(brief, *, footprint_xz, …) -> HouseResult` — single
  entry point wiring `resolve_programme → design_grid → build_tile_set →
  init_state → apply_position_priors → mcts_search → materialise →
  annotate`. `footprint_xz` is in **voxels**, not cells (a cell is
  `cell_voxel_size`-wide). Re-exported from `prefab_housing.__init__`.
- `annotate.annotate(state) -> list[SemanticCell]` — emits one record per
  occupied (non-EMPTY) cell with privacy_depth from a door/open-graph BFS
  rooted at every entry pod, daylight_score from boundary-aligned WINDOW
  faces, plus door_faces / window_faces lists.
- `score.ScoreWeights.hard_floor_on_circulation: bool = True` — when
  `functional_adequacy == 1.0` but `circulation < 1.0`, scale total by
  `0.5 * circulation`. Conditional on functional being met so the MCTS
  gradient toward placing required pods is preserved during early search.
- `mcts.mcts_search` — track first solved state via `has_solved` flag so
  layouts with hard-floor-zeroed scores still register as the best
  fully-solved candidate (avoids returning the unsolved initial state).
- Five-test smoke suite at `prefab-housing/tests/test_api_smoke.py`:
  full-pipeline run, both hard-floor satisfaction, required-pod presence
  and reachability, bit-exact determinism, seed sensitivity. Added
  `pytest` as dev dependency.

**Where:** `prefab-housing/src/prefab_housing/{api,annotate,__init__}.py`,
`prefab-housing/src/prefab_housing/search/{score,mcts}.py`,
`prefab-housing/tests/test_api_smoke.py`,
`prefab-housing/pyproject.toml`.

**Why:** The pipeline now has a single externally-facing entry point that
emits the contract HouseResult (blocks + semantic cells + metadata) and a
test fence that catches regressions in WFC, MCTS, scoring, or schema. The
conditional circulation hard-floor ensures MCTS prefers connected
habitable layouts without zeroing out gradient before functional is met.

Verified: seed=42, footprint=(24,24), 128 iters → total=0.906,
functional=1.0, circulation=1.0, all 6 required-pod slots present,
2 bedrooms, kitchen+bathroom reachable from entry, deterministic on rerun.

---

## 2026-05-08 — Cell-shell synthesis + whole-house materialisation

**What:** Two new modules turn a solved `SolverState` into renderable
blocks:

- `catalogue/shell.py` — `synthesise_cell_shell(faces, boundary_face_mask,
  cell_voxel_size, palette)` programmatically synthesises the floor slab,
  ceiling slab, and four wall planes for one cell at local origin. WALL
  planes paint exterior or interior material based on whether the face
  points outside the grid. DOOR cuts: 2×3 opening with a frame ring.
  WINDOW cuts: 4×2 opening with a frame ring + glass infill. ``OPEN`` and
  ``EMPTY`` faces emit nothing.
- `materialise.py` — `materialise(state, palette)` walks every assigned
  cell, calls the synthesiser with the *already-rotated* face profile from
  the tile set (rotation is baked into face categories — no
  property-rotation pass needed for v1 sci-fi shell), translates local
  blocks to world coords via `grid.cell_voxel_bbox`, and concatenates.

Smoke test: 3×2×3 grid, seed 42, 256 MCTS iters → score 0.8946, 2176
blocks emitted across 10 occupied cells (programme satisfied: entry,
living, kitchen, bathroom, 2 bedrooms, plus 2 stairwells aligned
vertically and 2 corridors).

**Where:** `prefab-housing/src/prefab_housing/catalogue/shell.py`,
`prefab-housing/src/prefab_housing/materialise.py`.

**Why:** Programmatic synthesis avoids a 7-pod × 4-rotation × N-voxel
hand-authored prefab table and lets the DOOR/WINDOW cut geometry track the
cell voxel size automatically. The trade-off is a known v1 quirk: every
cell paints all four walls, so a shared interior wall becomes 2 voxels
thick (sci-fi double-skin aesthetic). Single-skin shared walls require an
ownership convention that consults the neighbour's tile profile —
deferred to v2.

**Steelman vulnerability:** Cells with ``vy < 5`` or ``vx,vz < 4``
silently degrade their cuts. The default (8×6×8) is safely above this;
asserts are not yet on the boundary.

---

## 2026-05-08 — Per-cell positional priors

**What:** New `prefab_housing.search.priors` module with
`apply_position_priors(state, programme)` enforcing three hard pre-search
domain masks:

- **R1 — Entry placement.** `entry` allowed only on ground-floor cells
  (`iy==0`) with at least one horizontal boundary face, *and* only those
  rotation variants whose DOOR face aligns with one of those boundaries.
- **R2 — Windows on boundaries.** Tiles with WINDOW on any non-boundary face
  are forbidden at that cell.
- **R3 — External doors entry-exclusive.** Non-entry tiles with DOOR on a
  boundary face are forbidden.

The mask is AND-ed into `state.domain`, dirty cells re-propagated through
AC-3.

**Where:** `prefab-housing/src/prefab_housing/search/priors.py`.

**Why:** Scoring alone could not lift `privacy_gradient` above 0.333 on the
3×2×3 smoke grid because MCTS without positional priors places windows
anywhere, doors face boundaries arbitrarily, and entry pods spawn mid-grid.
On a 3×2×3 / 6-pod programme the priors prune total domain entropy 522 → 186
(≈64 % reduction), and at 256 MCTS iterations push the best score from
0.8356 (baseline) to 0.8946 (priors) while running ≈25 % faster (smaller
domains → cheaper propagation). At 64 iters baseline can occasionally win
by luck — the priors' advantage emerges with sufficient search budget,
which matches the v1 "no compute budget" posture.

---

## 2026-05-08 — MCTS-guided WFC search

**What:** New `prefab_housing.search.mcts` module: classical UCT
(`Q + c_puct * sqrt(ln N / n)`), expansion via `collapse_to`, weighted-random
rollouts biased by a per-step *programme prior* (tiles whose pod is required-
and-currently-unmet receive a multiplicative boost, EMPTY keeps baseline
1.0). Contradictions during expansion are silently dropped; the search
records the highest-scoring fully-solved state ever visited and returns it.
RNG is a single seeded `numpy.random.Generator`; bit-exact reproducibility
verified on seed=42.

**Where:** `prefab-housing/src/prefab_housing/search/mcts.py`.

**Why:** Pure WFC random collapse converges (smoke test: 100 % solved on
3×2×3) but cannot discriminate between programme-satisfying and
programme-violating layouts — the boundary EXTERIOR sentinel pruned zero
candidates because every face category is currently EXTERIOR-compatible.
MCTS with utility-driven backup is the discriminator: in a 3×2×3 grid with
a 6-pod programme it raises functional_adequacy and circulation to 1.0
within 64 iterations. Privacy gradient remains weak (0.333) by design until
per-cell programme priors land (next step).

---

## 2026-05-08 — Prefab-housing core: grid, palette, catalogue, WFC, programme, score

**What:** Six modules forming the pre-search substrate:
- `grid.py` — `CellGrid`, face index tables, `design_grid` sizer.
- `palette.py` — theme registry (currently `sci_fi_modular` only).
- `catalogue/pod_types.py` — 8 pod labels (incl. `_empty`), face-category
  constants, `PodFaceProfile`, `FACE_CATEGORY_COMPAT` symmetric table,
  `profile_at_rotation`.
- `wfc/tiles.py` — `TileSet` (29 tiles = 1 EMPTY + 7×4 rotations), packed
  numpy arrays (`pod_index`, `rotation`, `faces`, `has_window`, …),
  precomputed `compat[T, 6, T+1]` with EXTERIOR sentinel column.
- `wfc/solver.py` — `SolverState`, AC-3 `propagate`, `collapse_to`,
  `lowest_entropy_cell`, `candidate_tiles`, copyable for backtracking.
- `programme.py` — `Brief × utility_type → Programme` (required/optional
  pod multisets); bedroom counts derived per `household_type`;
  `residential_multi` adds extra bathrooms + corridor.
- `search/score.py` — 7 utility components, `ScoreWeights` defaults
  (functional 0.25; circulation/privacy/daylight 0.15; vert-stack/aesthetic/
  structural 0.10), hard-floor on functional_adequacy soft-scales the total
  by 0.5×fa when fa<1.

**Where:** `prefab-housing/src/prefab_housing/{grid,palette,programme}.py`,
`prefab-housing/src/prefab_housing/catalogue/pod_types.py`,
`prefab-housing/src/prefab_housing/wfc/{tiles,solver}.py`,
`prefab-housing/src/prefab_housing/search/score.py`.

**Why:** DOD layout (structs of arrays, packed numpy tables, int IDs in hot
loops) keeps the seams open for a Numba/C# swap later without rearchitecture.
Tile-set redundancy (4 rotations even for symmetric pods) is accepted in v1
for code-path simplicity. The functional hard floor is a *soft* scaler, not a
hard zero, so MCTS still observes a gradient when climbing toward complete
programmes.

---

## 2026-05-08 — Orientation transformer in voxel-renderer

**What:** New `voxel_renderer.orientation` module with whitelisted Y-axis
property rotators (`facing`, `axis`, `rotation`). `prefab.rotate_y` now applies
property rotation by default, with `transform_properties=False` available for
legacy callers. Re-exported `rotate_block`, `rotate_block_properties`,
`rotate_y_property`, `KNOWN_ROTATABLE_PROPERTIES` from package root.

**Where:** `voxel-renderer/src/voxel_renderer/orientation.py`,
`voxel-renderer/src/voxel_renderer/prefab.py`,
`voxel-renderer/src/voxel_renderer/__init__.py`,
`voxel-renderer/tests/test_orientation.py`.

**Why:** WFC tile rotation needs Minecraft-faithful directional bits or stairs,
doors, and logs render incorrectly when their containing pod is rotated. Pure
data tables (no inheritance) keep the rule set inspectable and trivially
swappable to a Numba/native lookup later. Whitelist policy: unknown property
names pass through untouched — this is documented and tested rather than
silently lossy.

---

## 2026-05-08 — Bootstrap

**What:** Created uv workspace at repo root with two members (`voxel-renderer`,
`prefab-housing`). Added `docs/implementation-choices.md` and this file.
Extended `.gitignore` to exclude `out/` (renderer galleries) and tooling caches.

**Where:** `pyproject.toml`, `.gitignore`, `docs/`.

**Why:** Establish a clean two-package layout so the renderer remains an
independently extractable artefact while the new pipeline lives at the repo
root as a sibling. Galleries are regenerable and large — they belong out of git.
