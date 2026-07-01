# GDMC Procedural Prefab Housing Onboarding Guide

This document is the current-entry guide for the repository. It assumes senior engineering experience but no prior exposure to this codebase.

The decision log in `docs/implementation-choices.md` records binding architectural choices. The change log in `docs/implementation-history.md` records milestone history. This guide states the current runtime shape and the practical iteration loops.

## 1. Current Shape

`GDMC_Procedural-Prefab-Housing` is a topology-first procedural building pipeline for Minecraft-compatible prefab housing shells. It emits:

- renderable block output for visual review or downstream placement
- semantic-cell metadata for later interior systems
- ordered block-stage signals for construction animation and structure-template baking
- planning metadata for topology review and performance analysis

The active core is request-driven and stage-oriented:

- `HousingRequest` is the preferred planning boundary: `footprint_xz`, broad `utility_type`, optional exact `capacity_override`, optional `max_storeys`
- height is normally inferred from footprint and utility load rather than specified externally
- the planner resolves programme, scale class, storey distribution, planning grid, and massing profile before WFC/MCTS search runs
- MCTS search then solves the 3D cell topology using topology-native void tiles, structural scoring, and positional priors
- block generation is staged after topology solve: structural shell, connection carving, swappable decor, boundary clipping, and interior population

Current goal:

- keep the planner sequential where possible: topology first, explicit opening
  policy second, then shell carving and interior layout on top of that contract
- use dedicated room and stair previews as the primary review loops for interior
  work rather than relying only on whole-house shell renders
- keep structure-template baking free of swappable details: wall-face textures,
  roof, trim, foundation, and interior population are not part of the cached
  structural template
- treat the current compact wall-hugging stairwell as functionally stable;
  further stair work should add controlled variants, not re-open the
  shaft/landing correctness bug

What is in scope:

- topology planning for broad utility classes: `residential`, `commercial`, `service_building`, `storage_utility`
- sealed sci-fi shell rendering for the current `sci_fi_modular` material theme
- explicit opening policy, semantic opening export, cached room-layout planning, and interior block population
- fixed procedural stairwell interiors with dedicated preview scripts
- topology-only review loops, gallery renders, and stage timing logs
- deterministic test coverage around planning, scoring, materialisation, and API smoke behaviour

What is not in scope:

- terrain-aware siting, biome adaptation, or settlement placement
- a finished balcony/topology grammar layer for large stepped wings
- a full commercial texture language beyond the current shell face treatment

## 2. Working Model

Think in three layers.

1. Request layer

- Outside systems should describe a site cap and broad intent, not a fully-authored building type.
- `footprint_xz` is a hard cap.
- `utility_type` selects broad programme behaviour.
- `capacity_override` is only for exact occupancy targets.

2. Planning layer

- `resolve_brief_for_request(...)` infers occupant load when no exact capacity is provided.
- `_resolve_planning_stages(...)` derives internal scale/storey/massing policy.
- `_select_planning_grid(...)` picks a grid that fits within the cap while preferring usable vertical spread.

3. Search and render layer

- `build_tile_set(...)` materialises the tile catalogue.
- `init_state(...)` and `apply_position_priors(...)` set up the WFC state.
- `mcts_search(...)` solves the topology.
- `render_plan_exterior_stages(...)` emits block-stage signals in construction order.
- `compose_block_generation_stages(...)` replays those signals into the final exterior block list.
- `annotate(...)` and `generate_room_interiors(...)` produce semantic cells and the later `populate_interiors` block stage.

## 3. Repository Tour

Top-level layout:

- `prefab-housing/` — the housing pipeline package
- `voxel-renderer/` — dev-only headless renderer used for image galleries and visual review
- `scripts/` — iteration harnesses and render loops
- `docs/` — architectural decisions, history, and onboarding material
- `out/` — generated galleries and reports (gitignored outputs)

Key package paths:

- `prefab-housing/src/prefab_housing/housing_plan.py`
  - request resolution
  - planner-stage inference
  - topology search entry points
- `prefab-housing/src/prefab_housing/programme.py`
  - broad utility-type programme resolution
- `prefab-housing/src/prefab_housing/search/mcts.py`
  - MCTS-guided WFC collapse
- `prefab-housing/src/prefab_housing/search/priors.py`
  - hard placement/storey priors
- `prefab-housing/src/prefab_housing/search/score.py`
  - utility and structure scoring
- `prefab-housing/src/prefab_housing/structure.py`
  - structural analysis and the current JIT-accelerated support pass
- `prefab-housing/src/prefab_housing/catalogue/shell.py`
  - current per-cell face treatment and single-pod shell language
- `prefab-housing/src/prefab_housing/exterior.py`
  - staged whole-exterior composition boundary and block-stage replay
- `prefab-housing/src/prefab_housing/materialise.py`
  - structural shell emission, neighbour passage drilling, and facade overlay primitives
- `prefab-housing/src/prefab_housing/connection_policy.py`
  - explicit sequential opening derivation after topology solve
- `prefab-housing/src/prefab_housing/interior.py`
  - cached room-plan and room-layout generation
- `prefab-housing/src/prefab_housing/stairwell.py`
  - fixed stairwell geometry and shared vertical shaft footprint
- `prefab-housing/src/prefab_housing/plan_review.py`
  - topology review plots and reports
- `prefab-housing/src/prefab_housing/room_review.py`
  - room-layout review plots with `D` / `O` / `W` edge annotations

## 4. Preferred Iteration Loops

Use the smallest loop that answers the current question.

For current single-cell face design:

```bash
uv run python scripts/preview_pod.py
```

For topology-only planning review:

```bash
uv run python scripts/preview_housing_plan.py --log-level INFO
```

For whole-exterior profile review:

```bash
uv run python scripts/plan_profile_sweep.py --log-level INFO
```

For room-layout review:

```bash
uv run python scripts/preview_room_layout.py
```

For stair-specific review:

```bash
uv run python scripts/preview_stair_stack_three_storey.py
```

`preview_stair_stack_three_storey.py` is the authoritative inter-storey stair
review loop. It now renders shell-free `stairs_only_1_2` / `stairs_only_2_3`
inspection scenes and derives every preview scene from one shared full-stack
plan. This avoids a low-yield verification trap where partial preview stacks
reset traversal phase and show handoffs that do not exist in the live build.

For wall-face decoration review:

```bash
uv run python scripts/preview_wallface_design.py prefab-housing/designs/default_modular.wallface
uv run python scripts/wallface_editor.py --no-browser
```

For whole-pipeline smoke checks:

```bash
uv run pytest -q
```

For Minecraft/GDPC upgrade animation export:

```bash
uv run python scripts/animate_residential_upgrade_minecraft.py
uv run python scripts/animate_residential_upgrade_minecraft.py --live
```

The export writes full exterior level payloads, `structure_cache/` payloads for
the cacheable house body, and diff-only upgrade payloads for level 1→2 and
level 2→3. Live mode requires a running Minecraft client with GDMC-HTTP
listening on `localhost:9000`. Live playback waits 3 seconds before each
upgrade diff by default; pass `--upgrade-delay-s 0` for immediate upgrades.

For dependency sync across workspace packages:

```bash
uv sync --all-packages
```

`--log-level INFO` enables per-stage planning timings. At present, the dominant bottleneck is still `search_ms`, not programme/grid planning.

## 5. Current Shell Design

The accepted current shell language is a sealed sci-fi pod.

Core rules:

- cells are placed as fully boxed modules before any adjacency cuts are made
- horizontal occupied-neighbour pairs are drilled afterward so passages are consistent
- coloured exterior overlays render only on air-exposed side faces after connection carving
- each exposed face uses one proud outer rectangle plus one filled inner rectangle
- the outer and inner rectangles always keep at least one air block between them
- the site AABB clips structure/interiors, not wall-face texture skins; proud
  wall-face detail may protrude beyond the construction footprint by design
- the roof owns the upper surface; the shell does not decorate the ceiling face
- roof generation runs after wall-face textures and wins any same-position
  collision during final stage replay

The fastest way to review this design is `scripts/preview_pod.py`, which now calls the shared `build_face_texture_panel(...)` helper directly instead of maintaining divergent preview-only geometry.

Structure-template baking should replay only stages where
`include_in_structure_template` is true. At present this is limited to:

- `structural_shell`
- `connection_openings`

These deliberately exclude `wall_face_textures`, `foundation`, `trim_bands`,
`roof`, and `populate_interiors` so cached structure can be decorated and
furnished later.

## 6. Planner Behaviour You Should Assume

- Height should normally be inferred, not supplied.
- Residential has a hard minimum of 2 occupied storeys.
- Larger family housing now receives stronger vertical pressure and an eased tall-first curve.
- The planner treats footprint as a limit, not a fill target.
- Structural void and terrace void are separate topology-native tile classes.
- Terrace carving is active in search, but clustered-wing/courtyard grammar is still incomplete.

Implication: if a large residential profile still reads as a uniform box, the next likely failure point is topology grammar or score pressure, not the basic height selector.

## 7. Verification

Preferred commands:

```bash
uv sync --all-packages
uv run pytest -q
uv run python scripts/preview_pod.py
uv run python scripts/plan_profile_sweep.py --log-level INFO
uv run python scripts/preview_stair_stack_three_storey.py
```

Current stable expectations:

- tests pass end-to-end under `pytest`
- `preview_pod.py` emits the standalone current face treatment from the shared shell builder
- `plan_profile_sweep.py` emits whole-exterior profile galleries and timing logs
- `preview_stair_stack_three_storey.py` shows floor-to-floor stair handoff using
  one full-stack plan, with shell-free `stairs_only_*` scenes for landing-height
  and terminal-step inspection
- `courtyard_family` and `grand_mansion` infer 3-storey grids rather than collapsing to shallow 2-storey boxes

## 8. Open Tensions

- Search still dominates runtime. Structural JIT helps, but settlement-scale throughput is still primarily a search problem.
- Courtyard and winged residential massing is still heuristic, not a hard topology grammar.
- The current shell has a strong material identity, but the general cell-face texture language is still in active design.
- The stairwell is now mechanically coherent, but still only one parity-driven compact pattern. Variant diversity is deferred.
- Height inference is now the preferred path, so tests and scripts that hardcode `max_storeys` should exist only when they are explicitly testing cap semantics.

## 9. First Tasks For A New Contributor

1. Run `uv sync --all-packages` and `uv run pytest -q`.
2. Render `scripts/preview_pod.py` to see the standalone shared face treatment.
3. Render `scripts/plan_profile_sweep.py --log-level INFO` to inspect current exterior behaviour and runtime cost.
4. Read `housing_plan.py`, `search/mcts.py`, `search/score.py`, `exterior.py`, and `catalogue/shell.py` in that order.
5. Only then start changing either topology grammar or shell/texture design.

## 10. Glossary

- Brief: legacy direct programme input used by `build_house(...)`.
- HousingRequest: preferred external planning request contract.
- Programme: resolved required and optional pod multiset.
- Scale class: internal coarse size archetype inferred from utility load and site cap.
- Storey distribution: planned vertical split of public/private/service usage.
- Massing profile: internal terrace/asymmetry policy given to the search layer.
- Structural void: true empty/support-breaking void tile.
- Terrace void: softer setback/carve-out void tile intended for upper-storey massing.
- Semantic cell: downstream metadata record for one occupied cell.
- Block generation stage: replayable signal for construction animation and cache filtering.
- Stage timings: per-subsystem runtime measurements emitted by planning scripts when logging is enabled.
