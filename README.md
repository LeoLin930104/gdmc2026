# GDMC Procedural Prefab Housing

A procedural pipeline that emits Minecraft-block-compatible **exterior shells**
plus a **rich semantic-cell layer** for downstream interior generation. v1 targets
the GDMC competition; production placement is delegated to a sibling GDPC
adapter.

The current planning core is request-driven and topology-first:

- footprint is a hard cap, not a fill target
- height is inferred by the planner unless an explicit cap is provided
- broad utility classes drive internal programme, scale, storey and massing stages
- topology review and renderer galleries are first-class iteration loops

## Workspace layout

- `voxel-renderer/` — dev-only headless renderer (galleries + visual review)
- `prefab-housing/` — the pipeline
- `narrative/` — settlement identity, wallface baking, area discovery, relics, and premade narrative placement
- `narrative_wallfaces/` — checked-in mood/biome wallface designs used by baked prefab packages
- `builder.py`, `map_manager.py`, `plotter.py`, ... — root-level GDMC settlement generator modules
- `docs/implementation-choices.md` — binding architectural decisions and rationale
- `docs/implementation-history.md` — chronological change log
- `docs/onboarding.md` — current-entry guide for contributors

## Quick start

```bash
uv sync --all-packages
uv run python run_settlement.py --dry-run
uv run python run_settlement.py
uv run python main.py
uv run pytest
```

`run_settlement.py` is the integrated identity → wallface bake → town generation
→ narrative-layer orchestrator. Live runs require Minecraft plus the GDMC HTTP
interface; `--dry-run` validates command wiring without world writes.

## LLM config (`.env`)

The narrative layer calls a hosted LLM. Config lives in the
repo-root `.env` (tracked, so it's easy to find - real shell env vars override
it):

```dotenv
LLM_API_KEY=                                  # required; blank → offline fallback content
LLM_API_BASE_URL=https://api.openai.com/v1    # optional; any OpenAI-compatible endpoint
LLM_API_MODEL=gpt-4o-mini                      # optional
NARRATIVE_FALLBACK_VARIANT=1                    # offline lore when no key: 1=Emberwell, 2=Saltmere, 3=Karrhold
```

Without a key the pipeline still runs, generating authored offline content
instead of live text.

## Public API

```python
from prefab_housing import build_house, Brief

result = build_house(
    Brief(
        occupant_count=4,
        household_type="single_family",
        material_theme="sci_fi_modular",
        seed=42,
    ),
    footprint_xz=(50, 50),
    utility_type="residential",
)
# result.blocks          → list[SemanticBlockDict] (renderer/GDPC compatible)
# result.semantic_cells  → list[SemanticCell]
# result.block_stages    → ordered BlockGenerationStage signals
# result.metadata        → HouseMetadata
```

The newer request-driven planning entry point is:

```python
from prefab_housing import HousingRequest, generate_housing_plan_for_request

plan = generate_housing_plan_for_request(
    HousingRequest(
        footprint_xz=(32, 24),
        utility_type="residential",
        seed=42,
    )
)
```

Use `capacity_override` only when the outside controller truly needs an exact
occupancy target. Otherwise let the planner infer height and capacity from
footprint and utility type.

## Review Loops

```bash
uv run python scripts/preview_pod.py
uv run python scripts/preview_room_layout.py
uv run python scripts/preview_stair_stack_three_storey.py
uv run python scripts/preview_housing_plan.py --log-level INFO
uv run python scripts/plan_profile_sweep.py --log-level INFO
uv run python scripts/plan_profile_batch_sweep.py --log-level INFO
uv run python scripts/preview_wallface_design.py prefab-housing/designs/modular_default.wallface
uv run python scripts/animate_residential_upgrade_minecraft.py
```

`preview_pod.py` renders the shared exterior face builder in isolation.
`preview_room_layout.py` renders the current per-room layout planner with
opening annotations.
`preview_stair_stack_three_storey.py` is the dedicated cutaway vertical-stack
review loop for 1→2 stairs, 2→3 stairs, and top-floor aperture carry-through.
It now emits shell-free `stairs_only_1_2` / `stairs_only_2_3` views and slices
every scene from one shared three-storey stack plan, so cross-storey landing
checks do not drift due to preview-only phase resets.
`preview_housing_plan.py` remains the fast topology-only massing preview.
`plan_profile_sweep.py` now renders the solved plans through the full exterior
composition path and emits per-stage timings when logging is enabled.
`plan_profile_batch_sweep.py` is the planning-only budget/tuning loop.
`preview_wallface_design.py` and `scripts/wallface_editor.py` review swappable
wall-face decoration without rebaking structure.
`animate_residential_upgrade_minecraft.py` exports residential level states,
cacheable structure-shape payloads, and diff-only upgrade payloads; rerun it
with `--live` while GDMC-HTTP is running to animate level 1, then level 1→2 and
level 2→3 diffs in the Minecraft client. Live playback waits 3 seconds before
each upgrade diff by default; use `--upgrade-delay-s` to tune that pause.
For committed production payloads, pass `--input-package` to place the compact
cached package without regenerating:

```bash
uv run python scripts/animate_residential_upgrade_minecraft.py \
  --input-package prefab-housing/production_cache/residential_upgrade/seed_043.pbp \
  --live
```

Current shell review rule:

- structural cells are boxed first
- neighbour passages are drilled as the next structural stage
- wall-face textures, foundation, trim, roof, and interiors are later swappable stages
- exterior overlays apply only to air-exposed side faces
- each exposed face uses one outer frame rectangle and one filled inner accent rectangle
- the two rectangles always keep at least one air block between them on every side
- wall-face textures are decorative skin, so they deliberately ignore the
  structural AABB clip and may protrude beyond the construction footprint
  by their designed depth

Only `structural_shell` and `connection_openings` are marked for reusable
structure-template baking. Decoration stages (`wall_face_textures`,
`foundation`, `trim_bands`, `roof`) and `populate_interiors` are intentionally
excluded so themes and furnishings can be swapped after cache lookup.

See `docs/implementation-choices.md` and `docs/onboarding.md` for current design
constraints and working practices.
