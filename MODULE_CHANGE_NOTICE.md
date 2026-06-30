# Notice: change inside `prefab-housing` (housing generation & designs)

Per our agreement to **notify on any tweak around housing generation & designs**,
this is a heads-up that the vendored `prefab_housing` module was modified in the
`gdmc2026` integration. Everything else in the module remains byte-identical to
your deliverable.

## What changed and why

**Problem:** exterior walls rebake per-biome (the narrative bake feeds a
biome-specific `.wallface` whose layer-0 base block is the biome family material),
but the **interior shell** did not — the boxed cell's floor, ceiling and walls
were emitted from the static `wall_exterior` palette slot (`white_concrete`), so a
desert/jungle/snowy house had biome exterior walls wrapped around a white-concrete
floor, ceiling and interior walls.

**Fix:** the interior shell (floor + ceiling + walls of `build_placeholder_cell`)
now resolves a **biome-appropriate stone/brick** material from the active wallface
design's base plane. Exterior faces exposed to air are still repainted on top by
the wallface overlay (`build_exterior_face_overlay`), so the **outside walls keep
their own per-biome material and are untouched** — only interior surfaces and the
floor/ceiling change. Mapping (`_SHELL_BY_WALL_BASE` in `shell.py`):

| Biome (wall base) | Floor |
|-------------------|-------|
| temperate / default (oak) | `stone_bricks` |
| birch | `stone_bricks` |
| jungle | `mossy_stone_bricks` |
| snowy (spruce) | `polished_diorite` |
| savanna (acacia) | `polished_granite` |
| swamp (mangrove) | `mossy_cobblestone` |
| dark forest (dark oak) | `deepslate_bricks` |
| desert (sandstone) | `cut_sandstone` |
| badlands (red sandstone) | `cut_red_sandstone` |
| sci_fi default (white_concrete) | unchanged (`white_concrete`) |

Unmapped design bases fall back to `stone_bricks`. Applies to the floor, ceiling
and interior walls; exterior walls are untouched.

### Files touched

1. `src/prefab_housing/wallface.py`
   - Added `base_wall_block(design) -> str`: returns the bulk material of layer 0
     (the base wall plane) — i.e. the biome family `base` block the walls render
     in. Falls back to `DEFAULT_BASE_WALL_BLOCK` when layer 0 is empty. Exported
     in `__all__`.

2. `src/prefab_housing/catalogue/shell.py`
   - Added `_SHELL_BY_WALL_BASE` (biome wall-base → stone shell block) and
     `_resolve_shell_block(palette)`: reads the active wallface design (via the
     existing `_resolve_wall_face_design_path()`), maps its `base_wall_block` to
     a stone material (default `stone_bricks`), and falls back to
     `palette[SLOT_WALL_EXTERIOR]` when no design is resolvable.
   - `build_placeholder_cell(...)` now lays the floor, ceiling and all four walls
     with the resolved shell block instead of `palette[SLOT_WALL_EXTERIOR]`.
     (`_emit_floor_and_ceiling` is unchanged from upstream — single `block_id`.)

### Behaviour / compatibility

- **No format change** to `.wallface`, palette, or package formats.
- **No new module inputs** — uses the already-active wallface design.
- Default runs (no biome design active) fall back to
  `modular_default.wallface` → `white_concrete`, unchanged from before.
- Only the **interior** floor course changes; perimeter floor cells sit under the
  walls and are unaffected visually.
- Existing tests pass (`test_wallface.py`, `test_api_smoke.py`,
  `test_openings.py`).

### Open question for you

A real per-biome floor arguably belongs in the module's design/palette layer
rather than being inferred from the wallface base (e.g. a dedicated `SLOT_FLOOR`
populated per material theme, or a `floor` directive in the wallface format).
If you'd prefer to own this differently, happy to back this out and adopt your
approach.
