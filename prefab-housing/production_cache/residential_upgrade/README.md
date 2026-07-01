# Tracked Residential Upgrade Packages

These packages are committed intentionally. They are portable GDPC placement
payloads for live Minecraft review and should not be treated as regenerable
`out/` artefacts.

Each `.pbp` package contains:

- compact full level payloads for levels 1, 2, and 3
- compact core level payloads that exclude wallface texture blocks
- separate wallface sections that can be ignored or swapped by runtime tooling
- compact diff-only upgrade payloads for `1->2` and `2->3`
- compact structure-cache payloads for reusable structural templates
- a trailing JSON manifest and block-state palette

Prepared variants:

- `seed_043.pbp`: WFC/MCTS-generated layout, `modular_var1.wallface`,
  rustic-cabin base interiors plus generated room variants
- `seed_044.pbp`: WFC/MCTS-generated layout, `modular_var1.wallface`,
  modern-minimalist base interiors plus generated room variants
- `seed_045.pbp`: WFC/MCTS-generated layout, `modular_default.wallface`,
  industrial-loft base interiors plus generated room variants
- `seed_046.pbp`: WFC/MCTS-generated layout, `modular_var1.wallface`,
  tropical-breeze base interiors plus generated room variants
- `seed_047.pbp`: WFC/MCTS-generated layout, `modular_default.wallface`,
  cosy-Scandinavian base interiors plus generated room variants
- `seed_050.pbp`: WFC/MCTS-generated layout, `modular_var1.wallface`,
  modern-minimalist base interiors plus generated room variants

Live placement from a committed package:

```bash
env MPLCONFIGDIR=/tmp/matplotlib uv --cache-dir /tmp/uv-cache run python scripts/animate_residential_upgrade_minecraft.py \
  --input-package prefab-housing/production_cache/residential_upgrade/seed_043.pbp \
  --live
```

Use any other prepared seed for additional residential variants:

```bash
env MPLCONFIGDIR=/tmp/matplotlib uv --cache-dir /tmp/uv-cache run python scripts/animate_residential_upgrade_minecraft.py \
  --input-package prefab-housing/production_cache/residential_upgrade/seed_045.pbp \
  --live
```

```bash
env MPLCONFIGDIR=/tmp/matplotlib uv --cache-dir /tmp/uv-cache run python scripts/animate_residential_upgrade_minecraft.py \
  --input-package prefab-housing/production_cache/residential_upgrade/seed_047.pbp \
  --live
```

To place a non-colliding review line-up of cached houses, pass all packages:

```bash
env MPLCONFIGDIR=/tmp/matplotlib uv --cache-dir /tmp/uv-cache run python scripts/animate_residential_upgrade_minecraft.py \
  --input-packages \
  prefab-housing/production_cache/residential_upgrade/seed_043.pbp \
  prefab-housing/production_cache/residential_upgrade/seed_044.pbp \
  prefab-housing/production_cache/residential_upgrade/seed_045.pbp \
  prefab-housing/production_cache/residential_upgrade/seed_046.pbp \
  prefab-housing/production_cache/residential_upgrade/seed_047.pbp \
  prefab-housing/production_cache/residential_upgrade/seed_050.pbp \
  --live
```

For generated line-ups, omit `--input-package`/`--input-packages` and pass
`--lineup-count N` or `--lineup-seeds ...`.

Dry-test cached packages against quarantined settlement plots without GDPC:

```bash
env MPLCONFIGDIR=/tmp/matplotlib uv --cache-dir /tmp/uv-cache run python scripts/test_settlement_module_placement.py
```

The dry-test default uses the core package section, keeping wallface texture
selection outside the cached placement contract.

Add `--host http://<host>:9000` when GDMC-HTTP is not running on
`localhost:9000`, or pass `--origin X Y Z` to place at a fixed world origin.
