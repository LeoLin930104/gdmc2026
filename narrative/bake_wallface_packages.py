"""Bake mood/biome prefab packages (.pbp) with narrative-driven wallpaper.

For each base residential seed x mood this generates a narrative wallface (via
wallface_narrative) and invokes the module's existing
``scripts/animate_residential_upgrade_minecraft.py`` to build + export a package
named ``seed_<NNN>__<mood>__<family>.pbp`` into the production cache. The town
generator picks these by reading data/settlement_identity.json.

No module code is modified: this only adds wallface inputs and drives the
exporter the module already ships. Baking needs no running Minecraft.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from wallface_narrative import MOODS, biome_family, generate, package_signature

ANIMATE = REPO_ROOT / "scripts" / "animate_residential_upgrade_minecraft.py"
CACHE_DIR = REPO_ROOT / "prefab-housing" / "production_cache" / "residential_upgrade"
STAMP_FILE = CACHE_DIR / ".wallface_stamps.json"
WALLFACE_DIR = REPO_ROOT / "narrative_wallfaces"

# Base residential seeds, mirrored from the town's DEFAULT_TYPED_PACKAGES stems
# (seed_043..seed_050). Each bakes one mood/biome variant per seed.
BASE_SEEDS = (43, 44, 45, 46, 47, 50)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bake mood/biome wallpaper prefab packages.")
    ap.add_argument("--biome", default="", help="Biome id; sets the palette family (e.g. minecraft:desert).")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(BASE_SEEDS))
    ap.add_argument("--moods", nargs="*", default=list(MOODS), choices=list(MOODS))
    ap.add_argument("--material-theme", default=None, help="Passed through to the exporter when set.")
    ap.add_argument("--print", action="store_true", dest="print_only",
                    help="Print the bake commands without running them.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    family = biome_family(args.biome)
    WALLFACE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stamps: dict[str, str] = {}
    if STAMP_FILE.exists():
        try:
            stamps = json.loads(STAMP_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt stamp file just forces a rebake
            stamps = {}

    failures: list[str] = []
    for seed in args.seeds:
        for mood in args.moods:
            wf_path = WALLFACE_DIR / f"narrative_{mood}_{family}_{seed:03d}.wallface"
            wf_path.write_text(generate(mood, args.biome, seed=seed), encoding="utf-8")
            out = CACHE_DIR / f"seed_{seed:03d}__{mood}__{family}.pbp"
            cmd = [
                "uv", "run", "python", str(ANIMATE),
                "--seed", str(seed),
                "--wallface-design", str(wf_path),
                "--package-out", str(out),
            ]
            if args.material_theme:
                cmd += ["--material-theme", args.material_theme]
            print("  " + " ".join(cmd))
            if args.print_only:
                continue
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if result.returncode != 0:
                failures.append(out.name)
            else:
                stamps[out.name] = package_signature(mood, args.biome, seed)

    if not args.print_only:
        STAMP_FILE.write_text(json.dumps(stamps, indent=2), encoding="utf-8")

    if failures:
        print(f"[bake] {len(failures)} package(s) failed: {failures}")
        return 1
    print(f"[bake] done -> {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
