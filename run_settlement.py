"""One-shot orchestrator: identity -> bake -> town -> narrative.

Runs the full pipeline in order, auto-detecting biome + mood from the player
position (via run_narrative --identity-only), then baking only the mood/biome
wallpaper variants that are needed, generating the town, and running the
narrative layer.

ENVIRONMENTS: launch this from your narrative env (the one with gdpc + the LLM
narrative deps, e.g. gdmc_env). The narrative steps use THIS interpreter; the
bake + town steps are run through `uv run` (the module workspace env). Minecraft
+ the GDMC HTTP interface must be running.

    python run_settlement.py
    python run_settlement.py --dry-run        # print the steps only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
IDENTITY = REPO_ROOT / "data" / "settlement_identity.json"
CACHE_DIR = REPO_ROOT / "prefab-housing" / "production_cache" / "residential_upgrade"
STAMP_FILE = CACHE_DIR / ".wallface_stamps.json"
RUN_NARRATIVE = REPO_ROOT / "narrative" / "run_narrative.py"
BAKE = REPO_ROOT / "narrative" / "bake_wallface_packages.py"
TOWN = REPO_ROOT / "scripts" / "generate_town_with_residential_prefabs.py"

sys.path.insert(0, str(REPO_ROOT / "narrative"))
from wallface_narrative import biome_family, design_signature

# Base residential seeds, mirrored from the town's DEFAULT_TYPED_PACKAGES stems.
BASE_SEEDS = (43, 44, 45, 46, 47, 50)


def _show(cmd: list[str]) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))


def _run(cmd: list[str], *, label: str, dry_run: bool) -> None:
    print("\n" + "=" * 72)
    print(f"[run_settlement] {label}")
    _show(cmd)
    print("=" * 72)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"[run_settlement] step failed: {label} (exit {result.returncode}); aborting.")


def _needs_bake(mood: str, family: str, biome: str) -> bool:
    """True if any variant package is missing or was baked from a stale design."""
    stamps: dict[str, str] = {}
    if STAMP_FILE.exists():
        try:
            stamps = json.loads(STAMP_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable stamps force a rebake
            stamps = {}
    for seed in BASE_SEEDS:
        pkg = f"seed_{seed:03d}__{mood}__{family}.pbp"
        if not (CACHE_DIR / pkg).exists():
            return True
        if stamps.get(pkg) != design_signature(mood, biome, seed):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full settlement pipeline in order.")
    ap.add_argument("--theme", default=None, help="Settlement theme (passed to run_narrative).")
    ap.add_argument("--biome", default=None, help="Force biome id; default: auto-detect from player.")
    ap.add_argument("--narrative-python", default=sys.executable,
                    help="Interpreter for the narrative steps (default: this one; must have the narrative deps).")
    ap.add_argument("--rebake", action="store_true", help="Re-bake even if variant packages already exist.")
    ap.add_argument("--skip-bake", action="store_true", help="Skip the bake step entirely.")
    ap.add_argument("--dry-run", action="store_true", help="Print the steps without running them.")
    args = ap.parse_args()

    if not args.dry_run and shutil.which("uv") is None:
        raise SystemExit("[run_settlement] 'uv' not found on PATH; install uv first.")

    npy = args.narrative_python
    theme_args = ["--theme", args.theme] if args.theme else []
    biome_args = ["--biome", args.biome] if args.biome else []

    # 1. Identity: detect biome + mood from the player and persist them.
    _run([npy, str(RUN_NARRATIVE), "--identity-only", *theme_args, *biome_args],
         label="1/4 identity (biome + mood)", dry_run=args.dry_run)

    if IDENTITY.exists():
        data = json.loads(IDENTITY.read_text(encoding="utf-8"))
        mood = data.get("mood_tier") or "strained"
        biome = data.get("biome") or (args.biome or "")
    else:
        # Only happens in --dry-run before any identity file exists.
        mood, biome = "strained", (args.biome or "")
        print("[run_settlement] (no identity file yet; assuming mood='strained')")
    family = biome_family(biome)
    print(f"[run_settlement] identity -> mood={mood!r} biome={biome!r} family={family!r}")

    # 2. Bake only the needed mood/biome wallpaper variants (uv env).
    if args.skip_bake:
        print("[run_settlement] 2/4 bake skipped (--skip-bake).")
    elif not args.rebake and not _needs_bake(mood, family, biome):
        print(f"[run_settlement] 2/4 bake skipped; variants for mood={mood!r} family={family!r} are up to date.")
    else:
        _run(["uv", "run", "python", str(BAKE), "--biome", biome, "--moods", mood],
             label=f"2/4 bake wallpaper variants (mood={mood}, family={family})", dry_run=args.dry_run)

    # 3. Town generation (uv env): places the mood/biome-matched prefabs.
    _run(["uv", "run", "python", str(TOWN)],
         label="3/4 town generation", dry_run=args.dry_run)

    # 4. Narrative layer (narrative env): datapack + premades, reusing the identity.
    _run([npy, str(RUN_NARRATIVE), *theme_args],
         label="4/4 narrative layer", dry_run=args.dry_run)

    print("\n[run_settlement] done. In-game: /reload, then once: /function area_discovery:setup")


if __name__ == "__main__":
    main()
