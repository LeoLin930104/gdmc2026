"""run_narrative.py - run this for all narrative part
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Wire every sibling package onto the path so we can import their entrypoints.
for _sub in ("LLM Narrative", "Area Discovery Generator", "Premade Builds"):
    p = ROOT / _sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 -> UTF-8 for symbols
    sys.stdout.reconfigure(encoding="utf-8")

# This file lives in <repo>/narrative/, so the gdmc2026 generator IS the parent.
_GEN_DIR = ROOT.parent
_NPZ = _GEN_DIR / "data" / "settlement_data.npz"
_IDENTITY = _GEN_DIR / "data" / "settlement_identity.json"

try:
    from wallface_narrative import biome_family as _biome_family
except Exception:  # noqa: BLE001 - generator is optional; fall back to a neutral family
    def _biome_family(_biome: str | None) -> str:
        return "temperate"

DEFAULT_THEME = "Fantasy"
DEFAULT_WORLD = "New World"


def write_identity(theme: str, biome: str | None, mood_tier: str | None,
                   *, name: str | None = None, era: str | None = None) -> None:
    _IDENTITY.parent.mkdir(parents=True, exist_ok=True)
    payload = {"theme": theme, "biome": biome, "biome_family": _biome_family(biome), "mood_tier": mood_tier}
    if name is not None:
        payload["name"] = name
    if era is not None:
        payload["era"] = era
    _IDENTITY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Identity persisted -> {_IDENTITY} (mood={mood_tier!r}, biome={biome!r}).")


def load_identity() -> dict | None:
    if not _IDENTITY.exists():
        return None
    try:
        return json.loads(_IDENTITY.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a bad identity file should not abort the run
        print(f"[warn] could not read {_IDENTITY} ({exc}); ignoring.")
        return None

# steps

def run_generator() -> None:
    main_py = _GEN_DIR / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Generator entrypoint not found: {main_py}")
    print("=" * 72)
    print("STEP 0  Running gdmc2026 generator (this builds the settlement in-world)")
    print("=" * 72)
    subprocess.run([sys.executable, "main.py"], cwd=str(_GEN_DIR), check=True)
    print()


def _detect_biome() -> str | None:
    try:
        from biome_context import sample_biome_at_player
        from gdpc import Editor
        biome, _pos = sample_biome_at_player(Editor())
        return biome
    except Exception as exc:  # noqa: BLE001 - biome is optional grounding
        print(f"[info] biome detection skipped ({exc!r}).")
        return None


def build_settlement(theme: str, biome: str | None):
    from settlement_generator import Settlement, generate_settlement
    from settlement_goal import generate_settlement_goal
    from shared_events import generate_shared_events
    from mood_tier import generate_mood_tier

    print("=" * 72)
    print("STEP 1  Generating shared settlement identity + pre-passes")
    print("=" * 72)
    try:
        s = generate_settlement(theme, biome=biome)
        print(f"  identity : {s.name} — {s.era}")
        s.goal = generate_settlement_goal(s)              # pre-pass 1
        if s.goal:
            print(f"  goal     : {s.goal.summary}")
        s.shared_events = generate_shared_events(s)        # pre-pass 2
        print(f"  events   : {len(s.shared_events or [])} shared event(s)")
        s.mood_tier = generate_mood_tier(s)                # pre-pass 3 (drives builds)
        print(f"  mood     : {s.mood_tier}")
        print()
        return s
    except Exception as exc:  # noqa: BLE001 - keep the geometry working without LLM
        print(f"[warn] settlement generation failed ({exc!r}); "
              f"using a neutral identity (mood tier 'strained').")
        print()
        return Settlement(name=theme or "Settlement", era="", founding_story="",
                          theme=theme, biome=biome)


def run_datapack(settlement, world: str, theme: str, biome: str | None,
                 npz: str | None) -> None:
    import integrate_settlement
    print("=" * 72)
    print("STEP 2  Area Discovery datapack (district names + title-on-entry)")
    print("=" * 72)
    integrate_settlement.main(world=world, theme=theme, npz=npz,
                              settlement=settlement, biome=biome)
    print()


def run_premades(settlement, theme: str, npz: str | None, plots: str | None,
                 dry_run: bool, max_builds: int | None, tier_override: str | None,
                 rotation_override: int | None, decay: bool, place_items: bool,
                 farm_fields: bool) -> None:
    import place_premades
    print("=" * 72)
    print("STEP 3  Premade builds on farm cells (mood palette + decay + district items + crop fields)")
    print("=" * 72)
    place_premades.main(
        theme=theme, npz=npz, plots=plots, dry_run=dry_run, max_builds=max_builds,
        tier_override=tier_override, rotation_override=rotation_override,
        decay=decay, settlement=settlement, place_items=place_items,
        farm_fields=farm_fields,
    )
    print()

# entrypoint

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full GDMC v2 narrative layer over the gdmc2026 generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--generate", action="store_true",
                    help="Run the gdmc2026 generator first (needs an open world + "
                         "GDMC HTTP interface). Default: assume it already ran.")
    ap.add_argument("--identity-only", action="store_true",
                    help="Compute the settlement identity (mood + biome) and write "
                         "data/settlement_identity.json, then exit. Run this BEFORE "
                         "the town generator so it can pick mood-matched prefabs.")
    ap.add_argument("--theme", default=DEFAULT_THEME,
                    help=f"Settlement theme (default: {DEFAULT_THEME!r}).")
    ap.add_argument("--world", default=DEFAULT_WORLD,
                    help=f"Minecraft save to write the datapack into (default: {DEFAULT_WORLD!r}).")
    ap.add_argument("--biome", default=None,
                    help="Override the auto-detected biome (e.g. 'minecraft:swamp').")
    ap.add_argument("--tier", default=None, choices=("thriving", "strained", "struggling"),
                    help="Force the mood tier for the builds (default: the LLM-decided tier).")
    ap.add_argument("--rotation", type=int, default=None, choices=[0, 1, 2, 3],
                    help="Force one rotation for all builds (default: per-cell, seeded).")
    ap.add_argument("--no-decay", action="store_false", dest="decay",
                    help="Disable struggling-tier decay (cobwebs + knocked-out blocks).")
    ap.add_argument("--no-items", action="store_false", dest="place_items",
                    help="Skip per-district diaries + tools + relics placed in build chests.")
    ap.add_argument("--no-farm-fields", action="store_false", dest="farm_fields",
                    help="Skip rendering the farm-role district's mood-scaled crop fields.")
    ap.add_argument("--max", type=int, default=None, dest="max_builds",
                    help="Cap the number of premade builds placed (testing).")
    ap.add_argument("--npz", default=None, help="Path to settlement_data.npz.")
    ap.add_argument("--plots", default=None, help="Path to settlement_plots.npz.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan the premades only; write no datapack and place nothing.")
    ap.add_argument("--skip-datapack", action="store_true",
                    help="Skip the Area Discovery datapack step.")
    ap.add_argument("--skip-premades", action="store_true",
                    help="Skip the premade-builds step.")
    args = ap.parse_args()

    identity = load_identity()

    if args.identity_only:
        biome = args.biome or (identity or {}).get("biome") or _detect_biome()
        settlement = build_settlement(args.theme, biome)
        mood_tier = args.tier or settlement.mood_tier
        write_identity(args.theme, biome, mood_tier,
                       name=settlement.name, era=settlement.era)
        return

    if args.generate:
        run_generator()

    if not args.npz and not _NPZ.exists():
        raise FileNotFoundError(
            f"Could not find {_NPZ}.\nRun the generator first (add --generate, or):\n"
            f"  python \"{_GEN_DIR / 'main.py'}\""
        )

    biome = args.biome or (identity or {}).get("biome") or _detect_biome()
    settlement = build_settlement(args.theme, biome)
    if identity and identity.get("mood_tier") and not args.tier:
        settlement.mood_tier = identity["mood_tier"]
        print(f"  mood pinned from identity file: {settlement.mood_tier}")

    if args.dry_run:
        # Dry run: no world side effects. Datapack writes files, so skip it too.
        print("[dry-run] skipping datapack write; planning premades only.\n")
        run_premades(settlement, args.theme, args.npz, args.plots, dry_run=True,
                     max_builds=args.max_builds, tier_override=args.tier,
                     rotation_override=args.rotation, decay=args.decay,
                     place_items=args.place_items, farm_fields=args.farm_fields)
        return

    if not args.skip_datapack:
        run_datapack(settlement, args.world, args.theme, biome, args.npz)

    if not args.skip_premades:
        run_premades(settlement, args.theme, args.npz, args.plots, dry_run=False,
                     max_builds=args.max_builds, tier_override=args.tier,
                     rotation_override=args.rotation, decay=args.decay,
                     place_items=args.place_items, farm_fields=args.farm_fields)

    print("=" * 72)
    print("Done. In-game: /reload, then once: /function area_discovery:setup")
    print("=" * 72)


if __name__ == "__main__":
    main()
