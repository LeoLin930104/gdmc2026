from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from nbt_structure import parse_structure
from premade_placer import TIERS, build_premade


def _player_xz(editor):
    llm = _HERE.parent / "LLM Narrative"
    if str(llm) not in sys.path:
        sys.path.insert(0, str(llm))
    try:
        from biome_context import get_player_position
        pos = get_player_position()
        return int(pos[0]), int(pos[2])
    except Exception as exc:  # noqa: BLE001 - convenience path; --pos is the fallback
        print(f"[warn] could not read player position ({exc}); pass --pos X Z.")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Place one premade build in-world (smoke test).")
    ap.add_argument("nbt", nargs="?", default="nbt/barrack_7.nbt",
                    help="Path to the .nbt build (default: nbt/barrack_7.nbt).")
    ap.add_argument("--tier", default="strained", choices=TIERS,
                    help="Mood tier to render (default: strained).")
    ap.add_argument("--pos", nargs=2, type=int, metavar=("X", "Z"),
                    help="World X Z of the build's min corner. Default: near the player.")
    ap.add_argument("--rotation", type=int, default=0, choices=[0, 1, 2, 3],
                    help="90-degree steps about Y (default: 0).")
    ap.add_argument("--sink", type=int, default=0,
                    help="Blocks the floor sits below grade (default: 0).")
    ap.add_argument("--foundation", default="minecraft:cobblestone",
                    help="Foundation skirt block, mood-swapped at place-time "
                         "(default: minecraft:cobblestone; use deepslate_bricks for barracks).")
    ap.add_argument("--offset", type=int, default=6,
                    help="When placing near the player, blocks away on +X/+Z (default: 6).")
    ap.add_argument("--ground-offset", type=int, default=0, dest="ground_offset",
                    help="Optional Y nudge on detected grade before seating the "
                         "floor (default 0 = floor flush with surrounding ground). "
                         "Only needed to hand-tune an odd spot.")
    args = ap.parse_args()

    nbt_path = Path(args.nbt)
    if not nbt_path.is_absolute():
        nbt_path = _HERE / nbt_path
    structure = parse_structure(nbt_path)

    from gdpc import Editor, Rect  # lazy: only this path needs a world
    import numpy as np

    editor = Editor(buffering=True)

    if args.pos:
        ax, az = args.pos
    else:
        pxz = _player_xz(editor)
        if pxz is None:
            return 2
        ax, az = pxz[0] + args.offset, pxz[1] + args.offset

    sx, sy, sz = structure.size
    fw, fd = (sz, sx) if args.rotation % 2 == 1 else (sx, sz)

    # Live-world grade. MOTION_BLOCKING_NO_LEAVES[x][z] is the first-air y, so the
    # top solid block is value-1. --ground-offset (default 0) is just a manual nudge.
    world = editor.loadWorldSlice(Rect((ax, az), (fw, fd)))
    hm = np.asarray(world.heightmaps["MOTION_BLOCKING_NO_LEAVES"], dtype=int)

    def ground_y(wx, wz) -> int:
        ix = min(max(int(wx) - ax, 0), fw - 1)
        iz = min(max(int(wz) - az, 0), fd - 1)
        return int(hm[ix][iz]) - 1 + args.ground_offset

    print(f"[info] Placing {nbt_path.name} (tier={args.tier}, rot={args.rotation}) "
          f"at corner ({ax}, {az}), footprint {fw}x{fd}...")
    stats = build_premade(
        editor, structure, (ax, az), ground_y,
        tier=args.tier, rotation=args.rotation, sink=args.sink,
        foundation_block=args.foundation,
    )
    editor.flushBuffer()

    print(
        f"[done] base_y={stats['base_y']} floor_y={stats['floor_y']} | "
        f"placed={stats['placed']} dropped={stats['dropped']} "
        f"liquid-source={stats['liquids']} air-skipped={stats['skipped_air']} | "
        f"foundation filled={stats['filled']} capped={stats['capped']} cleared={stats['cleared']}"
    )
    print(f"[info] Go to ~({ax}, {stats['floor_y']}, {az}) to inspect the build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
