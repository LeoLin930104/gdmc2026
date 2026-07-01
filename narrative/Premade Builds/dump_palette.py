from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbtlib
except ImportError:
    sys.exit(
        "nbtlib not found. It is declared in the uv-managed project "
        "dependencies; run:\n"
        "    uv sync --all-packages\n"
        "then launch through:\n"
        "    uv run python narrative/Premade\\ Builds/dump_palette.py"
    )

# Blocks that carry function and must NOT be remapped by the mood palette swap.
# Substring match against the (namespaced-stripped) block id.
_FIXED_FUNCTIONAL = (
    "chest", "barrel", "furnace", "smoker", "blast_furnace", "lectern",
    "crafting_table", "loom", "smithing_table", "cartography_table",
    "fletching_table", "grindstone", "stonecutter", "brewing_stand",
    "anvil", "bell", "campfire", "bookshelf", "composter", "cauldron",
    "beehive", "bee_nest", "jukebox", "note_block", "flower_pot",
    "armor_stand", "item_frame", "sign", "banner", "head", "skull",
    "spawner", "beacon", "conduit",
)

# Air-like states are never placed/swapped.
_AIR = ("air", "cave_air", "void_air")


def _short(block_id: str) -> str:
    return block_id.split(":", 1)[-1]


def _is_fixed(block_id: str) -> bool:
    s = _short(block_id)
    return any(tok in s for tok in _FIXED_FUNCTIONAL)


def _is_air(block_id: str) -> bool:
    return _short(block_id) in _AIR


def _load_structure(path: Path):
    """Return (size, palette, blocks) from a structure .nbt.

    Handles the single-`palette` and multi-`palettes` (variant) layouts.
    """
    nbt = nbtlib.load(str(path))
    root = nbt  # structure files are an unnamed root compound
    size = [int(v) for v in root.get("size", [0, 0, 0])]

    if "palette" in root:
        palette = root["palette"]
    elif "palettes" in root and len(root["palettes"]) > 0:
        palette = root["palettes"][0]  # first variant
    else:
        palette = []

    blocks = root.get("blocks", [])
    return size, palette, blocks


def _state_label(entry) -> str:
    name = str(entry["Name"])
    props = entry.get("Properties")
    if not props:
        return name
    kv = ",".join(f"{k}={props[k]}" for k in props)
    return f"{name}[{kv}]"


def dump(path: Path) -> None:
    size, palette, blocks = _load_structure(path)

    # Count usage + track y-extent per palette index.
    counts = [0] * len(palette)
    min_y_idx: set[int] = set()
    max_y_idx: set[int] = set()
    ys = [int(b["pos"][1]) for b in blocks] if blocks else [0]
    y_lo, y_hi = (min(ys), max(ys)) if ys else (0, 0)

    for b in blocks:
        idx = int(b["state"])
        counts[idx] += 1
        y = int(b["pos"][1])
        if y == y_lo:
            min_y_idx.add(idx)
        if y == y_hi:
            max_y_idx.add(idx)

    print("=" * 70)
    print(f"{path.name}   size={size[0]}x{size[1]}x{size[2]}  ({len(blocks)} blocks)")
    print("=" * 70)
    print(f"{'idx':>3}  {'count':>5}  {'fix':>3}  {'pos':>4}  block-state")
    print("-" * 70)

    for idx, entry in enumerate(palette):
        bid = str(entry["Name"])
        if _is_air(bid):
            continue
        fixed = _is_fixed(bid)
        pos = ""
        if idx in min_y_idx:
            pos += "F"   # touches bottom layer -> floor/foundation candidate
        if idx in max_y_idx:
            pos += "R"   # touches top layer -> roof candidate
        tag = "FIX" if fixed else "  -"
        print(f"{idx:>3}  {counts[idx]:>5}  {tag:>3}  {pos:>4}  {_state_label(entry)}")

    print("-" * 70)
    swappable = [
        str(e["Name"]) for i, e in enumerate(palette)
        if not _is_air(str(e["Name"])) and not _is_fixed(str(e["Name"]))
    ]
    fixed = [
        str(e["Name"]) for e in palette
        if not _is_air(str(e["Name"])) and _is_fixed(str(e["Name"]))
    ]
    print(f"swappable families needed: {sorted({_short(b) for b in swappable})}")
    print(f"fixed-functional (no swap): {sorted({_short(b) for b in fixed})}")
    print("legend: fix=FIX excluded from mood swap | pos F=bottom layer R=top layer")
    print()


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    targets: list[Path] = []

    if len(argv) > 1:
        p = Path(argv[1])
        if not p.is_absolute():
            p = here / p
        if p.is_dir():
            targets = sorted(p.glob("*.nbt"))
        else:
            targets = [p]
    else:
        targets = sorted((here / "nbt").glob("*.nbt"))

    if not targets:
        print("No .nbt files found. Drop builds into 'Premade Builds/nbt/'.")
        return 0

    for path in targets:
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        try:
            dump(path)
        except Exception as exc:  # noqa: BLE001 - diagnostic tool, report and continue
            print(f"[error] {path.name}: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
