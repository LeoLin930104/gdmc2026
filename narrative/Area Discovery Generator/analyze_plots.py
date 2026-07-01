import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


# In <repo>/narrative/Area Discovery Generator/, so the generator's data/ is two levels up (the repo root).
_DEFAULT_NPZ = (
    Path(__file__).parent.parent.parent / "data" / "settlement_plots.npz"
)


def _find_plots_npz() -> Path:
    if _DEFAULT_NPZ.exists():
        return _DEFAULT_NPZ
    matches = list(Path(__file__).parent.parent.parent.glob("**/data/settlement_plots.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find settlement_plots.npz. Run the area generator first:\n"
        "  python main.py   (from the repo root)\n"
        f"(looked for {_DEFAULT_NPZ})"
    )


def _items_to_dict(arr) -> dict:
    out = {}
    for pair in arr:
        if len(pair) != 2:
            continue
        out[pair[0]] = pair[1]
    return out


def _largest_rect_from_points(points) -> tuple[int, int]:
    pts = [(int(x), int(z)) for x, z in points]
    if not pts:
        return (0, 0)
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    W = max_x - min_x + 1
    D = max_z - min_z + 1
    mask = np.zeros((D, W), dtype=bool)
    for x, z in pts:
        mask[z - min_z, x - min_x] = True

    heights = np.zeros(W, dtype=int)
    best_w = best_d = 0
    best_area = 0
    for z in range(D):
        heights = np.where(mask[z], heights + 1, 0)
        stack = []
        for x in range(W + 1):
            cur = heights[x] if x < W else 0
            while stack and heights[stack[-1]] > cur:
                h = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                w = x - left
                if w * h > best_area:
                    best_area = w * h
                    best_w, best_d = w, h
            stack.append(x)
    return (best_w, best_d)


def _stats(values: list[int]) -> str:
    a = np.array(values)
    return (f"min {a.min():>3}  p25 {int(np.percentile(a,25)):>3}  "
            f"median {int(np.median(a)):>3}  p75 {int(np.percentile(a,75)):>3}  "
            f"max {a.max():>3}  mean {a.mean():5.1f}")


# Fit-square buckets: the largest square premade that fits a rect is
# min(width, depth). Bucket by that so authoring targets are concrete.
_BUCKETS = [
    ("Small  (fits  7x7,  <11)", lambda s: s < 11),
    ("Medium (fits 11x11, 11-22)", lambda s: 11 <= s < 23),
    ("Large  (fits 23x23+, >=23)", lambda s: s >= 23),
]


def main(plots_path: str | None = None) -> None:
    npz_path = Path(plots_path) if plots_path else _find_plots_npz()
    if plots_path and not npz_path.exists():
        raise FileNotFoundError(f"--plots path does not exist: {npz_path}")
    print(f"Loading: {npz_path}\n")

    data = np.load(npz_path, allow_pickle=True)
    module_size = int(data["module_size"]) if "module_size" in data.files else None
    setback = float(data["setback"]) if "setback" in data.files else None

    building_rects = _items_to_dict(data["building_rects"]) if "building_rects" in data.files else {}
    plots = _items_to_dict(data["plots"]) if "plots" in data.files else {}
    farms = _items_to_dict(data["farms"]) if "farms" in data.files else {}

    print("=== generator parameters ===")
    print(f"  module_size : {module_size}")
    print(f"  setback     : {setback}")
    print()

    print("=== cell classification ===")
    print(f"  house cells (>=4 modules) : {len(plots)}   <- building team's domain (we don't touch)")
    print(f"  farm cells  (<4 modules)  : {len(farms)}   <- OURS: fields in farm district, premade builds elsewhere")
    print()

    # ---- OUR target: farm-cell footprints --------------------------------
    if farms:
        farm_block_counts = [len(cells) for cells in farms.values()]
        farm_rects = [_largest_rect_from_points(cells) for cells in farms.values()]
        f_w = [w for (w, d) in farm_rects]
        f_d = [d for (w, d) in farm_rects]
        f_sq = [min(w, d) for (w, d) in farm_rects]

        print(f"=== FARM-CELL footprints ({len(farms)} cells) — premade sizing targets ===")
        print(f"  painted blocks/cell : {_stats(farm_block_counts)}   (irregular field area)")
        print(f"  largest-rect width  : {_stats(f_w)}")
        print(f"  largest-rect depth  : {_stats(f_d)}")
        print(f"  fit-square          : {_stats(f_sq)}   (largest square premade that fits a farm cell)")
        print()

        print("=== farm-cell fit-square buckets ===")
        total = len(f_sq)
        for label, pred in _BUCKETS:
            n = sum(1 for s in f_sq if pred(s))
            bar = "#" * int(round(40 * n / total)) if total else ""
            print(f"  {label:<28} {n:>3} ({100*n/total:4.0f}%) {bar}")
        print()

        print("=== farm-cell fit-square histogram (block size -> count) ===")
        sq = np.array(f_sq)
        for size in range(int(sq.min()), int(sq.max()) + 1):
            n = int((sq == size).sum())
            if n:
                print(f"  {size:>3} : {'#' * n} ({n})")
        print()

    # ---- FYI only: house lots (building team) ----------------------------
    if building_rects:
        widths = [int(r["width"]) for r in building_rects.values()]
        depths = [int(r["depth"]) for r in building_rects.values()]
        b_sq = [min(int(r["width"]), int(r["depth"])) for r in building_rects.values()]
        print(f"=== house building_rects ({len(building_rects)}) — FYI, building team's lots (not our premades) ===")
        print(f"  width      : {_stats(widths)}")
        print(f"  depth      : {_stats(depths)}")
        print(f"  fit-square : {_stats(b_sq)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots", default=None,
        help="Path to settlement_plots.npz (default: auto-locate under the repo's data/).",
    )
    args = parser.parse_args()
    main(plots_path=args.plots)
