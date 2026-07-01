"""Narrative-driven wall-face ("wallpaper") generator.

Emits prefab_housing's ``wallface-v1`` text format directly from a settlement's
mood tier + biome, with no import of (or edit to) the prefab_housing module.

Feed the resulting .wallface file to the module through its existing public
seam, e.g. ``build_house(..., wall_face_design_path=...)`` when baking the
blueprint cache, or open it in prefab-housing/editor/wallface_editor.html.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

WIDTH, HEIGHT = 10, 6
MIN_LAYER, MAX_LAYER = -2, 2
EMPTY = "."
HEADER = "wallface-v1"
_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)

MOODS = ("thriving", "strained", "struggling")
PACKAGE_CONTENT_VERSION = "entry_door_module_v2"

# Per-biome materials: base = bulk wall block, accent = the border/frame block
# (replaces the old black border), trim = secondary block used for boarded-up
# windows. Roofs are NOT touched by wall-face designs.
_BIOME_FAMILIES = {
    "temperate": {
        "base": "minecraft:oak_planks",
        "accent": "minecraft:dark_oak_log",
        "trim": "minecraft:spruce_planks",
    },
    "birch": {
        "base": "minecraft:birch_planks",
        "accent": "minecraft:spruce_log",
        "trim": "minecraft:spruce_planks",
    },
    "jungle": {
        "base": "minecraft:jungle_planks",
        "accent": "minecraft:stripped_jungle_log",
        "trim": "minecraft:jungle_planks",
    },
    "savanna": {
        "base": "minecraft:acacia_planks",
        "accent": "minecraft:stripped_acacia_log",
        "trim": "minecraft:acacia_planks",
    },
    "snowy": {
        "base": "minecraft:spruce_planks",
        "accent": "minecraft:stripped_spruce_log",
        "trim": "minecraft:spruce_planks",
    },
    "swamp": {
        "base": "minecraft:mangrove_planks",
        "accent": "minecraft:stripped_mangrove_log",
        "trim": "minecraft:mangrove_planks",
    },
    "desert": {
        "base": "minecraft:smooth_sandstone",
        "accent": "minecraft:chiseled_sandstone",
        "trim": "minecraft:cut_sandstone",
    },
    "badlands": {
        "base": "minecraft:smooth_red_sandstone",
        "accent": "minecraft:chiseled_red_sandstone",
        "trim": "minecraft:cut_red_sandstone",
    },
    "dark_forest": {
        "base": "minecraft:dark_oak_planks",
        "accent": "minecraft:stripped_dark_oak_log",
        "trim": "minecraft:dark_oak_planks",
    },
}

_BIOME_KEYWORDS = (
    (("desert",), "desert"),
    (("badlands", "mesa"), "badlands"),
    (("savanna",), "savanna"),
    (("jungle", "bamboo"), "jungle"),
    (("birch",), "birch"),
    (("snow", "frozen", "ice", "grove", "taiga"), "snowy"),
    (("swamp", "mangrove"), "swamp"),
    (("dark_forest", "dark forest", "roofed"), "dark_forest"),
)

_MOOD = {
    "thriving": {
        "window": "minecraft:glass",
        "lights": ("minecraft:sea_lantern", "minecraft:shroomlight"),
        "boarded": False,
        "cobweb": False,
    },
    "strained": {
        "window": "minecraft:glass_pane",
        "lights": (),
        "boarded": False,
        "cobweb": False,
    },
    "struggling": {
        "window": "minecraft:brown_stained_glass_pane",
        "lights": (),
        "boarded": True,
        "cobweb": True,
    },
}


def biome_family(biome: str | None) -> str:
    b = (biome or "").lower()
    for keys, fam in _BIOME_KEYWORDS:
        if any(k in b for k in keys):
            return fam
    return "temperate"


def _seed_int(mood: str, biome: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{mood}:{biome}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def _grid() -> list[list[str | None]]:
    return [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]


def _set_rect(g: list[list[str | None]], x0: int, x1: int, y0: int, y1: int, block: str | None) -> None:
    for y in range(max(0, y0), min(HEIGHT, y1 + 1)):
        for x in range(max(0, x0), min(WIDTH, x1 + 1)):
            g[y][x] = block


def build_layers(mood: str, biome: str | None, seed: int = 0) -> dict[int, list[list[str | None]]]:
    if mood not in _MOOD:
        raise ValueError(f"mood must be one of {MOODS}; got {mood!r}")
    fam = _BIOME_FAMILIES[biome_family(biome)]
    spec = _MOOD[mood]
    base, accent, trim = fam["base"], fam["accent"], fam["trim"]
    rnd = _seed_int(mood, biome or "", seed)

    layers = {layer: _grid() for layer in range(MIN_LAYER, MAX_LAYER + 1)}

    l0 = layers[0]
    _set_rect(l0, 0, WIDTH - 1, 0, HEIGHT - 1, base)
    margin = 2
    wx0, wx1 = margin, WIDTH - 1 - margin
    wy0, wy1 = margin, HEIGHT - 1 - margin
    if spec["boarded"]:
        _set_rect(l0, wx0, wx1, wy0, wy1, trim)
        cx0, cx1 = WIDTH // 2 - 1, WIDTH // 2
        _set_rect(l0, cx0, cx1, wy0, wy1, spec["window"])
    else:
        _set_rect(l0, wx0, wx1, wy0, wy1, spec["window"])

    # Accent border frame on every house (biome accent, replaces the old black
    # border), protruding one block. Lit corners only for the thriving mood.
    l1 = layers[1]
    for x in range(WIDTH):
        l1[0][x] = accent
        l1[HEIGHT - 1][x] = accent
    for y in range(HEIGHT):
        l1[y][0] = accent
        l1[y][WIDTH - 1] = accent
    lights = spec["lights"]
    if lights:
        light = lights[rnd % len(lights)]
        for cx, cy in ((0, 0), (WIDTH - 1, 0), (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1)):
            l1[cy][cx] = light

    if spec["cobweb"]:
        lm1 = layers[-1]
        for cx, cy in ((wx0, wy0), (wx1, wy0), (wx0, wy1), (wx1, wy1)):
            lm1[cy][cx] = "minecraft:cobweb"

    return layers


def serialise(layers: dict[int, list[list[str | None]]]) -> str:
    blocks = sorted(
        {b for grid in layers.values() for row in grid for b in row if b is not None}
    )
    if len(blocks) > len(_ALPHABET):
        raise ValueError("design uses more unique blocks than available symbols")
    symbol = {b: _ALPHABET[i] for i, b in enumerate(blocks)}

    out = [HEADER, f"size {WIDTH} {HEIGHT}"]
    for b in blocks:
        out.append(f"symbol {symbol[b]} {b}")
    out.append("")
    for layer in range(MIN_LAYER, MAX_LAYER + 1):
        out.append(f"layer {layer}")
        for row in layers[layer]:
            line = "".join(EMPTY if b is None else symbol[b] for b in row)
            if len(line) != WIDTH:
                raise ValueError(f"layer {layer} row width {len(line)} != {WIDTH}")
            out.append(line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def generate(mood: str, biome: str | None, seed: int = 0) -> str:
    return serialise(build_layers(mood, biome, seed))


def design_signature(mood: str, biome: str | None, seed: int = 0) -> str:
    """Stable short hash of a design, so bakers can detect palette changes."""
    import hashlib

    return hashlib.sha256(generate(mood, biome, seed).encode("utf-8")).hexdigest()[:16]


def package_signature(mood: str, biome: str | None, seed: int = 0) -> str:
    """Stable short hash of package-affecting wallface and module inputs."""
    payload = "|".join(
        (
            PACKAGE_CONTENT_VERSION,
            design_signature(mood, biome, seed),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a narrative wallface design.")
    ap.add_argument("--mood", choices=MOODS, default="strained")
    ap.add_argument("--biome", default="", help="Biome id (e.g. minecraft:desert).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="Write here; default prints to stdout.")
    ap.add_argument("--matrix", action="store_true", help="Write all moods for --biome into --out (a directory).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.matrix:
        out_dir = args.out or Path("narrative_wallfaces")
        out_dir.mkdir(parents=True, exist_ok=True)
        fam = biome_family(args.biome)
        for mood in MOODS:
            target = out_dir / f"narrative_{mood}_{fam}.wallface"
            target.write_text(generate(mood, args.biome, args.seed), encoding="utf-8")
            print(f"wrote {target}")
        return
    text = generate(args.mood, args.biome, args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
