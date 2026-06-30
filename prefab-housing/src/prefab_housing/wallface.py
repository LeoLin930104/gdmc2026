"""Text-based wall-face overlay designs.

The saved format is intentionally LLM-editable: a header, a symbol table, then
five text grids representing thickness layers ``-2..2`` relative to the base
wall plane. Positive thickness goes outward from the face; negative goes inward.
The first row of each grid is the top row of the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string

from prefab_housing.types import SemanticBlockDict
from voxel_renderer.api import render_orthographic_views

MIN_LAYER = -2
MAX_LAYER = 2
EMPTY_SYMBOL = "."
FORMAT_HEADER = "wallface-v1"
DEFAULT_BASE_WALL_BLOCK = "minecraft:white_concrete"
_SYMBOL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "@%&*+=!?/<>[]{}()#$^~;:,|-_"


@dataclass(frozen=True, slots=True)
class WallFaceDesign:
    width: int
    height: int
    layers: dict[int, tuple[tuple[str | None, ...], ...]]


def _empty_rows(width: int, height: int) -> tuple[tuple[str | None, ...], ...]:
    return tuple(tuple(None for _ in range(width)) for _ in range(height))


def empty_wall_face_design(width: int, height: int) -> WallFaceDesign:
    if width < 1 or height < 1:
        raise ValueError("wall face size must be positive")
    layers = {layer: _empty_rows(width, height) for layer in range(MIN_LAYER, MAX_LAYER + 1)}
    layers[0] = tuple(
        tuple(DEFAULT_BASE_WALL_BLOCK for _ in range(width)) for _ in range(height)
    )
    return WallFaceDesign(
        width=width,
        height=height,
        layers=layers,
    )


def _normalise_layers(
    width: int,
    height: int,
    layers: dict[int, tuple[tuple[str | None, ...], ...]],
) -> dict[int, tuple[tuple[str | None, ...], ...]]:
    normalised: dict[int, tuple[tuple[str | None, ...], ...]] = {}
    for layer in range(MIN_LAYER, MAX_LAYER + 1):
        rows = layers.get(layer, _empty_rows(width, height))
        if len(rows) != height:
            raise ValueError(f"layer {layer} expected {height} rows, got {len(rows)}")
        checked_rows: list[tuple[str | None, ...]] = []
        for row in rows:
            if len(row) != width:
                raise ValueError(f"layer {layer} row expected width {width}, got {len(row)}")
            checked_rows.append(tuple(row))
        normalised[layer] = tuple(checked_rows)
    return normalised


def wall_face_design_from_dict(payload: dict[str, object]) -> WallFaceDesign:
    width = int(payload["width"])
    height = int(payload["height"])
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("layers must be a dict keyed by thickness")
    layers: dict[int, tuple[tuple[str | None, ...], ...]] = {}
    for layer_key, raw_rows in raw_layers.items():
        layer = int(layer_key)
        if layer < MIN_LAYER or layer > MAX_LAYER:
            raise ValueError(f"layer must be in [{MIN_LAYER}, {MAX_LAYER}]")
        if not isinstance(raw_rows, list):
            raise ValueError(f"layer {layer} rows must be a list")
        rows: list[tuple[str | None, ...]] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, list):
                raise ValueError(f"layer {layer} row must be a list")
            cells: list[str | None] = []
            for cell in raw_row:
                if cell is None:
                    cells.append(None)
                else:
                    cells.append(str(cell))
            rows.append(tuple(cells))
        layers[layer] = tuple(rows)
    return WallFaceDesign(width=width, height=height, layers=_normalise_layers(width, height, layers))


def wall_face_design_to_dict(design: WallFaceDesign) -> dict[str, object]:
    return {
        "width": design.width,
        "height": design.height,
        "layers": {
            str(layer): [[cell for cell in row] for row in design.layers[layer]]
            for layer in range(MIN_LAYER, MAX_LAYER + 1)
        },
    }


def parse_wall_face_design(text: str) -> WallFaceDesign:
    lines = [
        raw.rstrip("\n")
        for raw in text.splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    if not lines or lines[0] != FORMAT_HEADER:
        raise ValueError(f"wall face file must start with {FORMAT_HEADER!r}")
    if len(lines) < 2:
        raise ValueError("wall face file missing size declaration")

    size_parts = lines[1].split()
    if len(size_parts) != 3 or size_parts[0] != "size":
        raise ValueError("size line must be `size <width> <height>`")
    width = int(size_parts[1])
    height = int(size_parts[2])
    symbols: dict[str, str | None] = {EMPTY_SYMBOL: None}
    layers: dict[int, tuple[tuple[str | None, ...], ...]] = {}

    index = 2
    while index < len(lines) and lines[index].startswith("symbol "):
        parts = lines[index].split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"invalid symbol declaration: {lines[index]!r}")
        symbol = parts[1]
        if len(symbol) != 1:
            raise ValueError("symbols must be single characters")
        block_id = parts[2]
        if symbol == EMPTY_SYMBOL and block_id not in {"air", "none"}:
            raise ValueError("`.` is reserved for empty cells")
        symbols[symbol] = None if block_id in {"air", "none"} else block_id
        index += 1

    while index < len(lines):
        header = lines[index].split()
        if len(header) != 2 or header[0] != "layer":
            raise ValueError(f"expected layer declaration, got {lines[index]!r}")
        layer = int(header[1])
        if layer < MIN_LAYER or layer > MAX_LAYER:
            raise ValueError(f"layer must be in [{MIN_LAYER}, {MAX_LAYER}]")
        if layer in layers:
            raise ValueError(f"duplicate layer {layer}")
        index += 1
        if index + height > len(lines) + 1:
            raise ValueError(f"layer {layer} missing rows")
        rows: list[tuple[str | None, ...]] = []
        for _ in range(height):
            if index >= len(lines):
                raise ValueError(f"layer {layer} missing rows")
            row_text = lines[index]
            if len(row_text) != width:
                raise ValueError(
                    f"layer {layer} row width mismatch: expected {width}, got {len(row_text)}"
                )
            row: list[str | None] = []
            for symbol in row_text:
                if symbol not in symbols:
                    raise ValueError(f"unknown symbol {symbol!r}")
                row.append(symbols[symbol])
            rows.append(tuple(row))
            index += 1
        layers[layer] = tuple(rows)

    return WallFaceDesign(width=width, height=height, layers=_normalise_layers(width, height, layers))


def serialise_wall_face_design(design: WallFaceDesign) -> str:
    unique_blocks = sorted(
        {
            block_id
            for layer in design.layers.values()
            for row in layer
            for block_id in row
            if block_id is not None
        }
    )
    if len(unique_blocks) > len(_SYMBOL_ALPHABET):
        raise ValueError("wall face design uses more unique blocks than available symbols")
    block_symbols = {block_id: _SYMBOL_ALPHABET[idx] for idx, block_id in enumerate(unique_blocks)}

    lines = [FORMAT_HEADER, f"size {design.width} {design.height}"]
    for block_id in unique_blocks:
        lines.append(f"symbol {block_symbols[block_id]} {block_id}")
    lines.append("")

    for layer in range(MIN_LAYER, MAX_LAYER + 1):
        lines.append(f"layer {layer}")
        for row in design.layers[layer]:
            lines.append(
                "".join(EMPTY_SYMBOL if block_id is None else block_symbols[block_id] for block_id in row)
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_wall_face_design(path: str | Path) -> WallFaceDesign:
    return parse_wall_face_design(Path(path).read_text(encoding="utf-8"))


def save_wall_face_design(path: str | Path, design: WallFaceDesign) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialise_wall_face_design(design), encoding="utf-8")


def resample_wall_face_design(design: WallFaceDesign, width: int, height: int) -> WallFaceDesign:
    if width < 1 or height < 1:
        raise ValueError("target wall face size must be positive")
    if width == design.width and height == design.height:
        return design

    layers: dict[int, tuple[tuple[str | None, ...], ...]] = {}
    for layer in range(MIN_LAYER, MAX_LAYER + 1):
        rows: list[tuple[str | None, ...]] = []
        for y in range(height):
            src_y = min(design.height - 1, (y * design.height) // height)
            row: list[str | None] = []
            for x in range(width):
                src_x = min(design.width - 1, (x * design.width) // width)
                row.append(design.layers[layer][src_y][src_x])
            rows.append(tuple(row))
        layers[layer] = tuple(rows)
    return WallFaceDesign(width=width, height=height, layers=layers)


def base_wall_block(design: WallFaceDesign) -> str:
    """Return the bulk wall material of a design (most common cell in layer 0).

    Layer 0 is the base wall plane: for the biome-baked narrative designs it is
    filled with the biome family's ``base`` block, so this is the same material
    the exterior walls render in. Floors/ceilings can use it to track the biome
    instead of the static palette default. Falls back to
    ``DEFAULT_BASE_WALL_BLOCK`` when layer 0 is empty.
    """
    counts: dict[str, int] = {}
    for row in design.layers.get(0, ()):  # type: ignore[arg-type]
        for block_id in row:
            if block_id is not None:
                counts[block_id] = counts.get(block_id, 0) + 1
    if not counts:
        return DEFAULT_BASE_WALL_BLOCK
    # Highest count wins; ties broken by block id for determinism.
    return max(sorted(counts), key=lambda block_id: counts[block_id])


def emit_wall_face_blocks(
    design: WallFaceDesign,
    *,
    axis: str,
    fixed: int,
    outward_sign: int,
    a0: int,
    a1: int,
    y0: int,
    y1: int,
) -> list[SemanticBlockDict]:
    width = a1 - a0 + 1
    height = y1 - y0 + 1
    scaled = resample_wall_face_design(design, width, height)
    blocks: list[SemanticBlockDict] = []
    for layer in range(MIN_LAYER, MAX_LAYER + 1):
        rows = scaled.layers[layer]
        for row_idx, row in enumerate(rows):
            y = y1 - row_idx
            for col_idx, block_id in enumerate(row):
                if block_id is None:
                    continue
                if axis == "x":
                    blocks.append(
                        {
                            "x": a0 + col_idx,
                            "y": y,
                            "z": fixed + layer * outward_sign,
                            "id": block_id,
                        }
                    )
                elif axis == "z":
                    blocks.append(
                        {
                            "x": fixed + layer * outward_sign,
                            "y": y,
                            "z": a0 + col_idx,
                            "id": block_id,
                        }
                    )
                else:
                    raise ValueError(f"unsupported axis {axis!r}")
    return blocks


def render_wall_face_preview(design: WallFaceDesign) -> dict[str, str]:
    blocks = emit_wall_face_blocks(
        design,
        axis="x",
        fixed=0,
        outward_sign=1,
        a0=0,
        a1=design.width - 1,
        y0=0,
        y1=design.height - 1,
    )
    if not blocks:
        return render_orthographic_views([], width=512, height=384, backend="pyrender")
    min_x = min(int(block["x"]) for block in blocks)
    min_y = min(int(block["y"]) for block in blocks)
    min_z = min(int(block["z"]) for block in blocks)
    translated = [
        {
            "x": int(block["x"]) - min_x,
            "y": int(block["y"]) - min_y,
            "z": int(block["z"]) - min_z,
            "id": str(block["id"]),
        }
        for block in blocks
    ]
    return render_orthographic_views(translated, width=512, height=384, backend="pyrender")


__all__ = [
    "EMPTY_SYMBOL",
    "FORMAT_HEADER",
    "MAX_LAYER",
    "MIN_LAYER",
    "WallFaceDesign",
    "DEFAULT_BASE_WALL_BLOCK",
    "base_wall_block",
    "emit_wall_face_blocks",
    "empty_wall_face_design",
    "load_wall_face_design",
    "parse_wall_face_design",
    "render_wall_face_preview",
    "resample_wall_face_design",
    "save_wall_face_design",
    "serialise_wall_face_design",
    "wall_face_design_from_dict",
    "wall_face_design_to_dict",
]
