from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "animate_residential_upgrade_minecraft.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "animate_residential_upgrade_minecraft",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
live = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = live
_SPEC.loader.exec_module(live)


def _interval(origin: int, size: int) -> tuple[int, int]:
    half = size // 2
    return origin - half, origin - half + size - 1


def test_rotation_turns_block_coordinates_and_facing() -> None:
    bbox = (0, 0, 0, 1, 0, 2)
    block = {
        "dx": 0,
        "dy": 0,
        "dz": 0,
        "id": "minecraft:oak_stairs",
        "props": {"facing": "west", "half": "bottom"},
    }

    steps = live._rotation_steps_between("west", "north")
    rotated = live._rotated_block(block, bbox=bbox, steps=steps)

    assert steps == 1
    assert (rotated["dx"], rotated["dy"], rotated["dz"]) == (2, 0, 0)
    assert rotated["props"] == {"facing": "north", "half": "bottom"}


def test_centred_lineup_offsets_keep_footprints_separated() -> None:
    sizes = [10, 12, 8]
    gap = 3
    offsets = live._centred_lineup_offsets(sizes, gap)
    intervals = [_interval(offset, size) for offset, size in zip(offsets, sizes, strict=True)]

    assert len(offsets) == 3
    assert intervals[0][1] + gap < intervals[1][0] + 1
    assert intervals[1][1] + gap < intervals[2][0] + 1
