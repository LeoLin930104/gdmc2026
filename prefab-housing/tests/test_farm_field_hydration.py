from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_PREMADE_DIR = (
    Path(__file__).resolve().parents[2]
    / "narrative"
    / "Premade Builds"
)
if str(_PREMADE_DIR) not in sys.path:
    sys.path.insert(0, str(_PREMADE_DIR))

_SPEC = importlib.util.spec_from_file_location("farm_field", _PREMADE_DIR / "farm_field.py")
assert _SPEC is not None
assert _SPEC.loader is not None
farm_field = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = farm_field
_SPEC.loader.exec_module(farm_field)


def _rect(width: int, depth: int) -> list[tuple[int, int]]:
    return [(x, z) for x in range(width) for z in range(depth)]


def _is_hydrated(cell: tuple[int, int], water: set[tuple[int, int]]) -> bool:
    x, z = cell
    return any(
        max(abs(x - wx), abs(z - wz)) <= farm_field.HYDRATION_RADIUS
        for wx, wz in water
    )


@pytest.mark.parametrize("width,depth", [(24, 8), (8, 24), (18, 18)])
def test_farm_layout_preserves_footprint_and_hydrates_crop_land(width: int, depth: int) -> None:
    cell_set, border, water, crop_land = farm_field.farm_layout(_rect(width, depth))

    assert border.isdisjoint(water)
    assert border.isdisjoint(crop_land)
    assert water.isdisjoint(crop_land)
    assert cell_set == border | water | crop_land
    assert water
    assert all(_is_hydrated(cell, water) for cell in crop_land)


def test_struggling_farm_keeps_required_irrigation_wet(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBlock:
        def __init__(self, block_id: str, states: dict[str, str] | None = None) -> None:
            self.id = block_id
            self.states = dict(states or {})

    class FakeEditor:
        def __init__(self) -> None:
            self.placed: list[tuple[tuple[int, int, int], FakeBlock]] = []

        def placeBlock(self, position: tuple[int, int, int], block: FakeBlock) -> None:
            self.placed.append((position, block))

    monkeypatch.setitem(sys.modules, "gdpc", types.SimpleNamespace(Block=FakeBlock))

    cells = _rect(18, 18)
    _, _, water, crop_land = farm_field.farm_layout(cells)
    editor = FakeEditor()

    stats = farm_field.place_farm_field(
        editor,
        cells,
        origin=(0, 0, 0),
        ground_y=lambda x, z: (x + z) % 3,
        mood="struggling",
        seed_name="hydration-test",
        clear_height=1,
    )

    placed_ids = [block.id for _position, block in editor.placed]
    surface_y = {
        position[1]
        for position, block in editor.placed
        if block.id in {farm_field.SOIL_BLOCK, farm_field.WATER_BLOCK}
    }
    farmland = [block for _position, block in editor.placed if block.id == farm_field.SOIL_BLOCK]

    assert stats["water"] == len(water)
    assert placed_ids.count(farm_field.WATER_BLOCK) == len(water)
    assert "minecraft:coarse_dirt" not in placed_ids
    assert len(surface_y) == 1
    assert len(farmland) == len(crop_land)
    assert all(block.states == {"moisture": "7"} for block in farmland)
