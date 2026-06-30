from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "generate_town_with_residential_prefabs.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_town_with_residential_prefabs",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
town = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = town
_SPEC.loader.exec_module(town)

_MAP_MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "upstream"
    / "gdmc2026_main"
    / "map_manager.py"
)
_MAP_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "upstream_map_manager",
    _MAP_MANAGER_PATH,
)
assert _MAP_MANAGER_SPEC is not None
assert _MAP_MANAGER_SPEC.loader is not None
upstream_map_manager = importlib.util.module_from_spec(_MAP_MANAGER_SPEC)
sys.modules[_MAP_MANAGER_SPEC.name] = upstream_map_manager
_MAP_MANAGER_SPEC.loader.exec_module(upstream_map_manager)


def test_lot_plot_writer_creates_level_three_capable_rectangles(tmp_path: Path) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    heightmap = np.full((96, 96), 70, dtype=np.int64)
    core_mask = np.ones_like(heightmap, dtype=bool)
    path_mask = np.zeros_like(heightmap, dtype=bool)
    zone_map = np.zeros_like(heightmap, dtype=np.int16)
    zone_map[:, 48:] = 1
    np.savez(
        data_dir / "settlement_data.npz",
        heightmap=heightmap,
        origin=np.array([100, 0, 200], dtype=np.int64),
        core_cell_mask=core_mask,
        path_mask=path_mask,
        water_map=np.zeros_like(heightmap, dtype=bool),
        chasm_mask=np.zeros_like(heightmap, dtype=bool),
        zone_map=zone_map,
    )
    np.savez(
        data_dir / "settlement_plots.npz",
        plots=np.array([(7, [(10, 10)])], dtype=object),
        farms=np.array([(8, [(20, 20), (21, 20)])], dtype=object),
        building_rects=np.array([], dtype=object),
        module_size=np.array(10),
    )
    args = SimpleNamespace(
        upstream_dir=upstream_dir,
        lot_width=36,
        lot_depth=36,
        lot_gap=8,
        lot_margin=4,
        lot_buildable_threshold=0.70,
        lot_inner_buildable_threshold=0.70,
        lot_max_height_delta=0,
        max_prefabs=2,
        module_size=10,
        setback=2.0,
        min_slot_width=20,
        min_slot_depth=20,
    )

    town._write_lot_building_plots(args)

    plots = np.load(data_dir / "settlement_plots.npz", allow_pickle=True)
    rects = [town._extract_rect(record) for record in plots["building_rects"]]
    assert str(plots["plot_source"]) == "level3_lots"
    assert len(rects) == 2
    assert {rect["width"] for rect in rects} == {36}
    assert {rect["depth"] for rect in rects} == {36}
    assert {rect["y"] for rect in rects} == {70}
    assert {rect["area"] for rect in rects} == {36 * 36}
    assert all(rect["world_x"] == 100 + rect["x"] for rect in rects)
    assert all(rect["world_z"] == 200 + rect["z"] for rect in rects)
    assert dict(plots["plots"]) == {7: [(10, 10)]}
    assert dict(plots["farms"]) == {8: [(20, 20), (21, 20)]}


def test_varied_rect_selection_does_not_cap_to_only_largest() -> None:
    rects = [
        {"cell_id": 1, "width": 40, "depth": 40, "area": 1600},
        {"cell_id": 2, "width": 38, "depth": 38, "area": 1444},
        {"cell_id": 3, "width": 34, "depth": 34, "area": 1156},
        {"cell_id": 4, "width": 32, "depth": 24, "area": 768},
        {"cell_id": 5, "width": 30, "depth": 30, "area": 900},
        {"cell_id": 6, "width": 24, "depth": 24, "area": 576},
    ]

    selected = town._select_varied_rects(rects, max_count=4)
    tiers = [town._rect_fit_tier(rect) for rect in selected]

    assert tiers == [3, 2, 1, 3]
    assert {rect["cell_id"] for rect in selected} == {1, 2, 4, 5}


def test_filter_plot_buildings_for_plan_keeps_only_placed_prefab_cells(
    tmp_path: Path,
) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    np.savez(
        data_dir / "settlement_plots.npz",
        plots=np.array(
            [
                (1, [(10, 10)]),
                (2, [(20, 20)]),
                (4, [(40, 40)]),
            ],
            dtype=object,
        ),
        farms=np.array([(8, [(80, 80)])], dtype=object),
        building_rects=np.array(
            [
                (1, {"cell_id": 1, "x": 10, "z": 10, "width": 30, "depth": 30}),
                (2, {"cell_id": 2, "x": 20, "z": 20, "width": 32, "depth": 32}),
                (3, {"cell_id": 3, "x": 30, "z": 30, "width": 24, "depth": 24}),
            ],
            dtype=object,
        ),
        module_size=np.array(10),
    )
    plan = SimpleNamespace(
        placements=(
            SimpleNamespace(slot=SimpleNamespace(cell_id=2)),
            SimpleNamespace(slot=SimpleNamespace(cell_id=99)),
        )
    )

    stats = town._filter_plot_buildings_for_plan(upstream_dir, plan)

    reloaded = np.load(data_dir / "settlement_plots.npz", allow_pickle=True)
    assert stats == {
        "selected_cell_ids": 2,
        "plots_before": 3,
        "plots_after": 1,
        "rects_before": 3,
        "rects_after": 1,
    }
    assert dict(reloaded["plots"]) == {2: [(20, 20)]}
    assert dict(reloaded["farms"]) == {8: [(80, 80)]}
    assert dict(reloaded["building_rects"]) == {
        2: {"cell_id": 2, "x": 20, "z": 20, "width": 32, "depth": 32}
    }
    assert reloaded["prefab_selected_cell_ids"].tolist() == [2, 99]
    assert int(reloaded["prefab_unfilled_plot_count"]) == 2
    assert int(reloaded["prefab_unfilled_rect_count"]) == 2


def test_building_floor_y_from_plan_tracks_prefab_ground_level() -> None:
    plan = SimpleNamespace(
        placements=(
            SimpleNamespace(bbox=(2, 10, 3, 4, 14, 5)),
            SimpleNamespace(bbox=(4, 12, 4, 6, 15, 6)),
        )
    )

    floor_y = town._building_floor_y_from_plan(plan, shape=(8, 8))

    assert floor_y[0, 0] == -1
    assert np.all(floor_y[3:6, 2:4] == 9)
    assert floor_y[4, 4] == 11
    assert np.all(floor_y[4:7, 5:7] == 11)


def test_prefab_support_maps_use_actual_block_shape_not_slot_rectangle() -> None:
    plan = SimpleNamespace(
        placements=(
            SimpleNamespace(
                bbox=(1, 12, 1, 8, 15, 8),
                blocks=(
                    {"id": "minecraft:stone", "dx": 2, "dz": 2},
                    {"id": "minecraft:stone", "dx": 3, "dz": 2},
                    {"id": "minecraft:stone", "dx": 3, "dz": 3},
                    {"id": town.AIR_BLOCK, "dx": 6, "dz": 6},
                ),
            ),
        )
    )

    footprint, support, floor_y = town._prefab_support_maps_from_plan(
        plan,
        shape=(10, 10),
        buffer=1,
    )

    assert int(np.count_nonzero(footprint)) == 3
    assert footprint[2, 2]
    assert footprint[2, 3]
    assert footprint[3, 3]
    assert not footprint[6, 6]
    assert not footprint[1, 1]
    assert support[1, 1]
    assert support[4, 4]
    assert not support[8, 8]
    assert floor_y[2, 2] == 11
    assert floor_y[1, 1] == 11
    assert floor_y[8, 8] == -1


def test_player_origin_capture_disables_automatic_teleport() -> None:
    assert town._uses_player_origin_capture(
        SimpleNamespace(
            reuse_upstream_data=False,
            region_center=None,
            region_origin=None,
        )
    )
    assert not town._uses_player_origin_capture(
        SimpleNamespace(
            reuse_upstream_data=True,
            region_center=None,
            region_origin=None,
        )
    )
    assert not town._uses_player_origin_capture(
        SimpleNamespace(
            reuse_upstream_data=False,
            region_center=(100, 70, 200),
            region_origin=None,
        )
    )
    assert not town._uses_player_origin_capture(
        SimpleNamespace(
            reuse_upstream_data=False,
            region_center=None,
            region_origin=(100, 200),
        )
    )


def test_high_water_capture_is_rejected_without_mutating_cache(tmp_path: Path) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    path = data_dir / "data.npz"
    water_map = np.ones((8, 8), dtype=bool)
    np.savez_compressed(
        path,
        heightmap=np.full((8, 8), 62, dtype=np.int64),
        flat_mask=np.ones((8, 8), dtype=bool),
        water_map=water_map,
        slope=np.zeros((8, 8), dtype=float),
        origin=np.array([320, 0, 64], dtype=np.int64),
    )
    args = SimpleNamespace(
        upstream_dir=upstream_dir,
        max_water_ratio=0.35,
        allow_water_settlement=False,
    )

    with pytest.raises(RuntimeError, match="high-water"):
        town._validate_captured_map_data(args)

    reloaded = np.load(path, allow_pickle=True)
    assert np.array_equal(reloaded["water_map"], water_map)


def test_map_sample_offsets_try_player_centre_first() -> None:
    offsets = town._map_sample_offsets(radius=1, step=128)

    assert offsets[0] == (0, 0)
    assert set(offsets) == {
        (-128, -128),
        (0, -128),
        (128, -128),
        (-128, 0),
        (0, 0),
        (128, 0),
        (-128, 128),
        (0, 128),
        (128, 128),
    }


def test_capture_valid_map_resamples_neighbouring_slice(tmp_path: Path) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "data.npz"
    captured_centres: list[tuple[int, int, int]] = []

    def write_capture(path: Path, water_ratio: float, origin_x: int, origin_z: int) -> None:
        water_map = np.zeros((10, 10), dtype=bool)
        water_count = int(water_map.size * water_ratio)
        water_map.reshape(-1)[:water_count] = True
        np.savez_compressed(
            path,
            heightmap=np.full((10, 10), 64, dtype=np.int64),
            flat_mask=np.ones((10, 10), dtype=bool),
            water_map=water_map,
            slope=np.zeros((10, 10), dtype=float),
            origin=np.array([origin_x, 0, origin_z], dtype=np.int64),
        )

    class FakeMapManager:
        def resolve_center(self) -> tuple[int, int, int]:
            return (384, 70, 128)

        def load_environment_dataset(self, center: tuple[int, int, int] | None = None) -> dict:
            assert center is not None
            captured_centres.append(center)
            water_ratio = 1.0 if len(captured_centres) == 1 else 0.0
            write_capture(data_path, water_ratio, center[0] - 128, center[2] - 128)
            return {}

    args = SimpleNamespace(
        upstream_dir=upstream_dir,
        map_sample_radius=1,
        map_sample_step=128,
        max_water_ratio=0.35,
        allow_water_settlement=False,
    )

    quality = town._capture_valid_upstream_map_data(args, manager_cls=FakeMapManager)

    assert quality is not None
    assert quality.water_ratio == 0.0
    assert captured_centres[0] == (384, 70, 128)
    assert len(captured_centres) == 2
    reloaded = np.load(data_path, allow_pickle=True)
    assert not np.any(reloaded["water_map"])


def test_region_center_override_replaces_hardcoded_upstream_centre(tmp_path: Path) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "data.npz"
    captured_centres: list[tuple[int, int, int]] = []

    class FakeMapManager:
        def resolve_center(self) -> tuple[int, int, int]:
            return (384, 70, 128)

        def load_environment_dataset(
            self,
            center: tuple[int, int, int] | None = None,
            origin: tuple[int, int] | None = None,
        ) -> dict:
            assert center is not None
            assert origin is None
            captured_centres.append(center)
            np.savez_compressed(
                data_path,
                heightmap=np.full((10, 10), 64, dtype=np.int64),
                flat_mask=np.ones((10, 10), dtype=bool),
                water_map=np.zeros((10, 10), dtype=bool),
                slope=np.zeros((10, 10), dtype=float),
                origin=np.array([center[0] - 192, 0, center[2] - 192], dtype=np.int64),
            )
            return {}

    args = SimpleNamespace(
        upstream_dir=upstream_dir,
        town_area_size=384,
        region_center=(1000, 80, 2000),
        region_origin=None,
        map_sample_radius=0,
        map_sample_step=128,
        max_water_ratio=0.35,
        allow_water_settlement=False,
    )

    quality = town._capture_valid_upstream_map_data(args, manager_cls=FakeMapManager)

    assert quality is not None
    assert captured_centres == [(1000, 80, 2000)]


def test_region_origin_override_samples_exact_rect_origin(tmp_path: Path) -> None:
    upstream_dir = tmp_path / "upstream"
    data_dir = upstream_dir / "data"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "data.npz"
    captured: list[tuple[tuple[int, int, int], tuple[int, int] | None]] = []

    class FakeMapManager:
        def load_environment_dataset(
            self,
            center: tuple[int, int, int] | None = None,
            origin: tuple[int, int] | None = None,
        ) -> dict:
            assert center is not None
            assert origin is not None
            captured.append((center, origin))
            np.savez_compressed(
                data_path,
                heightmap=np.full((10, 10), 64, dtype=np.int64),
                flat_mask=np.ones((10, 10), dtype=bool),
                water_map=np.zeros((10, 10), dtype=bool),
                slope=np.zeros((10, 10), dtype=float),
                origin=np.array([origin[0], 0, origin[1]], dtype=np.int64),
            )
            return {}

    args = SimpleNamespace(
        upstream_dir=upstream_dir,
        town_area_size=384,
        region_center=None,
        region_origin=(1024, 2048),
        map_sample_radius=0,
        map_sample_step=128,
        max_water_ratio=0.35,
        allow_water_settlement=False,
    )

    quality = town._capture_valid_upstream_map_data(args, manager_cls=FakeMapManager)

    assert quality is not None
    assert quality.origin == (1024, 0, 2048)
    assert captured == [((1216, 0, 2240), (1024, 2048))]


def test_player_position_parser_accepts_gdmc_player_data() -> None:
    data = "Pos:[12.7d,64.0d,-9.2d], Rotation:[180.0f,0.0f]"

    assert town._parse_player_position(data) == (12, 64, -10)
    assert upstream_map_manager._parse_player_pose(data) == (12, 64, -10)


def test_upstream_map_manager_uses_players_endpoint_when_editor_has_no_get_player_pos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_hosts: list[str] = []

    def fake_get_player_position(host: str) -> tuple[int, int, int]:
        captured_hosts.append(host)
        return (123, 70, 456)

    monkeypatch.setattr(
        upstream_map_manager,
        "_get_player_position_from_http",
        fake_get_player_position,
    )
    manager = upstream_map_manager.MapManager(host="localhost:9010")

    assert not hasattr(manager.editor, "getPlayerPos")
    assert manager.resolve_center() == (123, 70, 456)
    assert captured_hosts == ["http://localhost:9010"]


def test_wrapper_center_fallback_uses_players_endpoint_without_editor_get_player_pos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_hosts: list[str] = []

    def fake_get_player_position(host: str) -> tuple[int, int, int]:
        captured_hosts.append(host)
        return (-4, 63, 88)

    class FakeEditor:
        host = "http://localhost:9020"

    class FakeManager:
        editor = FakeEditor()
        default_center = None

    monkeypatch.setattr(town, "_get_player_position_from_http", fake_get_player_position)

    assert town._resolve_map_manager_center(FakeManager()) == (-4, 63, 88)
    assert captured_hosts == ["http://localhost:9020"]
