from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PREMADE_PATH = (
    Path(__file__).resolve().parents[2]
    / "narrative"
    / "Premade Builds"
    / "premade_placer.py"
)
_SPEC = importlib.util.spec_from_file_location("premade_placer", _PREMADE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
premade_placer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = premade_placer
_SPEC.loader.exec_module(premade_placer)


def test_chain_remaps_to_older_server_compatible_block() -> None:
    block_id, props, remapped = premade_placer._server_compatible_block(
        "minecraft:chain",
        {"axis": "y"},
    )

    assert remapped is True
    assert block_id == "minecraft:iron_bars"
    assert props == {}


def test_supported_blocks_keep_copied_properties() -> None:
    source_props = {"facing": "north"}

    block_id, props, remapped = premade_placer._server_compatible_block(
        "minecraft:oak_stairs",
        source_props,
    )

    assert remapped is False
    assert block_id == "minecraft:oak_stairs"
    assert props == source_props
    assert props is not source_props
