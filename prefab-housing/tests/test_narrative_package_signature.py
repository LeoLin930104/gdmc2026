from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_NARRATIVE_DIR = Path(__file__).resolve().parents[2] / "narrative"
if str(_NARRATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_NARRATIVE_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "wallface_narrative",
    _NARRATIVE_DIR / "wallface_narrative.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
wallface_narrative = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wallface_narrative
_SPEC.loader.exec_module(wallface_narrative)


def test_package_signature_invalidates_wallface_only_stamps() -> None:
    design_signature = wallface_narrative.design_signature(
        "strained",
        "minecraft:plains",
        43,
    )
    package_signature = wallface_narrative.package_signature(
        "strained",
        "minecraft:plains",
        43,
    )

    assert package_signature != design_signature
    assert package_signature == wallface_narrative.package_signature(
        "strained",
        "minecraft:plains",
        43,
    )
