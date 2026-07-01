from __future__ import annotations

import os
from pathlib import Path

ASSET_ROOT_ENV_VAR = "VOXEL_RENDERER_ASSET_ROOT"


def _discover_packaged_asset_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "assets" / "minecraft"
        if candidate.is_dir():
            return candidate
    return None


def get_asset_root() -> Path:
    override = os.environ.get(ASSET_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    packaged = _discover_packaged_asset_root()
    if packaged is not None:
        return packaged

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets"
        if candidate.is_dir():
            return candidate

    return Path(__file__).resolve().parents[1] / "assets" / "minecraft"


DEFAULT_ASSET_ROOT = get_asset_root()
