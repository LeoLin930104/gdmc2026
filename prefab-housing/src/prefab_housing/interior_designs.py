"""Validated room-style override catalogue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOM_STYLE_DESIGN_PATH = (
    Path(__file__).resolve().parents[2]
    / "designs"
    / "interior_room_styles.generated.json"
)

ALLOWED_ROOM_STYLE_VALUES: dict[str, frozenset[str]] = {
    "bed_block": frozenset(
        {
            "minecraft:red_bed",
            "minecraft:green_bed",
            "minecraft:gray_bed",
            "minecraft:white_bed",
            "minecraft:blue_bed",
            "minecraft:yellow_bed",
        }
    ),
    "wardrobe_block": frozenset(
        {
            "minecraft:barrel",
            "minecraft:bookshelf",
            "minecraft:dark_oak_planks",
            "minecraft:spruce_planks",
            "minecraft:birch_planks",
            "minecraft:bamboo_planks",
            "minecraft:white_concrete",
            "minecraft:gray_concrete",
            "minecraft:smooth_stone",
            "minecraft:stone_bricks",
        }
    ),
    "desk_top_block": frozenset(
        {
            "minecraft:birch_slab",
            "minecraft:spruce_slab",
            "minecraft:oak_slab",
            "minecraft:bamboo_slab",
            "minecraft:quartz_slab",
            "minecraft:smooth_stone_slab",
            "minecraft:dark_oak_planks",
        }
    ),
    "table_top_block": frozenset(
        {
            "minecraft:birch_slab",
            "minecraft:spruce_slab",
            "minecraft:oak_slab",
            "minecraft:bamboo_slab",
            "minecraft:quartz_slab",
            "minecraft:smooth_stone_slab",
            "minecraft:dark_oak_planks",
        }
    ),
    "sofa_base_block": frozenset(
        {
            "minecraft:gray_wool",
            "minecraft:spruce_planks",
            "minecraft:bamboo_planks",
            "minecraft:white_concrete",
            "minecraft:gray_concrete",
            "minecraft:birch_planks",
        }
    ),
    "sofa_top_block": frozenset(
        {
            "minecraft:gray_carpet",
            "minecraft:light_gray_carpet",
            "minecraft:white_carpet",
            "minecraft:green_carpet",
            "minecraft:light_blue_carpet",
            "minecraft:yellow_carpet",
        }
    ),
    "rug_primary_block": frozenset(
        {
            "minecraft:gray_carpet",
            "minecraft:light_gray_carpet",
            "minecraft:white_carpet",
            "minecraft:green_carpet",
            "minecraft:blue_carpet",
            "minecraft:light_blue_carpet",
            "minecraft:lime_carpet",
            "minecraft:yellow_carpet",
        }
    ),
    "rug_secondary_block": frozenset(
        {
            "minecraft:gray_carpet",
            "minecraft:light_gray_carpet",
            "minecraft:white_carpet",
            "minecraft:green_carpet",
            "minecraft:blue_carpet",
            "minecraft:light_blue_carpet",
            "minecraft:lime_carpet",
            "minecraft:yellow_carpet",
        }
    ),
    "kitchen_counter_block": frozenset(
        {
            "minecraft:smooth_stone",
            "minecraft:white_concrete",
            "minecraft:gray_concrete",
            "minecraft:bamboo_planks",
            "minecraft:birch_planks",
            "minecraft:stone_bricks",
        }
    ),
    "kitchen_top_block": frozenset(
        {
            "minecraft:smooth_stone_slab",
            "minecraft:quartz_slab",
            "minecraft:bamboo_slab",
            "minecraft:birch_slab",
            "minecraft:spruce_slab",
        }
    ),
    "bathroom_wall_block": frozenset(
        {
            "minecraft:light_blue_concrete",
            "minecraft:quartz_block",
            "minecraft:smooth_stone",
            "minecraft:stone_bricks",
            "minecraft:white_wool",
            "minecraft:bamboo_planks",
            "minecraft:spruce_planks",
        }
    ),
    "shower_glass_block": frozenset(
        {
            "minecraft:light_blue_stained_glass",
            "minecraft:glass_pane",
        }
    ),
    "storage_block": frozenset({"minecraft:barrel", "minecraft:bookshelf"}),
    "accent_light_block": frozenset({"minecraft:lantern", "minecraft:sea_lantern"}),
}


@dataclass(frozen=True, slots=True)
class RoomStyleVariant:
    room_type: str
    id: str
    label: str
    overrides: dict[str, str]


def _validate_override(
    *,
    room_type: str,
    variant_id: str,
    key: str,
    value: Any,
) -> tuple[str, str]:
    if key not in ALLOWED_ROOM_STYLE_VALUES:
        raise ValueError(
            f"room style {room_type}/{variant_id} uses unsupported field {key!r}"
        )
    block_id = str(value)
    if block_id not in ALLOWED_ROOM_STYLE_VALUES[key]:
        raise ValueError(
            f"room style {room_type}/{variant_id} uses unsupported {key}={block_id!r}"
        )
    return key, block_id


def _variant_from_mapping(room_type: str, payload: Mapping[str, Any]) -> RoomStyleVariant:
    variant_id = str(payload.get("id", "")).strip()
    label = str(payload.get("label", "")).strip()
    if not variant_id:
        raise ValueError(f"room style for {room_type!r} is missing id")
    if not label:
        raise ValueError(f"room style {room_type}/{variant_id} is missing label")
    raw_overrides = payload.get("overrides")
    if not isinstance(raw_overrides, Mapping) or not raw_overrides:
        raise ValueError(f"room style {room_type}/{variant_id} has no overrides")
    overrides = dict(
        _validate_override(
            room_type=room_type,
            variant_id=variant_id,
            key=str(key),
            value=value,
        )
        for key, value in raw_overrides.items()
    )
    return RoomStyleVariant(
        room_type=room_type,
        id=variant_id,
        label=label,
        overrides=overrides,
    )


def load_room_style_variants(
    path: str | Path = ROOM_STYLE_DESIGN_PATH,
) -> dict[str, tuple[RoomStyleVariant, ...]]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    room_styles = payload.get("room_styles")
    if not isinstance(room_styles, Mapping):
        raise ValueError(f"room style design has no room_styles mapping: {source}")

    variants: dict[str, tuple[RoomStyleVariant, ...]] = {}
    for room_type, records in room_styles.items():
        if not isinstance(records, list):
            raise ValueError(f"room style records for {room_type!r} must be a list")
        variants[str(room_type)] = tuple(
            _variant_from_mapping(str(room_type), record)
            for record in records
            if isinstance(record, Mapping)
        )
        if len(variants[str(room_type)]) != len(records):
            raise ValueError(f"room style records for {room_type!r} contain non-mappings")
    return variants


__all__ = [
    "ALLOWED_ROOM_STYLE_VALUES",
    "ROOM_STYLE_DESIGN_PATH",
    "RoomStyleVariant",
    "load_room_style_variants",
]
