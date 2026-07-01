"""Interior production validation for generated houses.

The room grammar is intentionally deterministic, so production quality should
be checked at the generated-house boundary: did planning select the expected
room types, and did each selected room carry the block motifs that make the
room recognisable downstream?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from prefab_housing.programme import Programme
from prefab_housing.types import HouseResult


MotifSpec = frozenset[str] | tuple[frozenset[str], ...]


ROOM_INTERIOR_MOTIFS: dict[str, MotifSpec] = {
    "bedroom": (
        frozenset(
            {
                "minecraft:red_bed",
                "minecraft:green_bed",
                "minecraft:gray_bed",
                "minecraft:white_bed",
                "minecraft:blue_bed",
                "minecraft:yellow_bed",
            }
        ),
        frozenset({"minecraft:barrel", "minecraft:bookshelf"}),
        frozenset({"minecraft:oak_stairs"}),
    ),
    "living": (
        frozenset(
            {
                "minecraft:gray_wool",
                "minecraft:spruce_planks",
                "minecraft:bamboo_planks",
                "minecraft:white_concrete",
                "minecraft:gray_concrete",
                "minecraft:birch_planks",
            }
        ),
        frozenset(
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
        frozenset({"minecraft:bookshelf", "minecraft:barrel"}),
    ),
    "kitchen": (
        frozenset(
            {
                "minecraft:smooth_stone",
                "minecraft:white_concrete",
                "minecraft:gray_concrete",
                "minecraft:bamboo_planks",
                "minecraft:birch_planks",
                "minecraft:stone_bricks",
            }
        ),
        frozenset({"minecraft:cauldron"}),
        frozenset({"minecraft:furnace"}),
    ),
    "bathroom": (
        frozenset({"minecraft:quartz_stairs"}),
        frozenset({"minecraft:light_blue_stained_glass", "minecraft:glass_pane"}),
    ),
    "entry": (
        frozenset(
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
        frozenset({"minecraft:barrel", "minecraft:bookshelf"}),
    ),
    "corridor": (
        frozenset(
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
        frozenset({"minecraft:barrel", "minecraft:bookshelf"}),
        frozenset({"minecraft:wall_torch"}),
    ),
    "stairwell": (
        frozenset({"minecraft:lantern", "minecraft:sea_lantern"}),
        frozenset({"minecraft:smooth_stone", "minecraft:stone_brick_stairs"}),
    ),
}


@dataclass(frozen=True, slots=True)
class RoomInteriorProductionStatus:
    room_type: str
    room_count: int
    required_block_ids: tuple[str, ...]
    present_block_ids: tuple[str, ...]
    missing_block_ids: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_block_ids


@dataclass(frozen=True, slots=True)
class InteriorProductionReport:
    room_counts: tuple[tuple[str, int], ...]
    expected_room_counts: tuple[tuple[str, int], ...]
    missing_room_counts: tuple[tuple[str, int], ...]
    room_statuses: tuple[RoomInteriorProductionStatus, ...]
    interior_block_count: int
    property_block_count: int

    @property
    def is_valid(self) -> bool:
        return not self.missing_room_counts and all(status.is_valid for status in self.room_statuses)


def expected_room_counts_from_programme(programme: Programme) -> dict[str, int]:
    return dict(programme.required_pods)


def _motif_groups(motif: MotifSpec) -> tuple[frozenset[str], ...]:
    if all(isinstance(item, str) for item in motif):
        return tuple(frozenset({str(item)}) for item in motif)
    return tuple(frozenset(group) for group in motif)  # type: ignore[arg-type]


def _motif_label(group: Iterable[str]) -> str:
    return "|".join(sorted(group))


def analyse_interior_production(
    result: HouseResult,
    *,
    expected_room_counts: Mapping[str, int] | None = None,
    room_motifs: Mapping[str, MotifSpec] = ROOM_INTERIOR_MOTIFS,
) -> InteriorProductionReport:
    room_counts = Counter(room.room_type for room in result.room_interiors)
    expected = dict(expected_room_counts or {})
    missing_room_counts = tuple(
        sorted(
            (room_type, required_count - room_counts.get(room_type, 0))
            for room_type, required_count in expected.items()
            if room_counts.get(room_type, 0) < required_count
        )
    )

    statuses: list[RoomInteriorProductionStatus] = []
    checked_room_types = sorted(set(room_counts) | set(expected))
    for room_type in checked_room_types:
        motif = room_motifs.get(room_type)
        if not motif:
            continue
        required_groups = _motif_groups(motif)
        present_ids = {
            str(block["id"])
            for room in result.room_interiors
            if room.room_type == room_type
            for block in room.blocks
        }
        missing_ids = tuple(
            _motif_label(group)
            for group in required_groups
            if not group & present_ids
        )
        statuses.append(
            RoomInteriorProductionStatus(
                room_type=room_type,
                room_count=room_counts.get(room_type, 0),
                required_block_ids=tuple(_motif_label(group) for group in required_groups),
                present_block_ids=tuple(sorted(present_ids)),
                missing_block_ids=missing_ids,
            )
        )

    return InteriorProductionReport(
        room_counts=tuple(sorted(room_counts.items())),
        expected_room_counts=tuple(sorted(expected.items())),
        missing_room_counts=missing_room_counts,
        room_statuses=tuple(statuses),
        interior_block_count=len(result.interior_blocks),
        property_block_count=sum(1 for block in result.interior_blocks if "properties" in block),
    )


__all__ = [
    "InteriorProductionReport",
    "ROOM_INTERIOR_MOTIFS",
    "RoomInteriorProductionStatus",
    "analyse_interior_production",
    "expected_room_counts_from_programme",
]
