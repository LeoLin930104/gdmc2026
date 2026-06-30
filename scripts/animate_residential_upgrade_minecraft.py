"""Export and optionally animate residential upgrades in Minecraft via GDPC.

The live path builds level 1, then applies only the block diffs for level 2
and level 3.  Per-level states and diff states are always exported first so
the generated results can be inspected without a running Minecraft client.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from prefab_housing.minecraft_animation import (
    AnimationStrategy,
    BlueprintBlock,
    UpgradeDiff,
    build_residential_upgrade_sequence,
    build_upgrade_diffs,
    compute_bounding_box,
    export_residential_upgrade_package,
    export_residential_upgrade_sequence,
    iter_batches,
    load_residential_upgrade_package,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "out" / "minecraft" / "residential_upgrade"

_HORIZONTAL_FACES = ("north", "east", "south", "west")
_FACE_TO_VECTOR = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
_VECTOR_TO_FACE = {vector: face for face, vector in _FACE_TO_VECTOR.items()}
_PASS_THROUGH_BLOCK_IDS = {
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
    "minecraft:water",
    "minecraft:lava",
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:grass",
    "minecraft:fern",
    "minecraft:large_fern",
    "minecraft:dead_bush",
    "minecraft:snow",
    "minecraft:vine",
}


@dataclass(frozen=True, slots=True)
class PlayerContext:
    x: int
    y: int
    z: int
    yaw: float
    forward: tuple[int, int]


@dataclass(frozen=True, slots=True)
class HousePayload:
    label: str
    seed: int | None
    states: list[Any]
    diffs: list[UpgradeDiff] | None
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LiveHousePlacement:
    label: str
    seed: int | None
    states: list[Any]
    diffs: list[UpgradeDiff] | None
    origin: tuple[int, int, int]
    bbox: tuple[int, int, int, int, int, int]
    entrance_face: str | None
    rotation_steps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and optionally live-animate residential housing upgrades.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Residential upgrade levels to generate in order.",
    )
    parser.add_argument("--material-theme", default="sci_fi_modular")
    parser.add_argument(
        "--input-package",
        type=Path,
        default=None,
        help=(
            "Load a compact residential upgrade package from this file "
            "instead of regenerating it."
        ),
    )
    parser.add_argument(
        "--input-packages",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Load multiple compact residential upgrade packages and place them as "
            "a line-up without regenerating them."
        ),
    )
    parser.add_argument(
        "--wallface-design",
        type=Path,
        default=None,
        help=(
            "Optional .wallface preset path. Defaults to a deterministic modular_* "
            "preset chosen from the seed."
        ),
    )
    parser.add_argument(
        "--lineup-count",
        "--aio-lineup-count",
        dest="lineup_count",
        type=int,
        default=1,
        help=(
            "Generate/place this many residential houses in a player-facing line-up. "
            "When --lineup-seeds is omitted, seeds increment from --seed."
        ),
    )
    parser.add_argument(
        "--lineup-seeds",
        "--aio-lineup-seeds",
        dest="lineup_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list for a generated line-up.",
    )
    parser.add_argument(
        "--lineup-gap",
        type=int,
        default=8,
        help="Minimum horizontal gap, in blocks, between line-up houses.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--package-out",
        type=Path,
        default=None,
        help="Optional compact binary package to write after generation.",
    )
    parser.add_argument(
        "--strategy",
        choices=("y_up", "y_down", "radial_out"),
        default="y_up",
        help="Batch ordering strategy for build and diff placements.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Place the generated upgrade sequence through GDPC.",
    )
    parser.add_argument("--host", default="http://localhost:9000")
    parser.add_argument(
        "--origin",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World-space origin for live placement. Defaults to player-relative.",
    )
    parser.add_argument("--player-clearance", type=int, default=3)
    parser.add_argument("--player-margin", type=int, default=2)
    parser.add_argument(
        "--face-player",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate live placement so the main exterior entrance faces the player.",
    )
    parser.add_argument(
        "--ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raycast down before live placement and put the house floor on terrain.",
    )
    parser.add_argument("--ground-search-up", type=int, default=96)
    parser.add_argument("--ground-search-down", type=int, default=192)
    parser.add_argument(
        "--clear-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the full upgrade bounding box before placing level 1.",
    )
    parser.add_argument("--per-block-delay-ms", type=int, default=0)
    parser.add_argument("--per-layer-delay-ms", type=int, default=120)
    parser.add_argument(
        "--upgrade-delay-s",
        "--between-upgrade-delay-s",
        dest="upgrade_delay_s",
        type=float,
        default=3.0,
        help="Seconds to pause before each live upgrade diff starts.",
    )
    parser.add_argument("--flush-every", type=int, default=64)
    args = parser.parse_args()
    if args.upgrade_delay_s < 0:
        parser.error("--upgrade-delay-s must be non-negative")
    if args.lineup_count < 1:
        parser.error("--lineup-count must be at least 1")
    if args.lineup_gap < 0:
        parser.error("--lineup-gap must be non-negative")
    if args.ground_search_up < 0:
        parser.error("--ground-search-up must be non-negative")
    if args.ground_search_down < 1:
        parser.error("--ground-search-down must be at least 1")
    if args.input_package is not None and args.input_packages is not None:
        parser.error("--input-package and --input-packages are mutually exclusive")
    if args.input_packages is not None:
        args.lineup_count = len(args.input_packages)
    elif args.lineup_seeds is not None:
        args.lineup_count = len(args.lineup_seeds)
    if args.input_package is not None and args.lineup_seeds is not None:
        parser.error("--lineup-seeds only applies when generating, not loading a package")
    if args.input_packages is not None and args.lineup_seeds is not None:
        parser.error("--lineup-seeds only applies when generating, not loading packages")
    if args.input_package is not None and not args.input_package.exists():
        parser.error(f"--input-package does not exist: {args.input_package}")
    if args.input_packages is not None:
        missing = [path for path in args.input_packages if not path.exists()]
        if missing:
            parser.error(f"--input-packages contains missing files: {missing}")
    return args


def _normalise_host(host: str) -> str:
    if not host.startswith(("http://", "https://")):
        return f"http://{host}"
    return host


def _import_gdpc() -> tuple[Any, Any]:
    try:
        from gdpc import Editor
        from gdpc.block import Block
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live placement requires the optional 'gdpc' package and a running "
            "GDMC-HTTP Minecraft server. Export-only mode does not require it."
        ) from exc
    return Editor, Block


def _parse_player_pose(data: str) -> tuple[int, int, int, float] | None:
    pos_match = re.search(r"Pos:\[([^\]]+)\]", data)
    rot_match = re.search(r"Rotation:\[([^\]]+)\]", data)
    if not pos_match or not rot_match:
        return None
    pos_values = [part.strip().rstrip("dD") for part in pos_match.group(1).split(",")]
    rot_values = [part.strip().rstrip("fF") for part in rot_match.group(1).split(",")]
    if len(pos_values) < 3 or len(rot_values) < 1:
        return None
    return (
        int(math.floor(float(pos_values[0]))),
        int(math.floor(float(pos_values[1]))),
        int(math.floor(float(pos_values[2]))),
        float(rot_values[0]),
    )


def _get_player_pose(host: str) -> tuple[int, int, int, float]:
    import requests

    response = requests.get(
        f"{_normalise_host(host).rstrip('/')}/players",
        params={"includeData": "true"},
        timeout=1.0,
    )
    response.raise_for_status()
    players = response.json()
    if not players:
        raise RuntimeError("GDMC server returned no players for player-relative origin")
    pose = _parse_player_pose(players[0].get("data", ""))
    if pose is None:
        raise RuntimeError("could not parse player position from GDMC player data")
    return pose


def _yaw_to_forward(yaw_degrees: float) -> tuple[int, int]:
    yaw = yaw_degrees % 360.0
    if 45.0 <= yaw < 135.0:
        return -1, 0
    if 135.0 <= yaw < 225.0:
        return 0, -1
    if 225.0 <= yaw < 315.0:
        return 1, 0
    return 0, 1


def _get_player_context(host: str) -> PlayerContext:
    player_x, player_y, player_z, yaw = _get_player_pose(host)
    return PlayerContext(
        x=player_x,
        y=player_y,
        z=player_z,
        yaw=yaw,
        forward=_yaw_to_forward(yaw),
    )


def _right_vector(forward: tuple[int, int]) -> tuple[int, int]:
    forward_x, forward_z = forward
    return -forward_z, forward_x


def _opposite_vector(vector: tuple[int, int]) -> tuple[int, int]:
    return -vector[0], -vector[1]


def _vector_to_face(vector: tuple[int, int]) -> str:
    try:
        return _VECTOR_TO_FACE[vector]
    except KeyError as exc:
        raise ValueError(f"not a cardinal horizontal vector: {vector}") from exc


def _face_after_rotation(face: str, steps: int) -> str:
    if face not in _HORIZONTAL_FACES:
        return face
    return _HORIZONTAL_FACES[(_HORIZONTAL_FACES.index(face) + steps) % 4]


def _rotation_steps_between(source_face: str, target_face: str) -> int:
    if source_face not in _HORIZONTAL_FACES:
        raise ValueError(f"cannot rotate non-horizontal source face: {source_face}")
    if target_face not in _HORIZONTAL_FACES:
        raise ValueError(f"cannot rotate non-horizontal target face: {target_face}")
    return (
        _HORIZONTAL_FACES.index(target_face)
        - _HORIZONTAL_FACES.index(source_face)
    ) % 4


def _rotated_props(props: Any, steps: int) -> dict[str, str]:
    if not isinstance(props, Mapping):
        return {}
    rotated = {str(key): str(value) for key, value in props.items()}
    facing = rotated.get("facing")
    if facing in _HORIZONTAL_FACES:
        rotated["facing"] = _face_after_rotation(facing, steps)
    axis = rotated.get("axis")
    if steps % 2 == 1 and axis in {"x", "z"}:
        rotated["axis"] = "z" if axis == "x" else "x"
    return rotated


def _rotated_block(
    block: BlueprintBlock,
    *,
    bbox: tuple[int, int, int, int, int, int],
    steps: int,
) -> BlueprintBlock:
    min_dx, _min_dy, min_dz, max_dx, _max_dy, max_dz = bbox
    width = max_dx - min_dx
    depth = max_dz - min_dz
    x = int(block["dx"]) - min_dx
    z = int(block["dz"]) - min_dz
    rotation = steps % 4
    if rotation == 0:
        nx, nz = x, z
    elif rotation == 1:
        nx, nz = depth - z, x
    elif rotation == 2:
        nx, nz = width - x, depth - z
    else:
        nx, nz = z, width - x

    rotated = dict(block)
    rotated["dx"] = min_dx + nx
    rotated["dz"] = min_dz + nz
    props = _rotated_props(block.get("props", {}), rotation)
    if props:
        rotated["props"] = props
    else:
        rotated.pop("props", None)
    return rotated


def _rotated_blocks(
    blocks: list[BlueprintBlock],
    *,
    bbox: tuple[int, int, int, int, int, int],
    steps: int,
) -> list[BlueprintBlock]:
    rotation = steps % 4
    if rotation == 0:
        return [dict(block) for block in blocks]
    return [_rotated_block(block, bbox=bbox, steps=rotation) for block in blocks]


def _state_entrance_face(states: list[Any]) -> str | None:
    for state in states:
        entrance_face = getattr(state, "entrance_face", None)
        if isinstance(entrance_face, str) and entrance_face in _HORIZONTAL_FACES:
            return entrance_face
    return None


def _oriented_house(
    payload: HousePayload,
    *,
    target_entrance_face: str | None,
) -> tuple[HousePayload, int, str | None]:
    entrance_face = _state_entrance_face(payload.states)
    if target_entrance_face is None or entrance_face is None:
        return payload, 0, entrance_face
    rotation_steps = _rotation_steps_between(entrance_face, target_entrance_face)
    if rotation_steps == 0:
        return payload, 0, entrance_face

    all_blocks = [block for state in payload.states for block in state.blocks]
    bbox = compute_bounding_box(all_blocks)
    rotated_states = [
        replace(
            state,
            blocks=_rotated_blocks(state.blocks, bbox=bbox, steps=rotation_steps),
            structure_blocks=_rotated_blocks(
                state.structure_blocks,
                bbox=bbox,
                steps=rotation_steps,
            ),
            entrance_face=_face_after_rotation(entrance_face, rotation_steps),
        )
        for state in payload.states
    ]
    rotated_diffs = None
    if payload.diffs is not None:
        rotated_diffs = [
            replace(
                diff,
                blocks=_rotated_blocks(diff.blocks, bbox=bbox, steps=rotation_steps),
            )
            for diff in payload.diffs
        ]
    return (
        HousePayload(
            label=payload.label,
            seed=payload.seed,
            states=rotated_states,
            diffs=rotated_diffs,
            manifest=payload.manifest,
        ),
        rotation_steps,
        _face_after_rotation(entrance_face, rotation_steps),
    )


def _resolve_player_relative_origin(
    *,
    player: PlayerContext,
    blocks: list[BlueprintBlock],
    clearance: int,
    margin: int,
    lateral_axis: tuple[int, int] = (1, 0),
    lateral_offset: int = 0,
) -> tuple[int, int, int]:
    min_dx, min_dy, min_dz, max_dx, _max_dy, max_dz = compute_bounding_box(blocks)
    size_x = max_dx - min_dx + 1
    size_z = max_dz - min_dz + 1
    forward_x, forward_z = player.forward
    depth = size_x if forward_x != 0 else size_z
    distance = clearance + margin + depth
    target_x = (
        player.x
        + forward_x * distance
        + lateral_axis[0] * lateral_offset
    )
    target_z = (
        player.z
        + forward_z * distance
        + lateral_axis[1] * lateral_offset
    )

    if forward_x > 0:
        origin_x = target_x - min_dx
    elif forward_x < 0:
        origin_x = target_x - max_dx
    else:
        origin_x = target_x - min_dx - (size_x // 2)

    if forward_z > 0:
        origin_z = target_z - min_dz
    elif forward_z < 0:
        origin_z = target_z - max_dz
    else:
        origin_z = target_z - min_dz - (size_z // 2)

    origin_y = player.y - min_dy
    print(
        f"[origin] player=({player.x}, {player.y}, {player.z}) yaw={player.yaw:.1f} "
        f"-> origin=({origin_x}, {origin_y}, {origin_z})"
    )
    return origin_x, origin_y, origin_z


def _footprint_size_along_axis(
    bbox: tuple[int, int, int, int, int, int],
    axis: tuple[int, int],
) -> int:
    min_dx, _min_dy, min_dz, max_dx, _max_dy, max_dz = bbox
    if axis[0] != 0:
        return max_dx - min_dx + 1
    if axis[1] != 0:
        return max_dz - min_dz + 1
    raise ValueError(f"line-up axis must be horizontal cardinal: {axis}")


def _centred_lineup_offsets(sizes: list[int], gap: int) -> list[int]:
    if not sizes:
        return []
    centres: list[float] = []
    cursor = 0.0
    for size in sizes:
        centres.append(cursor + (size / 2.0))
        cursor += size + gap
    total_width = cursor - gap
    return [int(round(centre - (total_width / 2.0))) for centre in centres]


def _first_origin_lineup_offsets(sizes: list[int], gap: int) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size + gap
    return offsets


def _block_id(block: Any) -> str:
    if isinstance(block, str):
        return block
    for attr in ("id", "namespaced_name", "name"):
        value = getattr(block, attr, None)
        if isinstance(value, str):
            return value
    text = str(block)
    return text.split("[", maxsplit=1)[0]


def _is_pass_through_block(block_id: str) -> bool:
    if block_id in _PASS_THROUGH_BLOCK_IDS:
        return True
    return (
        block_id.endswith("_leaves")
        or block_id.endswith("_sapling")
        or block_id.endswith("_roots")
        or block_id.endswith("_flower")
    )


def _grounded_origin(
    *,
    editor: Any,
    origin: tuple[int, int, int],
    bbox: tuple[int, int, int, int, int, int],
    search_up: int,
    search_down: int,
) -> tuple[int, int, int]:
    if not hasattr(editor, "getBlock"):
        raise RuntimeError("GDPC Editor does not expose getBlock for grounding raycast")

    ox, oy, oz = origin
    min_dx, min_dy, min_dz, max_dx, max_dy, max_dz = bbox
    probe_x = ox + ((min_dx + max_dx) // 2)
    probe_z = oz + ((min_dz + max_dz) // 2)
    start_y = min(319, oy + max_dy + search_up)
    end_y = max(-64, oy + min_dy - search_down)

    for y in range(start_y, end_y - 1, -1):
        block_id = _block_id(editor.getBlock((probe_x, y, probe_z)))
        if not _is_pass_through_block(block_id):
            grounded = (ox, y + 1 - min_dy, oz)
            print(
                "[ground] "
                f"probe=({probe_x}, {probe_z}) range={start_y}->{end_y} "
                f"hit={block_id}@{y} -> origin={grounded}"
            )
            return grounded

    print(
        "[ground] no terrain hit for "
        f"probe=({probe_x}, {probe_z}) range={start_y}->{end_y}; keeping {origin}"
    )
    return origin


def _place_blocks(
    *,
    editor: Any,
    block_cls: Any,
    blocks: list[BlueprintBlock],
    origin: tuple[int, int, int],
    strategy: AnimationStrategy,
    per_block_delay_ms: int,
    per_layer_delay_ms: int,
    flush_every: int,
    label: str,
) -> int:
    ox, oy, oz = origin
    total = 0
    per_block_s = per_block_delay_ms / 1000.0
    per_layer_s = per_layer_delay_ms / 1000.0
    previous_do_block_updates = getattr(editor, "doBlockUpdates", True)
    previous_spawn_drops = getattr(editor, "spawnDrops", True)
    editor.doBlockUpdates = False
    editor.spawnDrops = False
    try:
        for batch_index, batch in enumerate(
            iter_batches(blocks, strategy=strategy),
            start=1,
        ):
            for index, block in enumerate(batch):
                editor.placeBlock(
                    (
                        ox + int(block["dx"]),
                        oy + int(block["dy"]),
                        oz + int(block["dz"]),
                    ),
                    block_cls(str(block["id"]), block.get("props", {})),
                )
                total += 1
                if total % flush_every == 0:
                    editor.flushBuffer()
                if per_block_s > 0 and index < len(batch) - 1:
                    editor.flushBuffer()
                    time.sleep(per_block_s)

            editor.flushBuffer()
            if per_layer_s > 0:
                time.sleep(per_layer_s)
            if batch_index <= 3 or batch_index % 5 == 0:
                print(f"[{label}] batch {batch_index}: +{len(batch)} blocks")
    finally:
        editor.flushBuffer()
        editor.doBlockUpdates = previous_do_block_updates
        editor.spawnDrops = previous_spawn_drops

    editor.flushBuffer()
    print(f"[{label}] placed {total} block updates")
    return total


def _clear_bbox(
    *,
    editor: Any,
    block_cls: Any,
    bbox: tuple[int, int, int, int, int, int],
    origin: tuple[int, int, int],
) -> None:
    min_dx, min_dy, min_dz, max_dx, max_dy, max_dz = bbox
    ox, oy, oz = origin
    air = block_cls("minecraft:air")
    previous_do_block_updates = getattr(editor, "doBlockUpdates", True)
    previous_spawn_drops = getattr(editor, "spawnDrops", True)
    editor.doBlockUpdates = False
    editor.spawnDrops = False
    try:
        for dx in range(min_dx, max_dx + 1):
            for dy in range(min_dy, max_dy + 1):
                for dz in range(min_dz, max_dz + 1):
                    editor.placeBlock((ox + dx, oy + dy, oz + dz), air)
    finally:
        editor.flushBuffer()
        editor.doBlockUpdates = previous_do_block_updates
        editor.spawnDrops = previous_spawn_drops
    print(
        "[clear] cleared "
        f"({ox + min_dx}, {oy + min_dy}, {oz + min_dz}) -> "
        f"({ox + max_dx}, {oy + max_dy}, {oz + max_dz})"
    )


def _orient_live_houses(
    houses: list[HousePayload],
    *,
    player: PlayerContext | None,
    face_player: bool,
) -> list[tuple[HousePayload, int, str | None]]:
    if player is None or not face_player:
        return [(house, 0, _state_entrance_face(house.states)) for house in houses]

    target_face = _vector_to_face(_opposite_vector(player.forward))
    oriented: list[tuple[HousePayload, int, str | None]] = []
    for house in houses:
        oriented_house, rotation_steps, entrance_face = _oriented_house(
            house,
            target_entrance_face=target_face,
        )
        print(
            f"[orient] {house.label}: entrance={_state_entrance_face(house.states) or 'unknown'} "
            f"target={target_face} rotation={rotation_steps * 90}deg"
        )
        oriented.append((oriented_house, rotation_steps, entrance_face))
    return oriented


def _prepare_live_placements(
    *,
    args: argparse.Namespace,
    editor: Any,
    oriented_houses: list[tuple[HousePayload, int, str | None]],
    player: PlayerContext | None,
) -> list[LiveHousePlacement]:
    if not oriented_houses:
        return []

    row_axis = _right_vector(player.forward) if player is not None else (1, 0)
    house_blocks = [
        [block for state in house.states for block in state.blocks]
        for house, _rotation_steps, _entrance_face in oriented_houses
    ]
    bboxes = [compute_bounding_box(blocks) for blocks in house_blocks]
    sizes = [_footprint_size_along_axis(bbox, row_axis) for bbox in bboxes]
    offsets = (
        _centred_lineup_offsets(sizes, args.lineup_gap)
        if args.origin is None
        else _first_origin_lineup_offsets(sizes, args.lineup_gap)
    )

    placements: list[LiveHousePlacement] = []
    explicit_origin = (
        tuple(int(value) for value in args.origin)
        if args.origin is not None
        else None
    )
    if explicit_origin is not None:
        print(f"[origin] explicit origin={explicit_origin}")

    for index, (house, rotation_steps, entrance_face) in enumerate(oriented_houses):
        blocks = house_blocks[index]
        bbox = bboxes[index]
        lateral_offset = offsets[index]
        if player is not None and explicit_origin is None:
            origin = _resolve_player_relative_origin(
                player=player,
                blocks=blocks,
                clearance=args.player_clearance,
                margin=args.player_margin,
                lateral_axis=row_axis,
                lateral_offset=lateral_offset,
            )
        elif explicit_origin is not None:
            origin = (
                explicit_origin[0] + row_axis[0] * lateral_offset,
                explicit_origin[1],
                explicit_origin[2] + row_axis[1] * lateral_offset,
            )
            if index > 0:
                print(f"[origin] {house.label}: line-up origin={origin}")
        else:
            raise RuntimeError("player-relative live placement requires player data")

        if args.ground:
            origin = _grounded_origin(
                editor=editor,
                origin=origin,
                bbox=bbox,
                search_up=args.ground_search_up,
                search_down=args.ground_search_down,
            )

        placements.append(
            LiveHousePlacement(
                label=house.label,
                seed=house.seed,
                states=house.states,
                diffs=house.diffs,
                origin=origin,
                bbox=bbox,
                entrance_face=entrance_face,
                rotation_steps=rotation_steps,
            )
        )
    return placements


def _place_live_house(
    *,
    args: argparse.Namespace,
    editor: Any,
    block_cls: Any,
    placement: LiveHousePlacement,
) -> int:
    if args.clear_first:
        _clear_bbox(
            editor=editor,
            block_cls=block_cls,
            bbox=placement.bbox,
            origin=placement.origin,
        )

    strategy: AnimationStrategy = args.strategy
    total = _place_blocks(
        editor=editor,
        block_cls=block_cls,
        blocks=placement.states[0].blocks,
        origin=placement.origin,
        strategy=strategy,
        per_block_delay_ms=args.per_block_delay_ms,
        per_layer_delay_ms=args.per_layer_delay_ms,
        flush_every=args.flush_every,
        label=f"{placement.label}_level_{placement.states[0].level}",
    )

    for diff in (
        placement.diffs
        if placement.diffs is not None
        else build_upgrade_diffs(placement.states)
    ):
        if args.upgrade_delay_s > 0:
            print(
                f"[delay] {placement.label}: waiting {args.upgrade_delay_s:g}s "
                f"before level {diff.to_level}"
            )
            time.sleep(args.upgrade_delay_s)
        total += _place_blocks(
            editor=editor,
            block_cls=block_cls,
            blocks=diff.blocks,
            origin=placement.origin,
            strategy=strategy,
            per_block_delay_ms=args.per_block_delay_ms,
            per_layer_delay_ms=args.per_layer_delay_ms,
            flush_every=args.flush_every,
            label=f"{placement.label}_diff_{diff.from_level}_to_{diff.to_level}",
        )
    return total


def _run_live(args: argparse.Namespace, houses: list[HousePayload]) -> None:
    Editor, Block = _import_gdpc()
    editor = Editor(buffering=True, host=_normalise_host(args.host))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    player = None
    if args.origin is None or args.face_player:
        player = _get_player_context(args.host)

    oriented_houses = _orient_live_houses(
        houses,
        player=player,
        face_player=args.face_player,
    )
    placements = _prepare_live_placements(
        args=args,
        editor=editor,
        oriented_houses=oriented_houses,
        player=player,
    )

    total = 0
    for placement in placements:
        total += _place_live_house(
            args=args,
            editor=editor,
            block_cls=Block,
            placement=placement,
        )

    session = {
        "host": _normalise_host(args.host),
        "lineup_count": len(placements),
        "lineup_gap": args.lineup_gap,
        "levels": [state.level for state in placements[0].states] if placements else [],
        "final_level": placements[0].states[-1].level if placements else None,
        "upgrade_delay_s": args.upgrade_delay_s,
        "face_player": args.face_player,
        "ground": args.ground,
        "placed_updates": total,
        "houses": [
            {
                "label": placement.label,
                "seed": placement.seed,
                "origin": placement.origin,
                "bbox": placement.bbox,
                "entrance_face": placement.entrance_face,
                "rotation_degrees": placement.rotation_steps * 90,
            }
            for placement in placements
        ],
    }
    (args.output_dir / "minecraft_session.json").write_text(
        json.dumps(session, indent=2),
        encoding="utf-8",
    )
    print(f"[live] complete: {total} block updates across {len(placements)} houses")


def _lineup_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.lineup_seeds is not None:
        return tuple(int(seed) for seed in args.lineup_seeds)
    return tuple(args.seed + index for index in range(args.lineup_count))


def _output_dir_for_house(args: argparse.Namespace, *, index: int, seed: int) -> Path:
    if args.lineup_count == 1:
        return args.output_dir
    return args.output_dir / f"house_{index + 1:02d}_seed_{seed:03d}"


def _package_out_for_house(args: argparse.Namespace, *, index: int, seed: int) -> Path | None:
    if args.package_out is None:
        return None
    if args.lineup_count == 1:
        return args.package_out
    suffix = args.package_out.suffix or ".pbp"
    return args.package_out.with_name(
        f"{args.package_out.stem}_house_{index + 1:02d}_seed_{seed:03d}{suffix}"
    )


def _print_manifest(manifest: dict[str, Any]) -> None:
    for record in manifest["levels"]:
        level_ref = record.get("path") or f"package:{record['section']}"
        structure_ref = (
            record.get("structure_cache_path")
            or f"package:{record['structure_section']}"
        )
        print(
            f"  level {record['level']} {record['name']}: "
            f"{record['block_count']} blocks -> {level_ref}"
        )
        print(f"    wallface: {record.get('wall_face_preset') or 'none'}")
        print(f"    entrance: {record.get('entrance_face') or 'unknown'}")
        print(f"    layout: {record.get('layout_variant_id') or 'unknown'}")
        print(f"    interior: {record.get('interior_style_id') or 'unknown'}")
        print(
            f"    structure cache: {record['structure_block_count']} blocks -> "
            f"{structure_ref}"
        )
    for record in manifest["diffs"]:
        diff_ref = record.get("path") or f"package:{record['section']}"
        print(
            f"  diff {record['from_level']}->{record['to_level']}: "
            f"{record['block_count']} updates -> {diff_ref}"
        )


def _load_or_generate_houses(args: argparse.Namespace) -> list[HousePayload]:
    if args.input_packages is not None:
        houses: list[HousePayload] = []
        for index, package_path in enumerate(args.input_packages):
            states, cached_diffs, manifest = load_residential_upgrade_package(package_path)
            print(f"[package] loaded {index + 1}/{len(args.input_packages)} {package_path}")
            _print_manifest(manifest)
            houses.append(
                HousePayload(
                    label=f"house_{index + 1:02d}",
                    seed=getattr(states[0], "seed", None),
                    states=states,
                    diffs=cached_diffs,
                    manifest=manifest,
                )
            )
        return houses

    if args.input_package is not None:
        states, cached_diffs, manifest = load_residential_upgrade_package(args.input_package)
        print(f"[package] loaded {args.input_package}")
        _print_manifest(manifest)
        if args.lineup_count > 1:
            print(f"[lineup] repeating package {args.lineup_count} times")
        return [
            HousePayload(
                label=f"house_{index + 1:02d}",
                seed=getattr(states[0], "seed", None),
                states=states,
                diffs=cached_diffs,
                manifest=manifest,
            )
            for index in range(args.lineup_count)
        ]

    houses: list[HousePayload] = []
    for index, seed in enumerate(_lineup_seeds(args)):
        output_dir = _output_dir_for_house(args, index=index, seed=seed)
        states = build_residential_upgrade_sequence(
            seed=seed,
            material_theme=args.material_theme,
            levels=tuple(args.levels),
            wall_face_design_path=args.wallface_design,
        )
        manifest = export_residential_upgrade_sequence(states, output_dir)
        print(f"[export] {index + 1}/{args.lineup_count} seed={seed} -> {output_dir}")
        if (package_out := _package_out_for_house(args, index=index, seed=seed)) is not None:
            package_manifest = export_residential_upgrade_package(states, package_out)
            package_size = package_out.stat().st_size
            print(
                f"[package] wrote {package_out} "
                f"({package_size:,} bytes, {len(package_manifest['palette'])} block states)"
            )
        _print_manifest(manifest)
        houses.append(
            HousePayload(
                label=f"house_{index + 1:02d}",
                seed=seed,
                states=states,
                diffs=None,
                manifest=manifest,
            )
        )
    return houses


def main() -> int:
    args = _parse_args()
    houses = _load_or_generate_houses(args)

    if args.live:
        try:
            _run_live(args, houses)
        except Exception as exc:
            print(f"[live] failed: {exc}", file=sys.stderr)
            return 2
    else:
        if args.input_package is not None or args.input_packages is not None:
            print("[package] live placement skipped; rerun with --live to place in Minecraft")
        else:
            print("[export] live placement skipped; rerun with --live to place in Minecraft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
