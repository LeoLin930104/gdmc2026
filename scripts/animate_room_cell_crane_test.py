"""Experimental GDPC test for moving one hollow box cell like a crane.

This script is deliberately not part of the production generation pipeline. It
builds one empty concrete box cell, lifts it, moves it sideways while suspended,
then drops it at the target location. The live path replaces blocks at each
frame, so run it only in a disposable test area.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

Vector3 = tuple[int, int, int]
HorizontalOffset = tuple[int, int]
BlockState = tuple[str, tuple[tuple[str, str], ...]]
WorldGrid = dict[Vector3, BlockState]

_AIR_BLOCK_ID = "minecraft:air"
_DEFAULT_CELL_SIZE: Vector3 = (8, 6, 8)


@dataclass(frozen=True, slots=True)
class LocalBlock:
    x: int
    y: int
    z: int
    block_id: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental test-only GDPC crane animation for one hollow box.",
    )
    parser.add_argument("--host", default="http://localhost:9000")
    parser.add_argument(
        "--origin",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World-space lower north-west cell origin. Defaults to player-relative.",
    )
    parser.add_argument(
        "--player-distance",
        type=int,
        default=8,
        help="Distance in front of the player when --origin is omitted.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=_DEFAULT_CELL_SIZE,
    )
    parser.add_argument("--block-id", default="minecraft:white_concrete")
    parser.add_argument(
        "--lift-blocks",
        "--move-up-blocks",
        dest="lift_blocks",
        type=int,
        default=8,
        help="Vertical lift distance before the horizontal crane move.",
    )
    parser.add_argument(
        "--side-offset",
        type=int,
        nargs=2,
        metavar=("DX", "DZ"),
        default=(12, 0),
        help="Horizontal target offset after lifting.",
    )
    parser.add_argument(
        "--step-blocks",
        type=int,
        default=1,
        help="Maximum block distance advanced per animation frame.",
    )
    parser.add_argument(
        "--motion-interval-s",
        "--frame-interval-s",
        dest="motion_interval_s",
        type=float,
        default=0.15,
        help="Seconds to wait between motion frames.",
    )
    parser.add_argument(
        "--clear-swept",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clear the full swept bounding box before starting.",
    )
    parser.add_argument("--flush-every", type=int, default=128)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated path and exit without importing GDPC.",
    )
    args = parser.parse_args()
    if args.player_distance < 1:
        parser.error("--player-distance must be >= 1")
    if any(value < 1 for value in args.cell_size):
        parser.error("--cell-size values must be >= 1")
    if args.lift_blocks < 0:
        parser.error("--lift-blocks must be >= 0")
    if args.side_offset == [0, 0]:
        parser.error("--side-offset must move at least one horizontal axis")
    if args.step_blocks < 1:
        parser.error("--step-blocks must be >= 1")
    if args.motion_interval_s < 0:
        parser.error("--motion-interval-s must be >= 0")
    if args.flush_every < 1:
        parser.error("--flush-every must be >= 1")
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
            "Live placement requires gdpc and a running GDMC-HTTP Minecraft server."
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


def _resolve_player_relative_origin(
    *,
    host: str,
    cell_size: Vector3,
    distance: int,
) -> Vector3:
    player_x, player_y, player_z, yaw = _get_player_pose(host)
    forward_x, forward_z = _yaw_to_forward(yaw)
    vx, _vy, vz = cell_size

    if forward_x != 0:
        origin_x = player_x + forward_x * distance
        origin_z = player_z - vz // 2
        if forward_x < 0:
            origin_x -= vx - 1
    else:
        origin_x = player_x - vx // 2
        origin_z = player_z + forward_z * distance
        if forward_z < 0:
            origin_z -= vz - 1

    origin = (origin_x, player_y, origin_z)
    print(
        f"[origin] player=({player_x}, {player_y}, {player_z}) yaw={yaw:.1f} "
        f"-> origin={origin}"
    )
    return origin


def _build_empty_box_cell(*, cell_size: Vector3, block_id: str) -> list[LocalBlock]:
    vx, vy, vz = cell_size
    blocks: list[LocalBlock] = []
    for x in range(vx):
        for y in range(vy):
            for z in range(vz):
                if x in (0, vx - 1) or y in (0, vy - 1) or z in (0, vz - 1):
                    blocks.append(LocalBlock(x=x, y=y, z=z, block_id=block_id))
    return blocks


def _stepped_values(start: int, end: int, step: int) -> list[int]:
    if start == end:
        return []
    direction = 1 if end > start else -1
    out: list[int] = []
    current = start
    while current != end:
        current += direction * step
        if direction > 0:
            current = min(current, end)
        else:
            current = max(current, end)
        out.append(current)
    return out


def _dedupe_frames(frames: list[Vector3]) -> list[Vector3]:
    out: list[Vector3] = []
    for frame in frames:
        if not out or out[-1] != frame:
            out.append(frame)
    return out


def build_crane_path(
    *,
    lift_blocks: int,
    side_offset: HorizontalOffset,
    step_blocks: int,
) -> list[Vector3]:
    dx, dz = side_offset
    frames: list[Vector3] = [(0, 0, 0)]

    for y in _stepped_values(0, lift_blocks, step_blocks):
        frames.append((0, y, 0))

    horizontal_steps = max(
        1,
        math.ceil(max(abs(dx), abs(dz)) / step_blocks),
    )
    for index in range(1, horizontal_steps + 1):
        x = round(dx * index / horizontal_steps)
        z = round(dz * index / horizontal_steps)
        frames.append((x, lift_blocks, z))

    for y in _stepped_values(lift_blocks, 0, step_blocks):
        frames.append((dx, y, dz))

    return _dedupe_frames(frames)


def _translated_grid(
    blocks: list[LocalBlock],
    origin: Vector3,
    frame_offset: Vector3,
) -> WorldGrid:
    ox, oy, oz = origin
    fx, fy, fz = frame_offset
    return {
        (ox + block.x + fx, oy + block.y + fy, oz + block.z + fz): (
            block.block_id,
            (),
        )
        for block in blocks
    }


def _diff_grids(before: WorldGrid, after: WorldGrid) -> list[tuple[Vector3, BlockState]]:
    updates: list[tuple[Vector3, BlockState]] = []
    for pos, block in before.items():
        if pos not in after:
            updates.append((pos, (_AIR_BLOCK_ID, ())))
        elif after[pos] != block:
            updates.append((pos, after[pos]))
    for pos, block in after.items():
        if pos not in before:
            updates.append((pos, block))
    return sorted(updates, key=lambda item: item[0])


def _swept_bbox(
    blocks: list[LocalBlock],
    origin: Vector3,
    frames: list[Vector3],
) -> tuple[Vector3, Vector3]:
    positions = [
        pos
        for frame in frames
        for pos in _translated_grid(blocks, origin, frame)
    ]
    xs, ys, zs = zip(*positions, strict=False)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _place_updates(
    *,
    editor: Any,
    block_cls: Any,
    updates: list[tuple[Vector3, BlockState]],
    flush_every: int,
) -> None:
    for index, (pos, block_state) in enumerate(updates, start=1):
        block_id, props_tuple = block_state
        editor.placeBlock(pos, block_cls(block_id, dict(props_tuple)))
        if index % flush_every == 0:
            editor.flushBuffer()
    editor.flushBuffer()


def _clear_bbox(
    *,
    editor: Any,
    block_cls: Any,
    bbox: tuple[Vector3, Vector3],
    flush_every: int,
) -> None:
    (x0, y0, z0), (x1, y1, z1) = bbox
    updates = [
        ((x, y, z), (_AIR_BLOCK_ID, ()))
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
        for z in range(z0, z1 + 1)
    ]
    _place_updates(
        editor=editor,
        block_cls=block_cls,
        updates=updates,
        flush_every=flush_every,
    )
    print(f"[clear] swept bbox {bbox[0]} -> {bbox[1]} ({len(updates)} blocks)")


def _animate(
    *,
    editor: Any,
    block_cls: Any,
    blocks: list[LocalBlock],
    origin: Vector3,
    frames: list[Vector3],
    interval_s: float,
    flush_every: int,
) -> None:
    previous: WorldGrid = {}
    for index, frame in enumerate(frames, start=1):
        current = _translated_grid(blocks, origin, frame)
        updates = _diff_grids(previous, current)
        _place_updates(
            editor=editor,
            block_cls=block_cls,
            updates=updates,
            flush_every=flush_every,
        )
        print(f"[frame {index}/{len(frames)}] offset={frame} updates={len(updates)}")
        previous = current
        if interval_s > 0 and index < len(frames):
            time.sleep(interval_s)


def main() -> int:
    args = _parse_args()
    cell_size: Vector3 = tuple(args.cell_size)
    side_offset: HorizontalOffset = tuple(args.side_offset)
    blocks = _build_empty_box_cell(
        cell_size=cell_size,
        block_id=args.block_id,
    )
    frames = build_crane_path(
        lift_blocks=args.lift_blocks,
        side_offset=side_offset,
        step_blocks=args.step_blocks,
    )

    if args.origin is None:
        origin: Vector3 | None = None
    else:
        origin = tuple(int(value) for value in args.origin)

    if args.dry_run:
        dry_origin = origin or (0, 0, 0)
        bbox = _swept_bbox(blocks, dry_origin, frames)
        print("[dry-run] hollow-box crane test")
        print(f"  cell_size={cell_size} blocks={len(blocks)}")
        print(f"  frames={len(frames)} interval_s={args.motion_interval_s}")
        print(f"  path_start={frames[0]} path_end={frames[-1]} swept_bbox={bbox}")
        return 0

    Editor, Block = _import_gdpc()
    editor = Editor(buffering=True, host=_normalise_host(args.host))
    editor.doBlockUpdates = False
    editor.spawnDrops = False

    if origin is None:
        origin = _resolve_player_relative_origin(
            host=args.host,
            cell_size=cell_size,
            distance=args.player_distance,
        )
    else:
        print(f"[origin] explicit origin={origin}")

    bbox = _swept_bbox(blocks, origin, frames)
    if args.clear_swept:
        _clear_bbox(
            editor=editor,
            block_cls=Block,
            bbox=bbox,
            flush_every=args.flush_every,
        )

    print(
        "[animate] starting test-only crane move: "
        f"frames={len(frames)} blocks={len(blocks)} swept_bbox={bbox}"
    )
    _animate(
        editor=editor,
        block_cls=Block,
        blocks=blocks,
        origin=origin,
        frames=frames,
        interval_s=args.motion_interval_s,
        flush_every=args.flush_every,
    )
    print(f"[animate] installed at offset={frames[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
