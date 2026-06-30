"""Compact binary packages for blueprint block payloads.

The format follows the same storage idea as the SunkenCityProject ``EROS``
world dumps: a small magic header, compressed binary payload records, and a
JSON table at the end containing the palette and manifest.  This variant stays
sparse because prefab upgrade payloads are small block-update lists rather than
dense 16xH x16 world chunks.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

BlueprintBlock = dict[str, Any]

FORMAT_NAME = "prefab-housing-blueprint-package-v1"
MAGIC = b"PBH1"
_HEADER_STRUCT = struct.Struct("<Q")
_SECTION_HEADER_STRUCT = struct.Struct("<HII")
_BLOCK_COUNT_STRUCT = struct.Struct("<I")
_BLOCK_STRUCT = struct.Struct("<hhhH")
_INT16_MIN = -(2**15)
_INT16_MAX = 2**15 - 1
_UINT16_MAX = 2**16 - 1


def _normalise_props(block: Mapping[str, Any]) -> dict[str, str]:
    props = block.get("props", {})
    if not isinstance(props, Mapping):
        return {}
    return {str(key): str(value) for key, value in sorted(props.items())}


def _palette_key(block: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    props = _normalise_props(block)
    return str(block["id"]), tuple(props.items())


def _palette_entry(key: tuple[str, tuple[tuple[str, str], ...]]) -> dict[str, Any]:
    block_id, props = key
    entry: dict[str, Any] = {"id": block_id}
    if props:
        entry["props"] = dict(props)
    return entry


def _build_palette(
    sections: Mapping[str, list[BlueprintBlock]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, tuple[tuple[str, str], ...]], int]]:
    keys = sorted(
        {_palette_key(block) for blocks in sections.values() for block in blocks}
    )
    if len(keys) > _UINT16_MAX + 1:
        raise ValueError(f"blueprint package palette is too large: {len(keys)} states")
    return (
        [_palette_entry(key) for key in keys],
        {key: index for index, key in enumerate(keys)},
    )


def _pack_coordinate(value: Any, *, axis: str) -> int:
    integer = int(value)
    if integer < _INT16_MIN or integer > _INT16_MAX:
        raise ValueError(f"blueprint {axis} coordinate out of int16 range: {integer}")
    return integer


def _pack_blocks(
    blocks: list[BlueprintBlock],
    palette_index: Mapping[tuple[str, tuple[tuple[str, str], ...]], int],
) -> bytes:
    payload = bytearray()
    payload.extend(_BLOCK_COUNT_STRUCT.pack(len(blocks)))
    for block in blocks:
        index = palette_index[_palette_key(block)]
        payload.extend(
            _BLOCK_STRUCT.pack(
                _pack_coordinate(block["dx"], axis="dx"),
                _pack_coordinate(block["dy"], axis="dy"),
                _pack_coordinate(block["dz"], axis="dz"),
                index,
            )
        )
    return bytes(payload)


def _unpack_blocks(raw: bytes, palette: list[dict[str, Any]]) -> list[BlueprintBlock]:
    if len(raw) < _BLOCK_COUNT_STRUCT.size:
        raise ValueError("compressed blueprint section is missing its block count")
    block_count = _BLOCK_COUNT_STRUCT.unpack_from(raw, 0)[0]
    expected_size = _BLOCK_COUNT_STRUCT.size + block_count * _BLOCK_STRUCT.size
    if len(raw) != expected_size:
        raise ValueError(
            f"compressed blueprint section has {len(raw)} bytes, expected {expected_size}"
        )

    blocks: list[BlueprintBlock] = []
    offset = _BLOCK_COUNT_STRUCT.size
    for _ in range(block_count):
        dx, dy, dz, palette_id = _BLOCK_STRUCT.unpack_from(raw, offset)
        offset += _BLOCK_STRUCT.size
        try:
            state = palette[palette_id]
        except IndexError as exc:
            raise ValueError(f"blueprint section references palette index {palette_id}") from exc
        block: BlueprintBlock = {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "id": state["id"],
        }
        props = state.get("props")
        if isinstance(props, Mapping) and props:
            block["props"] = {str(key): str(value) for key, value in props.items()}
        blocks.append(block)
    return blocks


def write_blueprint_package(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    sections: Mapping[str, list[BlueprintBlock]],
) -> dict[str, Any]:
    """Write compressed blueprint sections and return the stored metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    palette, palette_index = _build_palette(sections)
    stored_metadata = dict(metadata)
    stored_metadata["format"] = FORMAT_NAME
    stored_metadata["palette"] = palette

    section_records: list[dict[str, Any]] = []
    temp_path = target.with_name(f".{target.name}.tmp")
    with temp_path.open("wb") as file:
        file.write(MAGIC)
        file.write(_HEADER_STRUCT.pack(0))
        for name, blocks in sections.items():
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > _UINT16_MAX:
                raise ValueError(f"blueprint section name is too long: {name}")
            raw = _pack_blocks(blocks, palette_index)
            compressed = zlib.compress(raw)
            file.write(
                _SECTION_HEADER_STRUCT.pack(len(name_bytes), len(raw), len(compressed))
            )
            file.write(name_bytes)
            file.write(compressed)
            section_records.append(
                {
                    "name": name,
                    "block_count": len(blocks),
                    "raw_size": len(raw),
                    "compressed_size": len(compressed),
                }
            )

        metadata_ptr = file.tell()
        stored_metadata["sections"] = section_records
        file.write(json.dumps(stored_metadata, separators=(",", ":")).encode("utf-8"))
        file.seek(len(MAGIC))
        file.write(_HEADER_STRUCT.pack(metadata_ptr))

    os.replace(temp_path, target)
    return stored_metadata


def read_blueprint_package(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, list[BlueprintBlock]]]:
    """Read a compact blueprint package into metadata and named block sections."""
    source = Path(path)
    compressed_sections: dict[str, bytes] = {}
    with source.open("rb") as file:
        if file.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"invalid blueprint package magic: {source}")
        metadata_ptr = _HEADER_STRUCT.unpack(file.read(_HEADER_STRUCT.size))[0]
        if metadata_ptr <= len(MAGIC) + _HEADER_STRUCT.size:
            raise ValueError(f"invalid blueprint package metadata pointer: {source}")

        while file.tell() < metadata_ptr:
            header = file.read(_SECTION_HEADER_STRUCT.size)
            if len(header) != _SECTION_HEADER_STRUCT.size:
                raise ValueError(f"truncated blueprint section header: {source}")
            name_size, _raw_size, compressed_size = _SECTION_HEADER_STRUCT.unpack(header)
            name = file.read(name_size).decode("utf-8")
            compressed_sections[name] = file.read(compressed_size)

        metadata = json.loads(file.read().decode("utf-8"))

    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported blueprint package format: {metadata.get('format')!r}")
    palette = metadata.get("palette")
    if not isinstance(palette, list):
        raise ValueError("blueprint package metadata is missing a palette list")

    sections: dict[str, list[BlueprintBlock]] = {}
    for name, compressed in compressed_sections.items():
        sections[name] = _unpack_blocks(zlib.decompress(compressed), palette)
    return metadata, sections


__all__ = [
    "FORMAT_NAME",
    "read_blueprint_package",
    "write_blueprint_package",
]
