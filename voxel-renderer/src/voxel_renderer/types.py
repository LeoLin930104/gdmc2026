from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

CoordinateKey: TypeAlias = tuple[int, int, int]
BlockProperties: TypeAlias = dict[str, str]
SemanticBlockDict: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BlockEntry:
    x: int
    y: int
    z: int
    id: str
    properties: BlockProperties = field(default_factory=dict)

    def to_semantic_dict(self) -> SemanticBlockDict:
        payload: SemanticBlockDict = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "id": self.id,
        }
        if self.properties:
            payload["properties"] = dict(self.properties)
        return payload

    @classmethod
    def from_semantic_dict(cls, raw: SemanticBlockDict) -> "BlockEntry":
        return cls(
            x=int(raw["x"]),
            y=int(raw["y"]),
            z=int(raw["z"]),
            id=str(raw["id"]),
            properties=dict(raw.get("properties") or {}),
        )


@dataclass(frozen=True, slots=True)
class VoxelStoreConfig:
    enforce_palette: bool = False
