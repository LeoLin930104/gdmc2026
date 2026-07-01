from __future__ import annotations

from collections.abc import Callable, Iterable

from voxel_renderer.types import BlockEntry, CoordinateKey, SemanticBlockDict, VoxelStoreConfig


def canonicalise_block_array(
    block_array: Iterable[SemanticBlockDict],
) -> tuple[list[SemanticBlockDict], int]:
    canonical: dict[CoordinateKey, SemanticBlockDict] = {}
    duplicate_count = 0

    for raw in block_array:
        entry = BlockEntry.from_semantic_dict(raw)
        key = (entry.x, entry.y, entry.z)
        if key in canonical:
            duplicate_count += 1
        canonical[key] = entry.to_semantic_dict()

    return list(canonical.values()), duplicate_count


class VoxelStore:
    def __init__(
        self,
        *,
        config: VoxelStoreConfig | None = None,
        block_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config or VoxelStoreConfig()
        self._block_validator = block_validator
        self._blocks: dict[CoordinateKey, BlockEntry] = {}

    def commit(self, block_array: Iterable[SemanticBlockDict]) -> int:
        entries = [BlockEntry.from_semantic_dict(raw) for raw in block_array]

        if self._config.enforce_palette and self._block_validator is not None:
            rejected = sorted({entry.id for entry in entries if not self._block_validator(entry.id)})
            if rejected:
                raise ValueError(f"Block IDs not in allowed palette: {rejected}")

        for entry in entries:
            key = (entry.x, entry.y, entry.z)
            if entry.id == "minecraft:air":
                self._blocks.pop(key, None)
            else:
                self._blocks[key] = entry

        return len(entries)

    def get_all(self) -> list[SemanticBlockDict]:
        return [entry.to_semantic_dict() for entry in self._blocks.values()]

    def get_bounding_box(self) -> tuple[int, int, int, int, int, int] | None:
        if not self._blocks:
            return None

        xs, ys, zs = zip(*self._blocks.keys(), strict=False)
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def get_count(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()

    def reset(self, block_array: Iterable[SemanticBlockDict] | None = None) -> int:
        self.clear()
        if block_array is None:
            return 0
        return self.commit(block_array)
