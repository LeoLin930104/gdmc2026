from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _require_nbtlib():
    """Lazy-import nbtlib so this module (and its dataclasses) import without it.

    Only parsing actually needs nbtlib; keeping the import lazy lets the rest of
    the placement code — and the pure geometry helpers in premade_placer — load
    in environments where nbtlib isn't installed.
    """
    try:
        import nbtlib
        from nbtlib import serialize_tag
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "nbtlib is required to parse .nbt structures. It is declared in "
            "the uv-managed project dependencies; run:\n"
            "    uv sync --all-packages\n"
            "then launch through:\n"
            "    uv run python run_settlement.py"
        ) from exc
    return nbtlib, serialize_tag

# Block-entity keys that are position/identity bookkeeping, not data gdpc needs
# in a `data=` payload — stripped so the SNBT is a clean component/data blob.
_BE_DROP_KEYS = ("id", "x", "y", "z", "keepPacked")


@dataclass
class StructureBlock:
    """One placed block from a structure: where, what, and its state/data."""

    pos: tuple[int, int, int]            # structure-local (x, y, z)
    name: str                            # namespaced id, e.g. "minecraft:oak_stairs"
    properties: dict[str, str] = field(default_factory=dict)  # block-state props
    nbt: str | None = None               # block-entity SNBT for gdpc data=, or None


@dataclass
class Structure:
    """A parsed structure: its extent plus every (non-implicit) block."""

    size: tuple[int, int, int]
    blocks: list[StructureBlock]
    name: str = ""                       # source file stem, e.g. "barrack2_7" (for logs/variant id)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def distinct_names(self) -> list[str]:
        return sorted({b.name for b in self.blocks})


def _serialize_block_entity(nbt_tag) -> str | None:
    """Serialize a block-entity compound to SNBT for gdpc's `data=`, or None.

    Strips positional/identity keys (id/x/y/z) that the structure carries but
    that don't belong in a place-time data payload. Returns None if nothing
    meaningful is left, or if serialization fails (warn-and-recover — a missing
    data blob is never fatal; the block still places without it).
    """
    try:
        nbtlib, serialize_tag = _require_nbtlib()
        compound = nbtlib.Compound(nbt_tag)
        for key in _BE_DROP_KEYS:
            compound.pop(key, None)
        if not compound:
            return None
        return serialize_tag(compound, compact=True)
    except Exception as exc:  # noqa: BLE001 - defensive: data is optional
        print(f"[warn] could not serialize block-entity nbt ({exc}); placing without data.")
        return None


def _resolve_palette(root):
    """Return the active palette list, handling single + variant layouts."""
    if "palette" in root:
        return root["palette"]
    if "palettes" in root and len(root["palettes"]) > 0:
        return root["palettes"][0]
    return []


def parse_structure(path: str | Path) -> Structure:
    """Load a structure `.nbt` from disk into a `Structure`.

    Raises FileNotFoundError if the path is missing; ValueError if the file
    has no `blocks`/`palette` (i.e. isn't a structure-block export).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such .nbt file: {path}")

    nbtlib, _ = _require_nbtlib()
    root = nbtlib.load(str(path))
    palette = _resolve_palette(root)
    raw_blocks = root.get("blocks", [])
    if not palette or not raw_blocks:
        raise ValueError(
            f"{path.name} has no palette/blocks — is it a structure-block .nbt?"
        )

    size_raw = root.get("size", [0, 0, 0])
    size = (int(size_raw[0]), int(size_raw[1]), int(size_raw[2]))

    blocks: list[StructureBlock] = []
    for b in raw_blocks:
        entry = palette[int(b["state"])]
        name = str(entry["Name"])
        props_tag = entry.get("Properties")
        properties = (
            {str(k): str(v) for k, v in props_tag.items()} if props_tag else {}
        )
        pos_raw = b["pos"]
        pos = (int(pos_raw[0]), int(pos_raw[1]), int(pos_raw[2]))
        nbt = _serialize_block_entity(b["nbt"]) if "nbt" in b else None
        blocks.append(StructureBlock(pos=pos, name=name, properties=properties, nbt=nbt))

    return Structure(size=size, blocks=blocks, name=path.stem)


if __name__ == "__main__":
    import sys

    here = Path(__file__).resolve().parent
    arg = sys.argv[1] if len(sys.argv) > 1 else "nbt"
    target = Path(arg)
    if not target.is_absolute():
        target = here / target
    files = sorted(target.glob("*.nbt")) if target.is_dir() else [target]

    for f in files:
        s = parse_structure(f)
        print(f"{f.name}: size={s.size}, {s.block_count} blocks, "
              f"{len(s.distinct_names())} distinct, "
              f"{sum(1 for b in s.blocks if b.nbt)} with block-entity data")
