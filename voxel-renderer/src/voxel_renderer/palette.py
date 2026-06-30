from __future__ import annotations

from pathlib import Path

from voxel_renderer.assets import get_asset_root

_EXCLUDED_BLOCKS: frozenset[str] = frozenset(
    [
        "minecraft:air",
        "minecraft:cave_air",
        "minecraft:void_air",
        "minecraft:command_block",
        "minecraft:chain_command_block",
        "minecraft:repeating_command_block",
        "minecraft:structure_block",
        "minecraft:structure_void",
        "minecraft:jigsaw",
        "minecraft:barrier",
        "minecraft:light",
        "minecraft:test_block",
        "minecraft:test_instance_block",
        "minecraft:moving_piston",
        "minecraft:piston_head",
        "minecraft:end_gateway",
        "minecraft:end_portal",
        "minecraft:nether_portal",
        "minecraft:fire",
        "minecraft:soul_fire",
        "minecraft:water",
        "minecraft:lava",
        "minecraft:bubble_column",
        "minecraft:frosted_ice",
        "minecraft:frogspawn",
        "minecraft:wheat",
        "minecraft:potatoes",
        "minecraft:carrots",
        "minecraft:beetroots",
        "minecraft:melon_stem",
        "minecraft:pumpkin_stem",
        "minecraft:attached_melon_stem",
        "minecraft:attached_pumpkin_stem",
        "minecraft:cocoa",
        "minecraft:torchflower_crop",
        "minecraft:pitcher_crop",
        "minecraft:nether_wart",
        "minecraft:sweet_berry_bush",
        "minecraft:cave_vines",
        "minecraft:cave_vines_plant",
        "minecraft:kelp",
        "minecraft:kelp_plant",
        "minecraft:twisting_vines",
        "minecraft:twisting_vines_plant",
        "minecraft:weeping_vines",
        "minecraft:weeping_vines_plant",
        "minecraft:redstone_wire",
        "minecraft:comparator",
        "minecraft:repeater",
        "minecraft:tripwire",
        "minecraft:suspicious_sand",
        "minecraft:suspicious_gravel",
        "minecraft:powder_snow",
        "minecraft:infested_stone",
        "minecraft:infested_cobblestone",
        "minecraft:infested_stone_bricks",
        "minecraft:infested_mossy_stone_bricks",
        "minecraft:infested_cracked_stone_bricks",
        "minecraft:infested_chiseled_stone_bricks",
        "minecraft:infested_deepslate",
    ]
)


def build_minecraft_palette(asset_root: str | Path | None = None) -> frozenset[str]:
    root = Path(asset_root) if asset_root is not None else get_asset_root()
    blockstates_dir = root / "blockstates"
    if not blockstates_dir.is_dir():
        return frozenset({"minecraft:air"})

    blocks = {"minecraft:air"}
    for path in blockstates_dir.glob("*.json"):
        block_id = f"minecraft:{path.stem}"
        if block_id not in _EXCLUDED_BLOCKS:
            blocks.add(block_id)

    return frozenset(blocks)


DEFAULT_MINECRAFT_PALETTE = build_minecraft_palette()


def is_minecraft_placeable(block_id: str, palette: frozenset[str] | None = None) -> bool:
    active_palette = DEFAULT_MINECRAFT_PALETTE if palette is None else palette
    return block_id != "minecraft:air" and block_id in active_palette
