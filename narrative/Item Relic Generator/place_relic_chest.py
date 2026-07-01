from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from gdpc import Editor, Block

# Ensure the terminal handles Unicode on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELICS_FILE = Path(__file__).parent / "relics.json"
CHEST_POS = (0, -60, 0)
MAX_RELICS = 27
REQUIRED_FIELDS = {"name", "item_type"}

# Minecraft does NOT auto-wrap item lore lines — a long single line runs off
# the tooltip / screen edge. We word-wrap the gray description into multiple
# lore lines so it stays on-screen. ~38 chars keeps a "<15 word" description to
# ~2 lines while staying comfortably inside the tooltip width. Tune if needed.
DESCRIPTION_WRAP_WIDTH = 38


# ---------------------------------------------------------------------------
# JSON / SNBT helpers
# ---------------------------------------------------------------------------

def wrap_lore_text(text: str, width: int = DESCRIPTION_WRAP_WIDTH) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)


def snbt_string(json_dict: dict) -> str:
    raw_json = json.dumps(json_dict, ensure_ascii=False, separators=(",", ":"))
    escaped = raw_json.replace("'", "\\'")
    return f"'{escaped}'"


# ---------------------------------------------------------------------------
# Relic loading and validation
# ---------------------------------------------------------------------------

def validate_relic(index: int, raw) -> "dict | None":
    if not isinstance(raw, dict):
        print(f"[warn] Entry {index} is not an object — skipping.")
        return None

    for field in REQUIRED_FIELDS:
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            print(f"[warn] Entry {index} missing or empty '{field}' — skipping.")
            return None

    item_type = raw["item_type"].strip()
    if ":" not in item_type:
        print(
            f"[warn] Entry {index} item_type '{item_type}' has no namespace "
            f"(expected 'minecraft:...') — skipping."
        )
        return None

    return {
        "name":        raw["name"].strip(),
        "item_type":   item_type,
        "description": str(raw.get("description", "")).strip(),
        "lore":        str(raw.get("lore", "")).strip(),
        "color":       str(raw.get("color", "white")).strip(),
    }


def _finalize(raw_list: list, source_label: str) -> list:
    if not isinstance(raw_list, list):
        sys.exit(f"[error] Expected a list of relics from {source_label}.")

    valid = []
    for i, entry in enumerate(raw_list):
        relic = validate_relic(i, entry)
        if relic is not None:
            valid.append(relic)

    if not valid:
        sys.exit(f"[error] No valid relics found in {source_label}.")

    if len(valid) > MAX_RELICS:
        print(f"[warn] {len(valid)} relics loaded; truncating to {MAX_RELICS} (chest limit).")
        valid = valid[:MAX_RELICS]

    return valid


def load_relics(path: Path) -> list:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"[error] Could not find '{path}'. Make sure relics.json is in the same folder.")
    except json.JSONDecodeError as exc:
        sys.exit(f"[error] relics.json contains invalid JSON: {exc}")

    return _finalize(data.get("relics", []), source_label="relics.json")


def _ensure_llm_on_path() -> None:
    llm_path = Path(__file__).parent.parent / "LLM Narrative"
    if str(llm_path) not in sys.path:
        sys.path.insert(0, str(llm_path))


def load_relics_from_llm(theme: str, count: int, settlement=None, biome: str | None = None) -> list:
    _ensure_llm_on_path()
    from relic_generator import generate_relics  # lazy import — only needed on --llm

    raw = generate_relics(theme, count=count, settlement=settlement, biome=biome)
    return _finalize(raw, source_label=f'LLM theme "{theme}"')


def generate_settlement_from_theme(settlement_theme: str, biome: str | None = None):
    _ensure_llm_on_path()
    from settlement_generator import generate_settlement
    return generate_settlement(settlement_theme, biome=biome)


def sample_biome_at(editor, pos) -> str:
    _ensure_llm_on_path()
    from biome_context import sample_biome
    return sample_biome(editor, pos)


# ---------------------------------------------------------------------------
# NBT construction
# ---------------------------------------------------------------------------

def build_item_nbt(relic: dict, slot: int, glint: bool = True) -> str:
    name_snbt = snbt_string({
        "text":   relic["name"],
        "color":  relic["color"],
        "italic": False,        # Override Minecraft's default italic on display names
    })

    lore_parts = []
    if relic["description"]:
        # Word-wrap into multiple gray lore lines so a long description doesn't
        # run off the tooltip; each wrapped line is its own lore component.
        for line in wrap_lore_text(relic["description"]):
            lore_parts.append(snbt_string({
                "text":   line,
                "color":  "gray",
                "italic": False,
            }))
    if relic["lore"]:
        # Same word-wrap as the description so the italic story doesn't run off
        # the tooltip; each wrapped line is its own italic lore component.
        for line in wrap_lore_text(relic["lore"]):
            lore_parts.append(snbt_string({
                "text":   line,
                "italic": True,
            }))

    lore_snbt = "[" + ",".join(lore_parts) + "]"

    glint_part = ',"minecraft:enchantment_glint_override":true' if glint else ""

    # Minecraft 1.20.5+ replaced tag.display with components.
    # Keys containing ":" must be quoted in SNBT.
    return (
        f'{{Slot:{slot}b,'
        f'id:"{relic["item_type"]}",'
        f'count:1,'
        f'components:{{"minecraft:custom_name":{name_snbt},'
        f'"minecraft:lore":{lore_snbt}{glint_part}}}}}'
    )


def build_chest_snbt(relics: list, glint: bool = True) -> str:
    items_nbt = [build_item_nbt(relic, slot, glint=glint) for slot, relic in enumerate(relics)]
    return "{Items:[" + ",".join(items_nbt) + "]}"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def place_chest(editor: Editor, pos: tuple, snbt: str) -> None:
    block = Block("minecraft:chest", data=snbt)
    editor.placeBlock(pos, block)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    llm_theme: str | None = None,
    count: int = 3,
    settlement_theme: str | None = None,
    biome_override: str | None = None,
) -> None:
    editor = Editor(buffering=True)

    biome: str | None = None
    if llm_theme or settlement_theme:
        if biome_override:
            biome = biome_override
            print(f'[info] Biome (override): "{biome}".')
        else:
            biome = sample_biome_at(editor, CHEST_POS)
            print(f'[info] Biome (auto-sampled at {CHEST_POS}): "{biome}".')

    settlement = None
    if settlement_theme:
        settlement = generate_settlement_from_theme(settlement_theme, biome=biome)
        print(
            f'[info] Settlement identity: "{settlement.name}" '
            f"({settlement.era}) — theme: \"{settlement_theme}\"."
        )

    if llm_theme:
        relics = load_relics_from_llm(llm_theme, count, settlement=settlement, biome=biome)
        print(f'[info] Generated {len(relics)} relic(s) from LLM theme: "{llm_theme}".')
    else:
        if settlement is not None:
            print("[warn] --settlement-theme has no effect without --llm; loading relics.json.")
        relics = load_relics(RELICS_FILE)
        print(f"[info] Loaded {len(relics)} relic(s) from {RELICS_FILE.name}.")

    chest_snbt = build_chest_snbt(relics)
    place_chest(editor, CHEST_POS, chest_snbt)
    editor.flushBuffer()

    x, y, z = CHEST_POS
    print(f"[info] Chest placed at ({x}, {y}, {z}) with {len(relics)} item(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Place a chest of relics from JSON or LLM.")
    parser.add_argument(
        "--llm",
        metavar="THEME",
        help='Generate relics from LLM instead of relics.json (e.g. --llm "haunted coast")',
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of relics to generate when using --llm (default: 3, max 27).",
    )
    parser.add_argument(
        "--settlement-theme",
        metavar="THEME",
        dest="settlement_theme",
        help=(
            "Generate a Settlement identity from THEME first, then thread it "
            "into --llm relic generation so lore coheres "
            '(e.g. --settlement-theme "haunted coastal village").'
        ),
    )
    parser.add_argument(
        "--biome",
        metavar="ID",
        dest="biome",
        help=(
            "Override the auto-sampled biome "
            '(e.g. --biome "minecraft:dark_forest"). '
            "When --llm or --settlement-theme is used without --biome, the "
            "biome at CHEST_POS is sampled from GDPC and threaded into all LLM prompts."
        ),
    )
    args = parser.parse_args()
    main(
        llm_theme=args.llm,
        count=args.count,
        settlement_theme=args.settlement_theme,
        biome_override=args.biome,
    )
