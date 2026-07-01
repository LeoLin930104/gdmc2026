from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models import Zone
import templates


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """
    Top-level settings for a generated datapack.

    Parameters
    ----------
    namespace       : Minecraft datapack namespace (lowercase, no spaces).
                      All function paths will be  namespace:path/to/func
    pack_description: Text shown in-game on the datapack selection screen.
    pack_format     : Datapack format version.  26 = MC 1.21.x
    output_dir      : Where to write the datapack folder.
    overwrite       : If True, delete and recreate output_dir on each run.
                      If False, raise if output_dir already exists.
    write_templates : Include _template.mcfunction helper files.
    """
    namespace:        str
    pack_description: str  = "Area Discovery — MMO-style zone title system"
    pack_format:      int  = 26
    output_dir:       Path = Path("./area_discovery_datapack")
    overwrite:        bool = True
    write_templates:  bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.namespace != self.namespace.lower():
            raise ValueError(f"namespace must be lowercase, got: {self.namespace!r}")
        if not self.namespace.replace("_", "").isalpha():
            raise ValueError(
                f"namespace must contain only letters and underscores, got: {self.namespace!r}"
            )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DatapackGenerator:
    """
    Generates a complete Minecraft datapack from a list of Zone objects.

    Usage
    -----
    >>> config = GeneratorConfig(namespace="area_discovery")
    >>> gen    = DatapackGenerator(config)
    >>> gen.add_zone(my_zone)
    >>> gen.generate()          # writes to disk
    >>> gen.summary()           # prints a report

    Or build in one shot:
    >>> DatapackGenerator.from_zones(config, [zone1, zone2]).generate()
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._zones: list[Zone] = []

    # -- Zone registration ---------------------------------------------------

    def add_zone(self, zone: Zone) -> "DatapackGenerator":
        if any(z.zone_id == zone.zone_id for z in self._zones):
            raise ValueError(f"Duplicate zone_id: {zone.zone_id!r}")
        self._zones.append(zone)
        return self

    def add_zones(self, zones: list[Zone]) -> "DatapackGenerator":
        for z in zones:
            self.add_zone(z)
        return self

    @classmethod
    def from_zones(
        cls,
        config: GeneratorConfig,
        zones: list[Zone],
    ) -> "DatapackGenerator":
        gen = cls(config)
        gen.add_zones(zones)
        return gen

    # -- Active zone helpers -------------------------------------------------

    @property
    def active_zones(self) -> list[Zone]:
        return [z for z in self._zones if z.enabled]

    @property
    def skipped_zones(self) -> list[Zone]:
        return [z for z in self._zones if not z.enabled]

    # -- Path helpers --------------------------------------------------------

    def _root(self) -> Path:
        return self.config.output_dir

    def _functions(self) -> Path:
        ns = self.config.namespace
        return self._root() / "data" / ns / "function"

    def _tags_functions(self) -> Path:
        # Minecraft only reads tick/load function tags from the `minecraft`
        # namespace, regardless of the datapack's own namespace.
        return self._root() / "data" / "minecraft" / "tags" / "function"

    # -- Low-level file write ------------------------------------------------

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._written.append(path)

    # -- Public generate -----------------------------------------------------

    def generate(self) -> Path:
        self._written: list[Path] = []
        cfg = self.config

        # Optionally clean previous output
        root = self._root()
        if root.exists():
            if cfg.overwrite:
                shutil.rmtree(root)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {root}\n"
                    "Set overwrite=True to replace it."
                )

        active = self.active_zones
        zone_ids = [z.zone_id for z in active]

        # 1 — pack.mcmeta
        self._write(
            root / "pack.mcmeta",
            templates.pack_mcmeta(cfg.pack_description, cfg.pack_format),
        )

        # 2 — tags/functions/tick.json  (registers the tick loop)
        self._write(
            self._tags_functions() / "tick.json",
            templates.tick_tag_json(cfg.namespace),
        )

        # 3 — setup.mcfunction
        self._write(
            self._functions() / "setup.mcfunction",
            templates.setup_function(cfg.namespace),
        )

        # 4 — tick.mcfunction
        self._write(
            self._functions() / "tick.mcfunction",
            templates.tick_function(cfg.namespace, zone_ids),
        )

        # 5 — one zone + one title file per active zone
        for zone in active:
            self._write(
                self._functions() / "zones" / f"{zone.zone_id}.mcfunction",
                templates.zone_function(cfg.namespace, zone),
            )
            self._write(
                self._functions() / "titles" / f"{zone.zone_id}.mcfunction",
                templates.title_function(cfg.namespace, zone),
            )

        # 6 — optional _template helper files
        if cfg.write_templates:
            self._write(
                self._functions() / "zones"  / "_template.mcfunction",
                templates.zone_template(cfg.namespace),
            )
            self._write(
                self._functions() / "titles" / "_template.mcfunction",
                templates.title_template(cfg.namespace),
            )

        return root

    # -- Reporting -----------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "",
            "━" * 60,
            f"  Datapack:  {self.config.namespace}",
            f"  Output:    {self.config.output_dir.resolve()}",
            f"  Zones:     {len(self.active_zones)} active"
            + (f", {len(self.skipped_zones)} skipped" if self.skipped_zones else ""),
            "━" * 60,
        ]

        for zone in self.active_zones:
            bb = zone.aabb
            lines.append(
                f"  ✓  {zone.zone_id:<30}  "
                f"{bb.width}×{bb.height}×{bb.depth} blocks  "
                f"({bb.volume:,} vol)"
            )

        for zone in self.skipped_zones:
            lines.append(f"  ○  {zone.zone_id:<30}  [disabled]")

        if hasattr(self, "_written"):
            lines += [
                "━" * 60,
                f"  Files written: {len(self._written)}",
            ]
            for p in self._written:
                rel = p.relative_to(self.config.output_dir)
                lines.append(f"     {rel}")

        lines.append("━" * 60)
        report = "\n".join(lines)
        print(report)
        return report
