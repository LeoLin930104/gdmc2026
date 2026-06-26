from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MCColor(str, Enum):
    """All valid Minecraft JSON text colors."""
    BLACK        = "black"
    DARK_BLUE    = "dark_blue"
    DARK_GREEN   = "dark_green"
    DARK_AQUA    = "dark_aqua"
    DARK_RED     = "dark_red"
    DARK_PURPLE  = "dark_purple"
    GOLD         = "gold"
    GRAY         = "gray"
    DARK_GRAY    = "dark_gray"
    BLUE         = "blue"
    GREEN        = "green"
    AQUA         = "aqua"
    RED          = "red"
    LIGHT_PURPLE = "light_purple"
    YELLOW       = "yellow"
    WHITE        = "white"


class SoundSource(str, Enum):
    """Minecraft sound source categories."""
    MASTER  = "master"
    MUSIC   = "music"
    RECORD  = "record"
    WEATHER = "weather"
    BLOCK   = "block"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    PLAYER  = "player"
    AMBIENT = "ambient"
    VOICE   = "voice"


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

@dataclass
class AABB:
    """
    Axis-Aligned Bounding Box for a zone.

    x, y, z   — the minimum corner (any integer coordinate)
    dx, dy, dz — extent in each axis (width - 1, height - 1, depth - 1)

    Minecraft's execute selector uses dx/dy/dz as *offsets* from the origin
    corner, so a 10-block wide space needs dx=9, not dx=10.

    Example
    -------
    >>> AABB(x=-50, y=60, z=-50, dx=99, dy=39, dz=99)
    # Covers a 100×40×100 block volume
    """
    x:  int
    y:  int
    z:  int
    dx: int   # width  - 1
    dy: int   # height - 1
    dz: int   # depth  - 1

    @classmethod
    def from_corners(
        cls,
        x1: int, y1: int, z1: int,
        x2: int, y2: int, z2: int,
    ) -> "AABB":
        """
        Build an AABB from two opposite corners (order doesn't matter).
        This is often more natural when working with GDMC world coordinates.

        Example
        -------
        >>> AABB.from_corners(0, 60, 0, 99, 99, 99)
        AABB(x=0, y=60, z=0, dx=99, dy=39, dz=99)
        """
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        min_z, max_z = min(z1, z2), max(z1, z2)
        return cls(
            x=min_x,  y=min_y,  z=min_z,
            dx=max_x - min_x,
            dy=max_y - min_y,
            dz=max_z - min_z,
        )

    @classmethod
    def from_center(
        cls,
        cx: int, cy: int, cz: int,
        half_x: int, half_y: int, half_z: int,
    ) -> "AABB":
        """
        Build an AABB from a center point and half-extents.
        Useful when a GDMC algorithm reports centroids rather than corners.

        Example
        -------
        >>> AABB.from_center(0, 70, 0, 50, 20, 50)
        # 100×40×100 block volume centered on (0, 70, 0)
        """
        return cls.from_corners(
            cx - half_x, cy - half_y, cz - half_z,
            cx + half_x, cy + half_y, cz + half_z,
        )

    @property
    def width(self)  -> int: return self.dx + 1
    @property
    def height(self) -> int: return self.dy + 1
    @property
    def depth(self)  -> int: return self.dz + 1
    @property
    def volume(self) -> int: return self.width * self.height * self.depth

    def selector_args(self) -> str:
        """Return the coordinate portion of an entity selector string."""
        return f"x={self.x},y={self.y},z={self.z},dx={self.dx},dy={self.dy},dz={self.dz}"

    def __repr__(self) -> str:
        return (
            f"AABB(x={self.x}, y={self.y}, z={self.z}, "
            f"dx={self.dx}, dy={self.dy}, dz={self.dz}) "
            f"[{self.width}×{self.height}×{self.depth}]"
        )


# ---------------------------------------------------------------------------
# Title configuration
# ---------------------------------------------------------------------------

@dataclass
class TextComponent:
    """A single Minecraft JSON text component."""
    text:        str
    color:       MCColor          = MCColor.WHITE
    bold:        bool             = False
    italic:      bool             = False
    underlined:  bool             = False
    obfuscated:  bool             = False

    def to_json_dict(self) -> dict:
        d: dict = {"text": self.text, "color": self.color.value}
        if self.bold:       d["bold"]       = True
        if self.italic:     d["italic"]     = True
        if self.underlined: d["underlined"] = True
        if self.obfuscated: d["obfuscated"] = True
        return d


@dataclass
class TitleConfig:
    """
    Visual configuration for the discovery title shown on zone entry.

    Parameters
    ----------
    main_title   : The large area name (e.g. "Town Center")
    subtitle     : The smaller flavor line (e.g. "A hub of trade")
    main_color   : Color of the main title text
    sub_color    : Color of the subtitle text
    main_bold    : Whether the main title is bold (default True)
    sub_italic   : Whether the subtitle is italic (default True)
    prefix       : Optional text prepended to the subtitle (e.g. "❧ ")
    prefix_color : Color of the prefix symbol
    fade_in      : Ticks for the title to fade in  (default 20 = 1s)
    stay         : Ticks for the title to stay     (default 80 = 4s)
    fade_out     : Ticks for the title to fade out (default 20 = 1s)
    """
    main_title:   str
    subtitle:     str
    main_color:   MCColor = MCColor.GOLD
    sub_color:    MCColor = MCColor.WHITE
    main_bold:    bool    = True
    sub_italic:   bool    = True
    prefix:       str     = ""
    prefix_color: MCColor = MCColor.YELLOW
    fade_in:      int     = 20
    stay:         int     = 80
    fade_out:     int     = 20

    def _main_component(self) -> TextComponent:
        return TextComponent(
            text   = self.main_title,
            color  = self.main_color,
            bold   = self.main_bold,
        )

    def _subtitle_components(self) -> list[TextComponent]:
        parts: list[TextComponent] = []
        if self.prefix:
            parts.append(TextComponent(text=self.prefix, color=self.prefix_color))
        parts.append(TextComponent(
            text   = self.subtitle,
            color  = self.sub_color,
            italic = self.sub_italic,
        ))
        return parts


@dataclass
class SoundConfig:
    """
    A playsound command played on zone entry.

    Parameters
    ----------
    sound_id : Minecraft namespaced sound ID
    source   : Sound category (master, ambient, block, …)
    volume   : 0.0–1.0 (higher = louder / broader range)
    pitch    : 0.5–2.0 (lower = deeper, higher = sharper)
    """
    sound_id: str
    source:   SoundSource = SoundSource.MASTER
    volume:   float       = 1.0
    pitch:    float       = 1.0

    # Handy presets -----------------------------------------------------------
    @classmethod
    def town(cls) -> "SoundConfig":
        return cls("minecraft:block.note_block.bell", SoundSource.MASTER, 0.8, 1.2)

    @classmethod
    def ruins(cls) -> "SoundConfig":
        return cls("minecraft:ambient.cave", SoundSource.AMBIENT, 0.6, 0.5)

    @classmethod
    def dungeon(cls) -> "SoundConfig":
        return cls("minecraft:ambient.basalt_deltas.mood", SoundSource.AMBIENT, 0.7, 0.8)

    @classmethod
    def victory(cls) -> "SoundConfig":
        return cls("minecraft:ui.toast.challenge_complete", SoundSource.MASTER, 1.0, 0.9)

    @classmethod
    def silent(cls) -> "SoundConfig":
        """No sound — use this to explicitly suppress the playsound line."""
        return cls("", SoundSource.MASTER)

    @property
    def is_silent(self) -> bool:
        return not self.sound_id


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """
    One discoverable area.  All generator logic reads from this object.

    Parameters
    ----------
    zone_id     : Snake_case identifier used for file names and scoreboard tags.
                  Must be unique within the datapack.
    display_name: Human-readable area name shown in-game (e.g. "The Dark Forest").
                  Defaults to a title-cased version of zone_id if not set.
    aabb        : The bounding volume for detection.
    title       : Visual title configuration.
    sound       : Sound played on entry.  Defaults to SoundConfig.town().
    enabled     : Set False to skip this zone during generation (useful for WIP zones).
    notes       : Free-text comments embedded in the generated .mcfunction files.
    """
    zone_id:      str
    aabb:         AABB
    title:        TitleConfig
    display_name: str             = ""
    sound:        SoundConfig     = field(default_factory=SoundConfig.town)
    enabled:      bool            = True
    notes:        str             = ""

    def __post_init__(self) -> None:
        # Default display name to title-cased zone_id
        if not self.display_name:
            self.display_name = self.zone_id.replace("_", " ").title()
        # Validate zone_id (Minecraft function paths are lowercase, no spaces)
        if not self.zone_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"zone_id must be alphanumeric with underscores/hyphens, got: {self.zone_id!r}"
            )
        if self.zone_id != self.zone_id.lower():
            raise ValueError(f"zone_id must be lowercase, got: {self.zone_id!r}")

    @property
    def tag_name(self) -> str:
        """The entity tag used for in/out state tracking."""
        return f"ad_in_{self.zone_id}"
