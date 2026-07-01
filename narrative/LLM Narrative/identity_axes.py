from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from settlement_generator import Settlement


# Example values for the five categorical axes. Used only to seed
# `settlement_generator.SYSTEM_PROMPT` — never used to validate output.
AXIS_OPTIONS: dict[str, list[str]] = {
    "primary_industry": [
        "mining", "fishing", "trade", "logging", "pilgrimage",
        "military outpost", "farming", "herding",
    ],
    "central_virtue": [
        "discipline", "hospitality", "resilience", "secrecy",
        "faith", "craftsmanship", "loyalty", "scholarship",
    ],
    "collective_fear": [
        "famine", "winter", "flooding", "outsiders",
        "collapse", "disease", "spirits", "isolation",
    ],
    "social_structure": [
        "clan-based", "merchant council", "military command",
        "religious authority", "worker cooperative", "hereditary stewardship",
    ],
    "outsider_reputation": [
        "stubborn", "greedy", "honorable", "cursed",
        "isolationist", "scholarly", "hospitable", "feared",
    ],
}


# Display label used in `axes_hint`. Order matters — this is the order they
# appear in the prompt block.
_AXIS_LABELS: tuple[tuple[str, str], ...] = (
    ("primary_industry",    "Industry"),
    ("central_virtue",      "Virtue"),
    ("collective_fear",     "Fear"),
    ("historical_wound",    "Wound"),
    ("motif",               "Motif"),
    ("worldview",           "Worldview"),
    ("social_structure",    "Social"),
    ("outsider_reputation", "Reputation"),
)


def axes_hint(settlement: "Settlement") -> str:
    lines: list[str] = []
    for attr, label in _AXIS_LABELS:
        value = getattr(settlement, attr, None)
        if isinstance(value, str) and value.strip():
            lines.append(f"  {label}: {value.strip()}")
    if not lines:
        return ""
    return "Identity:\n" + "\n".join(lines)
