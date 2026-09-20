from __future__ import annotations

DEFAULT_CLASSES = (
    "background",
    "dent",
    "scratch",
    "crack_or_breakage",
    "paint_damage",
    "deformation_or_detachment",
)

HARD_NEGATIVE_TAGS = frozenset(
    {
        "glare",
        "reflection",
        "dirt",
        "water_spot",
        "panel_gap",
        "body_contour",
        "shadow",
        "decal",
        "previous_repair",
    }
)


def validate_classes(classes: list[str] | tuple[str, ...]) -> None:
    if not classes or classes[0] != "background":
        raise ValueError("class id 0 must be named 'background'")
    if len(set(classes)) != len(classes):
        raise ValueError("class names must be unique")

