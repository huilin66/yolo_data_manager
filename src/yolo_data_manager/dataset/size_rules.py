"""Reusable annotation size-rule helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def keep_by_min_pixel_rules(
    pixel_width: float | None,
    pixel_height: float | None,
    rules: Any,
    *,
    logic: str = "or",
    comparison: str = "gte",
) -> bool:
    """Return whether a box satisfies a list of minimum pixel-size rules.

    Each rule is an ``and`` condition over width and height.  The rules are
    combined using ``logic``.  Rule values may use either the concise
    ``width``/``height`` keys or the explicit ``min_width``/``min_height``
    keys.  ``comparison`` accepts ``gte`` (the default) or ``gt``.
    """

    if logic not in {"or", "and"}:
        raise ValueError("pixel_logic must be 'or' or 'and'")
    if comparison not in {"gt", "gte"}:
        raise ValueError("pixel_comparison must be 'gt' or 'gte'")
    if pixel_width is None or pixel_height is None:
        raise ValueError(
            "min_pixel_rules require known image width and height"
        )

    if isinstance(rules, Mapping) or isinstance(rules, (str, bytes, bytearray)):
        raise TypeError("min_pixel_rules must be a list of rule mappings")
    try:
        rule_list = list(rules)
    except TypeError as exc:
        raise TypeError("min_pixel_rules must be a list of rule mappings") from exc
    if not rule_list:
        raise ValueError("min_pixel_rules must contain at least one rule")

    matches: list[bool] = []
    for index, rule in enumerate(rule_list):
        if not isinstance(rule, Mapping):
            raise TypeError(
                f"min_pixel_rules[{index}] must be a mapping of width/height thresholds"
            )
        min_width = rule.get("min_width", rule.get("width"))
        min_height = rule.get("min_height", rule.get("height"))
        if min_width is None and min_height is None:
            raise ValueError(
                f"min_pixel_rules[{index}] requires width or height"
            )

        width_ok = _compare_pixel_size(pixel_width, min_width, comparison)
        height_ok = _compare_pixel_size(pixel_height, min_height, comparison)
        matches.append(width_ok and height_ok)

    return any(matches) if logic == "or" else all(matches)


def _compare_pixel_size(
    actual: float,
    threshold: Any,
    comparison: str,
) -> bool:
    if threshold is None:
        return True
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("pixel size thresholds must be numeric") from exc
    if value < 0:
        raise ValueError("pixel size thresholds must be non-negative")
    if comparison == "gt":
        return actual > value
    return actual >= value


def min_pixel_rules_from_rule(rule: Mapping[str, Any] | None) -> tuple[Any, str, str] | None:
    """Extract the optional pixel-rule configuration from one class rule."""

    if rule is None or rule.get("min_pixel_rules") is None:
        return None
    return (
        rule["min_pixel_rules"],
        str(rule.get("pixel_logic", "or")),
        str(rule.get("pixel_comparison", "gte")),
    )

