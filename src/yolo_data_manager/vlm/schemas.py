"""Strict, provider-independent parsing for VLM responses."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


class VLMOutputError(ValueError):
    """Raised when a VLM response cannot be converted to the requested schema."""


@dataclass(frozen=True)
class VLMBox:
    class_name: str | None
    bbox: tuple[float, float, float, float]
    class_id: int | None = None
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMDecision:
    action: str
    target_class: str | int | None = None
    attribute_name: str | None = None
    attribute_value: Any = None
    confidence: float | None = None
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def extract_json_object(text: str) -> Any:
    """Extract the first valid JSON value from plain or fenced model output."""

    cleaned = str(text).strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith(fence):
            lines = lines[:-1]
        cleaned = chr(10).join(lines).strip()
    decoder = json.JSONDecoder()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise VLMOutputError("VLM response did not contain valid JSON")


def parse_detection_response(
    text: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> list[VLMBox]:
    """Parse a detection response containing normalized YOLO boxes."""

    payload = extract_json_object(text)
    if isinstance(payload, Mapping):
        values = payload.get("boxes", payload.get("detections", payload.get("annotations", [])))
    else:
        values = payload
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise VLMOutputError("detection JSON must be a list or contain a boxes list")

    result: list[VLMBox] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise VLMOutputError(f"detection {index} must be an object")
        class_value = item.get("class_name", item.get("class", item.get("name")))
        class_id = _optional_int(item.get("class_id", item.get("id")))
        if class_value is None and class_id is None:
            raise VLMOutputError(f"detection {index} has no class_name or class_id")
        bbox = item.get("bbox", item.get("box"))
        if bbox is None:
            raise VLMOutputError(f"detection {index} has no bbox")
        parsed_bbox = _parse_bbox(bbox, item, image_size=image_size)
        attrs = item.get("attributes", item.get("attrs", {}))
        if attrs is None:
            attrs = {}
        if not isinstance(attrs, Mapping):
            raise VLMOutputError(f"detection {index} attributes must be an object")
        result.append(
            VLMBox(
                class_name=None if class_value is None else str(class_value),
                class_id=class_id,
                bbox=parsed_bbox,
                confidence=_optional_float(item.get("confidence", item.get("score"))),
                attributes={str(key): value for key, value in attrs.items()},
            )
        )
    return result


def parse_error_decision(text: str) -> VLMDecision:
    """Parse and normalize a correction decision."""

    payload = extract_json_object(text)
    if not isinstance(payload, Mapping):
        raise VLMOutputError("error decision must be a JSON object")
    action = _normalise_action(payload.get("action", payload.get("decision", "keep")))
    target_class = payload.get("target_class", payload.get("class_name"))
    if target_class is None and "class_id" in payload:
        target_class = _optional_int(payload.get("class_id"))
    attribute_name = payload.get("attribute_name", payload.get("attribute"))
    attribute_value = payload.get("attribute_value", payload.get("value"))
    return VLMDecision(
        action=action,
        target_class=target_class,
        attribute_name=str(attribute_name) if attribute_name is not None else None,
        attribute_value=attribute_value,
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload["reason"]) if payload.get("reason") is not None else None,
        raw={str(key): value for key, value in payload.items()},
    )


def _parse_bbox(
    value: Any,
    item: Mapping[str, Any],
    *,
    image_size: tuple[int, int] | None,
) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VLMOutputError("bbox must be a list of four numbers")
    if len(value) != 4:
        raise VLMOutputError("bbox must contain four values")
    numbers = [_number(item_value) for item_value in value]
    fmt = str(item.get("bbox_format", item.get("format", "xywh"))).lower()
    if fmt in {"xyxy", "pixel_xyxy", "pixels_xyxy"}:
        x1, y1, x2, y2 = numbers
        if image_size is None:
            raise VLMOutputError("pixel xyxy bbox requires image_size")
        width, height = image_size
        numbers = [
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
    elif fmt in {"pixel_xywh", "pixels_xywh"}:
        if image_size is None:
            raise VLMOutputError("pixel xywh bbox requires image_size")
        width, height = image_size
        numbers = [numbers[0] / width, numbers[1] / height, numbers[2] / width, numbers[3] / height]
    elif fmt not in {"xywh", "normalized_xywh", "yolo"}:
        raise VLMOutputError(f"unsupported bbox format: {fmt}")
    if any(not math.isfinite(number) for number in numbers):
        raise VLMOutputError("bbox contains a non-finite number")
    cx, cy, box_width, box_height = numbers
    if box_width <= 0 or box_height <= 0:
        raise VLMOutputError("bbox width and height must be positive")
    if any(number < 0 or number > 1 for number in numbers):
        raise VLMOutputError("normalized bbox values must be between 0 and 1")
    return cx, cy, box_width, box_height


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise VLMOutputError(f"bbox value is not numeric: {value!r}") from exc


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise VLMOutputError(f"expected integer, got {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise VLMOutputError(f"expected number, got {value!r}") from exc


def _normalise_action(value: Any) -> str:
    text = str(value or "keep").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "correct": "correct_class",
        "change_class": "correct_class",
        "fix_class": "correct_class",
        "delete": "delete_gt",
        "remove": "delete_gt",
        "drop": "delete_gt",
        "correct_attr": "correct_attribute",
        "change_attribute": "correct_attribute",
        "fix_attribute": "correct_attribute",
        "add": "add_prediction",
        "replace": "replace_gt_from_prediction",
        "ignore": "keep",
        "unchanged": "keep",
    }
    return aliases.get(text, text)


__all__ = [
    "VLMBox",
    "VLMDecision",
    "VLMOutputError",
    "extract_json_object",
    "parse_detection_response",
    "parse_error_decision",
]
