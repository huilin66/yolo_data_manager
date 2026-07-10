from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from yolo_data_manager.core.models import YoloAnnotation, YoloDataset


AnnotationPredicate = Callable[[YoloAnnotation], bool]


def filter_annotations(dataset: YoloDataset, predicate: AnnotationPredicate) -> YoloDataset:
    result = copy.deepcopy(dataset)
    for image in result.images:
        image.annotations = [annotation for annotation in image.annotations if predicate(annotation)]
    return result


def keep_classes(dataset: YoloDataset, class_ids: set[int]) -> YoloDataset:
    return filter_annotations(dataset, lambda ann: ann.class_id in class_ids)


def filter_by_geometry(
    dataset: YoloDataset,
    class_ids: set[int] | None = None,
    min_width: float | None = None,
    min_height: float | None = None,
    min_size_logic: str = "or",
    min_area: float | None = None,
    max_area: float | None = None,
    min_confidence: float | None = None,
    class_rules: dict[int | str, dict[str, Any]] | None = None,
) -> YoloDataset:
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")

    resolved_rules = _resolve_class_rules(dataset, class_rules)
    global_rule = {
        "min_width": min_width,
        "min_height": min_height,
        "min_size_logic": min_size_logic,
        "min_area": min_area,
        "max_area": max_area,
        "min_confidence": min_confidence,
    }

    def predicate(annotation: YoloAnnotation) -> bool:
        if class_ids is not None and annotation.class_id not in class_ids:
            return False
        rule = resolved_rules.get(annotation.class_id, global_rule)
        return _keep_by_rule(annotation, rule)

    return filter_annotations(dataset, predicate)


def _keep_by_rule(annotation: YoloAnnotation, rule: dict[str, Any]) -> bool:
    min_confidence = rule.get("min_confidence")
    if min_confidence is not None and annotation.confidence is not None and annotation.confidence < float(min_confidence):
        return False
    box = annotation.geometry_box()
    if box is None:
        return False
    area = box.width * box.height
    min_width = rule.get("min_width")
    min_height = rule.get("min_height")
    min_size_logic = rule.get("min_size_logic", "or")
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")
    width_too_small = min_width is not None and box.width < float(min_width)
    height_too_small = min_height is not None and box.height < float(min_height)
    if min_size_logic == "and":
        if width_too_small and height_too_small:
            return False
    elif width_too_small or height_too_small:
        return False
    min_area = rule.get("min_area")
    if min_area is not None and area < float(min_area):
        return False
    max_area = rule.get("max_area")
    if max_area is not None and area > float(max_area):
        return False
    return True


def _resolve_class_rules(dataset: YoloDataset, class_rules: dict[int | str, dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not class_rules:
        return {}
    resolved: dict[int, dict[str, Any]] = {}
    for class_value, rule in class_rules.items():
        class_id = dataset.class_id(class_value)
        resolved[class_id] = dict(rule)
    return resolved
