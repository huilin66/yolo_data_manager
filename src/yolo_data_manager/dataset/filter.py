from __future__ import annotations

import copy
from collections.abc import Callable

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
) -> YoloDataset:
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")

    def predicate(annotation: YoloAnnotation) -> bool:
        if class_ids is not None and annotation.class_id not in class_ids:
            return False
        if min_confidence is not None and annotation.confidence is not None and annotation.confidence < min_confidence:
            return False
        box = annotation.geometry_box()
        if box is None:
            return False
        area = box.width * box.height
        width_too_small = min_width is not None and box.width < min_width
        height_too_small = min_height is not None and box.height < min_height
        if min_size_logic == "and":
            if width_too_small and height_too_small:
                return False
        elif width_too_small or height_too_small:
            return False
        if min_area is not None and area < min_area:
            return False
        if max_area is not None and area > max_area:
            return False
        return True

    return filter_annotations(dataset, predicate)
