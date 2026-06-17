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

