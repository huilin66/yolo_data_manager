from __future__ import annotations

import copy

from yolo_data_manager.core.models import YoloDataset


def predictions_to_pseudo_labels(
    dataset: YoloDataset,
    confidence_threshold: float = 0.0,
    drop_confidence: bool = True,
) -> YoloDataset:
    result = copy.deepcopy(dataset)
    for image in result.images:
        kept = []
        for annotation in image.annotations:
            if annotation.confidence is not None and annotation.confidence < confidence_threshold:
                continue
            if drop_confidence:
                annotation.confidence = None
            kept.append(annotation)
        image.annotations = kept
    return result

