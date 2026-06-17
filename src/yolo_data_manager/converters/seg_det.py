from __future__ import annotations

import copy

from yolo_data_manager.core.models import Box, YoloDataset


def segmentation_to_detection(dataset: YoloDataset) -> YoloDataset:
    result = copy.deepcopy(dataset)
    result.task = "detect"
    for image in result.images:
        for annotation in image.annotations:
            if annotation.polygon is None:
                continue
            box = annotation.geometry_box()
            annotation.box = Box(*box.as_tuple()) if box else None
            annotation.polygon = None
    return result

