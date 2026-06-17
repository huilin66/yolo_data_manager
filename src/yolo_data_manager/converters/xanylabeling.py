from __future__ import annotations

import json
from pathlib import Path

from yolo_data_manager.core.geometry import normalized_points_to_pixels, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloDataset, YoloImage


def export_xanylabeling(dataset: YoloDataset, out_dir: str | Path) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for image in dataset.images:
        data = image_to_xanylabeling(dataset, image)
        (output / f"{image.stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def image_to_xanylabeling(dataset: YoloDataset, image: YoloImage) -> dict[str, object]:
    width = image.width or 0
    height = image.height or 0
    shapes = []
    for annotation in image.annotations:
        attributes = {}
        if dataset.attributes is not None and annotation.attributes:
            attributes = dataset.attributes.decode(annotation.attributes)

        if annotation.polygon is not None:
            points = normalized_points_to_pixels(annotation.polygon.points, width, height)
            shape_type = "polygon"
        else:
            box = annotation.geometry_box()
            if box is None:
                continue
            xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
            points = [
                [xyxy.x1, xyxy.y1],
                [xyxy.x2, xyxy.y1],
                [xyxy.x2, xyxy.y2],
                [xyxy.x1, xyxy.y2],
            ]
            shape_type = "rectangle"

        shapes.append(
            {
                "label": dataset.class_name(annotation.class_id),
                "shape_type": shape_type,
                "flags": {},
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": False,
                "attributes": attributes,
            }
        )

    return {
        "version": "2.3.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": image.file_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

