from __future__ import annotations

import json
from pathlib import Path

from yolo_data_manager.core.geometry import normalized_points_to_pixels, polygon_area, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloDataset


def dataset_to_coco(dataset: YoloDataset) -> dict[str, object]:
    coco: dict[str, object] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "none"}
            for idx, name in enumerate(dataset.classes.names)
        ],
    }
    images = coco["images"]
    annotations = coco["annotations"]
    ann_id = 1

    for image_id, image in enumerate(dataset.images, start=1):
        width = image.width or 0
        height = image.height or 0
        images.append(
            {
                "id": image_id,
                "file_name": image.file_name,
                "width": width,
                "height": height,
            }
        )

        for annotation in image.annotations:
            if annotation.polygon is not None:
                points = normalized_points_to_pixels(annotation.polygon.points, width, height)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                segmentation = [[coord for point in points for coord in point]]
                area = polygon_area(points)
            else:
                box = annotation.geometry_box()
                if box is None:
                    continue
                xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
                bbox = [xyxy.x1, xyxy.y1, xyxy.x2 - xyxy.x1, xyxy.y2 - xyxy.y1]
                segmentation = [[
                    xyxy.x1,
                    xyxy.y1,
                    xyxy.x2,
                    xyxy.y1,
                    xyxy.x2,
                    xyxy.y2,
                    xyxy.x1,
                    xyxy.y2,
                ]]
                area = bbox[2] * bbox[3]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": annotation.class_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco


def export_coco(dataset: YoloDataset, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset_to_coco(dataset), ensure_ascii=False), encoding="utf-8")

