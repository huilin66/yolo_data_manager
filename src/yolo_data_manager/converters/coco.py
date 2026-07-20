from __future__ import annotations

import json
from pathlib import Path

from yolo_data_manager.core.geometry import normalized_points_to_pixels, polygon_area, xywhn_to_xyxy
from yolo_data_manager.core.geometry import xyxy_to_xywhn
from yolo_data_manager.core.models import Box, ClassSchema, Polygon, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.io.writer import write_yolo_dataset
from yolo_data_manager.runtime import iter_progress


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


def import_coco(
    json_path: str | Path,
    images_dir: str | Path,
    out_root: str | Path | None = None,
    task: str = "detect",
    class_names: list[str] | None = None,
    copy_images: bool = True,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> YoloDataset:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    if class_names is None:
        class_names = [str(item["name"]) for item in categories]
    classes = ClassSchema(list(class_names))
    cat_to_class = {item["id"]: classes.ensure(str(item["name"])) for item in categories}

    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    img_root = Path(images_dir)
    images: list[YoloImage] = []
    image_items = list(data.get("images", []))
    for image_item in iter_progress(image_items, enabled=progress, total=len(image_items), desc="import coco", leave=progress_leave):
        image_id = int(image_item["id"])
        file_name = str(image_item["file_name"])
        width = int(image_item.get("width") or 0)
        height = int(image_item.get("height") or 0)
        image_path = img_root / file_name
        if not image_path.exists():
            image_path = img_root / Path(file_name).name
        yolo_annotations = [
            _coco_annotation_to_yolo(annotation, cat_to_class, width, height, task)
            for annotation in annotations_by_image.get(image_id, [])
        ]
        yolo_annotations = [annotation for annotation in yolo_annotations if annotation is not None]
        images.append(
            YoloImage(
                path=image_path,
                width=width,
                height=height,
                annotations=yolo_annotations,
            )
        )

    dataset = YoloDataset(root=Path(out_root or Path(json_path).parent), images=images, classes=classes, task=task)
    if out_root is not None:
        write_yolo_dataset(
            dataset,
            out_root,
            copy_images=copy_images,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
    return dataset


def _coco_annotation_to_yolo(
    annotation: dict[str, object],
    cat_to_class: dict[int, int],
    width: int,
    height: int,
    task: str,
) -> YoloAnnotation | None:
    category_id = int(annotation["category_id"])
    if category_id not in cat_to_class:
        return None
    class_id = cat_to_class[category_id]

    segmentation = annotation.get("segmentation")
    if task == "segment" and isinstance(segmentation, list) and segmentation:
        raw_polygon = segmentation[0]
        if isinstance(raw_polygon, list) and len(raw_polygon) >= 6:
            points = [
                (float(raw_polygon[idx]) / width, float(raw_polygon[idx + 1]) / height)
                for idx in range(0, len(raw_polygon), 2)
            ]
            return YoloAnnotation(class_id=class_id, polygon=Polygon(points))

    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x, y, box_width, box_height = [float(value) for value in bbox[:4]]
    cx, cy, bw, bh = xyxy_to_xywhn(x, y, x + box_width, y + box_height, width, height)
    return YoloAnnotation(class_id=class_id, box=Box(cx, cy, bw, bh))
