from __future__ import annotations

import base64
import io
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.geometry import xyxy_to_xywhn
from yolo_data_manager.core.models import Box, ClassSchema, Polygon, YoloAnnotation, YoloDataset, YoloImage


def import_labelme_dir(
    json_dir: str | Path,
    out_root: str | Path | None = None,
    task: str = "auto",
    class_names: list[str] | None = None,
) -> YoloDataset:
    root = Path(json_dir)
    classes = ClassSchema(list(class_names or []))
    images: list[YoloImage] = []

    for json_path in sorted(root.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = _resolve_or_decode_image(data, json_path, out_root)
        width, height = _image_size(data, image_path)
        annotations: list[YoloAnnotation] = []

        for line_no, shape in enumerate(data.get("shapes", []), start=1):
            label = str(shape.get("label", "")).strip()
            if not label:
                continue
            class_id = classes.ensure(label)
            shape_type = shape.get("shape_type") or "polygon"
            points = [(float(x), float(y)) for x, y in shape.get("points", [])]
            annotation = _shape_to_annotation(class_id, points, shape_type, width, height, task)
            annotation.line_no = line_no
            annotations.append(annotation)

        images.append(
            YoloImage(
                path=image_path,
                label_path=None,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    dataset = YoloDataset(root=Path(out_root or root), images=images, classes=classes, task=task)
    if out_root is not None:
        from yolo_data_manager.io.writer import write_yolo_dataset

        write_yolo_dataset(dataset, out_root, copy_images=False)
    return dataset


def _shape_to_annotation(
    class_id: int,
    points: list[tuple[float, float]],
    shape_type: str,
    width: int,
    height: int,
    task: str,
) -> YoloAnnotation:
    if shape_type == "circle" and len(points) >= 2:
        points = _circle_points(points[0], points[1])

    if task == "segment" and len(points) >= 3:
        return YoloAnnotation(
            class_id=class_id,
            polygon=Polygon([(x / width, y / height) for x, y in points]),
        )

    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    cx, cy, bw, bh = xyxy_to_xywhn(x1, y1, x2, y2, width, height)
    return YoloAnnotation(class_id=class_id, box=Box(cx, cy, bw, bh))


def _circle_points(center: tuple[float, float], edge: tuple[float, float], count: int = 24) -> list[tuple[float, float]]:
    cx, cy = center
    radius = math.dist(center, edge)
    return [
        (cx + math.cos(2 * math.pi * idx / count) * radius, cy + math.sin(2 * math.pi * idx / count) * radius)
        for idx in range(count)
    ]


def _resolve_or_decode_image(data: dict[str, object], json_path: Path, out_root: str | Path | None) -> Path:
    image_name = str(data.get("imagePath") or f"{json_path.stem}.png")
    src_image = json_path.parent / image_name
    if out_root is None:
        if src_image.exists():
            return src_image
        return json_path.with_suffix(".png")

    image_dir = Path(out_root) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dst_image = image_dir / Path(image_name).name
    if src_image.exists():
        shutil.copy2(src_image, dst_image)
        return dst_image

    image_data = data.get("imageData")
    if isinstance(image_data, str) and image_data:
        raw = base64.b64decode(image_data)
        Image.open(io.BytesIO(raw)).save(dst_image)
    return dst_image


def _image_size(data: dict[str, object], image_path: Path) -> tuple[int, int]:
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if width and height:
        return int(width), int(height)
    with Image.open(image_path) as image:
        return image.size
