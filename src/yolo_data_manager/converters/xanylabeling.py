from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from yolo_data_manager.core.geometry import normalized_points_to_pixels, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.runtime import iter_progress, normalize_workers


def export_xanylabeling(
    dataset: YoloDataset,
    out_dir: str | Path,
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    def write_image(image: YoloImage) -> None:
        data = image_to_xanylabeling(dataset, image)
        (output / f"{image.stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for image in iter_progress(dataset.images, enabled=progress, total=len(dataset.images), desc="export xany", leave=progress_leave):
            write_image(image)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(write_image, image) for image in dataset.images]
        for future in iter_progress(as_completed(futures), enabled=progress, total=len(futures), desc="export xany", leave=progress_leave):
            future.result()


def image_to_xanylabeling(dataset: YoloDataset, image: YoloImage) -> dict[str, object]:
    width = image.width or 0
    height = image.height or 0
    shapes = []
    for annotation in image.annotations:
        attributes = dataset.annotation_attributes(annotation)

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
