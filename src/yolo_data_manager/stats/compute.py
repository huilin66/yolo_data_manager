from __future__ import annotations

from statistics import mean

from yolo_data_manager.core.models import YoloDataset


def compute_stats(dataset: YoloDataset) -> dict[str, object]:
    class_counts = {name: 0 for name in dataset.classes.names}
    class_id_counts: dict[int, int] = {}
    objects_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    polygon_points: list[int] = []

    for image in dataset.images:
        objects_per_image.append(len(image.annotations))
        for annotation in image.annotations:
            class_id_counts[annotation.class_id] = class_id_counts.get(annotation.class_id, 0) + 1
            class_counts[dataset.class_name(annotation.class_id)] = class_counts.get(dataset.class_name(annotation.class_id), 0) + 1
            box = annotation.geometry_box()
            if box is not None:
                widths.append(box.width)
                heights.append(box.height)
                areas.append(box.width * box.height)
            if annotation.polygon is not None:
                polygon_points.append(len(annotation.polygon.points))

    return {
        "image_count": len(dataset.images),
        "label_count": len(dataset.labels()),
        "orphan_label_count": len(dataset.orphan_labels),
        "annotation_count": dataset.annotation_count(),
        "empty_image_count": sum(1 for value in objects_per_image if value == 0),
        "class_counts": class_counts,
        "class_id_counts": class_id_counts,
        "objects_per_image": _describe(objects_per_image),
        "box_width": _describe(widths),
        "box_height": _describe(heights),
        "box_area": _describe(areas),
        "polygon_points": _describe(polygon_points),
    }


def _describe(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }

