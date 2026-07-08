from __future__ import annotations

from statistics import mean

from yolo_data_manager.core.models import AttributeSchema, YoloDataset

SHAPE_RATE_BINS = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2,
    2.1,
    2.2,
    2.4,
    2.6,
    3,
    3.5,
    4,
    5,
]


def compute_stats(dataset: YoloDataset) -> dict[str, object]:
    class_counts = {name: 0 for name in dataset.classes.names}
    class_id_counts: dict[int, int] = {}
    objects_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    polygon_points: list[int] = []
    image_widths: list[int] = []
    image_heights: list[int] = []
    box_width_pixels: list[float] = []
    box_height_pixels: list[float] = []
    box_shape_rates: list[float] = []
    box_start_x: list[float] = []
    box_start_y: list[float] = []
    box_center_x: list[float] = []
    box_center_y: list[float] = []
    box_end_x: list[float] = []
    box_end_y: list[float] = []
    attribute_counts: dict[str, dict[str, int]] = {}
    class_attribute_counts: dict[str, dict[str, dict[str, int]]] = {}
    boxes_with_attribute = 0
    images_with_attribute: set[str] = set()

    for image in dataset.images:
        objects_per_image.append(len(image.annotations))
        if image.width is not None:
            image_widths.append(image.width)
        if image.height is not None:
            image_heights.append(image.height)
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            class_id_counts[annotation.class_id] = class_id_counts.get(annotation.class_id, 0) + 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            box = annotation.geometry_box()
            if box is not None:
                widths.append(box.width)
                heights.append(box.height)
                areas.append(box.width * box.height)
                box_center_x.append(box.cx)
                box_center_y.append(box.cy)
                box_start_x.append(box.cx - box.width * 0.5)
                box_start_y.append(box.cy - box.height * 0.5)
                box_end_x.append(box.cx + box.width * 0.5)
                box_end_y.append(box.cy + box.height * 0.5)
                if box.height:
                    box_shape_rates.append(box.width / box.height)
                if image.width is not None:
                    box_width_pixels.append(box.width * image.width)
                if image.height is not None:
                    box_height_pixels.append(box.height * image.height)
            if annotation.polygon is not None:
                polygon_points.append(len(annotation.polygon.points))
            attrs = dataset.annotation_attributes(annotation)
            if attrs:
                has_positive_attr = False
                for attr_name, attr_value in attrs.items():
                    value_text = str(attr_value)
                    attribute_counts.setdefault(attr_name, {})
                    attribute_counts[attr_name][value_text] = attribute_counts[attr_name].get(value_text, 0) + 1
                    class_attribute_counts.setdefault(class_name, {}).setdefault(attr_name, {})
                    class_attribute_counts[class_name][attr_name][value_text] = class_attribute_counts[class_name][attr_name].get(value_text, 0) + 1
                    if not AttributeSchema.is_no_value(attr_value):
                        has_positive_attr = True
                if has_positive_attr:
                    boxes_with_attribute += 1
                    images_with_attribute.add(image.file_name)

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
        "image_width": _describe(image_widths),
        "image_height": _describe(image_heights),
        "box_width_pix": _describe(box_width_pixels),
        "box_height_pix": _describe(box_height_pixels),
        "box_shape_rate": _describe(box_shape_rates),
        "box_pos_start_x": _describe(box_start_x),
        "box_pos_start_y": _describe(box_start_y),
        "box_pos_center_x": _describe(box_center_x),
        "box_pos_center_y": _describe(box_center_y),
        "box_pos_end_x": _describe(box_end_x),
        "box_pos_end_y": _describe(box_end_y),
        "box_shape_rate_bins": _bin_counts(box_shape_rates, SHAPE_RATE_BINS),
        "polygon_points": _describe(polygon_points),
        "attribute_counts": attribute_counts,
        "class_attribute_counts": class_attribute_counts,
        "boxes_with_attribute": boxes_with_attribute,
        "boxes_without_attribute": dataset.annotation_count() - boxes_with_attribute,
        "images_with_attribute": len(images_with_attribute),
        "images_without_attribute": len(dataset.images) - len(images_with_attribute),
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


def _bin_counts(values: list[float], bins: list[float]) -> dict[str, int]:
    counts = {f"({bins[idx]}, {bins[idx + 1]}]": 0 for idx in range(len(bins) - 1)}
    for value in values:
        for idx in range(len(bins) - 1):
            left = bins[idx]
            right = bins[idx + 1]
            if (idx == 0 and left <= value <= right) or left < value <= right:
                counts[f"({left}, {right}]"] += 1
                break
    return counts
