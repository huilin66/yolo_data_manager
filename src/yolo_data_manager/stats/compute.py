from __future__ import annotations

from statistics import mean

from yolo_data_manager.core.models import AttributeSchema, YoloDataset


def compute_stats(dataset: YoloDataset) -> dict[str, object]:
    class_counts = {name: 0 for name in dataset.classes.names}
    class_id_counts: dict[int, int] = {}
    objects_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    polygon_points: list[int] = []
    attribute_counts: dict[str, dict[str, int]] = {}
    class_attribute_counts: dict[str, dict[str, dict[str, int]]] = {}
    boxes_with_attribute = 0
    images_with_attribute: set[str] = set()

    for image in dataset.images:
        objects_per_image.append(len(image.annotations))
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            class_id_counts[annotation.class_id] = class_id_counts.get(annotation.class_id, 0) + 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            box = annotation.geometry_box()
            if box is not None:
                widths.append(box.width)
                heights.append(box.height)
                areas.append(box.width * box.height)
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
