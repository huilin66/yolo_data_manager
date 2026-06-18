from __future__ import annotations

import csv
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset


def write_annotation_csv(dataset: YoloDataset, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label_path",
        "line_no",
        "class_id",
        "class_name",
        "task",
        "cx",
        "cy",
        "width",
        "height",
        "polygon_points",
        "confidence",
    ]
    attr_names = dataset.attributes.names if dataset.attributes is not None else []
    fieldnames.extend([f"attr:{name}" for name in attr_names])
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for image in dataset.images:
            for annotation in image.annotations:
                box = annotation.geometry_box()
                attrs = dataset.annotation_attributes(annotation)
                row = {
                    "image": image.file_name,
                    "label_path": str(image.label_path) if image.label_path else "",
                    "line_no": annotation.line_no or "",
                    "class_id": annotation.class_id,
                    "class_name": dataset.class_name(annotation.class_id),
                    "task": "segment" if annotation.polygon else "detect",
                    "cx": box.cx if box else "",
                    "cy": box.cy if box else "",
                    "width": box.width if box else "",
                    "height": box.height if box else "",
                    "polygon_points": len(annotation.polygon.points) if annotation.polygon else "",
                    "confidence": annotation.confidence if annotation.confidence is not None else "",
                }
                for name in attr_names:
                    row[f"attr:{name}"] = attrs.get(name, "")
                writer.writerow(
                    row
                )


def write_attribute_csv(dataset: YoloDataset, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["image", "line_no", "class_id", "class_name", "attribute", "raw_value", "value"],
        )
        writer.writeheader()
        if dataset.attributes is None:
            return
        for image in dataset.images:
            for annotation in image.annotations:
                class_name = dataset.class_name(annotation.class_id)
                names = dataset.attributes.names_for_class(class_name)
                decoded = dataset.annotation_attributes(annotation)
                for idx, name in enumerate(names):
                    writer.writerow(
                        {
                            "image": image.file_name,
                            "line_no": annotation.line_no or "",
                            "class_id": annotation.class_id,
                            "class_name": class_name,
                            "attribute": name,
                            "raw_value": annotation.attributes[idx] if idx < len(annotation.attributes) else "",
                            "value": decoded.get(name, ""),
                        }
                    )


def write_stats_plots(dataset: YoloDataset, out_dir: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plots-dir; install with `pip install -e .[plot]`") from exc

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    class_counts = {name: 0 for name in dataset.classes.names}
    objects_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    attribute_counts: dict[str, dict[str, int]] = {}

    for image in dataset.images:
        objects_per_image.append(len(image.annotations))
        for annotation in image.annotations:
            class_counts[dataset.class_name(annotation.class_id)] = class_counts.get(dataset.class_name(annotation.class_id), 0) + 1
            box = annotation.geometry_box()
            if box is not None:
                widths.append(box.width)
                heights.append(box.height)
                areas.append(box.width * box.height)
            for attr_name, attr_value in dataset.annotation_attributes(annotation).items():
                attribute_counts.setdefault(attr_name, {})
                value_text = str(attr_value)
                attribute_counts[attr_name][value_text] = attribute_counts[attr_name].get(value_text, 0) + 1

    _bar_plot(plt, class_counts, output / "class_counts.png", "Class Counts", "class", "count")
    _hist_plot(plt, objects_per_image, output / "objects_per_image.png", "Objects Per Image", "objects")
    _hist_plot(plt, widths, output / "box_width.png", "Box Width", "normalized width")
    _hist_plot(plt, heights, output / "box_height.png", "Box Height", "normalized height")
    _hist_plot(plt, areas, output / "box_area.png", "Box Area", "normalized area")
    for attr_name, counts in attribute_counts.items():
        _bar_plot(plt, counts, output / f"attribute_{_safe_name(attr_name)}.png", f"Attribute: {attr_name}", "value", "count")


def _bar_plot(plt, values: dict[str, int], path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(max(8, len(values) * 0.6), 5))
    keys = list(values.keys())
    vals = [values[key] for key in keys]
    plt.bar(keys, vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _hist_plot(plt, values: list[float | int], path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 5))
    if values:
        plt.hist(values, bins=min(50, max(5, len(set(values)))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
