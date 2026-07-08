from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.models import AttributeSchema, YoloDataset
from yolo_data_manager.stats.compute import SHAPE_RATE_BINS

DEFAULT_STATS = {
    "class_counts",
    "box_number",
    "box_width",
    "box_height",
    "box_area",
    "attribute",
}
ALL_STATS = DEFAULT_STATS | {
    "image_shape",
    "box_shape",
    "box_shape_pix",
    "box_shape_rate",
    "box_pos_start",
    "box_pos_center",
    "box_pos_end",
    "legacy_csv",
}


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


def normalize_stats_list(stats_list: str | Iterable[str] | None) -> set[str]:
    if stats_list is None:
        return set(DEFAULT_STATS)
    if isinstance(stats_list, str):
        values = [item.strip() for item in stats_list.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in stats_list if str(item).strip()]
    if not values:
        return set(DEFAULT_STATS)
    if "all" in values:
        return set(ALL_STATS)
    unknown = sorted(set(values) - ALL_STATS)
    if unknown:
        raise ValueError(f"unknown stats item(s): {', '.join(unknown)}; available: {', '.join(sorted(ALL_STATS | {'all'}))}")
    return set(values)


def write_stats_plots(dataset: YoloDataset, out_dir: str | Path, stats_list: str | Iterable[str] | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plots-dir; install with `pip install .`") from exc

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = normalize_stats_list(stats_list)

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

    if selected & {"legacy_csv", "box_shape", "box_shape_pix", "box_shape_rate", "box_pos_start", "box_pos_center", "box_pos_end"}:
        write_legacy_box_csv(dataset, output / "sta_box.csv")
    if selected & {"legacy_csv", "attribute"} and dataset.attributes is not None:
        write_legacy_attribute_csv(dataset, output / "sta_attribute.csv")

    if "class_counts" in selected:
        _bar_plot(plt, class_counts, output / "class_counts.png", "Class Counts", "class", "count")
        _write_counts_csv(class_counts, output / "box_category.csv")
    if "box_number" in selected:
        _hist_plot(plt, objects_per_image, output / "objects_per_image.png", "Objects Per Image", "objects")
        _box_number_plot(plt, objects_per_image, output / "box_number.png")
    if "box_width" in selected:
        _hist_plot(plt, widths, output / "box_width.png", "Box Width", "normalized width")
    if "box_height" in selected:
        _hist_plot(plt, heights, output / "box_height.png", "Box Height", "normalized height")
    if "box_shape" in selected:
        _box_shape_plot(plt, _annotation_rows(dataset), output / "box_shape.png")
    if "box_area" in selected:
        _hist_plot(plt, areas, output / "box_area.png", "Box Area", "normalized area")
    if "image_shape" in selected:
        write_image_shape_csv(dataset, output / "image_shape.csv")
        _image_shape_plot(plt, dataset, output / "image_shape.png")
    if "box_shape_pix" in selected:
        _box_shape_pix_plot(plt, _annotation_rows(dataset), output / "box_shape_pix.png")
    if "box_shape_rate" in selected:
        _box_shape_rate_plot(plt, _annotation_rows(dataset), output / "box_shape_rate.png")
    if "box_pos_start" in selected:
        _box_position_plot(plt, _annotation_rows(dataset), "start_x", "start_y", output / "box_pos_start.png", "Box Start Position")
    if "box_pos_center" in selected:
        _box_position_plot(plt, _annotation_rows(dataset), "center_x", "center_y", output / "box_pos_center.png", "Box Center Position")
    if "box_pos_end" in selected:
        _box_position_plot(plt, _annotation_rows(dataset), "end_x", "end_y", output / "box_pos_end.png", "Box End Position")
    if "attribute" in selected:
        for attr_name, counts in attribute_counts.items():
            _bar_plot(plt, counts, output / f"attribute_{_safe_name(attr_name)}.png", f"Attribute: {attr_name}", "value", "count")
        if dataset.attributes is not None:
            _attribute_distribution_outputs(plt, dataset, output)


def write_image_shape_csv(dataset: YoloDataset, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "width", "height", "shape_rate"])
        writer.writeheader()
        for image in dataset.images:
            rate = image.width / image.height if image.width is not None and image.height else ""
            writer.writerow(
                {
                    "image": image.file_name,
                    "width": image.width if image.width is not None else "",
                    "height": image.height if image.height is not None else "",
                    "shape_rate": rate,
                }
            )


def write_legacy_box_csv(dataset: YoloDataset, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["category", "center_x", "center_y", "width", "height", "image"])
        writer.writeheader()
        for row in _annotation_rows(dataset):
            writer.writerow(
                {
                    "category": row["category"],
                    "center_x": row["center_x"],
                    "center_y": row["center_y"],
                    "width": row["width"],
                    "height": row["height"],
                    "image": row["image"],
                }
            )


def write_legacy_attribute_csv(dataset: YoloDataset, path: str | Path) -> None:
    if dataset.attributes is None:
        return
    attr_names = dataset.attributes.names
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["category", "image", *attr_names, "attribute sum", "with attribute"],
        )
        writer.writeheader()
        for image in dataset.images:
            for annotation in image.annotations:
                class_name = dataset.class_name(annotation.class_id)
                decoded = dataset.annotation_attributes(annotation)
                row = {"category": class_name, "image": image.file_name}
                positive_count = 0
                for name in attr_names:
                    value = decoded.get(name, "")
                    row[name] = value
                    if value != "" and not AttributeSchema.is_no_value(value):
                        positive_count += 1
                row["attribute sum"] = positive_count
                row["with attribute"] = 1 if positive_count else 0
                writer.writerow(row)


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


def _hexbin_plot(plt, x_values: list[float], y_values: list[float], path: Path, title: str, xlabel: str, ylabel: str, invert_y: bool = False) -> None:
    plt.figure(figsize=(8, 6))
    if x_values and y_values:
        plt.hexbin(x_values, y_values, gridsize=35, mincnt=1)
        plt.colorbar(label="count")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if invert_y:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _annotation_rows(dataset: YoloDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image in dataset.images:
        for annotation in image.annotations:
            box = annotation.geometry_box()
            if box is None:
                continue
            width_pix = box.width * image.width if image.width is not None else None
            height_pix = box.height * image.height if image.height is not None else None
            rows.append(
                {
                    "image": image.file_name,
                    "category": dataset.class_name(annotation.class_id),
                    "center_x": box.cx,
                    "center_y": box.cy,
                    "width": box.width,
                    "height": box.height,
                    "width_pix": width_pix,
                    "height_pix": height_pix,
                    "shape_rate": box.width / box.height if box.height else None,
                    "start_x": box.cx - box.width * 0.5,
                    "start_y": box.cy - box.height * 0.5,
                    "end_x": box.cx + box.width * 0.5,
                    "end_y": box.cy + box.height * 0.5,
                }
            )
    return rows


def _box_shape_plot(plt, rows: list[dict[str, object]], path: Path) -> None:
    _hexbin_plot(
        plt,
        [float(row["height"]) for row in rows],
        [float(row["width"]) for row in rows],
        path,
        "Box Shape",
        "normalized height",
        "normalized width",
    )


def _box_shape_pix_plot(plt, rows: list[dict[str, object]], path: Path) -> None:
    filtered = [row for row in rows if row["width_pix"] is not None and row["height_pix"] is not None]
    _hexbin_plot(
        plt,
        [float(row["height_pix"]) for row in filtered],
        [float(row["width_pix"]) for row in filtered],
        path,
        "Box Shape Pixels",
        "box height (px)",
        "box width (px)",
    )


def _box_position_plot(plt, rows: list[dict[str, object]], x_key: str, y_key: str, path: Path, title: str) -> None:
    _hexbin_plot(
        plt,
        [float(row[x_key]) for row in rows],
        [float(row[y_key]) for row in rows],
        path,
        title,
        x_key,
        y_key,
        invert_y=True,
    )


def _image_shape_plot(plt, dataset: YoloDataset, path: Path) -> None:
    widths = [image.width for image in dataset.images if image.width is not None and image.height is not None]
    heights = [image.height for image in dataset.images if image.width is not None and image.height is not None]
    _hexbin_plot(plt, widths, heights, path, "Image Shape", "image width", "image height")


def _box_shape_rate_plot(plt, rows: list[dict[str, object]], path: Path) -> None:
    rates = [float(row["shape_rate"]) for row in rows if row["shape_rate"] is not None]
    counts = {f"({SHAPE_RATE_BINS[idx]}, {SHAPE_RATE_BINS[idx + 1]}]": 0 for idx in range(len(SHAPE_RATE_BINS) - 1)}
    for rate in rates:
        for idx in range(len(SHAPE_RATE_BINS) - 1):
            left = SHAPE_RATE_BINS[idx]
            right = SHAPE_RATE_BINS[idx + 1]
            if (idx == 0 and left <= rate <= right) or left < rate <= right:
                counts[f"({left}, {right}]"] += 1
                break
    _bar_plot(plt, counts, path, "Box Shape Rate", "width / height", "count")


def _box_number_plot(plt, values: list[int], path: Path) -> None:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    _bar_plot(plt, counts, path, "Box Number Per Image", "boxes", "image count")


def _attribute_distribution_outputs(plt, dataset: YoloDataset, out_dir: Path) -> None:
    attr_names = dataset.attributes.names if dataset.attributes is not None else []
    if not attr_names:
        return

    defects_per_box: dict[str, int] = {}
    category_defects: dict[str, dict[str, int]] = {}
    total_by_attr = {name: 0 for name in attr_names}

    for image in dataset.images:
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            decoded = dataset.annotation_attributes(annotation)
            positive_count = 0
            category_defects.setdefault(class_name, {name: 0 for name in attr_names})
            for name in attr_names:
                value = decoded.get(name, "")
                if value != "" and not AttributeSchema.is_no_value(value):
                    positive_count += 1
                    category_defects[class_name][name] += 1
                    total_by_attr[name] += 1
            key = str(positive_count)
            defects_per_box[key] = defects_per_box.get(key, 0) + 1

    _bar_plot(plt, defects_per_box, out_dir / "defects_num.png", "Defects Number Per Box", "defect count", "box count")
    _write_attribute_distribution_csv(category_defects, total_by_attr, out_dir / "sta_attribute_distributions.csv")
    _attribute_distribution_plot(plt, category_defects, total_by_attr, out_dir / "attribute_num.png")


def _write_attribute_distribution_csv(category_defects: dict[str, dict[str, int]], total_by_attr: dict[str, int], path: Path) -> None:
    categories = list(category_defects)
    fieldnames = ["attribute", *categories, "total"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for attr_name in total_by_attr:
            row = {"attribute": attr_name, "total": total_by_attr[attr_name]}
            for category in categories:
                row[category] = category_defects[category].get(attr_name, 0)
            writer.writerow(row)


def _attribute_distribution_plot(plt, category_defects: dict[str, dict[str, int]], total_by_attr: dict[str, int], path: Path) -> None:
    attrs = list(total_by_attr)
    categories = list(category_defects)
    x_positions = list(range(len(attrs)))
    bottom = [0] * len(attrs)
    plt.figure(figsize=(max(8, len(attrs) * 0.8), 6))
    for category in categories:
        values = [category_defects[category].get(attr_name, 0) for attr_name in attrs]
        plt.bar(x_positions, values, bottom=bottom, label=category)
        bottom = [old + value for old, value in zip(bottom, values)]
    plt.title("Attribute Distribution")
    plt.xlabel("attribute")
    plt.ylabel("count")
    plt.xticks(x_positions, attrs, rotation=30, ha="right")
    if categories:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_counts_csv(values: dict[str, int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["name", "count"])
        writer.writeheader()
        for key, value in values.items():
            writer.writerow({"name": key, "count": value})


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
