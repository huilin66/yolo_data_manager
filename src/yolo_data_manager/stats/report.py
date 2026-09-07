from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset
from yolo_data_manager.runtime import iter_progress

SPLIT_NAMES = ("train", "val", "test")
BASIC_INFO_FIELDS = (
    "section",
    "class_name",
    "attribute",
    "value",
    "total",
    "train",
    "val",
    "test",
)


def write_json_report(data: dict[str, object], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_class_counts_csv(data: dict[str, object], path: str | Path) -> None:
    counts = data.get("class_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["class_name", "count"])
        writer.writeheader()
        for class_name, count in counts.items():
            writer.writerow({"class_name": class_name, "count": count})


def build_basic_info_rows(
    dataset: YoloDataset,
    *,
    progress: bool = False,
    progress_leave: bool = False,
) -> list[dict[str, object]]:
    """Build compact image, class, and standalone attribute counts by split."""

    split_by_image = infer_image_splits(dataset)
    image_counts = _empty_split_counts()
    class_counts: dict[str, dict[str, int]] = {
        name: _empty_split_counts() for name in dataset.classes.names
    }
    attribute_counts: dict[tuple[str, str], dict[str, int]] = {}
    class_order = list(dataset.classes.names)
    attribute_order: list[tuple[str, str]] = []

    for image in iter_progress(
        dataset.images,
        enabled=progress,
        total=len(dataset.images),
        desc="stats basic info",
        leave=progress_leave,
    ):
        split = split_by_image.get(id(image))
        _increment_split_count(image_counts, split)
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            if class_name not in class_counts:
                class_counts[class_name] = _empty_split_counts()
                class_order.append(class_name)
            _increment_split_count(class_counts[class_name], split)

            for attribute_name, attribute_value in dataset.annotation_attributes(annotation).items():
                key = (attribute_name, str(attribute_value))
                if key not in attribute_counts:
                    attribute_counts[key] = _empty_split_counts()
                    attribute_order.append(key)
                _increment_split_count(attribute_counts[key], split)

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "section": "image",
            "class_name": "image",
            "attribute": "",
            "value": "",
            **image_counts,
        }
    )
    for class_name in class_order:
        counts = class_counts[class_name]
        rows.append(
            {
                "section": "box",
                "class_name": class_name,
                "attribute": "",
                "value": "",
                **counts,
            }
        )
    for attribute_name, attribute_value in attribute_order:
        rows.append(
            {
                "section": "attribute",
                "class_name": "",
                "attribute": attribute_name,
                "value": attribute_value,
                **attribute_counts[(attribute_name, attribute_value)],
            }
        )
    return rows


def write_basic_info_csv(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write compact image, box, and attribute counts to one CSV file."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(BASIC_INFO_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def format_basic_info_tables(rows: Iterable[dict[str, object]]) -> str:
    """Format the compact statistics as terminal-friendly tables."""

    row_list = list(rows)
    image_rows = [
        [row["class_name"], row["total"], row["train"], row["val"], row["test"]]
        for row in row_list
        if row.get("section") == "image"
    ]
    box_rows = [
        [row["class_name"], row["total"], row["train"], row["val"], row["test"]]
        for row in row_list
        if row.get("section") == "box"
    ]
    attribute_rows = [
        [
            row["attribute"],
            row["value"],
            row["total"],
            row["train"],
            row["val"],
            row["test"],
        ]
        for row in row_list
        if row.get("section") == "attribute"
    ]

    sections = [
        "Image counts",
        _format_table(
            ["scope", "total", "train", "val", "test"],
            image_rows,
        ),
        "Box counts",
        _format_table(
            ["class_name", "total", "train", "val", "test"],
            box_rows,
        ),
        "Attribute counts",
    ]
    if attribute_rows:
        sections.append(
            _format_table(
                ["attribute", "value", "total", "train", "val", "test"],
                attribute_rows,
            )
        )
    else:
        sections.append("(no attributes)")
    return "\n".join(sections)


def infer_image_splits(dataset: YoloDataset) -> dict[int, str]:
    """Infer train/val/test membership for the images in *dataset*.

    Split directories and the conventional ``train.txt``/``val.txt``/
    ``test.txt`` lists are both supported. An image with ambiguous membership
    is left without a split assignment.
    """

    root = Path(dataset.root).resolve()
    assignments: dict[str, set[str]] = {}

    for split in SPLIT_NAMES:
        split_file = root / f"{split}.txt"
        if not split_file.is_file():
            continue
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            for key in _split_key_variants(root, raw_line):
                assignments.setdefault(key, set()).add(split)

    result: dict[int, str] = {}
    for image in dataset.images:
        candidates: set[str] = set()
        for key in _split_key_variants(root, image.path):
            candidates.update(assignments.get(key, set()))
        candidates.update(_split_names_from_image_path(root, image.path))
        if len(candidates) == 1:
            result[id(image)] = next(iter(candidates))
    return result


def _empty_split_counts() -> dict[str, int]:
    return {name: 0 for name in ("total", *SPLIT_NAMES)}


def _increment_split_count(counts: dict[str, int], split: str | None) -> None:
    counts["total"] += 1
    if split in SPLIT_NAMES:
        counts[split] += 1


def _format_table(headers: list[str], rows: list[list[object]]) -> str:
    values = [[str(value) for value in row] for row in rows]
    all_rows = [headers, *values]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in values
    )
    return "\n".join(lines)


def _split_key_variants(root: Path, value: str | Path) -> set[str]:
    text = str(value).strip()
    if not text:
        return set()

    path = Path(text)
    keys = {_normalise_split_key(text), _normalise_split_key(path.name)}
    if path.stem:
        keys.add(_normalise_split_key(path.stem))
    try:
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        keys.add(_normalise_split_key(resolved))
        keys.add(_normalise_split_key(resolved.relative_to(root)))
    except (OSError, ValueError):
        pass
    return {key for key in keys if key}


def _split_names_from_image_path(root: Path, image_path: Path) -> set[str]:
    try:
        relative = image_path.resolve().relative_to(root)
    except (OSError, ValueError):
        relative = image_path

    parts = [part.casefold() for part in relative.parts]
    candidates: set[str] = set()
    for index, part in enumerate(parts):
        if part not in SPLIT_NAMES:
            continue
        previous = parts[index - 1] if index > 0 else ""
        following = parts[index + 1] if index + 1 < len(parts) else ""
        if (
            index == len(parts) - 2
            or previous in {"images", "image"}
            or following in {"images", "image"}
        ):
            candidates.add(part)
    return candidates


def _normalise_split_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().rstrip("/").casefold()
