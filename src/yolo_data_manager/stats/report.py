from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def format_split_distribution_table(
    distribution: dict[str, dict[str, int]],
) -> str:
    """Format train/val/test image and annotation counts for the terminal."""

    split_names = ("train", "val", "test")
    rows = [
        [
            split_name,
            distribution.get(split_name, {}).get("images", 0),
            distribution.get(split_name, {}).get("label_files", 0),
            distribution.get(split_name, {}).get("annotations", 0),
        ]
        for split_name in split_names
    ]
    rows.append(
        [
            "total",
            sum(row[1] for row in rows),
            sum(row[2] for row in rows),
            sum(row[3] for row in rows),
        ]
    )
    return "Split distribution\n" + _format_table(
        ["split", "images", "label_files", "annotations"],
        rows,
    )


def snapshot_label_distribution(dataset: YoloDataset) -> dict[str, object]:
    """Capture class and split counts before a label-edit operation."""

    split_by_image = infer_image_splits(dataset)
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    image_splits: dict[str, str] = {}
    for image in dataset.images:
        split = split_by_image.get(id(image), "unassigned")
        split_counts[split] += len(image.annotations)
        for annotation in image.annotations:
            class_counts[dataset.class_name(annotation.class_id)] += 1
        for value in (
            image.file_name,
            image.stem,
            image.path,
            image.path.name,
            image.path.resolve(),
            image.label_path,
        ):
            if value is not None:
                image_splits.setdefault(_summary_key(value), split)

    for class_name in dataset.classes.names:
        class_counts.setdefault(class_name, 0)
    for split_name in ("train", "val", "test"):
        split_counts.setdefault(split_name, 0)
    return {
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "image_splits": image_splits,
    }


def build_label_edit_summary(
    before: dict[str, object],
    rows: Iterable[Any],
) -> dict[str, object]:
    """Build before/after class and split summaries from edit rows."""

    before_class_counts = Counter(
        {str(name): int(count) for name, count in dict(before["class_counts"]).items()}
    )
    after_class_counts = Counter(before_class_counts)
    before_split_counts = Counter(
        {str(name): int(count) for name, count in dict(before["split_counts"]).items()}
    )
    after_split_counts = Counter(before_split_counts)
    changed_classes: Counter[tuple[str, str, str]] = Counter()
    split_changes: dict[str, dict[str, int]] = {
        split_name: {
            "before": before_split_counts.get(split_name, 0),
            "after": after_split_counts.get(split_name, 0),
            "changed": 0,
            "added": 0,
            "deleted": 0,
        }
        for split_name in ("train", "val", "test")
    }
    image_splits = {
        str(key): str(value)
        for key, value in dict(before.get("image_splits", {})).items()
    }

    for row in rows:
        action = str(getattr(row, "action", None) or "update")
        old_name = str(getattr(row, "old_class_name", ""))
        new_value = getattr(row, "new_class_name", None)
        new_name = str(new_value) if new_value not in (None, "") else None
        if action == "add":
            from_name = "<none>"
            to_name = new_name or old_name
            after_class_counts[to_name] += 1
        elif action == "delete" or new_name is None:
            from_name = old_name
            to_name = "<deleted>"
            after_class_counts[old_name] -= 1
        else:
            from_name = old_name
            to_name = new_name
            after_class_counts[old_name] -= 1
            after_class_counts[new_name] += 1
        changed_classes[(action, from_name, to_name)] += 1

        split = _edit_row_split(row, image_splits)
        split_entry = split_changes.setdefault(
            split,
            {
                "before": before_split_counts.get(split, 0),
                "after": after_split_counts.get(split, 0),
                "changed": 0,
                "added": 0,
                "deleted": 0,
            },
        )
        if action == "add":
            split_entry["added"] += 1
            split_entry["after"] += 1
            after_split_counts[split] += 1
        elif action == "delete" or new_name is None:
            split_entry["deleted"] += 1
            split_entry["after"] -= 1
            after_split_counts[split] -= 1
        else:
            split_entry["changed"] += 1

    class_names = list(before_class_counts)
    for name in after_class_counts:
        if name not in class_names:
            class_names.append(name)
    changed_rows = [
        {
            "action": action,
            "from_class": from_name,
            "to_class": to_name,
            "count": count,
        }
        for (action, from_name, to_name), count in sorted(changed_classes.items())
    ]
    split_names = [name for name in ("train", "val", "test") if name in split_changes]
    split_names.extend(
        name for name in split_changes if name not in {"train", "val", "test"}
    )
    return {
        "before_class_counts": {
            name: before_class_counts.get(name, 0) for name in class_names
        },
        "after_class_counts": {
            name: after_class_counts.get(name, 0) for name in class_names
        },
        "changed_class_counts": changed_rows,
        "split_distribution": {
            name: split_changes[name] for name in split_names
        },
    }


def format_label_edit_summary(summary: dict[str, object]) -> str:
    """Format class and split distributions for crop-driven class updates."""

    before = dict(summary.get("before_class_counts", {}))
    after = dict(summary.get("after_class_counts", {}))
    changed = list(summary.get("changed_class_counts", []))
    splits = dict(summary.get("split_distribution", {}))
    sections = [
        "Before annotation class distribution",
        _format_table(
            ["class_name", "annotations"],
            [[name, count] for name, count in before.items()],
        ),
        "After annotation class distribution",
        _format_table(
            ["class_name", "annotations"],
            [[name, count] for name, count in after.items()],
        ),
        "Changed annotation class distribution",
        _format_table(
            ["action", "from_class", "to_class", "annotations"],
            [
                [
                    row.get("action", ""),
                    row.get("from_class", ""),
                    row.get("to_class", ""),
                    row.get("count", 0),
                ]
                for row in changed
            ],
        ),
        "Annotation split distribution",
        _format_table(
            ["split", "before", "after", "changed", "added", "deleted"],
            [
                [
                    split_name,
                    values.get("before", 0),
                    values.get("after", 0),
                    values.get("changed", 0),
                    values.get("added", 0),
                    values.get("deleted", 0),
                ]
                for split_name, values in splits.items()
            ],
        ),
    ]
    return "\n".join(sections)


def infer_image_splits(dataset: YoloDataset) -> dict[int, str]:
    """Infer train/val/test membership for the images in *dataset*.

    The conventional ``train.txt``/``val.txt``/``test.txt`` lists take
    precedence when any of them exists at the dataset root. Split directories
    are used only when none of those list files exists. An image with
    ambiguous membership is left without a split assignment.
    """

    root = Path(dataset.root).resolve()
    split_files = {
        split: root / f"{split}.txt"
        for split in SPLIT_NAMES
        if (root / f"{split}.txt").is_file()
    }
    assignments: dict[str, set[str]] = {}

    for split, split_file in split_files.items():
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            for key in _split_key_variants(root, raw_line):
                assignments.setdefault(key, set()).add(split)

    result: dict[int, str] = {}
    for image in dataset.images:
        candidates: set[str] = set()
        if image.split in SPLIT_NAMES:
            result[id(image)] = image.split
            continue
        for key in _split_key_variants(root, image.path):
            candidates.update(assignments.get(key, set()))
        if not split_files:
            candidates.update(_split_names_from_image_path(root, image.path))
        if len(candidates) == 1:
            result[id(image)] = next(iter(candidates))
    return result


def _edit_row_split(row: Any, image_splits: dict[str, str]) -> str:
    for value in (
        getattr(row, "image", None),
        getattr(row, "label_path", None),
    ):
        if value is None:
            continue
        split = image_splits.get(_summary_key(value))
        if split is not None:
            return split
        path = Path(str(value))
        for candidate in (path.name, path.stem):
            split = image_splits.get(_summary_key(candidate))
            if split is not None:
                return split
    return "unassigned"


def _summary_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").casefold()


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
