from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yolo_data_manager.annotation.edit import EditReport, delete_class, merge_classes, rename_class
from yolo_data_manager.core.models import YoloDataset


def apply_class_map(
    dataset: YoloDataset,
    map_file: str | Path,
    compact: bool = True,
) -> tuple[YoloDataset, list[EditReport]]:
    data: dict[str, Any] = yaml.safe_load(Path(map_file).read_text(encoding="utf-8")) or {}
    current = dataset
    reports: list[EditReport] = []

    for old_name, new_name in (data.get("rename") or {}).items():
        current, report = rename_class(current, old_name, str(new_name))
        reports.append(report)

    for target, sources in (data.get("merge") or {}).items():
        current, report = merge_classes(current, sources, target, compact=compact, add_missing=True)
        reports.append(report)

    drop = data.get("drop") or []
    if drop:
        current, report = delete_class(current, drop, compact=compact)
        reports.append(report)

    return current, reports

