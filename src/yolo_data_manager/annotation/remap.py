from __future__ import annotations

from collections.abc import Mapping
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
    return apply_class_map_data(dataset, data, compact=compact)


def apply_class_map_data(
    dataset: YoloDataset,
    class_map: Mapping[str, Any],
    compact: bool = True,
) -> tuple[YoloDataset, list[EditReport]]:
    """Apply a class-operation mapping without reading a YAML file.

    Supported operations are ``rename`` (old name to new name), ``merge``
    (target name to source names), and ``drop`` (names to remove). Operations
    are applied in that order so one mapping can combine them.
    """

    if not isinstance(class_map, Mapping):
        raise TypeError("class_map must be a mapping")

    data = dict(class_map)
    current = dataset
    reports: list[EditReport] = []

    rename_map = data.get("rename") or {}
    if not isinstance(rename_map, Mapping):
        raise TypeError("class_map['rename'] must be a mapping")
    for old_name, new_name in rename_map.items():
        current, report = rename_class(current, old_name, str(new_name))
        reports.append(report)

    merge_map = data.get("merge") or {}
    if not isinstance(merge_map, Mapping):
        raise TypeError("class_map['merge'] must be a mapping")
    for target, sources in merge_map.items():
        if isinstance(sources, (str, int)):
            sources = [sources]
        current, report = merge_classes(current, sources, target, compact=compact, add_missing=True)
        reports.append(report)

    drop = data.get("drop") or []
    if drop:
        if isinstance(drop, (str, int)):
            drop = [drop]
        current, report = delete_class(current, drop, compact=compact)
        reports.append(report)

    return current, reports

