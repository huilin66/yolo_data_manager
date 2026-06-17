from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.models import ClassSchema, YoloAnnotation, YoloDataset


@dataclass
class EditRow:
    operation: str
    image: str
    label_path: str
    line_no: int | None
    old_class_id: int
    old_class_name: str
    new_class_id: int | None = None
    new_class_name: str | None = None
    action: str = "update"


@dataclass
class EditReport:
    rows: list[EditRow] = field(default_factory=list)
    class_remap: dict[int, int] = field(default_factory=dict)

    def add(self, row: EditRow) -> None:
        self.rows.append(row)

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "operation": row.operation,
                "image": row.image,
                "label_path": row.label_path,
                "line_no": row.line_no or "",
                "old_class_id": row.old_class_id,
                "old_class_name": row.old_class_name,
                "new_class_id": row.new_class_id if row.new_class_id is not None else "",
                "new_class_name": row.new_class_name or "",
                "action": row.action,
            }
            for row in self.rows
        ]

    def write_csv(self, path: str | Path) -> None:
        rows = self.to_rows()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "operation",
            "image",
            "label_path",
            "line_no",
            "old_class_id",
            "old_class_name",
            "new_class_id",
            "new_class_name",
            "action",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def delete_class(
    dataset: YoloDataset,
    classes: Iterable[int | str],
    compact: bool = False,
) -> tuple[YoloDataset, EditReport]:
    result = copy.deepcopy(dataset)
    report = EditReport()
    target_ids = _resolve_class_ids(result, classes)

    for image in result.images:
        kept: list[YoloAnnotation] = []
        for annotation in image.annotations:
            if annotation.class_id in target_ids:
                report.add(
                    EditRow(
                        operation="delete_class",
                        image=image.file_name,
                        label_path=str(image.label_path) if image.label_path else "",
                        line_no=annotation.line_no,
                        old_class_id=annotation.class_id,
                        old_class_name=result.class_name(annotation.class_id),
                        action="delete",
                    )
                )
            else:
                kept.append(annotation)
        image.annotations = kept

    if compact:
        report.class_remap = _compact_classes(result, remove_ids=target_ids)
    return result, report


def replace_class(
    dataset: YoloDataset,
    from_classes: Iterable[int | str],
    to_class: int | str,
    compact: bool = False,
    add_missing: bool = True,
) -> tuple[YoloDataset, EditReport]:
    result = copy.deepcopy(dataset)
    report = EditReport()
    source_ids = _resolve_class_ids(result, from_classes)
    if add_missing:
        target_id = result.classes.ensure(to_class)
    else:
        target_id = result.class_id(to_class)

    for image in result.images:
        for annotation in image.annotations:
            if annotation.class_id in source_ids:
                old_id = annotation.class_id
                annotation.class_id = target_id
                report.add(
                    EditRow(
                        operation="replace_class",
                        image=image.file_name,
                        label_path=str(image.label_path) if image.label_path else "",
                        line_no=annotation.line_no,
                        old_class_id=old_id,
                        old_class_name=result.class_name(old_id),
                        new_class_id=target_id,
                        new_class_name=result.class_name(target_id),
                    )
                )

    if compact:
        remove_ids = {class_id for class_id in source_ids if class_id != target_id}
        report.class_remap = _compact_classes(result, remove_ids=remove_ids)
    return result, report


def merge_classes(
    dataset: YoloDataset,
    from_classes: Iterable[int | str],
    to_class: int | str,
    compact: bool = True,
    add_missing: bool = True,
) -> tuple[YoloDataset, EditReport]:
    merged, report = replace_class(
        dataset,
        from_classes=from_classes,
        to_class=to_class,
        compact=compact,
        add_missing=add_missing,
    )
    for row in report.rows:
        row.operation = "merge_class"
    return merged, report


def rename_class(
    dataset: YoloDataset,
    old_class: int | str,
    new_name: str,
) -> tuple[YoloDataset, EditReport]:
    result = copy.deepcopy(dataset)
    report = EditReport()
    old_id = result.class_id(old_class)
    old_name = result.class_name(old_id)
    result.classes = result.classes.renamed(old_id, new_name)
    for image in result.images:
        for annotation in image.annotations:
            if annotation.class_id == old_id:
                report.add(
                    EditRow(
                        operation="rename_class",
                        image=image.file_name,
                        label_path=str(image.label_path) if image.label_path else "",
                        line_no=annotation.line_no,
                        old_class_id=old_id,
                        old_class_name=old_name,
                        new_class_id=old_id,
                        new_class_name=new_name,
                        action="rename",
                    )
                )
    return result, report


def _resolve_class_ids(dataset: YoloDataset, classes: Iterable[int | str]) -> set[int]:
    return {dataset.class_id(value) for value in classes}


def _compact_classes(dataset: YoloDataset, remove_ids: set[int]) -> dict[int, int]:
    old_names = list(dataset.classes.names)
    mapping: dict[int, int] = {}
    new_names: list[str] = []
    for old_id, name in enumerate(old_names):
        if old_id in remove_ids:
            continue
        mapping[old_id] = len(new_names)
        new_names.append(name)

    for image in dataset.images:
        for annotation in image.annotations:
            if annotation.class_id in mapping:
                annotation.class_id = mapping[annotation.class_id]
    dataset.classes = ClassSchema(new_names)
    return mapping

