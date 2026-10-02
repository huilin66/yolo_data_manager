from __future__ import annotations

import copy
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
    attr_name: str | None = None
    old_attr_value: object | None = None
    new_attr_value: object | None = None


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
                "attr_name": row.attr_name or "",
                "old_attr_value": row.old_attr_value if row.old_attr_value is not None else "",
                "new_attr_value": row.new_attr_value if row.new_attr_value is not None else "",
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
            "attr_name",
            "old_attr_value",
            "new_attr_value",
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


def set_attribute(
    dataset: YoloDataset,
    attribute_name: str,
    value: str | int | float,
    classes: Iterable[int | str] | None = None,
    where_value: str | int | float | None = None,
) -> tuple[YoloDataset, EditReport]:
    result = copy.deepcopy(dataset)
    report = EditReport()
    if result.attributes is None or not _has_attribute(result, attribute_name):
        return result, report

    class_ids = _resolve_class_ids(result, classes) if classes is not None else None

    for image in result.images:
        for annotation in image.annotations:
            if class_ids is not None and annotation.class_id not in class_ids:
                continue
            class_name = result.class_name(annotation.class_id)
            attr_names = result.attributes.names_for_class(class_name)
            if attribute_name not in attr_names:
                continue
            attr_idx = attr_names.index(attribute_name)
            new_raw = result.attributes.value_to_raw(attribute_name, value, class_name=class_name)
            where_raw = result.attributes.value_to_raw(attribute_name, where_value, class_name=class_name) if where_value is not None else None
            _ensure_attribute_len(annotation, attr_idx + 1)
            old_raw = annotation.attributes[attr_idx]
            if where_raw is not None and old_raw != where_raw:
                continue
            annotation.attributes[attr_idx] = new_raw
            report.add(
                EditRow(
                    operation="set_attribute",
                    image=image.file_name,
                    label_path=str(image.label_path) if image.label_path else "",
                    line_no=annotation.line_no,
                    old_class_id=annotation.class_id,
                    old_class_name=result.class_name(annotation.class_id),
                    action="set_attribute",
                    attr_name=attribute_name,
                    old_attr_value=old_raw,
                    new_attr_value=new_raw,
                )
            )
    return result, report


def set_attributes_from_map(
    dataset: YoloDataset,
    attribute_map: Mapping[str, Any],
) -> tuple[YoloDataset, EditReport]:
    """Apply several attribute updates from a Python mapping.

    The canonical form is::

        {
            "update": {
                "defect": {"no": "yes", "unknown": "yes"},
                "material": {"value": "metal", "class": ["sign"]},
            }
        }

    A rule may also be written as ``{"name": ..., "value": ..., ...}``,
    with optional ``where_value``/``from`` and ``class``/``class_`` fields.
    The returned dataset is a copy and the report combines all operations.
    """

    operations = _attribute_update_operations(attribute_map)
    current = dataset
    combined = EditReport()
    for name, value, classes, where_value in operations:
        current, report = set_attribute(
            current,
            name,
            value,
            classes=classes,
            where_value=where_value,
        )
        combined.rows.extend(report.rows)
    return current, combined


def _attribute_update_operations(
    attribute_map: Mapping[str, Any],
) -> list[tuple[str, Any, list[str] | None, Any]]:
    if not isinstance(attribute_map, Mapping):
        raise TypeError("attribute_map must be a mapping")

    payload: Any = attribute_map
    for key in ("update", "set"):
        if key in attribute_map:
            payload = attribute_map[key]
            break

    if isinstance(payload, Mapping):
        if _looks_like_attribute_rule(payload):
            return _attribute_rule_operations(payload)

        operations: list[tuple[str, Any, list[str] | None, Any]] = []
        for name, spec in payload.items():
            operations.extend(_attribute_name_operations(str(name), spec))
        return operations

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        operations = []
        for rule in payload:
            if not isinstance(rule, Mapping):
                raise TypeError("attribute update list entries must be mappings")
            operations.extend(_attribute_rule_operations(rule))
        return operations

    raise TypeError("attribute_map['update'] must be a mapping or list")


def _looks_like_attribute_rule(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in {
            "name",
            "attribute",
            "attribute_name",
            "value",
            "to",
            "target",
            "target_value",
            "where_value",
            "from",
            "old",
            "class",
            "class_",
            "classes",
        }
    )


def _attribute_rule_operations(
    rule: Mapping[str, Any],
    *,
    default_name: str | None = None,
) -> list[tuple[str, Any, list[str] | None, Any]]:
    name = rule.get("name", rule.get("attribute_name", rule.get("attribute", default_name)))
    if name is None:
        raise ValueError("attribute update rule requires name")

    value_key = next(
        (key for key in ("value", "to", "target_value", "target") if key in rule),
        None,
    )
    if value_key is None:
        raise ValueError(f"attribute update rule for {name!r} requires value/to")
    value = rule[value_key]
    where = rule.get("where_value", rule.get("from", rule.get("old")))
    classes = rule.get("class_", rule.get("class", rule.get("classes")))
    if isinstance(classes, str):
        class_values = [classes]
    elif classes is None:
        class_values = None
    else:
        class_values = [str(item) for item in classes]

    where_values = (
        list(where)
        if isinstance(where, Sequence) and not isinstance(where, (str, bytes, bytearray))
        else [where]
    )
    return [(str(name), value, class_values, old_value) for old_value in where_values]


def _attribute_name_operations(
    name: str,
    spec: Any,
) -> list[tuple[str, Any, list[str] | None, Any]]:
    if isinstance(spec, Mapping):
        if _looks_like_attribute_rule(spec):
            return _attribute_rule_operations(spec, default_name=name)
        operations = []
        for old_value, new_value in spec.items():
            operations.append((name, new_value, None, old_value))
        return operations

    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
        if len(spec) != 2:
            raise ValueError(
                f"attribute mapping for {name!r} must be [old_value, new_value]"
            )
        return [(name, spec[1], None, spec[0])]

    return [(name, spec, None, None)]


def delete_by_attribute(
    dataset: YoloDataset,
    attribute_name: str,
    values: Iterable[str] | None = None,
    nonzero: bool = False,
) -> tuple[YoloDataset, EditReport]:
    result = copy.deepcopy(dataset)
    report = EditReport()
    if result.attributes is None or not _has_attribute(result, attribute_name):
        return result, report

    for image in result.images:
        kept: list[YoloAnnotation] = []
        for annotation in image.annotations:
            class_name = result.class_name(annotation.class_id)
            attr_names = result.attributes.names_for_class(class_name)
            if attribute_name not in attr_names:
                kept.append(annotation)
                continue
            attr_idx = attr_names.index(attribute_name)
            raw_values = {result.attributes.value_to_raw(attribute_name, value, class_name=class_name) for value in values} if values is not None else None
            raw = annotation.attributes[attr_idx] if attr_idx < len(annotation.attributes) else 0
            matched = (nonzero and float(raw) != 0) or (raw_values is not None and raw in raw_values)
            if matched:
                report.add(
                    EditRow(
                        operation="delete_by_attribute",
                        image=image.file_name,
                        label_path=str(image.label_path) if image.label_path else "",
                        line_no=annotation.line_no,
                        old_class_id=annotation.class_id,
                        old_class_name=result.class_name(annotation.class_id),
                        action="delete",
                        attr_name=attribute_name,
                        old_attr_value=raw,
                    )
                )
            else:
                kept.append(annotation)
        image.annotations = kept
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


def _ensure_attribute_len(annotation: YoloAnnotation, length: int) -> None:
    while len(annotation.attributes) < length:
        annotation.attributes.append(0.0)


def _has_attribute(dataset: YoloDataset, attribute_name: str) -> bool:
    if dataset.attributes is None:
        return False
    if attribute_name in dataset.attributes.names:
        return True
    return any(attribute_name in dataset.attributes.names_for_class(class_name) for class_name in dataset.classes.names)
