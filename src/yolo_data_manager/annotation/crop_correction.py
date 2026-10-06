"""Correct per-instance YOLO classes and attributes from visual crop filenames."""

from __future__ import annotations

from copy import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from yolo_data_manager.core.models import YoloAnnotation, YoloDataset, YoloImage, is_image_file
from yolo_data_manager.annotation.edit import EditReport, EditRow
from yolo_data_manager.io.loader import parse_label_file
from yolo_data_manager.io.backup import LabelBackup, ensure_source_labels_backup
from yolo_data_manager.io.writer import clone_yolo_dataset_for_output


_CROP_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<index>[1-9][0-9]*)$")
_ERROR_CROP_NAME_RE = re.compile(
    r"^(?P<stem>.+)_pred(?P<pred>none|[1-9][0-9]*)_gt(?P<gt>none|[1-9][0-9]*)$"
)
_ATTRIBUTE_ERROR_CROP_NAME_RE = re.compile(
    r"^(?P<stem>.+)_pred(?P<pred>none|[1-9][0-9]*)_gt(?P<gt>none|[1-9][0-9]*)(?:_(?P<attribute>.+))?$"
)


@dataclass
class CropCorrectionResult:
    """Summary of a crop-driven class correction operation."""

    target_class_id: int | None
    target_class_name: str | None
    crop_files: int = 0
    unique_targets: int = 0
    changed: int = 0
    added: int = 0
    replaced: int = 0
    deduplicated: int = 0
    deleted: int = 0
    unchanged: int = 0
    duplicate_targets: int = 0
    invalid_crops: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    ambiguous_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    invalid_indices: list[str] = field(default_factory=list)
    missing_prediction_labels: list[str] = field(default_factory=list)
    ambiguous_prediction_labels: list[str] = field(default_factory=list)
    invalid_prediction_labels: list[str] = field(default_factory=list)
    invalid_prediction_indices: list[str] = field(default_factory=list)
    backup_dir: str | None = None
    backup_timestamp: str | None = None
    backup_files: int = 0
    target_classes: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "target_classes": self.target_classes,
            "crop_files": self.crop_files,
            "unique_targets": self.unique_targets,
            "changed": self.changed,
            "added": self.added,
            "replaced": self.replaced,
            "deduplicated": self.deduplicated,
            "deleted": self.deleted,
            "unchanged": self.unchanged,
            "duplicate_targets": self.duplicate_targets,
            "invalid_crops": self.invalid_crops,
            "missing_images": self.missing_images,
            "ambiguous_images": self.ambiguous_images,
            "missing_labels": self.missing_labels,
            "invalid_indices": self.invalid_indices,
            "missing_prediction_labels": self.missing_prediction_labels,
            "ambiguous_prediction_labels": self.ambiguous_prediction_labels,
            "invalid_prediction_labels": self.invalid_prediction_labels,
            "invalid_prediction_indices": self.invalid_prediction_indices,
            "backup_dir": self.backup_dir,
            "backup_timestamp": self.backup_timestamp,
            "backup_files": self.backup_files,
            "skipped": (
                len(self.invalid_crops)
                + len(self.missing_images)
                + len(self.ambiguous_images)
                + len(self.missing_labels)
                + len(self.invalid_indices)
                + len(self.missing_prediction_labels)
                + len(self.ambiguous_prediction_labels)
                + len(self.invalid_prediction_labels)
                + len(self.invalid_prediction_indices)
                + self.deduplicated
            ),
        }


@dataclass
class AttributeCropCorrectionResult:
    """Summary of an attribute update driven by annotation crops."""

    attribute_name: str
    target_value: str | int | float | None
    crop_files: int = 0
    unique_targets: int = 0
    changed: int = 0
    unchanged: int = 0
    duplicate_targets: int = 0
    invalid_crops: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    ambiguous_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    invalid_indices: list[str] = field(default_factory=list)
    invalid_attributes: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    backup_dir: str | None = None
    backup_timestamp: str | None = None
    backup_files: int = 0
    attribute_updates: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "attribute_name": self.attribute_name,
            "target_value": self.target_value,
            "crop_files": self.crop_files,
            "unique_targets": self.unique_targets,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "duplicate_targets": self.duplicate_targets,
            "invalid_crops": self.invalid_crops,
            "missing_images": self.missing_images,
            "ambiguous_images": self.ambiguous_images,
            "missing_labels": self.missing_labels,
            "invalid_indices": self.invalid_indices,
            "invalid_attributes": self.invalid_attributes,
            "invalid_values": self.invalid_values,
            "backup_dir": self.backup_dir,
            "backup_timestamp": self.backup_timestamp,
            "backup_files": self.backup_files,
            "attribute_updates": self.attribute_updates,
            "skipped": (
                len(self.invalid_crops)
                + len(self.missing_images)
                + len(self.ambiguous_images)
                + len(self.missing_labels)
                + len(self.invalid_indices)
                + len(self.invalid_attributes)
                + len(self.invalid_values)
                + self.duplicate_targets
            ),
        }


def correct_labels_from_crops(
    dataset: YoloDataset,
    crops_dir: str | Path | Mapping[str | Path, int | str | None],
    target_class: int | str | None = None,
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[CropCorrectionResult, EditReport]:
    """Update label classes identified by standard ``vis crop`` filenames.

    A crop named ``image_stem_3.jpg`` maps to the third annotation in
    ``image_stem.txt``. The crop directory is searched recursively, so both
    class folders and ``by_attribute`` subfolders are supported. When
    ``target_class`` is ``None``, the mapped annotation line is deleted.
    ``crops_dir`` may alternatively be a mapping of crop directories to
    target classes; that form is processed in one backup session.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_labels_from_crops(
            output_dataset,
            crops_dir,
            target_class,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if isinstance(crops_dir, Mapping):
        if target_class is not None:
            raise ValueError("target_class must be omitted when crops_dir is a mapping")
        return correct_labels_from_crop_map(
            dataset,
            crops_dir,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if not dry_run:
        _ensure_crop_source_backup(dataset, backup_dir)
    backup = (
        LabelBackup(dataset.root, backup_dir, method="ann.correct_from_crops")
        if not dry_run
        else None
    )
    result, edit_report = _correct_labels_from_crop_dir(
        dataset,
        crops_dir,
        target_class,
        backup=backup,
        dry_run=dry_run,
    )
    _set_crop_backup_info(result, backup, method="ann.correct_from_crops")
    return result, edit_report


def correct_labels_from_crop_map(
    dataset: YoloDataset,
    crops_to_classes: Mapping[str | Path, int | str | None],
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[CropCorrectionResult, EditReport]:
    """Apply several crop-directory class corrections in one backup session.

    ``crops_to_classes`` maps each standard ``vis crop`` directory to the
    target class for that directory. All directories are scanned before the
    label files are rewritten, so a label file is backed up at most once for
    the complete operation. If the same crop target occurs in more than one
    directory, the last mapping entry wins.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_labels_from_crop_map(
            output_dataset,
            crops_to_classes,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if not crops_to_classes:
        raise ValueError("crops_to_classes must contain at least one crop directory")

    if not dry_run:
        _ensure_crop_source_backup(dataset, backup_dir)
    backup = (
        LabelBackup(dataset.root, backup_dir, method="ann.correct_from_crops")
        if not dry_run
        else None
    )
    targets: dict[tuple[str, int], list[Path]] = {}
    target_class_by_target: dict[tuple[str, int], int | None] = {}
    target_specs: dict[str, dict[str, object]] = {}
    crop_files = 0
    invalid_crops: list[str] = []

    for raw_crop_dir, raw_target in crops_to_classes.items():
        crop_root = Path(raw_crop_dir)
        if not crop_root.is_dir():
            raise FileNotFoundError(f"crop directory not found: {crop_root}")

        target_value = _normalise_optional_target(raw_target)
        target_id = dataset.class_id(target_value) if target_value is not None else None
        target_specs[str(crop_root)] = {
            "target_class_id": target_id,
            "target_class_name": (
                dataset.class_name(target_id) if target_id is not None else None
            ),
        }

        for crop_path in sorted(crop_root.rglob("*")):
            if not crop_path.is_file() or not is_image_file(crop_path):
                continue
            crop_files += 1
            parsed = _parse_crop_name(crop_path)
            if parsed is None:
                invalid_crops.append(str(crop_path))
                continue
            targets.setdefault(parsed, []).append(crop_path)
            target_class_by_target[parsed] = target_id

    result, edit_report = _correct_target_map(
        dataset,
        targets,
        None,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        image_key=lambda image: image.stem,
        target_class_by_target=target_class_by_target,
        backup=backup,
        dry_run=dry_run,
    )
    result.target_classes = target_specs
    _set_crop_backup_info(result, backup, method="ann.correct_from_crops")
    return result, edit_report


def _correct_labels_from_crop_dir(
    dataset: YoloDataset,
    crops_dir: str | Path,
    target_class: int | str | None,
    *,
    backup: LabelBackup | None,
    dry_run: bool,
) -> tuple[CropCorrectionResult, EditReport]:
    crop_root = Path(crops_dir)
    if not crop_root.is_dir():
        raise FileNotFoundError(f"crop directory not found: {crop_root}")

    targets: dict[tuple[str, int], list[Path]] = {}
    crop_files = 0
    invalid_crops: list[str] = []
    for crop_path in sorted(crop_root.rglob("*")):
        if not crop_path.is_file() or not is_image_file(crop_path):
            continue
        crop_files += 1
        parsed = _parse_crop_name(crop_path)
        if parsed is None:
            invalid_crops.append(str(crop_path))
            continue
        targets.setdefault(parsed, []).append(crop_path)

    result, edit_report = _correct_target_map(
        dataset,
        targets,
        target_class,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        image_key=lambda image: image.stem,
        backup=backup,
        dry_run=dry_run,
    )
    return result, edit_report


def _set_crop_backup_info(
    result: CropCorrectionResult,
    backup: LabelBackup | None,
    *,
    method: str,
) -> None:
    if backup is not None and backup.count:
        result.backup_dir = str(backup.snapshot_dir)
        result.backup_timestamp = backup.timestamp
        result.backup_files = backup.count
        backup.write_metadata(
            method=method,
            result=_backup_result_summary(result.to_dict()),
        )


def _ensure_crop_source_backup(
    dataset: YoloDataset,
    backup_dir: str | Path | None,
) -> None:
    ensure_source_labels_backup(
        dataset.root,
        backup_dir,
        source_paths=(
            image.label_path
            for image in dataset.images
            if image.label_path is not None
        ),
        extra_paths=(
            Path(dataset.root) / name
            for name in ("class.txt", "dataset.yaml", "attribute.yaml")
        ),
    )


def _backup_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep backup metadata useful without duplicating long filename lists."""

    summary: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"backup_dir", "backup_timestamp", "backup_files"}:
            continue
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        else:
            summary[key] = value
    return summary


def _normalise_optional_target(value: int | str | None) -> int | str | None:
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def correct_gt_labels_from_error_crops(
    dataset: YoloDataset,
    crops_dir: str | Path | Mapping[str | Path, int | str | None],
    target_class: int | str | None = None,
    *,
    pred_labels_dir: str | Path | None = None,
    dedup_iou: float | None = 0.5,
    delete_pred_none: bool = False,
    replace_gt_from_pred: bool = False,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[CropCorrectionResult, EditReport]:
    """Correct GT classes from ``eval_error_analysis`` crop filenames.

    A crop named ``image_stem_pred2_gt3.jpg`` maps to the third GT
    annotation in ``image_stem.txt``. By default, the prediction index is
    retained in the filename for review context but is not needed for the GT
    class update.
    When ``pred_labels_dir`` is supplied, a crop with ``gt none`` appends the
    corresponding prediction annotation selected by ``predx`` to the GT
    label. Prediction confidence is omitted from the appended GT line.
    When ``delete_pred_none`` is true, ``prednone_gty`` crops force deletion
    of GT annotation ``y`` even if ``target_class`` is otherwise an update
    class.
    When ``replace_gt_from_pred`` is true, ``predx_gty`` crops replace the
    complete GT line with prediction ``x`` (class and geometry), while
    ``prednone_gty`` crops are deleted and ``predx_gtnone`` crops are appended.
    ``crops_dir`` may alternatively be a mapping of error-analysis crop
    directories to target classes; all directories are processed in one
    backup session.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_gt_labels_from_error_crops(
            output_dataset,
            crops_dir,
            target_class,
            pred_labels_dir=pred_labels_dir,
            dedup_iou=dedup_iou,
            delete_pred_none=delete_pred_none,
            replace_gt_from_pred=replace_gt_from_pred,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if isinstance(crops_dir, Mapping):
        if target_class is not None:
            raise ValueError("target_class must be omitted when crops_dir is a mapping")
        return correct_gt_labels_from_error_crop_map(
            dataset,
            crops_dir,
            pred_labels_dir=pred_labels_dir,
            dedup_iou=dedup_iou,
            delete_pred_none=delete_pred_none,
            replace_gt_from_pred=replace_gt_from_pred,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    return _correct_gt_labels_from_error_crop_specs(
        dataset,
        [(Path(crops_dir), None)],
        target_class=target_class,
        pred_labels_dir=pred_labels_dir,
        dedup_iou=dedup_iou,
        delete_pred_none=delete_pred_none,
        replace_gt_from_pred=replace_gt_from_pred,
        backup_dir=backup_dir,
        dry_run=dry_run,
        operation="ann.correct_from_error_crops",
    )


def correct_gt_labels_from_error_crop_map(
    dataset: YoloDataset,
    crops_to_classes: Mapping[str | Path, int | str | None],
    *,
    pred_labels_dir: str | Path | None = None,
    dedup_iou: float | None = 0.5,
    delete_pred_none: bool = False,
    replace_gt_from_pred: bool = False,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[CropCorrectionResult, EditReport]:
    """Apply several error-crop class corrections in one backup session.

    ``crops_to_classes`` maps each ``eval_error_analysis`` crop directory to
    the class assigned to its selected GT boxes. ``None`` deletes the selected
    GT box. Prediction-backed append/replace behavior remains controlled by
    ``pred_labels_dir`` and the other correction options.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_gt_labels_from_error_crop_map(
            output_dataset,
            crops_to_classes,
            pred_labels_dir=pred_labels_dir,
            dedup_iou=dedup_iou,
            delete_pred_none=delete_pred_none,
            replace_gt_from_pred=replace_gt_from_pred,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if not crops_to_classes:
        raise ValueError("crops_to_classes must contain at least one crop directory")

    crop_specs: list[tuple[Path, int | None]] = []
    target_class_by_target: dict[tuple[str, int], int | None] = {}
    target_specs: dict[str, dict[str, object]] = {}
    for raw_crop_dir, raw_target in crops_to_classes.items():
        crop_root = Path(raw_crop_dir)
        if not crop_root.is_dir():
            raise FileNotFoundError(
                f"error-analysis crop directory not found: {crop_root}"
            )
        target_value = _normalise_optional_target(raw_target)
        target_id = (
            dataset.class_id(target_value) if target_value is not None else None
        )
        crop_specs.append((crop_root, target_id))
        target_specs[str(crop_root)] = {
            "target_class_id": target_id,
            "target_class_name": (
                dataset.class_name(target_id) if target_id is not None else None
            ),
        }

    return _correct_gt_labels_from_error_crop_specs(
        dataset,
        crop_specs,
        target_class=None,
        target_class_by_target=target_class_by_target,
        target_specs=target_specs,
        pred_labels_dir=pred_labels_dir,
        dedup_iou=dedup_iou,
        delete_pred_none=delete_pred_none,
        replace_gt_from_pred=replace_gt_from_pred,
        backup_dir=backup_dir,
        dry_run=dry_run,
        operation="ann.correct_from_error_crops",
    )


def _correct_gt_labels_from_error_crop_specs(
    dataset: YoloDataset,
    crop_specs: list[tuple[Path, int | None]],
    *,
    target_class: int | str | None,
    target_class_by_target: dict[tuple[str, int], int | None] | None = None,
    target_specs: dict[str, dict[str, object]] | None = None,
    pred_labels_dir: str | Path | None,
    dedup_iou: float | None,
    delete_pred_none: bool,
    replace_gt_from_pred: bool,
    backup_dir: str | Path | None,
    dry_run: bool,
    operation: str = "ann.correct_from_error_crops",
) -> tuple[CropCorrectionResult, EditReport]:

    if dedup_iou is not None and not 0.0 < float(dedup_iou) <= 1.0:
        raise ValueError("dedup_iou must be between 0 and 1, or None to disable deduplication")

    targets: dict[tuple[str, int], list[Path]] = {}
    prediction_targets: dict[tuple[str, int], list[Path]] = {}
    replacement_targets: dict[tuple[str, int], list[tuple[int, Path]]] = {}
    forced_delete_targets: set[tuple[str, int]] = set()
    crop_files = 0
    invalid_crops: list[str] = []
    for crop_root, mapped_target_id in crop_specs:
        if not crop_root.is_dir():
            raise FileNotFoundError(
                f"error-analysis crop directory not found: {crop_root}"
            )
        for crop_path in sorted(crop_root.rglob("*")):
            if not crop_path.is_file() or not is_image_file(crop_path):
                continue
            crop_files += 1
            parsed = _parse_error_crop_name(crop_path)
            if parsed is None:
                invalid_crops.append(str(crop_path))
                continue
            stem, pred_index, gt_index = parsed
            if gt_index is None:
                if pred_index is None:
                    invalid_crops.append(str(crop_path))
                else:
                    prediction_targets.setdefault((stem, pred_index), []).append(crop_path)
                continue
            if replace_gt_from_pred and pred_index is not None:
                replacement_targets.setdefault((stem, gt_index), []).append(
                    (pred_index, crop_path)
                )
                continue
            if (delete_pred_none or replace_gt_from_pred) and pred_index is None:
                forced_delete_targets.add((stem, gt_index))
            targets.setdefault((stem, gt_index), []).append(crop_path)
            if target_class_by_target is not None:
                target_class_by_target[(stem, gt_index)] = mapped_target_id

    if not dry_run:
        _ensure_crop_source_backup(dataset, backup_dir)
    backup = (
        LabelBackup(dataset.root, backup_dir, method=operation)
        if not dry_run
        else None
    )
    result, edit_report = _correct_target_map(
        dataset,
        targets,
        target_class,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        image_key=lambda image: _safe_file_name(image.stem),
        target_class_by_target=target_class_by_target,
        replacement_targets=replacement_targets,
        pred_labels_dir=pred_labels_dir,
        dedup_iou=dedup_iou,
        backup=backup,
        forced_delete_targets=forced_delete_targets,
        dry_run=dry_run,
    )
    if target_specs is not None:
        result.target_classes = target_specs
    result.unique_targets += len(prediction_targets)
    result.duplicate_targets += sum(
        max(0, len(paths) - 1) for paths in prediction_targets.values()
    )
    _append_prediction_targets(
        dataset,
        prediction_targets,
        pred_labels_dir,
        result,
        edit_report,
        dedup_iou=dedup_iou,
        backup=backup,
        dry_run=dry_run,
    )
    if backup is not None:
        if backup.count:
            result.backup_dir = str(backup.snapshot_dir)
            result.backup_timestamp = backup.timestamp
            result.backup_files = backup.count
            backup.write_metadata(
                method=operation,
                result=_backup_result_summary(result.to_dict()),
            )
    return result, edit_report


def correct_gt_attributes_from_crops(
    dataset: YoloDataset,
    crops_dir: str | Path | Mapping[str | Path, Any],
    attribute_name: str | None = None,
    target_value: str | int | float | None = None,
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[AttributeCropCorrectionResult, EditReport]:
    """Update GT attributes selected by standard ``vis crop`` filenames.

    A crop named ``image_stem_3.jpg`` maps to the third GT annotation in
    ``image_stem.txt``.  The crop directory is searched recursively, so
    attribute folders can be used to organize manually selected crops.
    ``crops_dir`` may also be a mapping from crop directories to attribute
    rules, for example ``{"crops_yes": {"name": "defect", "value": "yes"}}``.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_gt_attributes_from_crops(
            output_dataset,
            crops_dir,
            attribute_name,
            target_value,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if isinstance(crops_dir, Mapping):
        specs = _attribute_crop_specs_from_map(
            crops_dir,
            default_name=attribute_name,
            default_value=target_value,
        )
    else:
        if attribute_name is None or target_value is None:
            raise ValueError("attribute_name and target_value are required")
        specs = [(Path(crops_dir), str(attribute_name).strip(), target_value)]

    return _correct_gt_attributes_from_error_crop_specs(
        dataset,
        specs,
        backup_dir=backup_dir,
        dry_run=dry_run,
        crop_parser=_parse_crop_name,
        require_attribute_suffix=False,
        operation="correct_attribute_from_crops",
        backup_method="ann.att_correct_from_crops",
    )


def correct_gt_attributes_from_error_crops(
    dataset: YoloDataset,
    crops_dir: str | Path | Mapping[str | Path, Any],
    attribute_name: str | None = None,
    target_value: str | int | float | None = None,
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    out_data: str | Path | None = None,
) -> tuple[AttributeCropCorrectionResult, EditReport]:
    """Update GT attributes on boxes selected by error-analysis crops.

    A crop named ``image_stem_pred2_gt3_defect.jpg`` maps to the third GT
    annotation in ``image_stem.txt``.  The ``pred`` index and the trailing
    attribute suffix are retained as review context; only ``gt3`` is used to
    select the annotation.  The selected annotation's attribute is updated
    while its class and geometry remain unchanged.

    The crop directory is searched recursively, so a complete
    ``attribute_<name>/gt_<value>_pred_<value>/crops`` directory or a folder
    containing only manually selected crop files can be supplied.  ``crops_dir``
    may also be a mapping from crop directories to attribute rules, for
    example ``{"crops_yes": {"name": "defect", "value": "yes"}}``.
    """

    if out_data is not None and not dry_run:
        output_dataset = clone_yolo_dataset_for_output(dataset, out_data)
        return correct_gt_attributes_from_error_crops(
            output_dataset,
            crops_dir,
            attribute_name,
            target_value,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )

    if isinstance(crops_dir, Mapping):
        specs = _attribute_crop_specs_from_map(
            crops_dir,
            default_name=attribute_name,
            default_value=target_value,
        )
    else:
        if attribute_name is None or target_value is None:
            raise ValueError("attribute_name and target_value are required")
        specs = [(Path(crops_dir), str(attribute_name).strip(), target_value)]

    return _correct_gt_attributes_from_error_crop_specs(
        dataset,
        specs,
        backup_dir=backup_dir,
        dry_run=dry_run,
        crop_parser=_parse_attribute_error_crop_name,
        require_attribute_suffix=True,
        backup_method="ann.att_correct_from_error_crops",
    )


def _attribute_crop_specs_from_map(
    crops_to_attributes: Mapping[str | Path, Any],
    *,
    default_name: str | None,
    default_value: str | int | float | None,
) -> list[tuple[Path, str, str | int | float]]:
    if not crops_to_attributes:
        raise ValueError("crops_dir mapping must contain at least one crop directory")

    specs: list[tuple[Path, str, str | int | float]] = []
    for raw_crop_dir, rule in crops_to_attributes.items():
        name: Any = default_name
        value: Any = default_value
        if isinstance(rule, Mapping):
            name = rule.get(
                "name",
                rule.get("attribute_name", rule.get("attribute", default_name)),
            )
            value = rule.get(
                "value",
                rule.get("to", rule.get("attribute_value", default_value)),
            )
            if len(rule) == 1 and not any(
                key in rule
                for key in {
                    "name",
                    "attribute_name",
                    "attribute",
                    "value",
                    "to",
                    "attribute_value",
                }
            ):
                mapped_name, mapped_value = next(iter(rule.items()))
                name = default_name or mapped_name
                value = mapped_value
        elif isinstance(rule, (list, tuple)) and len(rule) == 2:
            name, value = rule
        elif rule is not None:
            value = rule

        if name is None or value is None:
            raise ValueError(
                "attribute crop mapping values must provide name and value; "
                "use {'name': ..., 'value': ...}"
            )
        specs.append((Path(raw_crop_dir), str(name).strip(), value))
    return specs


def _correct_gt_attributes_from_error_crop_specs(
    dataset: YoloDataset,
    specs: list[tuple[Path, str, str | int | float]],
    *,
    backup_dir: str | Path | None,
    dry_run: bool,
    crop_parser: Any = None,
    require_attribute_suffix: bool = True,
    operation: str = "correct_attribute_from_error_crops",
    backup_method: str | None = None,
) -> tuple[AttributeCropCorrectionResult, EditReport]:
    """Apply multiple attribute crop rules in one backup session."""

    if crop_parser is None:
        crop_parser = _parse_attribute_error_crop_name

    if dataset.attributes is None:
        raise ValueError(
            "attribute schema is required; pass attribute_file when loading the dataset"
        )

    known_attributes = set(dataset.attributes.names)
    known_attributes.update(
        name
        for class_name in dataset.classes.names
        for name in dataset.attributes.names_for_class(class_name)
    )

    normalized_specs: list[tuple[Path, str, str | int | float]] = []
    for crop_root, raw_name, target in specs:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("attribute_name must not be empty")
        if name not in known_attributes:
            raise ValueError(f"attribute name not found: {name}")
        if not crop_root.is_dir():
            raise FileNotFoundError(
                f"attribute error crop directory not found: {crop_root}"
            )
        normalized_specs.append((crop_root, name, target))

    unique_names = {name for _path, name, _target in normalized_specs}
    result = AttributeCropCorrectionResult(
        attribute_name=(next(iter(unique_names)) if len(unique_names) == 1 else "multiple"),
        target_value=(
            normalized_specs[0][2]
            if len(normalized_specs) == 1
            else None
        ),
        attribute_updates=[
            {
                "crop_dir": str(crop_root),
                "attribute_name": name,
                "target_value": target,
            }
            for crop_root, name, target in normalized_specs
        ],
    )
    targets: dict[tuple[str, int, str, str | int | float], list[Path]] = {}
    crop_files = 0
    for crop_root, name, target in normalized_specs:
        for crop_path in sorted(crop_root.rglob("*")):
            if not crop_path.is_file() or not is_image_file(crop_path):
                continue
            crop_files += 1
            parsed = crop_parser(crop_path)
            if parsed is None:
                result.invalid_crops.append(str(crop_path))
                continue
            if len(parsed) == 2:
                stem, gt_index = parsed
                crop_attribute = None
            else:
                stem, _pred_index, gt_index, crop_attribute = parsed
            if gt_index is None or (
                require_attribute_suffix
                and crop_attribute != _safe_file_name(name)
            ):
                result.invalid_crops.append(str(crop_path))
                continue
            targets.setdefault((stem, gt_index, name, target), []).append(crop_path)

    result.crop_files = crop_files
    result.unique_targets = len(targets)
    result.duplicate_targets = sum(
        max(0, len(paths) - 1) for paths in targets.values()
    )
    edit_report = EditReport()
    image_candidates: dict[str, list[YoloImage]] = {}
    for image in dataset.images:
        image_candidates.setdefault(_safe_file_name(image.stem), []).append(image)

    pending: dict[Path, dict[int, str]] = {}
    working: dict[tuple[int, int], YoloAnnotation] = {}
    changed_annotations: dict[
        tuple[int, int], tuple[YoloImage, YoloAnnotation, YoloAnnotation, str]
    ] = {}
    target_items = sorted(
        targets.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            str(item[0][3]),
        ),
    )
    for (
        stem,
        crop_index,
        current_attribute_name,
        target_value,
    ), _crop_paths in target_items:
        candidates = image_candidates.get(stem, [])
        if not candidates:
            result.missing_images.append(stem)
            continue
        if len(candidates) > 1:
            result.ambiguous_images.append(stem)
            continue

        image = candidates[0]
        if image.label_path is None or not image.label_path.is_file():
            result.missing_labels.append(stem)
            continue

        annotation = _annotation_for_label_index(image, crop_index)
        if annotation is None:
            result.invalid_indices.append(f"{stem}_gt{crop_index}")
            continue

        class_name = dataset.class_name(annotation.class_id)
        attribute_names = dataset.attributes.names_for_class(class_name)
        if current_attribute_name not in attribute_names:
            result.invalid_attributes.append(
                f"{stem}_gt{crop_index}:{current_attribute_name}:{class_name}"
            )
            continue
        attribute_index = attribute_names.index(current_attribute_name)
        try:
            new_raw = dataset.attributes.value_to_raw(
                current_attribute_name,
                target_value,
                class_name=class_name,
            )
        except (TypeError, ValueError) as exc:
            result.invalid_values.append(
                f"{stem}_gt{crop_index}:{current_attribute_name}={target_value!s} ({exc})"
            )
            continue

        line_no = annotation.line_no or crop_index
        state_key = (id(image), line_no)
        current_annotation = working.get(state_key, annotation)
        old_raw = (
            current_annotation.attributes[attribute_index]
            if attribute_index < len(current_annotation.attributes)
            else 0.0
        )
        if old_raw == new_raw:
            result.unchanged += 1
            continue

        updated = copy(current_annotation)
        updated.attributes = list(current_annotation.attributes)
        _ensure_attribute_values(updated.attributes, attribute_index + 1)
        updated.attributes[attribute_index] = new_raw
        line = updated.to_yolo_line(include_confidence=updated.confidence is not None)
        working[state_key] = updated
        pending.setdefault(image.label_path, {})[line_no] = line
        changed_annotations[state_key] = (image, annotation, updated, line)
        edit_report.add(
            EditRow(
                operation=operation,
                image=image.file_name,
                label_path=str(image.label_path),
                line_no=line_no,
                old_class_id=annotation.class_id,
                old_class_name=class_name,
                action="set_attribute",
                attr_name=current_attribute_name,
                old_attr_value=old_raw,
                new_attr_value=new_raw,
            )
        )
        result.changed += 1

    resolved_backup_method = backup_method or operation
    if not dry_run:
        _ensure_crop_source_backup(dataset, backup_dir)
    backup = (
        LabelBackup(dataset.root, backup_dir, method=resolved_backup_method)
        if not dry_run
        else None
    )
    if not dry_run:
        for label_path, changes in pending.items():
            if backup is not None:
                backup.backup(label_path)
            _rewrite_label_annotations(label_path, list(changes.items()))
        for image, annotation, updated, line in changed_annotations.values():
            updated.source_line = line
            for index, current in enumerate(image.annotations):
                if current is annotation:
                    image.annotations[index] = updated
                    break

    if backup is not None and backup.count:
        result.backup_dir = str(backup.snapshot_dir)
        result.backup_timestamp = backup.timestamp
        result.backup_files = backup.count
        backup.write_metadata(
            method=resolved_backup_method,
            result=_backup_result_summary(result.to_dict()),
        )
    return result, edit_report


def _correct_target_map(
    dataset: YoloDataset,
    targets: dict[tuple[str, int], list[Path]],
    target_class: int | str | None,
    *,
    crop_files: int,
    invalid_crops: list[str],
    image_key,
    dry_run: bool,
    target_class_by_target: Mapping[tuple[str, int], int | None] | None = None,
    replacement_targets: dict[tuple[str, int], list[tuple[int, Path]]] | None = None,
    pred_labels_dir: str | Path | None = None,
    dedup_iou: float | None = None,
    backup: LabelBackup | None = None,
    forced_delete_targets: set[tuple[str, int]] | None = None,
) -> tuple[CropCorrectionResult, EditReport]:
    target_id = dataset.class_id(target_class) if target_class is not None else None
    if target_class_by_target is not None:
        target_ids = set(target_class_by_target.values())
        target_id = next(iter(target_ids)) if len(target_ids) == 1 else None
    target_name = dataset.class_name(target_id) if target_id is not None else None
    result = CropCorrectionResult(
        target_class_id=target_id,
        target_class_name=target_name,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        unique_targets=len(targets) + len(replacement_targets or {}),
        duplicate_targets=(
            sum(max(0, len(paths) - 1) for paths in targets.values())
            + sum(max(0, len(paths) - 1) for paths in (replacement_targets or {}).values())
        ),
    )
    edit_report = EditReport()

    image_candidates: dict[str, list[YoloImage]] = {}
    for image in dataset.images:
        image_candidates.setdefault(image_key(image), []).append(image)

    pending: dict[Path, list[tuple[int, int | None]]] = {}
    changed_annotations: list[tuple[YoloImage, YoloAnnotation, int | None]] = []
    forced_delete_targets = forced_delete_targets or set()
    _replace_target_map_from_predictions(
        dataset,
        replacement_targets or {},
        pred_labels_dir,
        result,
        edit_report,
        image_key=image_key,
        dedup_iou=dedup_iou,
        backup=backup,
        dry_run=dry_run,
    )
    for (stem, crop_index), crop_paths in sorted(targets.items()):
        candidates = image_candidates.get(stem, [])
        if not candidates:
            result.missing_images.append(stem)
            continue
        if len(candidates) > 1:
            result.ambiguous_images.append(stem)
            continue

        image = candidates[0]
        if image.label_path is None or not image.label_path.is_file():
            result.missing_labels.append(stem)
            continue
        if crop_index > len(image.annotations):
            result.invalid_indices.append(f"{stem}_{crop_index}")
            continue

        annotation = image.annotations[crop_index - 1]
        line_no = annotation.line_no or crop_index
        force_delete = (stem, crop_index) in forced_delete_targets
        mapped_target_id = (
            target_class_by_target[(stem, crop_index)]
            if target_class_by_target is not None
            else target_id
        )
        new_class_id = None if force_delete else mapped_target_id
        mapped_target_name = (
            dataset.class_name(mapped_target_id)
            if mapped_target_id is not None
            else None
        )
        new_class_name = None if force_delete else mapped_target_name
        if (
            not force_delete
            and mapped_target_id is not None
            and annotation.class_id == mapped_target_id
        ):
            result.unchanged += 1
            continue

        old_id = annotation.class_id
        edit_report.add(
            EditRow(
                operation=(
                    "delete_gt_from_missing_prediction"
                    if force_delete
                    else "correct_class_from_crops"
                ),
                image=image.file_name,
                label_path=str(image.label_path),
                line_no=line_no,
                old_class_id=old_id,
                old_class_name=dataset.class_name(old_id),
                new_class_id=new_class_id,
                new_class_name=new_class_name,
                action="delete" if new_class_id is None else "update",
            )
        )
        pending.setdefault(image.label_path, []).append((line_no, new_class_id))
        changed_annotations.append((image, annotation, new_class_id))
        result.changed += 1
        if new_class_id is None:
            result.deleted += 1

    if not dry_run:
        for label_path, changes in pending.items():
            if backup is not None:
                backup.backup(label_path)
            _rewrite_label_classes(label_path, changes)
        for image, annotation, new_class_id in changed_annotations:
            if new_class_id is None:
                image.annotations = [item for item in image.annotations if item is not annotation]
            else:
                annotation.class_id = new_class_id

    return result, edit_report


def _replace_target_map_from_predictions(
    dataset: YoloDataset,
    targets: dict[tuple[str, int], list[tuple[int, Path]]],
    pred_labels_dir: str | Path | None,
    result: CropCorrectionResult,
    edit_report: EditReport,
    *,
    image_key,
    dedup_iou: float | None,
    backup: LabelBackup | None,
    dry_run: bool,
) -> None:
    """Replace existing GT rows with prediction rows selected by crop names."""
    if not targets:
        return
    if pred_labels_dir is None:
        result.missing_prediction_labels.extend(
            f"{stem}_pred{pred_index}_gt{gt_index}"
            for (stem, gt_index), candidates in sorted(targets.items())
            for pred_index, _crop_path in candidates[:1]
        )
        return

    pred_root = Path(pred_labels_dir)
    if not pred_root.is_dir():
        raise FileNotFoundError(f"prediction label directory not found: {pred_root}")

    prediction_files: dict[str, list[Path]] = {}
    for path in sorted(pred_root.rglob("*.txt")):
        prediction_files.setdefault(path.stem, []).append(path)

    image_candidates: dict[str, list[YoloImage]] = {}
    for image in dataset.images:
        image_candidates.setdefault(image_key(image), []).append(image)

    parsed_cache: dict[Path, list[YoloAnnotation] | None] = {}
    resolved: list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]] = []

    for (stem, gt_index), candidates_for_target in sorted(targets.items()):
        candidates = image_candidates.get(stem, [])
        if not candidates:
            result.missing_images.append(stem)
            continue
        if len(candidates) > 1:
            result.ambiguous_images.append(stem)
            continue

        image = candidates[0]
        if image.label_path is None or not image.label_path.is_file():
            result.missing_labels.append(stem)
            continue
        if gt_index > len(image.annotations):
            result.invalid_indices.append(f"{stem}_{gt_index}")
            continue

        annotation = image.annotations[gt_index - 1]
        line_no = annotation.line_no or gt_index
        predictions_for_target: list[tuple[YoloAnnotation, int]] = []
        for pred_index in sorted({pred_index for pred_index, _path in candidates_for_target}):
            pred_candidates: list[Path] = []
            for key in dict.fromkeys((stem, _safe_file_name(stem))):
                pred_candidates.extend(prediction_files.get(key, []))
            pred_candidates = list(dict.fromkeys(pred_candidates))
            if not pred_candidates:
                result.missing_prediction_labels.append(f"{stem}_pred{pred_index}_gt{gt_index}")
                continue
            if len(pred_candidates) > 1:
                result.ambiguous_prediction_labels.append(f"{stem}_pred{pred_index}_gt{gt_index}")
                continue

            pred_path = pred_candidates[0]
            if pred_path not in parsed_cache:
                try:
                    parsed_cache[pred_path] = parse_label_file(
                        pred_path,
                        task=dataset.task,
                        attributes=dataset.attributes,
                    )
                except (OSError, ValueError):
                    parsed_cache[pred_path] = None
                    result.invalid_prediction_labels.append(str(pred_path))
            predictions = parsed_cache[pred_path]
            if predictions is None:
                continue

            prediction = _prediction_at_index(predictions, pred_index)
            if prediction is None:
                result.invalid_prediction_indices.append(f"{stem}_pred{pred_index}_gt{gt_index}")
                continue
            predictions_for_target.append((prediction, pred_index))

        if not predictions_for_target:
            continue
        prediction, pred_index = max(
            predictions_for_target,
            key=lambda item: (
                1.0 if item[0].confidence is None else float(item[0].confidence),
                -item[1],
            ),
        )
        result.deduplicated += max(0, len(predictions_for_target) - 1)
        resolved.append((image, annotation, prediction, line_no, pred_index, gt_index))

    kept, suppressed = _deduplicate_replacement_candidates(
        resolved,
        dedup_iou=dedup_iou,
        result=result,
    )
    pending: dict[Path, list[tuple[int, str | None]]] = {}
    changed_annotations: list[tuple[YoloImage, YoloAnnotation, YoloAnnotation | None]] = []
    for image, annotation, prediction, line_no, _pred_index, _gt_index in kept:
        replacement = copy(prediction)
        replacement.confidence = None
        replacement.line_no = line_no
        if not replacement.attributes and annotation.attributes:
            replacement.attributes = list(annotation.attributes)
        line = replacement.to_yolo_line(include_confidence=False)
        replacement.source_line = line
        pending.setdefault(image.label_path, []).append((line_no, line))
        changed_annotations.append((image, annotation, replacement))
        edit_report.add(
            EditRow(
                operation="replace_gt_from_prediction",
                image=image.file_name,
                label_path=str(image.label_path),
                line_no=line_no,
                old_class_id=annotation.class_id,
                old_class_name=dataset.class_name(annotation.class_id),
                new_class_id=replacement.class_id,
                new_class_name=dataset.class_name(replacement.class_id),
                action="replace",
            )
        )
        result.changed += 1
        result.replaced += 1

    for image, annotation, _prediction, line_no, _pred_index, _gt_index in suppressed:
        pending.setdefault(image.label_path, []).append((line_no, None))
        changed_annotations.append((image, annotation, None))
        edit_report.add(
            EditRow(
                operation="delete_duplicate_gt_after_prediction_replacement",
                image=image.file_name,
                label_path=str(image.label_path),
                line_no=line_no,
                old_class_id=annotation.class_id,
                old_class_name=dataset.class_name(annotation.class_id),
                action="delete",
            )
        )
        result.changed += 1
        result.deleted += 1

    if dry_run:
        return
    for label_path, changes in pending.items():
        if backup is not None:
            backup.backup(label_path)
        _rewrite_label_annotations(label_path, changes)
    for image, annotation, replacement in changed_annotations:
        if replacement is None:
            image.annotations = [item for item in image.annotations if item is not annotation]
            continue
        for index, current in enumerate(image.annotations):
            if current is annotation:
                image.annotations[index] = replacement
                break


def _deduplicate_replacement_candidates(
    candidates: list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]],
    *,
    dedup_iou: float | None,
    result: CropCorrectionResult,
) -> tuple[
    list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]],
    list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]],
]:
    """Deduplicate overlapping prediction-backed GT replacements."""
    if dedup_iou is None or len(candidates) < 2:
        return candidates, []

    ordered = sorted(
        candidates,
        key=lambda item: (
            -(
                1.0
                if item[2].confidence is None
                else float(item[2].confidence)
            ),
            item[5],
            item[4],
        ),
    )
    kept: list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]] = []
    suppressed: list[tuple[YoloImage, YoloAnnotation, YoloAnnotation, int, int, int]] = []
    for candidate in ordered:
        prediction = candidate[2]
        is_duplicate = any(
            prediction.class_id == existing[2].class_id
            and _annotation_iou(prediction, existing[2]) >= float(dedup_iou)
            for existing in kept
            if existing[0].label_path == candidate[0].label_path
        )
        if is_duplicate:
            result.deduplicated += 1
            suppressed.append(candidate)
        else:
            kept.append(candidate)
    return kept, suppressed


def _append_prediction_targets(
    dataset: YoloDataset,
    targets: dict[tuple[str, int], list[Path]],
    pred_labels_dir: str | Path | None,
    result: CropCorrectionResult,
    edit_report: EditReport,
    *,
    dedup_iou: float | None,
    backup: LabelBackup | None,
    dry_run: bool,
) -> None:
    """Append prediction annotations selected by ``gtnone`` crops."""
    if not targets:
        return
    if pred_labels_dir is None:
        result.missing_prediction_labels.extend(
            f"{stem}_pred{pred_index}"
            for stem, pred_index in sorted(targets)
        )
        return

    pred_root = Path(pred_labels_dir)
    if not pred_root.is_dir():
        raise FileNotFoundError(f"prediction label directory not found: {pred_root}")

    prediction_files: dict[str, list[Path]] = {}
    for path in sorted(pred_root.rglob("*.txt")):
        prediction_files.setdefault(path.stem, []).append(path)

    image_candidates: dict[str, list[YoloImage]] = {}
    for image in dataset.images:
        image_candidates.setdefault(_safe_file_name(image.stem), []).append(image)

    pending: dict[Path, list[tuple[YoloImage, YoloAnnotation, str, int]]] = {}
    parsed_cache: dict[Path, list[YoloAnnotation] | None] = {}
    for (stem, pred_index), _crop_paths in sorted(targets.items()):
        candidates = image_candidates.get(stem, [])
        if not candidates:
            result.missing_images.append(stem)
            continue
        if len(candidates) > 1:
            result.ambiguous_images.append(stem)
            continue

        image = candidates[0]
        if image.label_path is None or not image.label_path.is_file():
            result.missing_labels.append(stem)
            continue

        pred_candidates: list[Path] = []
        for key in dict.fromkeys((stem, _safe_file_name(stem))):
            pred_candidates.extend(prediction_files.get(key, []))
        pred_candidates = list(dict.fromkeys(pred_candidates))
        if not pred_candidates:
            result.missing_prediction_labels.append(f"{stem}_pred{pred_index}")
            continue
        if len(pred_candidates) > 1:
            result.ambiguous_prediction_labels.append(f"{stem}_pred{pred_index}")
            continue

        pred_path = pred_candidates[0]
        if pred_path not in parsed_cache:
            try:
                parsed_cache[pred_path] = parse_label_file(
                    pred_path,
                    task=dataset.task,
                    attributes=dataset.attributes,
                )
            except (OSError, ValueError):
                parsed_cache[pred_path] = None
                result.invalid_prediction_labels.append(str(pred_path))
        predictions = parsed_cache[pred_path]
        if predictions is None:
            continue

        prediction = _prediction_at_index(predictions, pred_index)
        if prediction is None:
            result.invalid_prediction_indices.append(f"{stem}_pred{pred_index}")
            continue
        line = prediction.to_yolo_line(include_confidence=False)
        pending.setdefault(image.label_path, []).append(
            (image, prediction, line, pred_index)
        )

    for label_path, additions in pending.items():
        additions = _deduplicate_prediction_additions(
            additions,
            dedup_iou=dedup_iou,
            result=result,
        )
        if not additions:
            continue
        with label_path.open("r", encoding="utf-8", newline="") as fp:
            existing_lines = fp.read().splitlines(keepends=True)
        next_line_no = len(existing_lines) + 1
        append_lines: list[str] = []
        for image, prediction, line, _pred_index in additions:
            line_no = next_line_no
            next_line_no += 1
            append_lines.append(line)
            edit_report.add(
                EditRow(
                    operation="add_prediction_from_error_crops",
                    image=image.file_name,
                    label_path=str(label_path),
                    line_no=line_no,
                    old_class_id=prediction.class_id,
                    old_class_name=dataset.class_name(prediction.class_id),
                    new_class_id=prediction.class_id,
                    new_class_name=dataset.class_name(prediction.class_id),
                    action="add",
                )
            )
            result.changed += 1
            result.added += 1

        if dry_run:
            continue
        if backup is not None:
            backup.backup(label_path)
        _append_label_lines(label_path, append_lines)
        for image, prediction, line, _pred_index in additions:
            appended = copy(prediction)
            appended.line_no = len(image.annotations) + 1
            appended.source_line = line
            image.annotations.append(appended)


def _deduplicate_prediction_additions(
    additions: list[tuple[YoloImage, YoloAnnotation, str, int]],
    *,
    dedup_iou: float | None,
    result: CropCorrectionResult,
) -> list[tuple[YoloImage, YoloAnnotation, str, int]]:
    """Keep the highest-confidence overlapping prediction per class."""
    if dedup_iou is None or len(additions) < 2:
        return additions

    ordered = sorted(
        additions,
        key=lambda item: (
            -(1.0 if item[1].confidence is None else float(item[1].confidence)),
            item[3],
        ),
    )
    kept: list[tuple[YoloImage, YoloAnnotation, str, int]] = []
    for candidate in ordered:
        prediction = candidate[1]
        is_duplicate = any(
            prediction.class_id == existing[1].class_id
            and _annotation_iou(prediction, existing[1]) >= float(dedup_iou)
            for existing in kept
        )
        if is_duplicate:
            result.deduplicated += 1
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: item[3])


def _annotation_iou(first: YoloAnnotation, second: YoloAnnotation) -> float:
    first_box = first.geometry_box()
    second_box = second.geometry_box()
    if first_box is None or second_box is None:
        return 0.0

    first_x1 = first_box.cx - first_box.width / 2.0
    first_y1 = first_box.cy - first_box.height / 2.0
    first_x2 = first_box.cx + first_box.width / 2.0
    first_y2 = first_box.cy + first_box.height / 2.0
    second_x1 = second_box.cx - second_box.width / 2.0
    second_y1 = second_box.cy - second_box.height / 2.0
    second_x2 = second_box.cx + second_box.width / 2.0
    second_y2 = second_box.cy + second_box.height / 2.0

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _prediction_at_index(
    predictions: list[YoloAnnotation],
    pred_index: int,
) -> YoloAnnotation | None:
    """Find a prediction by original txt line number, with order fallback."""
    for prediction in predictions:
        if prediction.line_no == pred_index:
            return prediction
    if 1 <= pred_index <= len(predictions):
        return predictions[pred_index - 1]
    return None


def _annotation_for_label_index(image: YoloImage, label_index: int) -> YoloAnnotation | None:
    """Find an annotation by its source line number, with order fallback."""

    for annotation in image.annotations:
        if annotation.line_no == label_index:
            return annotation
    if 1 <= label_index <= len(image.annotations):
        return image.annotations[label_index - 1]
    return None


def _append_label_lines(label_path: Path, lines_to_append: list[str]) -> None:
    """Append normalised YOLO lines while preserving existing line endings."""
    if not lines_to_append:
        return
    with label_path.open("r", encoding="utf-8", newline="") as fp:
        lines = fp.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    lines.extend(line.rstrip("\r\n") + "\n" for line in lines_to_append)
    with label_path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("".join(lines))


def _parse_crop_name(path: Path) -> tuple[str, int] | None:
    match = _CROP_NAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index"))


def _parse_error_crop_name(path: Path) -> tuple[str, int | None, int | None] | None:
    match = _ERROR_CROP_NAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    pred_text = match.group("pred")
    gt_text = match.group("gt")
    return (
        match.group("stem"),
        None if pred_text == "none" else int(pred_text),
        None if gt_text == "none" else int(gt_text),
    )


def _parse_attribute_error_crop_name(
    path: Path,
) -> tuple[str, int | None, int | None, str | None] | None:
    match = _ATTRIBUTE_ERROR_CROP_NAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    pred_text = match.group("pred")
    gt_text = match.group("gt")
    return (
        match.group("stem"),
        None if pred_text == "none" else int(pred_text),
        None if gt_text == "none" else int(gt_text),
        match.group("attribute"),
    )


def _safe_file_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _ensure_attribute_values(values: list[float], length: int) -> None:
    while len(values) < length:
        values.append(0.0)


def _rewrite_label_classes(label_path: Path, changes: list[tuple[int, int | None]]) -> None:
    with label_path.open("r", encoding="utf-8", newline="") as fp:
        lines = fp.read().splitlines(keepends=True)

    for line_no, target_class_id in sorted(changes, key=lambda item: item[0], reverse=True):
        line_index = line_no - 1
        if not 0 <= line_index < len(lines):
            continue
        if target_class_id is None:
            del lines[line_index]
        else:
            lines[line_index] = _replace_class_token(lines[line_index], target_class_id)

    with label_path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("".join(lines))


def _rewrite_label_annotations(label_path: Path, changes: list[tuple[int, str | None]]) -> None:
    """Replace complete label lines by original 1-based line number."""
    with label_path.open("r", encoding="utf-8", newline="") as fp:
        lines = fp.read().splitlines(keepends=True)

    for line_no, replacement in sorted(changes, key=lambda item: item[0], reverse=True):
        line_index = line_no - 1
        if not 0 <= line_index < len(lines):
            continue
        if replacement is None:
            del lines[line_index]
            continue
        old_line = lines[line_index]
        body = old_line.rstrip("\r\n")
        ending = old_line[len(body) :]
        lines[line_index] = replacement.rstrip("\r\n") + ending

    with label_path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("".join(lines))


def _replace_class_token(line: str, target_class_id: int) -> str:
    body = line.rstrip("\r\n")
    ending = line[len(body) :]
    match = re.match(r"^(\s*)\S+(.*)$", body)
    if match is None:
        return line
    return f"{match.group(1)}{target_class_id}{match.group(2)}{ending}"
