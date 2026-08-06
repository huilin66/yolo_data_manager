"""Correct per-instance YOLO classes from visual crop filenames."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
import re

from yolo_data_manager.core.models import YoloAnnotation, YoloDataset, YoloImage, is_image_file
from yolo_data_manager.annotation.edit import EditReport, EditRow
from yolo_data_manager.io.loader import parse_label_file


_CROP_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<index>[1-9][0-9]*)$")
_ERROR_CROP_NAME_RE = re.compile(
    r"^(?P<stem>.+)_pred(?P<pred>none|[1-9][0-9]*)_gt(?P<gt>none|[1-9][0-9]*)$"
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

    def to_dict(self) -> dict[str, object]:
        return {
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "crop_files": self.crop_files,
            "unique_targets": self.unique_targets,
            "changed": self.changed,
            "added": self.added,
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


def correct_labels_from_crops(
    dataset: YoloDataset,
    crops_dir: str | Path,
    target_class: int | str | None,
    *,
    dry_run: bool = False,
) -> tuple[CropCorrectionResult, EditReport]:
    """Update label classes identified by standard ``vis crop`` filenames.

    A crop named ``image_stem_3.jpg`` maps to the third annotation in
    ``image_stem.txt``. The crop directory is searched recursively, so both
    class folders and ``by_attribute`` subfolders are supported. When
    ``target_class`` is ``None``, the mapped annotation line is deleted.
    """

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

    return _correct_target_map(
        dataset,
        targets,
        target_class,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        image_key=lambda image: image.stem,
        dry_run=dry_run,
    )


def correct_gt_labels_from_error_crops(
    dataset: YoloDataset,
    crops_dir: str | Path,
    target_class: int | str | None,
    *,
    pred_labels_dir: str | Path | None = None,
    dedup_iou: float | None = 0.5,
    dry_run: bool = False,
) -> tuple[CropCorrectionResult, EditReport]:
    """Correct GT classes from ``eval_error_analysis`` crop filenames.

    A crop named ``image_stem_pred2_gt3.jpg`` maps to the third GT
    annotation in ``image_stem.txt``. The prediction index is retained in
    the filename for review context but is not needed for the GT update.
    When ``pred_labels_dir`` is supplied, a crop with ``gt none`` appends the
    corresponding prediction annotation selected by ``predx`` to the GT
    label. Prediction confidence is omitted from the appended GT line.
    """

    crop_root = Path(crops_dir)
    if not crop_root.is_dir():
        raise FileNotFoundError(f"error-analysis crop directory not found: {crop_root}")
    if dedup_iou is not None and not 0.0 < float(dedup_iou) <= 1.0:
        raise ValueError("dedup_iou must be between 0 and 1, or None to disable deduplication")

    targets: dict[tuple[str, int], list[Path]] = {}
    prediction_targets: dict[tuple[str, int], list[Path]] = {}
    crop_files = 0
    invalid_crops: list[str] = []
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
        targets.setdefault((stem, gt_index), []).append(crop_path)

    result, edit_report = _correct_target_map(
        dataset,
        targets,
        target_class,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        image_key=lambda image: _safe_file_name(image.stem),
        dry_run=dry_run,
    )
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
        dry_run=dry_run,
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
) -> tuple[CropCorrectionResult, EditReport]:
    target_id = dataset.class_id(target_class) if target_class is not None else None
    target_name = dataset.class_name(target_id) if target_id is not None else None
    result = CropCorrectionResult(
        target_class_id=target_id,
        target_class_name=target_name,
        crop_files=crop_files,
        invalid_crops=invalid_crops,
        unique_targets=len(targets),
        duplicate_targets=sum(max(0, len(paths) - 1) for paths in targets.values()),
    )
    edit_report = EditReport()

    image_candidates: dict[str, list[YoloImage]] = {}
    for image in dataset.images:
        image_candidates.setdefault(image_key(image), []).append(image)

    pending: dict[Path, list[tuple[int, int | None]]] = {}
    changed_annotations: list[tuple[YoloImage, YoloAnnotation, int | None]] = []
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
        if target_id is not None and annotation.class_id == target_id:
            result.unchanged += 1
            continue

        old_id = annotation.class_id
        edit_report.add(
            EditRow(
                operation="correct_class_from_crops",
                image=image.file_name,
                label_path=str(image.label_path),
                line_no=line_no,
                old_class_id=old_id,
                old_class_name=dataset.class_name(old_id),
                new_class_id=target_id,
                new_class_name=target_name,
                action="delete" if target_id is None else "update",
            )
        )
        pending.setdefault(image.label_path, []).append((line_no, target_id))
        changed_annotations.append((image, annotation, target_id))
        result.changed += 1
        if target_id is None:
            result.deleted += 1

    if not dry_run:
        for label_path, changes in pending.items():
            _rewrite_label_classes(label_path, changes)
        for image, annotation, new_class_id in changed_annotations:
            if new_class_id is None:
                image.annotations = [item for item in image.annotations if item is not annotation]
            else:
                annotation.class_id = new_class_id

    return result, edit_report


def _append_prediction_targets(
    dataset: YoloDataset,
    targets: dict[tuple[str, int], list[Path]],
    pred_labels_dir: str | Path | None,
    result: CropCorrectionResult,
    edit_report: EditReport,
    *,
    dedup_iou: float | None,
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
        existing_lines = label_path.read_text(
            encoding="utf-8", newline=""
        ).splitlines(keepends=True)
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


def _append_label_lines(label_path: Path, lines_to_append: list[str]) -> None:
    """Append normalised YOLO lines while preserving existing line endings."""
    if not lines_to_append:
        return
    lines = label_path.read_text(encoding="utf-8", newline="").splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    lines.extend(line.rstrip("\r\n") + "\n" for line in lines_to_append)
    label_path.write_text("".join(lines), encoding="utf-8", newline="")


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


def _safe_file_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


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


def _replace_class_token(line: str, target_class_id: int) -> str:
    body = line.rstrip("\r\n")
    ending = line[len(body) :]
    match = re.match(r"^(\s*)\S+(.*)$", body)
    if match is None:
        return line
    return f"{match.group(1)}{target_class_id}{match.group(2)}{ending}"
