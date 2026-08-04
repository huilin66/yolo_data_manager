"""Correct per-instance YOLO classes from visual crop filenames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from yolo_data_manager.core.models import YoloDataset, is_image_file
from yolo_data_manager.annotation.edit import EditReport, EditRow


_CROP_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<index>[1-9][0-9]*)$")


@dataclass
class CropCorrectionResult:
    """Summary of a crop-driven class correction operation."""

    target_class_id: int
    target_class_name: str
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

    def to_dict(self) -> dict[str, object]:
        return {
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
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
            "skipped": (
                len(self.invalid_crops)
                + len(self.missing_images)
                + len(self.ambiguous_images)
                + len(self.missing_labels)
                + len(self.invalid_indices)
            ),
        }


def correct_labels_from_crops(
    dataset: YoloDataset,
    crops_dir: str | Path,
    target_class: int | str,
    *,
    dry_run: bool = False,
) -> tuple[CropCorrectionResult, EditReport]:
    """Update label classes identified by standard ``vis crop`` filenames.

    A crop named ``image_stem_3.jpg`` maps to the third annotation in
    ``image_stem.txt``. The crop directory is searched recursively, so both
    class folders and ``by_attribute`` subfolders are supported.
    """

    crop_root = Path(crops_dir)
    if not crop_root.is_dir():
        raise FileNotFoundError(f"crop directory not found: {crop_root}")

    target_id = dataset.class_id(target_class)
    target_name = dataset.class_name(target_id)
    result = CropCorrectionResult(target_class_id=target_id, target_class_name=target_name)
    edit_report = EditReport()

    image_candidates: dict[str, list] = {}
    for image in dataset.images:
        image_candidates.setdefault(image.stem, []).append(image)

    targets: dict[tuple[str, int], list[Path]] = {}
    for crop_path in sorted(crop_root.rglob("*")):
        if not crop_path.is_file() or not is_image_file(crop_path):
            continue
        result.crop_files += 1
        parsed = _parse_crop_name(crop_path)
        if parsed is None:
            result.invalid_crops.append(str(crop_path))
            continue
        targets.setdefault(parsed, []).append(crop_path)

    result.unique_targets = len(targets)
    result.duplicate_targets = sum(max(0, len(paths) - 1) for paths in targets.values())

    pending: dict[Path, list[tuple[int, int]]] = {}
    changed_annotations: list[tuple[object, int]] = []
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
        if annotation.class_id == target_id:
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
            )
        )
        pending.setdefault(image.label_path, []).append((line_no, target_id))
        changed_annotations.append((annotation, target_id))
        result.changed += 1

    if not dry_run:
        for label_path, changes in pending.items():
            _rewrite_label_classes(label_path, changes)
        for annotation, new_class_id in changed_annotations:
            annotation.class_id = new_class_id

    return result, edit_report


def _parse_crop_name(path: Path) -> tuple[str, int] | None:
    match = _CROP_NAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index"))


def _rewrite_label_classes(label_path: Path, changes: list[tuple[int, int]]) -> None:
    with label_path.open("r", encoding="utf-8", newline="") as fp:
        lines = fp.read().splitlines(keepends=True)

    for line_no, target_class_id in changes:
        line_index = line_no - 1
        if not 0 <= line_index < len(lines):
            continue
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
