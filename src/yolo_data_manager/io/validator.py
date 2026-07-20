from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from yolo_data_manager.core.geometry import polygon_self_intersects
from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.io.layout import infer_label_path_from_image
from yolo_data_manager.runtime import iter_progress, normalize_workers


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str
    image: str | None = None
    label: str | None = None
    line_no: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add(
        self,
        level: str,
        code: str,
        message: str,
        image: str | None = None,
        label: Path | str | None = None,
        line_no: int | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                level=level,
                code=code,
                message=message,
                image=image,
                label=str(label) if label is not None else None,
                line_no=line_no,
            )
        )

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            key = f"{issue.level}:{issue.code}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "level": issue.level,
                "code": issue.code,
                "message": issue.message,
                "image": issue.image,
                "label": issue.label,
                "line_no": issue.line_no,
            }
            for issue in self.issues
        ]


def validate_dataset(
    dataset: YoloDataset,
    *,
    workers: int = 1,
    progress: bool = False,
    progress_leave: bool = False,
) -> ValidationReport:
    report = ValidationReport()

    seen_image_names: set[str] = set()
    duplicate_image_indices: set[int] = set()
    for label in dataset.orphan_labels:
        report.add("warning", "orphan_label", "label has no matching image", label=label)
    for idx, image in enumerate(dataset.images):
        if image.file_name in seen_image_names:
            duplicate_image_indices.add(idx)
        seen_image_names.add(image.file_name)

    worker_count = normalize_workers(workers)
    class_count = len(dataset.classes)
    if worker_count == 1:
        for idx, image in iter_progress(enumerate(dataset.images), enabled=progress, total=len(dataset.images), desc="check", leave=progress_leave):
            report.issues.extend(_validate_image(image, class_count, idx in duplicate_image_indices))
        return report

    indexed_issues: list[tuple[int, list[ValidationIssue]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            (
                idx,
                executor.submit(_validate_image, image, class_count, idx in duplicate_image_indices),
            )
            for idx, image in enumerate(dataset.images)
        ]
        future_to_idx = {future: idx for idx, future in futures}
        for future in iter_progress(as_completed(future_to_idx), enabled=progress, total=len(future_to_idx), desc="check", leave=progress_leave):
            indexed_issues.append((future_to_idx[future], future.result()))
    for _, issues in sorted(indexed_issues, key=lambda item: item[0]):
        report.issues.extend(issues)
    return report


def _validate_image(image: YoloImage, class_count: int, duplicate_image_name: bool) -> list[ValidationIssue]:
    report = ValidationReport()
    if duplicate_image_name:
        report.add("warning", "duplicate_image_name", f"duplicate image output name: {image.file_name}", image=image.file_name)

    if image.label_path is None:
        report.add("warning", "missing_label", "image has no matching label", image=image.file_name)
    elif not image.label_path.exists():
        report.add("warning", "missing_label_file", "label path does not exist", image=image.file_name, label=image.label_path)

    seen_annotations: set[str] = set()
    for ann in image.annotations:
        ann_key = ann.to_yolo_line(include_confidence=True)
        if ann_key in seen_annotations:
            report.add(
                "warning",
                "duplicate_annotation",
                "duplicate annotation line in image",
                image=image.file_name,
                label=image.label_path,
                line_no=ann.line_no,
            )
        seen_annotations.add(ann_key)

        if class_count and not 0 <= ann.class_id < class_count:
            report.add(
                "error",
                "class_out_of_range",
                f"class id {ann.class_id} is out of range",
                image=image.file_name,
                label=image.label_path,
                line_no=ann.line_no,
            )
        if ann.box is not None:
            _validate_box(report, image.file_name, image.label_path, ann.line_no, ann.box.as_tuple())
        if ann.polygon is not None:
            if len(ann.polygon.points) < 3:
                report.add(
                    "error",
                    "invalid_polygon",
                    "polygon needs at least 3 points",
                    image=image.file_name,
                    label=image.label_path,
                    line_no=ann.line_no,
                )
            elif polygon_self_intersects(ann.polygon.points):
                report.add(
                    "warning",
                    "polygon_self_intersection",
                    "polygon appears to self-intersect",
                    image=image.file_name,
                    label=image.label_path,
                    line_no=ann.line_no,
                )
            for idx, point in enumerate(ann.polygon.points):
                _validate_values(
                    report,
                    image.file_name,
                    image.label_path,
                    ann.line_no,
                    point,
                    f"polygon point {idx}",
                )
        if ann.confidence is not None and not 0 <= ann.confidence <= 1:
            report.add(
                "warning",
                "confidence_out_of_range",
                f"confidence {ann.confidence} is outside [0, 1]",
                image=image.file_name,
                label=image.label_path,
                line_no=ann.line_no,
            )
    return report.issues


def fill_missing_label_files(dataset: YoloDataset) -> list[Path]:
    created: list[Path] = []
    for image in dataset.images:
        label_path = image.label_path or infer_label_path_from_image(image.path)
        if label_path.exists():
            continue
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
        created.append(label_path)
    return created


def _validate_box(
    report: ValidationReport,
    image_name: str,
    label_path: Path | None,
    line_no: int | None,
    values: tuple[float, float, float, float],
) -> None:
    cx, cy, width, height = values
    _validate_values(report, image_name, label_path, line_no, (cx, cy, width, height), "box")
    if width <= 0 or height <= 0:
        report.add(
            "error",
            "invalid_box_size",
            f"box width/height must be positive, got {width}/{height}",
            image=image_name,
            label=label_path,
            line_no=line_no,
        )


def _validate_values(
    report: ValidationReport,
    image_name: str,
    label_path: Path | None,
    line_no: int | None,
    values: tuple[float, ...],
    name: str,
) -> None:
    for value in values:
        if not 0 <= value <= 1:
            report.add(
                "warning",
                "coord_out_of_range",
                f"{name} value {value} is outside [0, 1]",
                image=image_name,
                label=label_path,
                line_no=line_no,
            )

