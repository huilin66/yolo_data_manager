"""Dataset image/label check orchestration.

The lower-level validators remain available on their original modules.  This
module provides the user-facing check groups used by ``ydm check``,
``ydm image-check`` and ``ydm label-check``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yolo_data_manager.dataset.duplicates import find_duplicate_images
from yolo_data_manager.dataset.quality import find_bad_images
from yolo_data_manager.evaluation.error_analysis import find_duplicate_gt
from yolo_data_manager.io.validator import ValidationReport, validate_dataset
from yolo_data_manager.runtime import progress_stage

IMAGE_CHECK_ITEMS = frozenset({"bad_images", "duplicates"})
LABEL_CHECK_ITEMS = frozenset({"format", "overlap"})
CHECK_GROUPS = frozenset({"image", "label"})


def _normalise_values(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_check_list(
    check_list: str | Iterable[str] | None,
    *,
    available: Iterable[str],
    aliases: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Normalize a comma-separated check selection and validate its names."""

    available_values = list(available)
    available_set = set(available_values)
    values = _normalise_values(check_list)
    if not values or "all" in values:
        return available_values

    resolved: list[str] = []
    alias_map = aliases or {}
    for value in values:
        expanded = alias_map.get(value, (value,))
        for item in expanded:
            if item not in available_set:
                choices = ", ".join(["all", *available_values])
                raise ValueError(
                    f"unknown check item(s): {value}; available: {choices}"
                )
            if item not in resolved:
                resolved.append(item)
    return [item for item in available_values if item in resolved]


def normalize_image_check_list(
    check_list: str | Iterable[str] | None,
) -> list[str]:
    return normalize_check_list(
        check_list,
        available=("bad_images", "duplicates"),
        aliases={
            "bad_image": ("bad_images",),
            "corrupt": ("bad_images",),
            "duplicate": ("duplicates",),
        },
    )


def normalize_label_check_list(
    check_list: str | Iterable[str] | None,
) -> list[str]:
    return normalize_check_list(
        check_list,
        available=("format", "overlap"),
        aliases={
            "validity": ("format",),
            "duplicates": ("overlap",),
            "duplicate": ("overlap",),
            "iou": ("overlap",),
        },
    )


def normalize_check_groups(
    check_list: str | Iterable[str] | None,
) -> list[str]:
    return normalize_check_list(
        check_list,
        available=("image", "label"),
        aliases={
            "images": ("image",),
            "labels": ("label",),
        },
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('level', 'warning')}:{row.get('code', 'unknown')}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def _payload(
    name: str,
    selected: Sequence[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "check": name,
        "selected": list(selected),
        "ok": not any(row.get("level") == "error" for row in rows),
        "summary": _summary(rows),
        "issues": rows,
    }


def _duplicate_image_name_rows(dataset) -> list[dict[str, Any]]:
    by_name: dict[str, list[str]] = {}
    for image in dataset.images:
        by_name.setdefault(image.file_name, []).append(str(image.path))

    rows: list[dict[str, Any]] = []
    for name, paths in by_name.items():
        if len(paths) < 2:
            continue
        for path in paths:
            rows.append(
                {
                    "level": "warning",
                    "code": "duplicate_image_name",
                    "message": f"duplicate image output name: {name}",
                    "image": name,
                    "label": None,
                    "line_no": None,
                    "paths": paths,
                    "path": path,
                }
            )
    return rows


def run_image_checks(
    dataset,
    check_list: str | Iterable[str] | None = None,
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    """Run selected image checks and return a JSON-serializable payload."""

    selected = normalize_image_check_list(check_list)
    rows: list[dict[str, Any]] = []

    if "bad_images" in selected:
        progress_stage("image check: missing/corrupt images", enabled=progress)
        rows.extend(
            {
                "level": "error",
                "code": issue.code,
                "message": issue.message,
                "image": issue.image,
                "label": None,
                "line_no": None,
                "path": issue.path,
            }
            for issue in find_bad_images(
                dataset,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
            )
        )

    if "duplicates" in selected:
        progress_stage("image check: duplicate names", enabled=progress)
        rows.extend(_duplicate_image_name_rows(dataset))

        progress_stage("image check: duplicate content", enabled=progress)
        for group in find_duplicate_images(
            dataset,
            algorithm=algorithm,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        ):
            for image_name in group.images:
                rows.append(
                    {
                        "level": "warning",
                        "code": "duplicate_image_content",
                        "message": (
                            f"image content is identical to {len(group.images) - 1} "
                            f"other image(s)"
                        ),
                        "image": image_name,
                        "label": None,
                        "line_no": None,
                        "digest": group.digest,
                        "group_images": group.images,
                    }
                )

    return _payload("image_check", selected, rows)


def _validation_rows(report: ValidationReport) -> list[dict[str, Any]]:
    return [dict(row) for row in report.to_rows()]


def _duplicate_gt_rows(dataset, duplicate_iou: float, *, workers: int, progress: bool, progress_leave: bool) -> list[dict[str, Any]]:
    image_by_stem = dataset.image_by_stem()
    rows: list[dict[str, Any]] = []
    for duplicate in find_duplicate_gt(
        dataset,
        duplicate_iou=duplicate_iou,
        workers=workers,
        progress=progress,
        progress_leave=progress_leave,
    ):
        image = image_by_stem.get(duplicate.image)
        rows.append(
            {
                "level": "warning",
                "code": "duplicate_gt_iou",
                "message": (
                    f"two GT annotations overlap at IoU={duplicate.iou:.6f} "
                    f"(threshold={duplicate_iou:.6f})"
                ),
                "image": duplicate.image,
                "label": str(image.label_path) if image and image.label_path else None,
                "line_no": None,
                "gt_idx_i": duplicate.gt_idx_i,
                "gt_idx_j": duplicate.gt_idx_j,
                "iou": duplicate.iou,
                "cls_i": duplicate.cls_i,
                "name_i": duplicate.name_i,
                "cls_j": duplicate.cls_j,
                "name_j": duplicate.name_j,
                "type": duplicate.type,
                "line_i": duplicate.line_i,
                "line_j": duplicate.line_j,
            }
        )
    return rows


def run_label_checks(
    dataset,
    check_list: str | Iterable[str] | None = None,
    *,
    duplicate_iou: float = 0.9,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, Any]:
    """Run selected label checks and return a JSON-serializable payload."""

    if not 0.0 < float(duplicate_iou) <= 1.0:
        raise ValueError("duplicate_iou must be greater than 0 and at most 1")

    selected = normalize_label_check_list(check_list)
    rows: list[dict[str, Any]] = []

    if "format" in selected:
        progress_stage("label check: format and geometry", enabled=progress)
        rows.extend(
            _validation_rows(
                validate_dataset(
                    dataset,
                    workers=workers,
                    progress=progress,
                    progress_leave=progress_leave,
                    check_image_names=False,
                )
            )
        )

    if "overlap" in selected:
        progress_stage("label check: high-IoU overlaps", enabled=progress)
        rows.extend(
            _duplicate_gt_rows(
                dataset,
                float(duplicate_iou),
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
            )
        )

    return _payload("label_check", selected, rows)


def combine_check_payloads(
    results: Mapping[str, Mapping[str, Any]],
    selected: Sequence[str],
    *,
    fixed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine image and label check payloads while preserving old check keys."""

    rows: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for name in ("image", "label"):
        result = results.get(name)
        if not result:
            continue
        rows.extend(dict(row) for row in result.get("issues", []))
        for key, count in dict(result.get("summary", {})).items():
            summary[key] = summary.get(key, 0) + int(count)

    return {
        "check": "check",
        "selected": list(selected),
        "ok": not any(row.get("level") == "error" for row in rows),
        "summary": summary,
        "issues": rows,
        "checks": dict(results),
        "fixed": dict(fixed or {}),
    }
