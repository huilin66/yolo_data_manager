"""Extract error results shared by multiple ``eval_error_analysis`` runs."""

from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yolo_data_manager.core.models import is_image_file
from yolo_data_manager.evaluation.error_analysis import (
    BACKGROUND_FP,
    CLASS_ERROR_PRED,
    ErrorDetail,
    FN_NO_PRED,
    _parse_box_json,
    _review_group_name,
    _safe_file_name,
)
from yolo_data_manager.evaluation.matching import box_iou_matrix
from yolo_data_manager.runtime import iter_progress, normalize_workers, progress_stage


_ERROR_COLUMNS = [
    "image",
    "status",
    "error_type",
    "class_id",
    "class_name",
    "pred_class_id",
    "pred_class_name",
    "pred_conf",
    "gt_class_id",
    "gt_class_name",
    "best_iou",
    "pred_box_xyxy",
    "gt_box_xyxy",
    "pred_line",
    "gt_line",
    "pred_idx",
    "gt_idx",
]

_COMMON_ERROR_COLUMNS = [
    "common_kind",
    "source_run",
    "source_error_dir",
    "matched_runs",
    *_ERROR_COLUMNS,
]

_SUMMARY_COLUMNS = [
    "run_name",
    "input_dir",
    "original_fn",
    "common_fn",
    "remaining_fn",
    "original_fp",
    "common_fp",
    "remaining_fp",
    "original_total",
    "common_total",
    "remaining_total",
]


@dataclass
class _ErrorRun:
    name: str
    root: Path
    rows: list[ErrorDetail]


@dataclass
class _CommonMatch:
    kind: str
    row: ErrorDetail
    refs: list[tuple[_ErrorRun, ErrorDetail]]


def extract_common_error_analysis(
    error_dirs: Sequence[str | Path] | Mapping[str, str | Path],
    out: str | Path,
    *,
    iou: float = 0.5,
    copy_crops: bool = True,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, Any]:
    """Extract errors that are present in every error-analysis result.

    ``error_dirs`` accepts either a sequence of output directories or a
    mapping from a user-defined run name to an output directory.  Each
    directory must contain the ``fp_report.csv`` and ``fn_report.csv`` files
    written by :func:`eval_error_analysis`.

    ``fn_no_pred`` records are matched by image, GT class, and GT-box IoU.
    FP records are restricted to background and class-error predictions, then
    matched by image, predicted class, and prediction-box IoU.  Restricting
    FN to ``fn_no_pred`` avoids counting the GT side of a class-error pair a
    second time; that pair is represented by its class-error prediction.
    The returned report and CSV files contain one canonical row for each
    intersection item.
    """

    _validate_iou(iou)
    worker_count = normalize_workers(workers)
    runs = _load_runs(
        error_dirs,
        progress=progress,
        progress_leave=progress_leave,
        workers=worker_count,
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)

    fn_rows = [
        [row for row in run.rows if row.status == "fn" and row.error_type == FN_NO_PRED]
        for run in runs
    ]
    fp_rows = [
        [row for row in run.rows if _is_common_prediction_error(row)]
        for run in runs
    ]

    common_fn = _find_common_matches(
        fn_rows,
        runs,
        kind="fn",
        iou=iou,
        workers=worker_count,
        progress=progress,
        progress_leave=progress_leave,
    )
    common_fp = _find_common_matches(
        fp_rows,
        runs,
        kind="fp",
        iou=iou,
        workers=worker_count,
        progress=progress,
        progress_leave=progress_leave,
    )
    common_matches = [*common_fn, *common_fp]

    crop_report: list[dict[str, Any]] = []
    if copy_crops:
        common_crops = output / "common_crops"
        if common_crops.exists():
            shutil.rmtree(common_crops)
        common_crops.mkdir(parents=True, exist_ok=True)
        crop_report = _copy_common_crops(
            common_matches,
            common_crops,
            workers=worker_count,
            progress=progress,
            progress_leave=progress_leave,
        )
    else:
        progress_stage("common errors skip crop copy", enabled=progress)
        crop_report = [
            {
                "common_kind": match.kind,
                "source_run": None,
                "source_crop": None,
                "destination_crop": None,
            }
            for match in common_matches
        ]

    progress_stage("common errors write reports", enabled=progress)
    summary_rows = _build_summary_rows(runs, common_fn, common_fp)
    common_rows = _common_rows(common_matches, crop_report)
    _write_csv(output / "common_error_summary.csv", _SUMMARY_COLUMNS, summary_rows)
    _write_csv(output / "common_error_rows.csv", _COMMON_ERROR_COLUMNS, common_rows)
    _write_csv(
        output / "common_fn.csv",
        _COMMON_ERROR_COLUMNS,
        [row for row in common_rows if row["common_kind"] == "fn"],
    )
    _write_csv(
        output / "common_fp.csv",
        _COMMON_ERROR_COLUMNS,
        [row for row in common_rows if row["common_kind"] == "fp"],
    )

    missing_crops = sum(1 for item in crop_report if item["source_crop"] is None)
    result: dict[str, Any] = {
        "out": str(output),
        "iou": float(iou),
        "runs": [
            {
                "name": run.name,
                "input_dir": str(run.root),
                "error_rows": len(run.rows),
            }
            for run in runs
        ],
        "common": {
            "fn": len(common_fn),
            "fp": len(common_fp),
            "total": len(common_matches),
        },
        "missing_crops": missing_crops,
        "summary": summary_rows,
    }
    (output / "common_error_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _load_runs(
    error_dirs: Sequence[str | Path] | Mapping[str, str | Path],
    *,
    progress: bool = False,
    progress_leave: bool = False,
    workers: int = 1,
) -> list[_ErrorRun]:
    if isinstance(error_dirs, Mapping):
        items = [(str(name), path) for name, path in error_dirs.items()]
    elif isinstance(error_dirs, (str, Path)):
        raise TypeError("error_dirs must contain at least two result directories")
    else:
        items = [(Path(path).name or f"run_{index + 1}", path) for index, path in enumerate(error_dirs)]

    if len(items) < 2:
        raise ValueError("at least two eval_error_analysis result directories are required")

    worker_count = normalize_workers(workers)
    names: set[str] = set()
    prepared: list[tuple[int, str, Path]] = []
    for index, (raw_name, raw_path) in enumerate(items, start=1):
        name = raw_name.strip() or f"run_{index}"
        if name in names:
            name = f"{name}_{index}"
        names.add(name)
        root = _resolve_error_dir(raw_path)
        prepared.append((index - 1, name, root))

    def load_one(item: tuple[int, str, Path]) -> tuple[int, _ErrorRun]:
        position, name, root = item
        return position, _ErrorRun(name=name, root=root, rows=_read_error_rows(root))

    loaded: list[_ErrorRun | None] = [None] * len(prepared)
    if worker_count == 1:
        items_with_progress = iter_progress(
            prepared,
            enabled=progress,
            total=len(prepared),
            desc="common errors load reports",
            leave=progress_leave,
        )
        for item in items_with_progress:
            position, run = load_one(item)
            loaded[position] = run
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(load_one, item) for item in prepared]
            completed = iter_progress(
                as_completed(futures),
                enabled=progress,
                total=len(futures),
                desc="common errors load reports",
                leave=progress_leave,
            )
            for future in completed:
                position, run = future.result()
                loaded[position] = run
    return [run for run in loaded if run is not None]


def _resolve_error_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name in {"fp_report.csv", "fn_report.csv"}:
        candidate = candidate.parent
    if candidate.name.lower() == "review" and (candidate.parent / "fp_report.csv").exists():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"error-analysis result directory not found: {candidate}")
    return candidate


def _read_error_rows(root: Path) -> list[ErrorDetail]:
    paths = [root / "fp_report.csv", root / "fn_report.csv"]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"{root} does not contain fp_report.csv or fn_report.csv"
        )
    rows: list[ErrorDetail] = []
    for path in existing:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                rows.append(_error_detail_from_csv(raw))
    return rows


def _error_detail_from_csv(raw: Mapping[str, str | None]) -> ErrorDetail:
    return ErrorDetail(
        image=str(raw.get("image") or ""),
        status=str(raw.get("status") or ""),
        error_type=str(raw.get("error_type") or ""),
        class_id=_parse_int(raw.get("class_id"), default=-1),
        class_name=str(raw.get("class_name") or ""),
        pred_class_id=_parse_optional_int(raw.get("pred_class_id")),
        pred_class_name=_parse_optional_text(raw.get("pred_class_name")),
        pred_conf=_parse_optional_float(raw.get("pred_conf")),
        gt_class_id=_parse_optional_int(raw.get("gt_class_id")),
        gt_class_name=_parse_optional_text(raw.get("gt_class_name")),
        best_iou=_parse_float(raw.get("best_iou"), default=0.0),
        pred_box_xyxy=_parse_optional_text(raw.get("pred_box_xyxy")),
        gt_box_xyxy=_parse_optional_text(raw.get("gt_box_xyxy")),
        pred_line=_parse_optional_text(raw.get("pred_line")),
        gt_line=_parse_optional_text(raw.get("gt_line")),
        pred_idx=_parse_optional_int(raw.get("pred_idx")),
        gt_idx=_parse_optional_int(raw.get("gt_idx")),
    )


def _find_common_matches(
    rows_by_run: list[list[ErrorDetail]],
    runs: list[_ErrorRun],
    *,
    kind: str,
    iou: float,
    workers: int = 1,
    progress: bool = False,
    progress_leave: bool = False,
) -> list[_CommonMatch]:
    grouped_rows: list[dict[str, list[ErrorDetail]]] = []
    for rows in rows_by_run:
        by_image: dict[str, list[ErrorDetail]] = {}
        for row in rows:
            by_image.setdefault(row.image, []).append(row)
        grouped_rows.append(by_image)

    image_jobs = list(grouped_rows[0].items())
    worker_count = normalize_workers(workers)

    def match_one(
        position: int,
        image: str,
        base_rows: list[ErrorDetail],
    ) -> tuple[int, list[_CommonMatch]]:
        candidates_by_run = [
            grouped.get(image, []) for grouped in grouped_rows
        ]
        return position, _find_common_matches_for_image(
            base_rows,
            candidates_by_run,
            runs,
            kind=kind,
            iou=iou,
        )

    matched_by_image: list[list[_CommonMatch] | None] = [None] * len(image_jobs)
    if worker_count == 1:
        jobs = iter_progress(
            image_jobs,
            enabled=progress,
            total=len(image_jobs),
            desc=f"common errors match {kind}",
            leave=progress_leave,
        )
        for position, (image, base_rows) in enumerate(jobs):
            _, matches = match_one(position, image, base_rows)
            matched_by_image[position] = matches
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(match_one, position, image, base_rows)
                for position, (image, base_rows) in enumerate(image_jobs)
            ]
            completed = iter_progress(
                as_completed(futures),
                enabled=progress,
                total=len(futures),
                desc=f"common errors match {kind}",
                leave=progress_leave,
            )
            for future in completed:
                position, matches = future.result()
                matched_by_image[position] = matches

    return [
        match
        for matches in matched_by_image
        if matches is not None
        for match in matches
    ]


def _find_common_matches_for_image(
    base_rows: list[ErrorDetail],
    rows_by_run: list[list[ErrorDetail]],
    runs: list[_ErrorRun],
    *,
    kind: str,
    iou: float,
) -> list[_CommonMatch]:
    used: list[set[int]] = [set() for _ in rows_by_run]
    matches: list[_CommonMatch] = []
    for base_index, base_row in enumerate(base_rows):
        if _primary_box(base_row, kind) is None:
            continue
        selected: list[tuple[int, ErrorDetail]] = [(0, base_row)]
        selected_indices = [base_index]
        for run_index in range(1, len(rows_by_run)):
            candidate = _best_match(
                base_row,
                rows_by_run[run_index],
                used[run_index],
                kind=kind,
                iou=iou,
            )
            if candidate is None:
                break
            candidate_index, candidate_row = candidate
            selected.append((run_index, candidate_row))
            selected_indices.append(candidate_index)
        if len(selected) != len(rows_by_run):
            continue
        for run_index, row_index in enumerate(selected_indices):
            used[run_index].add(row_index)
        matches.append(_CommonMatch(kind=kind, row=base_row, refs=[(runs[i], row) for i, row in selected]))
    return matches


def _best_match(
    reference: ErrorDetail,
    candidates: list[ErrorDetail],
    used: set[int],
    *,
    kind: str,
    iou: float,
) -> tuple[int, ErrorDetail] | None:
    reference_box = _primary_box(reference, kind)
    if reference_box is None:
        return None
    best: tuple[float, float, int, ErrorDetail] | None = None
    for index, candidate in enumerate(candidates):
        if index in used or candidate.image != reference.image:
            continue
        if not _same_error_class(reference, candidate, kind):
            continue
        candidate_box = _primary_box(candidate, kind)
        if candidate_box is None:
            continue
        overlap = _box_iou(reference_box, candidate_box)
        if overlap < iou:
            continue
        confidence = candidate.pred_conf if candidate.pred_conf is not None else -1.0
        option = (overlap, confidence, index, candidate)
        if best is None or option[:3] > best[:3]:
            best = option
    return None if best is None else (best[2], best[3])


def _primary_box(row: ErrorDetail, kind: str) -> list[float] | None:
    return _parse_box_json(row.gt_box_xyxy if kind == "fn" else row.pred_box_xyxy)


def _same_error_class(left: ErrorDetail, right: ErrorDetail, kind: str) -> bool:
    if kind == "fn":
        left_id, right_id = left.gt_class_id, right.gt_class_id
        left_name, right_name = left.gt_class_name, right.gt_class_name
    else:
        left_id, right_id = left.pred_class_id, right.pred_class_id
        left_name, right_name = left.pred_class_name, right.pred_class_name
    if left_id is not None and right_id is not None:
        return left_id == right_id
    return bool(left_name and right_name and left_name == right_name)


def _box_iou(left: list[float], right: list[float]) -> float:
    matrix = box_iou_matrix(
        np.asarray([left], dtype=np.float64),
        np.asarray([right], dtype=np.float64),
    )
    return float(matrix[0, 0])


def _is_common_prediction_error(row: ErrorDetail) -> bool:
    if row.status != "fp":
        return False
    return row.error_type in {BACKGROUND_FP, CLASS_ERROR_PRED}


def _build_summary_rows(
    runs: list[_ErrorRun],
    common_fn: list[_CommonMatch],
    common_fp: list[_CommonMatch],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_fn_count = len(common_fn)
    common_fp_count = len(common_fp)
    for run in runs:
        original_fn = sum(row.status == "fn" for row in run.rows)
        original_fp = sum(row.status == "fp" for row in run.rows)
        rows.append(
            {
                "run_name": run.name,
                "input_dir": str(run.root),
                "original_fn": original_fn,
                "common_fn": common_fn_count,
                "remaining_fn": original_fn - common_fn_count,
                "original_fp": original_fp,
                "common_fp": common_fp_count,
                "remaining_fp": original_fp - common_fp_count,
                "original_total": original_fn + original_fp,
                "common_total": common_fn_count + common_fp_count,
                "remaining_total": original_fn + original_fp - common_fn_count - common_fp_count,
            }
        )
    return rows


def _common_rows(
    matches: list[_CommonMatch],
    crop_report: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match, crop in zip(matches, crop_report, strict=True):
        source_run = crop.get("source_run") or match.refs[0][0].name
        source_dir = next(
            (str(run.root) for run, _row in match.refs if run.name == source_run),
            str(match.refs[0][0].root),
        )
        data = dict(match.row.__dict__)
        data.update(
            {
                "common_kind": match.kind,
                "source_run": source_run,
                "source_error_dir": source_dir,
                "matched_runs": ";".join(run.name for run, _row in match.refs),
            }
        )
        rows.append(data)
    return rows


@dataclass
class _PreparedCropCopy:
    report: dict[str, Any]
    source: Path | None = None
    destination: Path | None = None


def _copy_common_crops(
    matches: list[_CommonMatch],
    destination_root: Path,
    *,
    workers: int,
    progress: bool,
    progress_leave: bool,
) -> list[dict[str, Any]]:
    sources = _find_common_crop_sources(
        matches,
        workers=workers,
        progress=progress,
        progress_leave=progress_leave,
    )
    reserved: set[str] = set()
    prepared = [
        _prepare_common_crop(match, source_info, destination_root, reserved)
        for match, source_info in zip(matches, sources, strict=True)
    ]
    reports: list[dict[str, Any] | None] = [None] * len(prepared)
    worker_count = normalize_workers(workers)
    if worker_count == 1:
        items = iter_progress(
            list(enumerate(prepared)),
            enabled=progress,
            total=len(prepared),
            desc="extract common error crops",
            leave=progress_leave,
        )
        for position, item in items:
            reports[position] = _copy_prepared_crop(item)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_position = {
                executor.submit(_copy_prepared_crop, item): position
                for position, item in enumerate(prepared)
            }
            completed = iter_progress(
                as_completed(future_to_position),
                enabled=progress,
                total=len(future_to_position),
                desc="extract common error crops",
                leave=progress_leave,
            )
            for future in completed:
                position = future_to_position[future]
                reports[position] = future.result()
    return [report for report in reports if report is not None]


def _find_common_crop_sources(
    matches: list[_CommonMatch],
    *,
    workers: int,
    progress: bool,
    progress_leave: bool,
) -> list[tuple[_ErrorRun, Path] | None]:
    worker_count = normalize_workers(workers)
    sources: list[tuple[_ErrorRun, Path] | None] = [None] * len(matches)

    def find_one(position: int, match: _CommonMatch) -> tuple[int, tuple[_ErrorRun, Path] | None]:
        for run, row in match.refs:
            source = _find_error_crop(run.root, row)
            if source is not None:
                return position, (run, source)
        return position, None

    if worker_count == 1:
        items = iter_progress(
            list(enumerate(matches)),
            enabled=progress,
            total=len(matches),
            desc="locate common error crops",
            leave=progress_leave,
        )
        for position, match in items:
            _, source_info = find_one(position, match)
            sources[position] = source_info
        return sources

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_position = {
            executor.submit(find_one, position, match): position
            for position, match in enumerate(matches)
        }
        completed = iter_progress(
            as_completed(future_to_position),
            enabled=progress,
            total=len(future_to_position),
            desc="locate common error crops",
            leave=progress_leave,
        )
        for future in completed:
            position, source_info = future.result()
            sources[position] = source_info
    return sources


def _prepare_common_crop(
    match: _CommonMatch,
    source_info: tuple[_ErrorRun, Path] | None,
    destination_root: Path,
    reserved: set[str],
) -> _PreparedCropCopy:
    if source_info is None:
        return _PreparedCropCopy(
            report={
                "common_kind": match.kind,
                "source_run": None,
                "source_crop": None,
                "destination_crop": None,
            }
        )

    run, source = source_info
    row = match.row
    group = _review_group_name(row)
    destination_dir = destination_root / group
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    destination_key = str(destination)
    if destination.exists() or destination_key in reserved:
        prefix = _safe_file_name(run.name)
        destination = destination_dir / f"{prefix}_{source.name}"
        counter = 2
        while destination.exists() or str(destination) in reserved:
            destination = destination_dir / f"{prefix}_{counter}_{source.name}"
            counter += 1
    reserved.add(str(destination))
    return _PreparedCropCopy(
        report={
            "common_kind": match.kind,
            "source_run": run.name,
            "source_crop": str(source),
            "destination_crop": str(destination),
        },
        source=source,
        destination=destination,
    )


def _copy_prepared_crop(item: _PreparedCropCopy) -> dict[str, Any]:
    if item.source is not None and item.destination is not None:
        shutil.copy2(item.source, item.destination)
    return item.report


def _find_error_crop(root: Path, row: ErrorDetail) -> Path | None:
    review_root = root / "review" if (root / "review").is_dir() else root
    crop_dir = review_root / _review_group_name(row) / "crops"
    if not crop_dir.is_dir():
        return None

    pred_id = "none" if row.error_type == FN_NO_PRED else str(row.pred_idx) if row.pred_idx is not None else "none"
    gt_id = "none" if row.error_type == BACKGROUND_FP else str(row.gt_idx) if row.gt_idx is not None else "none"
    prefix = f"{_safe_file_name(row.image)}_pred{pred_id}_gt{gt_id}"
    for path in sorted(crop_dir.rglob("*")):
        if path.is_file() and is_image_file(path) and path.stem == prefix:
            return path
    return None


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_iou(value: float) -> None:
    if not 0.0 < float(value) <= 1.0:
        raise ValueError("iou must be greater than 0 and no greater than 1")


def _parse_optional_text(value: str | None) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _parse_int(value: str | None, *, default: int) -> int:
    parsed = _parse_optional_int(value)
    return default if parsed is None else parsed


def _parse_optional_int(value: str | None) -> int | None:
    text = _parse_optional_text(value)
    if text is None:
        return None
    return int(float(text))


def _parse_float(value: str | None, *, default: float) -> float:
    text = _parse_optional_text(value)
    return default if text is None else float(text)


def _parse_optional_float(value: str | None) -> float | None:
    text = _parse_optional_text(value)
    return None if text is None else float(text)
