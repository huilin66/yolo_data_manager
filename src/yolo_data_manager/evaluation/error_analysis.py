"""Fine-grained prediction-vs-ground-truth error analysis.

Categorises every unmatched prediction and ground-truth annotation into
detailed error subtypes (background FP, localisation FP, duplicate
prediction, class-error, missed-due-to-class-error, missed-due-to-low-iou,
missed-no-prediction) and detects duplicate / overlapping GT annotations.

The matching logic is adapted from the ``demo.py`` reference script.
"""

from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np
from PIL import Image, ImageDraw

from yolo_data_manager.core.models import ClassSchema, YoloAnnotation, YoloDataset, YoloImage, is_image_file
from yolo_data_manager.io.loader import load_yolo_dataset, parse_label_file

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error type constants
# ---------------------------------------------------------------------------

#: Matched prediction – correct class and sufficient IoU.
TP = "tp"
#: Unmatched prediction with best IoU < low_iou – likely background or
#: unlabelled object.
BACKGROUND_FP = "background_fp"
#: Unmatched prediction whose best IoU is between low_iou and match_iou,
#: or whose best-matching GT was already claimed by a higher-confidence
#: prediction of the same class.
LOCALISATION_FP = "localisation_fp"
#: Unmatched prediction whose best IoU >= match_iou but the best GT was
#: already matched to another prediction (duplicate / over-detection).
DUPLICATE_PREDICTION = "duplicate_prediction"
#: Unmatched prediction where best IoU >= match_iou with a GT of a
#: *different* class (class confusion by the model).
CLASS_ERROR_PRED = "class_error_pred"
#: Unmatched GT whose best-matching prediction is of a different class
#: (the GT was "stolen" by a class-error prediction).
FN_CLASS_ERROR = "fn_class_error"
#: Unmatched GT with a prediction whose IoU is in [low_iou, match_iou).
FN_LOW_IOU = "fn_low_iou"
#: Unmatched GT with no prediction above low_iou – the model completely
#: missed this object.
FN_NO_PRED = "fn_no_pred"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ErrorDetail:
    """A single error (or TP) entry from prediction-vs-GT analysis."""

    image: str
    """Image file name (stem or relative path)."""

    status: str
    """High-level status: ``"tp"``, ``"fp"``, or ``"fn"``."""

    error_type: str
    """Fine-grained error subtype (one of the module-level constants)."""

    class_id: int
    """Class id of the primary annotation (GT for FN, pred for FP/TP)."""

    class_name: str
    """Human-readable class name."""

    pred_class_id: int | None = None
    pred_class_name: str | None = None
    pred_conf: float | None = None

    gt_class_id: int | None = None
    gt_class_name: str | None = None

    best_iou: float = 0.0
    """Best IoU with any annotation in the other set."""

    pred_box_xyxy: str | None = None
    """Normalised xyxy box of the prediction (JSON list string)."""

    gt_box_xyxy: str | None = None
    """Normalised xyxy box of the GT (JSON list string)."""

    pred_line: str | None = None
    """Raw text of the prediction label line."""

    gt_line: str | None = None
    """Raw text of the GT label line."""

    pred_idx: int | None = None
    """Index of this prediction within its label file."""

    gt_idx: int | None = None
    """Index of this GT within its label file."""


@dataclass
class DuplicateGt:
    """A pair of GT annotations on the same image with high overlap."""

    image: str
    gt_idx_i: int
    gt_idx_j: int
    iou: float
    cls_i: int
    name_i: str
    cls_j: int
    name_j: str
    type: str  # "same_class_duplicate" | "overlap_different_class"
    line_i: str = ""
    line_j: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _xywh_to_xyxy_norm(box: tuple[float, float, float, float]) -> list[float]:
    """Convert normalised *cx-cy-w-h* to normalised *x1-y1-x2-y2*."""
    cx, cy, w, h = box
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _annotation_box_norm(ann: YoloAnnotation) -> list[float]:
    """Return the normalised *x1-y1-x2-y2* bounding box of *ann*."""
    geo = ann.geometry_box()
    if geo is None:
        return [0.0, 0.0, 0.0, 0.0]
    return _xywh_to_xyxy_norm(geo.as_tuple())


def _box_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Vectorised IoU matrix (N x M) for two sets of xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0, None
    )
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def _serialise_box(box: list[float]) -> str:
    """JSON-list string for a box, rounded to 6 decimal places."""
    return "[" + ", ".join(f"{v:.6f}" for v in box) + "]"


def read_eval_class_schema(path: str | Path | None) -> ClassSchema:
    """Read a class-name file, accepting either ``name`` or ``id name`` lines."""
    if path is None:
        return ClassSchema([])
    class_path = Path(path)
    if not class_path.exists():
        return ClassSchema([])

    names_by_id: dict[int, str] = {}
    fallback_names: list[str] = []
    for idx, raw_line in enumerate(class_path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            names_by_id[int(parts[0])] = parts[1]
        else:
            fallback_names.append(line)
            names_by_id.setdefault(idx, line)

    if names_by_id:
        max_id = max(names_by_id)
        return ClassSchema([names_by_id.get(idx, str(idx)) for idx in range(max_id + 1)])
    return ClassSchema(fallback_names)


def collect_stems_from_source(source: str | Path | None) -> set[str] | None:
    """Collect image/label stems from an image directory or txt list."""
    if source is None:
        return None
    source_path = Path(source)
    stems: set[str] = set()
    if source_path.is_dir():
        for path in source_path.rglob("*"):
            if path.is_file() and (is_image_file(path) or path.suffix.lower() == ".txt"):
                stems.add(path.stem)
        return stems
    if source_path.is_file():
        for line in source_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                stems.add(Path(text).stem)
        return stems
    raise FileNotFoundError(f"val_source not found: {source_path}")


def load_error_analysis_dataset(
    root: str | Path,
    *,
    task: str = "auto",
    layout: str = "auto",
    images_dir: str | Path = "images",
    labels_dir: str | Path = "labels",
    class_file: str | Path | None = None,
    stems: set[str] | None = None,
) -> YoloDataset:
    """Load either a full YOLO dataset root or a plain label-txt directory."""
    root_path = Path(root)
    if _looks_like_label_dir(root_path, images_dir=images_dir, labels_dir=labels_dir):
        return _load_label_dir_dataset(root_path, task=task, class_file=class_file, stems=stems)

    dataset = load_yolo_dataset(
        root_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_file=class_file,
        task=task,
        layout=layout,
    )
    if class_file is not None:
        dataset.classes = read_eval_class_schema(class_file)
    return _filter_dataset_by_stems(dataset, stems)


def _looks_like_label_dir(root: Path, *, images_dir: str | Path, labels_dir: str | Path) -> bool:
    if not root.is_dir():
        return False
    image_root = root / images_dir
    label_root = root / labels_dir
    return not image_root.exists() and not label_root.exists() and any(root.glob("*.txt"))


def _load_label_dir_dataset(
    label_dir: Path,
    *,
    task: str,
    class_file: str | Path | None,
    stems: set[str] | None,
) -> YoloDataset:
    labels = sorted(path for path in label_dir.glob("*.txt") if stems is None or path.stem in stems)
    images: list[YoloImage] = []
    for label_path in labels:
        annotations = parse_label_file(label_path, task=task)
        images.append(
            YoloImage(
                path=label_dir / f"{label_path.stem}.jpg",
                label_path=label_path,
                annotations=annotations,
            )
        )
    return YoloDataset(
        root=label_dir,
        images=images,
        classes=read_eval_class_schema(class_file),
        task=task,
    )


def _filter_dataset_by_stems(dataset: YoloDataset, stems: set[str] | None) -> YoloDataset:
    if stems is None:
        return dataset
    return YoloDataset(
        root=dataset.root,
        images=[image for image in dataset.images if image.stem in stems],
        classes=dataset.classes,
        attributes=dataset.attributes,
        task=dataset.task,
        orphan_labels=[path for path in dataset.orphan_labels if path.stem in stems],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_errors(
    gt: YoloDataset,
    pred: YoloDataset,
    match_iou: float = 0.5,
    low_iou: float = 0.1,
    conf_thres: float = 0.0,
) -> tuple[list[ErrorDetail], dict[str, int]]:
    """Compare predictions against ground-truth with fine-grained error typing.

    Parameters
    ----------
    gt:
        Ground-truth dataset.
    pred:
        Prediction dataset.  Images are matched to GT by stem.
    match_iou:
        IoU threshold for a correct (TP) match.
    low_iou:
        IoU threshold below which an unmatched prediction is considered
        a background / false-alarm FP rather than a localisation issue.
    conf_thres:
        Minimum confidence for a prediction to be considered (predictions
        with ``confidence is None`` are always kept).

    Returns
    -------
    rows:
        One :class:`ErrorDetail` per annotation outcome.
    summary:
        Dict mapping error-type string to count.
    """
    pred_by_stem: dict[str, list[YoloAnnotation]] = {}
    pred_lines_by_stem: dict[str, list[str]] = {}
    for image in pred.images:
        annotations = list(image.annotations)
        if conf_thres > 0:
            annotations = [
                a
                for a in annotations
                if a.confidence is None or a.confidence >= conf_thres
            ]
        pred_by_stem[image.stem] = annotations
        pred_lines_by_stem[image.stem] = [a.source_line for a in annotations]

    rows: list[ErrorDetail] = []
    type_counter: Counter[str] = Counter()

    for gt_image in gt.images:
        stem = gt_image.stem
        gt_anns = gt_image.annotations
        pred_anns = pred_by_stem.get(stem, [])
        pred_lines = pred_lines_by_stem.get(stem, [])

        # --- build box arrays (normalised xyxy) ---
        gt_boxes_np = (
            np.array([_annotation_box_norm(a) for a in gt_anns], dtype=np.float64)
            if gt_anns
            else np.zeros((0, 4), dtype=np.float64)
        )
        pred_boxes_np = (
            np.array([_annotation_box_norm(a) for a in pred_anns], dtype=np.float64)
            if pred_anns
            else np.zeros((0, 4), dtype=np.float64)
        )

        ious = _box_iou_matrix(pred_boxes_np, gt_boxes_np)

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()

        # ---- 1. class-aware greedy matching (same class, IoU >= match_iou) ----
        candidates: list[tuple[float, int, int]] = []
        for pi, p_ann in enumerate(pred_anns):
            for gi, g_ann in enumerate(gt_anns):
                if p_ann.class_id == g_ann.class_id and ious[pi, gi] >= match_iou:
                    candidates.append((float(ious[pi, gi]), pi, gi))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for iou_val, pi, gi in candidates:
            if pi not in matched_pred and gi not in matched_gt:
                matched_pred.add(pi)
                matched_gt.add(gi)
                p_ann = pred_anns[pi]
                g_ann = gt_anns[gi]
                conf = p_ann.confidence
                rows.append(
                    ErrorDetail(
                        image=stem,
                        status="tp",
                        error_type=TP,
                        class_id=p_ann.class_id,
                        class_name=gt.class_name(p_ann.class_id),
                        pred_class_id=p_ann.class_id,
                        pred_class_name=gt.class_name(p_ann.class_id),
                        pred_conf=conf,
                        gt_class_id=g_ann.class_id,
                        gt_class_name=gt.class_name(g_ann.class_id),
                        best_iou=iou_val,
                        pred_box_xyxy=_serialise_box(_annotation_box_norm(p_ann)),
                        gt_box_xyxy=_serialise_box(_annotation_box_norm(g_ann)),
                        pred_line=p_ann.source_line,
                        gt_line=g_ann.source_line,
                        pred_idx=pi + 1,
                        gt_idx=gi + 1,
                    )
                )
                type_counter[TP] += 1

        # ---- 2. unmatched predictions → FP (with sub-type) ----
        for pi, p_ann in enumerate(pred_anns):
            if pi in matched_pred:
                continue

            pred_cls = p_ann.class_id
            conf = p_ann.confidence

            if len(gt_anns) > 0:
                best_gi = int(np.argmax(ious[pi]))
                best_iou = float(ious[pi, best_gi])
                best_gt = gt_anns[best_gi]
                best_gt_cls = best_gt.class_id
            else:
                best_gi = -1
                best_iou = 0.0
                best_gt = None
                best_gt_cls = -1

            error_type: str
            if best_gt is not None and best_iou >= match_iou and pred_cls != best_gt_cls:
                # High overlap but wrong class → class error (pred side)
                error_type = CLASS_ERROR_PRED
                gt_suffix = (
                    f" (gt) | cls_err: {gt.class_name(pred_cls)} → "
                    f"{gt.class_name(best_gt_cls)}"
                )
            elif (
                best_gt is not None
                and best_gi in matched_gt
                and best_iou >= match_iou
            ):
                # GT already claimed → duplicate / over-detection
                error_type = DUPLICATE_PREDICTION
                gt_suffix = (
                    f" (gt) | dup_pred: {gt.class_name(pred_cls)} "
                    f"(best GT already matched)"
                )
            elif best_iou < low_iou:
                error_type = BACKGROUND_FP
                gt_suffix = " (gt)" if best_gt is not None else ""
            else:
                error_type = LOCALISATION_FP
                gt_suffix = " (gt)" if best_gt is not None else ""

            rows.append(
                ErrorDetail(
                    image=stem,
                    status="fp",
                    error_type=error_type,
                    class_id=pred_cls,
                    class_name=gt.class_name(pred_cls),
                    pred_class_id=pred_cls,
                    pred_class_name=gt.class_name(pred_cls),
                    pred_conf=conf,
                    gt_class_id=best_gt_cls if best_gt_cls >= 0 else None,
                    gt_class_name=gt.class_name(best_gt_cls) if best_gt_cls >= 0 else None,
                    best_iou=best_iou,
                    pred_box_xyxy=_serialise_box(_annotation_box_norm(p_ann)),
                    gt_box_xyxy=(
                        _serialise_box(_annotation_box_norm(best_gt))
                        if best_gt is not None
                        else None
                    ),
                    pred_line=p_ann.source_line,
                    gt_line=best_gt.source_line if best_gt is not None else None,
                    pred_idx=pi + 1,
                    gt_idx=best_gi + 1 if best_gi >= 0 else None,
                )
            )
            type_counter[error_type] += 1

        # ---- 3. unmatched GT → FN (with sub-type) ----
        for gi, g_ann in enumerate(gt_anns):
            if gi in matched_gt:
                continue

            gt_cls = g_ann.class_id

            if len(pred_anns) > 0:
                best_pi = int(np.argmax(ious[:, gi]))
                best_iou = float(ious[best_pi, gi])
                best_pred = pred_anns[best_pi]
                best_pred_cls = best_pred.class_id
                best_pred_conf = best_pred.confidence
            else:
                best_pi = -1
                best_iou = 0.0
                best_pred = None
                best_pred_cls = -1
                best_pred_conf = None

            error_type: str
            if best_pred is not None and best_iou >= match_iou and best_pred_cls != gt_cls:
                error_type = FN_CLASS_ERROR
            elif best_pred is not None and best_iou >= low_iou:
                error_type = FN_LOW_IOU
            else:
                error_type = FN_NO_PRED

            rows.append(
                ErrorDetail(
                    image=stem,
                    status="fn",
                    error_type=error_type,
                    class_id=gt_cls,
                    class_name=gt.class_name(gt_cls),
                    pred_class_id=best_pred_cls if best_pred_cls >= 0 else None,
                    pred_class_name=(
                        gt.class_name(best_pred_cls) if best_pred_cls >= 0 else None
                    ),
                    pred_conf=best_pred_conf,
                    gt_class_id=gt_cls,
                    gt_class_name=gt.class_name(gt_cls),
                    best_iou=best_iou,
                    pred_box_xyxy=(
                        _serialise_box(_annotation_box_norm(best_pred))
                        if best_pred is not None
                        else None
                    ),
                    gt_box_xyxy=_serialise_box(_annotation_box_norm(g_ann)),
                    pred_line=best_pred.source_line if best_pred is not None else None,
                    gt_line=g_ann.source_line,
                    pred_idx=best_pi + 1 if best_pi >= 0 else None,
                    gt_idx=gi + 1,
                )
            )
            type_counter[error_type] += 1

    # ---- 4. predictions with no matching GT image stem → FP ----
    gt_stems = {img.stem for img in gt.images}
    for pred_image in pred.images:
        stem = pred_image.stem
        if stem in gt_stems:
            continue
        for pi, p_ann in enumerate(pred_image.annotations):
            if conf_thres > 0 and p_ann.confidence is not None and p_ann.confidence < conf_thres:
                continue
            rows.append(
                ErrorDetail(
                    image=stem,
                    status="fp",
                    error_type=BACKGROUND_FP,
                    class_id=p_ann.class_id,
                    class_name=pred.class_name(p_ann.class_id),
                    pred_class_id=p_ann.class_id,
                    pred_class_name=pred.class_name(p_ann.class_id),
                    pred_conf=p_ann.confidence,
                    best_iou=0.0,
                    pred_box_xyxy=_serialise_box(_annotation_box_norm(p_ann)),
                    pred_line=p_ann.source_line,
                    pred_idx=pi + 1,
                )
            )
            type_counter[BACKGROUND_FP] += 1

    summary = dict(type_counter)
    return rows, summary


def find_duplicate_gt(
    dataset: YoloDataset,
    duplicate_iou: float = 0.9,
) -> list[DuplicateGt]:
    """Find highly-overlapping GT annotation pairs on the same image.

    Parameters
    ----------
    dataset:
        Ground-truth dataset to inspect.
    duplicate_iou:
        IoU threshold above which two annotations are flagged as
        potential duplicates.

    Returns
    -------
    rows:
        One :class:`DuplicateGt` per overlapping pair.
    """
    rows: list[DuplicateGt] = []
    for image in dataset.images:
        anns = image.annotations
        if len(anns) <= 1:
            continue
        boxes = np.array(
            [_annotation_box_norm(a) for a in anns], dtype=np.float64
        )
        ious = _box_iou_matrix(boxes, boxes)
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if ious[i, j] >= duplicate_iou:
                    rows.append(
                        DuplicateGt(
                            image=image.stem,
                            gt_idx_i=i + 1,
                            gt_idx_j=j + 1,
                            iou=float(ious[i, j]),
                            cls_i=anns[i].class_id,
                            name_i=dataset.class_name(anns[i].class_id),
                            cls_j=anns[j].class_id,
                            name_j=dataset.class_name(anns[j].class_id),
                            type=(
                                "same_class_duplicate"
                                if anns[i].class_id == anns[j].class_id
                                else "overlap_different_class"
                            ),
                            line_i=anns[i].source_line,
                            line_j=anns[j].source_line,
                        )
                    )
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_ERROR_CSV_COLUMNS = [
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

_DUP_CSV_COLUMNS = [
    "image",
    "gt_idx_i",
    "gt_idx_j",
    "iou",
    "cls_i",
    "name_i",
    "cls_j",
    "name_j",
    "type",
    "line_i",
    "line_j",
]


def write_error_csvs(
    error_rows: list[ErrorDetail],
    out_dir: str | Path,
) -> None:
    """Write error-analysis CSV files to *out_dir*.

    Produces four files:

    * ``fp_report.csv`` — all false-positive entries
    * ``fn_report.csv`` — all false-negative entries
    * ``class_error.csv`` — class-confusion entries (pred side)
    * ``tp_report.csv`` — correctly matched entries
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_rows(
        out / "fp_report.csv",
        _ERROR_CSV_COLUMNS,
        [r for r in error_rows if r.status == "fp"],
    )
    _write_rows(
        out / "fn_report.csv",
        _ERROR_CSV_COLUMNS,
        [r for r in error_rows if r.status == "fn"],
    )
    _write_rows(
        out / "class_error.csv",
        _ERROR_CSV_COLUMNS,
        [
            r
            for r in error_rows
            if r.error_type in (CLASS_ERROR_PRED, FN_CLASS_ERROR)
        ],
    )
    _write_rows(
        out / "tp_report.csv",
        _ERROR_CSV_COLUMNS,
        [r for r in error_rows if r.status == "tp"],
    )
    # Compatibility with the original demo.py report names.
    _write_rows(
        out / "false_positive_background.csv",
        _ERROR_CSV_COLUMNS,
        [r for r in error_rows if r.status == "fp"],
    )
    _write_rows(
        out / "false_negative_missed_gt.csv",
        _ERROR_CSV_COLUMNS,
        [r for r in error_rows if r.status == "fn"],
    )


def write_duplicate_gt_csv(
    dup_rows: list[DuplicateGt],
    out_dir: str | Path,
) -> None:
    """Write duplicate-GT CSV to *out_dir / duplicate_gt.csv*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "duplicate_gt.csv", _DUP_CSV_COLUMNS, dup_rows)


def write_error_review_pack(
    error_rows: list[ErrorDetail],
    gt: YoloDataset,
    pred: YoloDataset,
    out_dir: str | Path,
    *,
    crop_padding: int = 12,
    workers: int = 1,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, int]:
    """Write visual review images and crops grouped by error type.

    The review pack is intentionally optional because it opens and writes many
    image files.  It creates:

    * ``review/<group>/images`` — full images with GT / prediction boxes
    * ``review/<group>/crops`` — local crops around the relevant box

    Review rows are grouped under ``pred_gt/pred_<pred_class>_gt_<gt_class>``
    using the same class/background coordinates as the Ultralytics-style
    confusion matrix.  For example, false positives become
    ``pred_<class>_gt_background`` and false negatives become
    ``pred_background_gt_<class>``.
    """
    output = Path(out_dir) / "review"
    output.mkdir(parents=True, exist_ok=True)
    gt_images = _images_by_stem(gt)
    pred_images = _images_by_stem(pred)
    counts: dict[str, int] = Counter()
    worker_count = max(1, int(workers))

    work_items = [(idx, row) for idx, row in enumerate(error_rows, start=1) if row.status != "tp"]
    if worker_count == 1:
        iterator = (
            _write_one_error_review(item, gt_images, pred_images, output, crop_padding)
            for item in work_items
        )
        for error_type in _progress(iterator, enabled=progress, total=len(work_items), desc="error review", leave=progress_leave):
            if error_type:
                counts[error_type] += 1
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_write_one_error_review, item, gt_images, pred_images, output, crop_padding)
                for item in work_items
            ]
            for future in _progress(as_completed(futures), enabled=progress, total=len(futures), desc="error review", leave=progress_leave):
                error_type = future.result()
                if error_type:
                    counts[error_type] += 1

    _write_ultralytics_confusion_matrix(error_rows, gt, pred, output / "pred_gt")
    return dict(counts)


def copy_prediction_txt_to_review(
    pred: YoloDataset,
    out_dir: str | Path,
    *,
    stems: set[str] | None = None,
) -> list[Path]:
    """Copy prediction label txt files to ``review/pred_txt``."""
    pred_txt_dir = Path(out_dir) / "review" / "pred_txt"
    pred_txt_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for image in pred.images:
        if stems is not None and image.stem not in stems and image.file_name not in stems:
            continue
        if image.label_path is None or not image.label_path.exists():
            continue
        out_path = pred_txt_dir / image.label_path.name
        shutil.copy2(image.label_path, out_path)
        copied.append(out_path)
    return copied


def print_error_summary(
    error_rows: list[ErrorDetail],
    dup_rows: list[DuplicateGt] | None = None,
) -> None:
    """Print a human-readable summary of error analysis results."""
    status_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    gt_class_counter: Counter[str] = Counter()
    pred_class_counter: Counter[str] = Counter()

    for r in error_rows:
        status_counter[r.status] += 1
        type_counter[r.error_type] += 1
        if r.status == "fn":
            gt_class_counter[f"{r.class_id} {r.class_name}"] += 1
        elif r.status in ("fp", "tp"):
            pred_class_counter[f"{r.class_id} {r.class_name}"] += 1
            if r.gt_class_id is not None and r.gt_class_name:
                gt_class_counter[f"{r.gt_class_id} {r.gt_class_name}"] += 1

    dup_count = len(dup_rows) if dup_rows else 0

    print("\n========== ERROR ANALYSIS SUMMARY ==========")
    print(f"Total rows: {len(error_rows)}")
    print(f"  TP: {status_counter.get('tp', 0)}")
    print(f"  FP: {status_counter.get('fp', 0)}")
    print(f"  FN: {status_counter.get('fn', 0)}")
    print(f"  Duplicate GT pairs: {dup_count}")
    print("\nError type breakdown:")
    for etype, count in sorted(type_counter.items()):
        print(f"  {etype}: {count}")

    if gt_class_counter:
        print("\nGT class distribution:")
        for cls_key, count in sorted(gt_class_counter.items()):
            print(f"  {cls_key}: {count}")

    if pred_class_counter:
        print("\nPred class distribution:")
        for cls_key, count in sorted(pred_class_counter.items()):
            print(f"  {cls_key}: {count}")
    print("==============================================\n")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _write_rows(
    path: Path,
    columns: list[str],
    rows: list[Any],
) -> None:
    """Write a list of objects/dataclasses to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            d = row.__dict__ if hasattr(row, "__dict__") else row
            writer.writerow(d)


def _images_by_stem(dataset: YoloDataset) -> dict[str, YoloImage]:
    return {image.stem: image for image in dataset.images}


def _write_one_error_review(
    item: tuple[int, ErrorDetail],
    gt_images: dict[str, YoloImage],
    pred_images: dict[str, YoloImage],
    output: Path,
    crop_padding: int,
) -> str | None:
    idx, row = item
    source_image = gt_images.get(row.image) or pred_images.get(row.image)
    if source_image is None or not source_image.path.exists():
        return None
    box = _row_primary_box(row)
    if box is None:
        return None

    group_name = _review_group_name(row)
    type_dir = output / group_name
    image_dir = type_dir / "images"
    crop_dir = type_dir / "crops"
    image_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_image.path) as src:
        canvas = src.convert("RGB")
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    _draw_review_box(draw, row.pred_box_xyxy, width, height, outline=(255, 42, 4), label="pred")
    _draw_review_box(draw, row.gt_box_xyxy, width, height, outline=(0, 192, 38), label="gt")

    safe_image = _safe_file_name(row.image)
    image_name = f"{safe_image}_{idx}_{row.status}_{row.error_type}{source_image.path.suffix}"
    canvas.save(image_dir / image_name)

    crop = _crop_norm_box(canvas, box, padding=crop_padding)
    crop.save(crop_dir / image_name)
    return group_name


def _review_group_name(row: ErrorDetail) -> str:
    pred_name = _class_label(row.pred_class_id, row.pred_class_name)
    gt_name = _class_label(row.gt_class_id, row.gt_class_name)
    if row.status == "fp" and row.error_type not in {CLASS_ERROR_PRED, FN_CLASS_ERROR}:
        gt_name = "background"
    elif row.status == "fn" and row.error_type not in {CLASS_ERROR_PRED, FN_CLASS_ERROR}:
        pred_name = "background"
    return f"pred_gt/pred_{_safe_file_name(pred_name)}_gt_{_safe_file_name(gt_name)}"


def _class_label(class_id: int | None, class_name: str | None) -> str:
    if class_name:
        return class_name
    if class_id is not None:
        return str(class_id)
    return "unknown"


def _write_ultralytics_confusion_matrix(
    error_rows: list[ErrorDetail],
    gt: YoloDataset,
    pred: YoloDataset,
    out_dir: Path,
) -> Path | None:
    matrix, labels = _ultralytics_confusion_matrix_data(error_rows, gt, pred)
    if matrix.size == 0:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix.png"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = min(24.0, max(6.0, 0.55 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    max_count = int(matrix.max()) if matrix.size else 0
    threshold = max_count / 2 if max_count else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            if value:
                color = "white" if value > threshold else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _ultralytics_confusion_matrix_data(
    error_rows: list[ErrorDetail],
    gt: YoloDataset,
    pred: YoloDataset,
) -> tuple[np.ndarray, list[str]]:
    max_class_id = _max_matrix_class_id(error_rows, gt, pred)
    if max_class_id < 0:
        return np.zeros((0, 0), dtype=np.int64), []

    class_ids = list(range(max_class_id + 1))
    labels = [_matrix_class_name(class_id, gt, pred) for class_id in class_ids] + ["background"]
    background_idx = len(labels) - 1
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    seen: set[tuple[Any, ...]] = set()

    for idx, row in enumerate(error_rows):
        pred_idx = row.pred_class_id
        gt_idx = row.gt_class_id

        if row.status == "tp":
            if pred_idx is not None and gt_idx is not None:
                matrix[pred_idx, gt_idx] += 1
            continue

        if row.error_type in {CLASS_ERROR_PRED, FN_CLASS_ERROR}:
            if pred_idx is None or gt_idx is None:
                continue
            key = _class_confusion_key(row, idx)
            if key in seen:
                continue
            seen.add(key)
            matrix[pred_idx, gt_idx] += 1
            continue

        if row.status == "fp":
            if pred_idx is not None:
                matrix[pred_idx, background_idx] += 1
        elif row.status == "fn":
            if gt_idx is not None:
                matrix[background_idx, gt_idx] += 1

    return matrix, labels


def _max_matrix_class_id(error_rows: list[ErrorDetail], gt: YoloDataset, pred: YoloDataset) -> int:
    max_id = max(len(gt.classes.names), len(pred.classes.names)) - 1
    for row in error_rows:
        for class_id in (row.pred_class_id, row.gt_class_id, row.class_id):
            if class_id is not None:
                max_id = max(max_id, int(class_id))
    return max_id


def _matrix_class_name(class_id: int, gt: YoloDataset, pred: YoloDataset) -> str:
    if 0 <= class_id < len(gt.classes.names):
        return gt.classes.names[class_id]
    if 0 <= class_id < len(pred.classes.names):
        return pred.classes.names[class_id]
    return str(class_id)


def _class_confusion_key(row: ErrorDetail, idx: int) -> tuple[Any, ...]:
    return (
        row.image,
        row.pred_idx if row.pred_idx is not None else f"pred_row_{idx}",
        row.gt_idx if row.gt_idx is not None else f"gt_row_{idx}",
        row.pred_class_id,
        row.gt_class_id,
    )


def _row_primary_box(row: ErrorDetail) -> list[float] | None:
    if row.status == "fn":
        return _parse_box_json(row.gt_box_xyxy)
    return _parse_box_json(row.pred_box_xyxy) or _parse_box_json(row.gt_box_xyxy)


def _parse_box_json(text: str | None) -> list[float] | None:
    if not text:
        return None
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list) or len(values) != 4:
        return None
    return [float(value) for value in values]


def _draw_review_box(
    draw: ImageDraw.ImageDraw,
    box_text: str | None,
    width: int,
    height: int,
    *,
    outline: tuple[int, int, int],
    label: str,
) -> None:
    box = _parse_box_json(box_text)
    if box is None:
        return
    left, top, right, bottom = _norm_box_to_pixels(box, width, height)
    draw.rectangle([left, top, right, bottom], outline=outline, width=3)
    draw.rectangle([left, max(0, top - 16), left + 42, top], fill=outline)
    draw.text((left + 2, max(0, top - 15)), label, fill=(0, 0, 0))


def _crop_norm_box(image: Image.Image, box: list[float], padding: int) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = _norm_box_to_pixels(box, width, height)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    return image.crop((left, top, right, bottom))


def _norm_box_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1 * width))))
    top = max(0, min(height, int(round(y1 * height))))
    right = max(0, min(width, int(round(x2 * width))))
    bottom = max(0, min(height, int(round(y2 * height))))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _safe_file_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _progress(items: Iterable[T], *, enabled: bool, total: int, desc: str, leave: bool) -> Iterable[T]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return _simple_progress(items, total=total, desc=desc)
    return tqdm(items, total=total, desc=desc, leave=leave)


def _simple_progress(items: Iterable[T], *, total: int, desc: str) -> Iterator[T]:
    step = max(1, total // 20) if total else 1
    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx == total or idx % step == 0:
            print(f"{desc}: {idx}/{total}")
        yield item
