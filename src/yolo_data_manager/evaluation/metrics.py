from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from yolo_data_manager.annotation.edit import merge_classes
from yolo_data_manager.core.errors import ClassNotFoundError
from yolo_data_manager.core.models import YoloAnnotation, YoloDataset
from yolo_data_manager.evaluation.matching import (
    greedy_match_indices,
    non_max_suppress_annotations,
    validate_nms_iou,
)


DEFAULT_IOU_THRESHOLDS = tuple(float(v) for v in np.linspace(0.5, 0.95, 10))
TARGET_SIZE_NAMES = ("small", "medium", "large")
TARGET_SIZE_THRESHOLDS = {
    "small_max_area": 32 * 32,
    "medium_max_area": 96 * 96,
}


@dataclass
class ClassMetric:
    class_id: int
    class_name: str
    images: int
    labels: int
    predictions: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap75: float
    map: float


@dataclass
class SizeMetric:
    size: str
    images: int
    labels: int
    predictions: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap75: float
    map: float
    fitness: float


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    map50: float
    map75: float
    map: float
    fitness: float
    labels: int
    predictions: int
    images: int
    classes: list[ClassMetric]
    selected_class_ids: list[int] | None
    excluded_class_ids: list[int] | None
    merge_class_map: dict[str, list[str]] | None
    iou_thresholds: list[float]
    nms_iou: float | None
    size_metrics: dict[str, SizeMetric]
    target_size_thresholds: dict[str, int]
    size_filter: dict[str, float | str | None]
    class_rules: dict[str, dict[str, Any]] | None
    ignore_empty_classes: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["report_type"] = "detection_metrics"
        return data


def compute_detection_metrics(
    gt: YoloDataset,
    pred: YoloDataset,
    *,
    class_ids: Iterable[int | str] | int | str | None = None,
    exclude_class_ids: Iterable[int | str] | int | str | None = None,
    merge_class_map: Mapping[int | str, int | str | Iterable[int | str]] | None = None,
    conf_thres: float = 0.0,
    nms_iou: float | None = 0.5,
    min_width: float | None = None,
    min_height: float | None = None,
    min_area: float | None = None,
    min_size_logic: str = "or",
    min_pixels: float | None = None,
    class_rules: Mapping[int | str, Mapping[str, Any]] | None = None,
    ignore_empty_classes: bool = True,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> DetectionMetrics:
    """Compute Ultralytics-style detection metrics from YOLO GT/prediction txt.

    Predictions without an explicit confidence are treated as confidence 1.0.
    Same-class predictions are deduplicated with confidence-prioritized NMS
    before matching; pass ``nms_iou=None`` to disable it.
    In addition to the aggregate metrics, ``size_metrics`` reports metrics for
    COCO-style target areas: small < 32² pixels, medium 32²–96² pixels, and
    large >= 96² pixels.  Images must have known dimensions for this breakdown.
    If *class_ids* is provided, GT and predictions outside that class set are
    ignored before matching and averaging. *exclude_class_ids* removes classes
    from evaluation even when no inclusion list is provided. If
    *merge_class_map* is provided, its target-to-source mapping is applied to
    both datasets before class selection, filtering, matching, and averaging.
    ``class_rules`` maps a class id or name to a size filter.  A matching rule
    replaces the global size filter for that class; classes without a rule use
    the global values.  Rule keys support ``width``/``min_width``,
    ``height``/``min_height``, ``min_area``, ``min_pixels``, and
    ``logic``/``min_size_logic``.
    """
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")
    validate_nms_iou(nms_iou)
    gt, pred, normalized_merge_map = _prepare_eval_datasets(gt, pred, merge_class_map)
    resolved_class_rules, normalized_class_rules = _resolve_eval_class_rules(
        gt, pred, class_rules
    )
    selected = None if class_ids is None else set(resolve_eval_class_ids(gt, class_ids))
    excluded = None if exclude_class_ids is None else set(resolve_eval_class_ids(gt, exclude_class_ids))
    if selected is not None and excluded:
        selected -= excluded
    iouv = np.array(iou_thresholds, dtype=np.float64)
    pred_by_stem = {image.stem: image for image in pred.images}

    tp_parts: list[np.ndarray] = []
    conf_values: list[float] = []
    pred_cls_values: list[int] = []
    target_cls_values: list[int] = []
    pred_count_by_class: dict[int, int] = {}
    image_count_by_class: dict[int, int] = {}
    size_target_cls_values: dict[str, list[int]] = {
        size: [] for size in TARGET_SIZE_NAMES
    }
    size_image_count: dict[str, int] = {size: 0 for size in TARGET_SIZE_NAMES}
    size_assignment_parts: list[str | None] = []

    seen_gt_stems: set[str] = set()
    for gt_image in gt.images:
        seen_gt_stems.add(gt_image.stem)
        pred_image = pred_by_stem.get(gt_image.stem)
        width = gt_image.width if gt_image.width is not None else (pred_image.width if pred_image is not None else None)
        height = gt_image.height if gt_image.height is not None else (pred_image.height if pred_image is not None else None)
        gt_anns = _filter_annotations(
            gt_image.annotations,
            selected,
            excluded=excluded,
            is_prediction=False,
            conf_thres=conf_thres,
            image_width=width,
            image_height=height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
            class_rules=resolved_class_rules,
        )
        pred_anns = _filter_annotations(
            pred_image.annotations if pred_image is not None else [],
            selected,
            excluded=excluded,
            is_prediction=True,
            conf_thres=conf_thres,
            image_width=width,
            image_height=height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
            class_rules=resolved_class_rules,
        )
        pred_anns = non_max_suppress_annotations(pred_anns, nms_iou)
        gt_sizes = [
            _target_size(ann, image_width=width, image_height=height)
            for ann in gt_anns
        ]
        pred_sizes = [
            _target_size(ann, image_width=width, image_height=height)
            for ann in pred_anns
        ]
        correct, matched_gt_sizes = _match_predictions_with_target_sizes(
            gt_anns,
            pred_anns,
            iouv,
            gt_sizes,
        )
        size_assignment_parts.extend(
            _assign_prediction_sizes(pred_sizes, matched_gt_sizes)
        )
        target_cls_values.extend(ann.class_id for ann in gt_anns)
        for class_id in {ann.class_id for ann in gt_anns}:
            image_count_by_class[class_id] = image_count_by_class.get(class_id, 0) + 1
        for size in set(gt_sizes):
            if size is not None:
                size_image_count[size] += 1
        for ann, size in zip(gt_anns, gt_sizes):
            if size is not None:
                size_target_cls_values[size].append(ann.class_id)
        for ann in pred_anns:
            pred_count_by_class[ann.class_id] = pred_count_by_class.get(ann.class_id, 0) + 1

        if pred_anns:
            tp_parts.append(correct)
            conf_values.extend(_confidence(ann) for ann in pred_anns)
            pred_cls_values.extend(ann.class_id for ann in pred_anns)

    for pred_image in pred.images:
        if pred_image.stem in seen_gt_stems:
            continue
        pred_anns = _filter_annotations(
            pred_image.annotations,
            selected,
            excluded=excluded,
            is_prediction=True,
            conf_thres=conf_thres,
            image_width=pred_image.width,
            image_height=pred_image.height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
            class_rules=resolved_class_rules,
        )
        pred_anns = non_max_suppress_annotations(pred_anns, nms_iou)
        if not pred_anns:
            continue
        pred_sizes = [
            _target_size(
                ann,
                image_width=pred_image.width,
                image_height=pred_image.height,
            )
            for ann in pred_anns
        ]
        size_assignment_parts.extend(pred_sizes)
        tp_parts.append(np.zeros((len(pred_anns), len(iouv)), dtype=bool))
        conf_values.extend(_confidence(ann) for ann in pred_anns)
        pred_cls_values.extend(ann.class_id for ann in pred_anns)
        for ann in pred_anns:
            pred_count_by_class[ann.class_id] = pred_count_by_class.get(ann.class_id, 0) + 1

    tp = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.array(conf_values, dtype=np.float64)
    pred_cls = np.array(pred_cls_values, dtype=np.int64)
    target_cls = np.array(target_cls_values, dtype=np.int64)
    size_assignments = np.array(size_assignment_parts, dtype=object)

    tp_count, fp_count, p, r, f1, ap, ap_class = _ap_per_class(tp, conf, pred_cls, target_cls)
    class_metrics = _build_class_metrics(
        gt,
        selected=selected,
        excluded=excluded,
        target_cls=target_cls,
        pred_count_by_class=pred_count_by_class,
        image_count_by_class=image_count_by_class,
        p=p,
        r=r,
        f1=f1,
        ap=ap,
        ap_class=ap_class,
    )
    if ignore_empty_classes:
        class_metrics = [row for row in class_metrics if row.labels > 0]

    metric_rows = [row for row in class_metrics if row.labels > 0 or row.predictions > 0]
    if metric_rows:
        precision = float(np.mean([row.precision for row in metric_rows]))
        recall = float(np.mean([row.recall for row in metric_rows]))
        map50 = float(np.mean([row.ap50 for row in metric_rows]))
        map75 = float(np.mean([row.ap75 for row in metric_rows]))
        map_value = float(np.mean([row.map for row in metric_rows]))
    else:
        precision = recall = map50 = map75 = map_value = 0.0

    size_metrics = _build_size_metrics(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        size_assignments=size_assignments,
        target_cls_by_size=size_target_cls_values,
        image_count_by_size=size_image_count,
        ignore_empty_classes=ignore_empty_classes,
    )

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map75=map75,
        map=map_value,
        fitness=map50 * 0.1 + map_value * 0.9,
        labels=int(len(target_cls)),
        predictions=int(len(pred_cls)),
        images=len(gt.images),
        classes=class_metrics,
        selected_class_ids=sorted(selected) if selected is not None else None,
        excluded_class_ids=sorted(excluded) if excluded is not None else None,
        merge_class_map=normalized_merge_map,
        iou_thresholds=[float(v) for v in iouv],
        nms_iou=None if nms_iou is None else float(nms_iou),
        size_metrics=size_metrics,
        target_size_thresholds=dict(TARGET_SIZE_THRESHOLDS),
        size_filter={
            "min_width": min_width,
            "min_height": min_height,
            "min_area": min_area,
            "min_size_logic": min_size_logic,
            "min_pixels": min_pixels,
        },
        class_rules=normalized_class_rules or None,
        ignore_empty_classes=ignore_empty_classes,
    )


def resolve_eval_class_ids(
    dataset: YoloDataset,
    values: Iterable[int | str] | int | str | None,
) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (int, str)):
        values = [values]
    return [dataset.class_id(value) for value in values]


def _resolve_eval_class_rules(
    gt: YoloDataset,
    pred: YoloDataset,
    class_rules: Mapping[int | str, Mapping[str, Any]] | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve per-class rules against GT, falling back to prediction names."""
    if not class_rules:
        return {}, {}
    resolved: dict[int, dict[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for class_value, rule in class_rules.items():
        if not isinstance(rule, Mapping):
            raise TypeError(
                f"class_rules[{class_value!r}] must be a mapping of filter options"
            )
        try:
            class_id = gt.class_id(class_value)
        except ClassNotFoundError:
            class_id = pred.class_id(class_value)
        resolved[class_id] = dict(rule)
        class_name = gt.class_name(class_id)
        if class_name == str(class_id) and pred.class_name(class_id) != str(class_id):
            class_name = pred.class_name(class_id)
        normalized[str(class_name)] = dict(rule)
    return resolved, normalized


def _prepare_eval_datasets(
    gt: YoloDataset,
    pred: YoloDataset,
    merge_class_map: Mapping[int | str, int | str | Iterable[int | str]] | None,
) -> tuple[YoloDataset, YoloDataset, dict[str, list[str]] | None]:
    if merge_class_map is None:
        return gt, pred, None

    normalized = _normalize_merge_class_map(merge_class_map)
    if not normalized:
        return gt, pred, {}

    gt_result = deepcopy(gt)
    pred_result = deepcopy(pred)
    if not gt_result.classes.names and pred_result.classes.names:
        gt_result.classes = deepcopy(pred_result.classes)
    elif not pred_result.classes.names and gt_result.classes.names:
        pred_result.classes = deepcopy(gt_result.classes)

    for target, sources in normalized.items():
        gt_result, _ = merge_classes(gt_result, sources, target, compact=True, add_missing=True)
        pred_result, _ = merge_classes(pred_result, sources, target, compact=True, add_missing=True)

    return gt_result, pred_result, {
        str(target): [str(source) for source in sources]
        for target, sources in normalized.items()
    }


def _normalize_merge_class_map(
    merge_class_map: Mapping[int | str, int | str | Iterable[int | str]],
) -> dict[int | str, list[int | str]]:
    if not isinstance(merge_class_map, Mapping):
        raise TypeError("merge_class_map must be a mapping of target class to source classes")

    normalized: dict[int | str, list[int | str]] = {}
    for target, sources in merge_class_map.items():
        if not isinstance(target, (int, str)):
            raise TypeError("merge_class_map targets must be class ids or names")
        if isinstance(sources, (int, str)):
            source_values = [sources]
        else:
            try:
                source_values = list(sources)
            except TypeError as exc:
                raise TypeError("merge_class_map values must be class ids/names or iterables of them") from exc
        if not source_values:
            raise ValueError(f"merge_class_map target {target!r} has no source classes")
        if not all(isinstance(source, (int, str)) for source in source_values):
            raise TypeError("merge_class_map sources must be class ids or names")
        normalized[target] = source_values
    return normalized


def write_metrics_json(metrics: DetectionMetrics, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def write_metrics_csv(metrics: DetectionMetrics, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "class_name",
        "images",
        "labels",
        "predictions",
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap75",
        "map",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics.classes:
            writer.writerow(asdict(row))


def write_size_metrics_csv(metrics: DetectionMetrics, path: str | Path) -> None:
    """Write aggregate small/medium/large target metrics to a CSV file."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size",
        "images",
        "labels",
        "predictions",
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap75",
        "map",
        "fitness",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics.size_metrics.values():
            writer.writerow(asdict(metric))


def format_metrics_table(metrics: DetectionMetrics, *, precision: int = 3) -> str:
    """Return an Ultralytics-style metrics table for terminal comparison."""
    headers = ["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95"]
    rows: list[list[str]] = [
        [
            "all",
            str(metrics.images),
            str(metrics.labels),
            _format_metric(metrics.precision, precision),
            _format_metric(metrics.recall, precision),
            _format_metric(metrics.map50, precision),
            _format_metric(metrics.map, precision),
        ]
    ]
    for row in metrics.classes:
        if row.labels == 0 and row.predictions == 0:
            continue
        rows.append(
            [
                row.class_name,
                str(row.images),
                str(row.labels),
                _format_metric(row.precision, precision),
                _format_metric(row.recall, precision),
                _format_metric(row.ap50, precision),
                _format_metric(row.map, precision),
            ]
        )

    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]
    lines = [
        " ".join(header.rjust(widths[idx]) if idx else header.ljust(widths[idx]) for idx, header in enumerate(headers))
    ]
    for row in rows:
        lines.append(" ".join(value.rjust(widths[idx]) if idx else value.ljust(widths[idx]) for idx, value in enumerate(row)))

    size_rows = [
        [
            size,
            str(metric.images),
            str(metric.labels),
            _format_metric(metric.precision, precision),
            _format_metric(metric.recall, precision),
            _format_metric(metric.ap50, precision),
            _format_metric(metric.map, precision),
        ]
        for size, metric in metrics.size_metrics.items()
    ]
    if size_rows:
        size_widths = [
            max(len(headers[col_idx]), *(len(row[col_idx]) for row in size_rows))
            for col_idx in range(len(headers))
        ]
        lines.extend(
            [
                "",
                (
                    "Target size metrics "
                    "(small area < 32² px, 32² <= medium area < 96² px, "
                    "large area >= 96² px):"
                ),
                " ".join(
                    header.rjust(size_widths[idx]) if idx else header.ljust(size_widths[idx])
                    for idx, header in enumerate(headers)
                ),
            ]
        )
        for row in size_rows:
            lines.append(
                " ".join(
                    value.rjust(size_widths[idx]) if idx else value.ljust(size_widths[idx])
                    for idx, value in enumerate(row)
                )
            )
    return "\n".join(lines)


def _filter_annotations(
    annotations: Sequence[YoloAnnotation],
    selected: set[int] | None,
    *,
    excluded: set[int] | None,
    is_prediction: bool,
    conf_thres: float,
    image_width: int | None,
    image_height: int | None,
    min_width: float | None,
    min_height: float | None,
    min_area: float | None,
    min_size_logic: str,
    min_pixels: float | None,
    class_rules: Mapping[int, Mapping[str, Any]] | None,
) -> list[YoloAnnotation]:
    rows = [
        ann
        for ann in annotations
        if (selected is None or ann.class_id in selected)
        and (excluded is None or ann.class_id not in excluded)
    ]

    def keep_by_size(ann: YoloAnnotation) -> bool:
        rule = class_rules.get(ann.class_id) if class_rules else None
        rule_min_width = (
            rule.get("min_width", rule.get("width"))
            if rule is not None
            else min_width
        )
        rule_min_height = (
            rule.get("min_height", rule.get("height"))
            if rule is not None
            else min_height
        )
        rule_min_area = rule.get("min_area") if rule is not None else min_area
        rule_logic = (
            rule.get("min_size_logic", rule.get("logic", "or"))
            if rule is not None
            else min_size_logic
        )
        rule_min_pixels = (
            rule.get("min_pixels") if rule is not None else min_pixels
        )
        return _keep_by_size(
            ann,
            image_width=image_width,
            image_height=image_height,
            min_width=rule_min_width,
            min_height=rule_min_height,
            min_area=rule_min_area,
            min_size_logic=rule_logic,
            min_pixels=rule_min_pixels,
        )

    def keep(ann: YoloAnnotation) -> bool:
        if (
            is_prediction
            and conf_thres > 0
            and ann.confidence is not None
            and ann.confidence < conf_thres
        ):
            return False
        return keep_by_size(ann)

    return [ann for ann in rows if keep(ann)]


def _keep_by_size(
    annotation: YoloAnnotation,
    *,
    image_width: int | None,
    image_height: int | None,
    min_width: float | None,
    min_height: float | None,
    min_area: float | None,
    min_size_logic: str,
    min_pixels: float | None,
) -> bool:
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")
    box = annotation.geometry_box()
    if box is None:
        return False
    width_too_small = min_width is not None and box.width < float(min_width)
    height_too_small = min_height is not None and box.height < float(min_height)
    if min_size_logic == "and":
        if width_too_small and height_too_small:
            return False
    elif width_too_small or height_too_small:
        return False
    if min_area is not None and box.width * box.height < float(min_area):
        return False
    if min_pixels is not None and image_width is not None and image_height is not None:
        pixel_width = box.width * image_width
        pixel_height = box.height * image_height
        if pixel_width < float(min_pixels) or pixel_height < float(min_pixels):
            return False
    return True


def _match_predictions(
    gt_anns: Sequence[YoloAnnotation],
    pred_anns: Sequence[YoloAnnotation],
    iouv: np.ndarray,
) -> np.ndarray:
    correct, _matched_gt_sizes = _match_predictions_with_target_sizes(
        gt_anns,
        pred_anns,
        iouv,
        [None] * len(gt_anns),
    )
    return correct


def _match_predictions_with_target_sizes(
    gt_anns: Sequence[YoloAnnotation],
    pred_anns: Sequence[YoloAnnotation],
    iouv: np.ndarray,
    gt_sizes: Sequence[str | None],
) -> tuple[np.ndarray, np.ndarray]:
    correct = np.zeros((len(pred_anns), len(iouv)), dtype=bool)
    matched_gt_sizes = np.full(
        (len(pred_anns), len(iouv)),
        None,
        dtype=object,
    )
    if len(gt_sizes) != len(gt_anns):
        raise ValueError("gt_sizes must have one entry per ground-truth annotation")
    if not gt_anns or not pred_anns:
        return correct, matched_gt_sizes

    for idx, threshold in enumerate(iouv):
        for _iou, gt_idx, pred_idx in greedy_match_indices(
            gt_anns, pred_anns, float(threshold)
        ):
            correct[pred_idx, idx] = True
            matched_gt_sizes[pred_idx, idx] = gt_sizes[gt_idx]
    return correct, matched_gt_sizes


def _target_size(
    annotation: YoloAnnotation,
    *,
    image_width: int | None,
    image_height: int | None,
) -> str | None:
    """Classify a target using COCO-style pixel-area thresholds."""

    if image_width is None or image_height is None:
        return None
    if image_width <= 0 or image_height <= 0:
        return None
    box = annotation.geometry_box()
    if box is None:
        return None
    area = float(box.width) * float(image_width) * float(box.height) * float(image_height)
    if area < TARGET_SIZE_THRESHOLDS["small_max_area"]:
        return "small"
    if area < TARGET_SIZE_THRESHOLDS["medium_max_area"]:
        return "medium"
    return "large"


def _assign_prediction_sizes(
    pred_sizes: Sequence[str | None],
    matched_gt_sizes: np.ndarray,
) -> list[str | None]:
    """Assign each prediction to its matched GT size, or its own size if unmatched."""

    assignments: list[str | None] = []
    for pred_idx, pred_size in enumerate(pred_sizes):
        matched_size = next(
            (
                value
                for value in matched_gt_sizes[pred_idx]
                if value is not None
            ),
            None,
        )
        assignments.append(matched_size or pred_size)
    return assignments


def _aggregate_metric_values(
    *,
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    ignore_empty_classes: bool,
) -> tuple[float, float, float, float, float, float]:
    """Aggregate class-level arrays into the standard summary metrics."""

    _tp_count, _fp_count, p, r, f1, ap, ap_class = _ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
    )
    target_counts = {
        int(class_id): int(count)
        for class_id, count in zip(*np.unique(target_cls, return_counts=True))
    }
    pred_counts = {
        int(class_id): int(count)
        for class_id, count in zip(*np.unique(pred_cls, return_counts=True))
    }
    class_ids = sorted(set(target_counts) | set(pred_counts))
    if ignore_empty_classes:
        class_ids = [class_id for class_id in class_ids if target_counts.get(class_id, 0) > 0]
    class_index = {int(class_id): idx for idx, class_id in enumerate(ap_class)}

    precision_values: list[float] = []
    recall_values: list[float] = []
    map50_values: list[float] = []
    map75_values: list[float] = []
    map_values: list[float] = []
    for class_id in class_ids:
        if target_counts.get(class_id, 0) == 0 and pred_counts.get(class_id, 0) == 0:
            continue
        metric_idx = class_index.get(class_id)
        if metric_idx is None:
            precision_values.append(0.0)
            recall_values.append(0.0)
            map50_values.append(0.0)
            map75_values.append(0.0)
            map_values.append(0.0)
            continue
        ap_values = ap[metric_idx]
        precision_values.append(float(p[metric_idx]))
        recall_values.append(float(r[metric_idx]))
        map50_values.append(float(ap_values[0]) if len(ap_values) else 0.0)
        map75_values.append(float(ap_values[5]) if len(ap_values) > 5 else 0.0)
        map_values.append(float(ap_values.mean()) if len(ap_values) else 0.0)

    if not precision_values:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    precision = float(np.mean(precision_values))
    recall = float(np.mean(recall_values))
    map50 = float(np.mean(map50_values))
    map75 = float(np.mean(map75_values))
    map_value = float(np.mean(map_values))
    fitness = map50 * 0.1 + map_value * 0.9
    return precision, recall, map50, map75, map_value, fitness


def _build_size_metrics(
    *,
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    size_assignments: np.ndarray,
    target_cls_by_size: dict[str, list[int]],
    image_count_by_size: dict[str, int],
    ignore_empty_classes: bool,
) -> dict[str, SizeMetric]:
    if size_assignments.shape[0] != tp.shape[0]:
        raise ValueError("size assignments must align with prediction metrics")

    size_metrics: dict[str, SizeMetric] = {}
    for size in TARGET_SIZE_NAMES:
        mask = size_assignments == size
        target_cls = np.asarray(target_cls_by_size[size], dtype=np.int64)
        size_tp = tp[mask]
        size_conf = conf[mask]
        size_pred_cls = pred_cls[mask]
        precision, recall, map50, map75, map_value, fitness = _aggregate_metric_values(
            tp=size_tp,
            conf=size_conf,
            pred_cls=size_pred_cls,
            target_cls=target_cls,
            ignore_empty_classes=ignore_empty_classes,
        )
        size_metrics[size] = SizeMetric(
            size=size,
            images=image_count_by_size[size],
            labels=int(len(target_cls)),
            predictions=int(mask.sum()),
            precision=precision,
            recall=recall,
            f1=(2 * precision * recall / (precision + recall))
            if precision + recall > 0
            else 0.0,
            ap50=map50,
            ap75=map75,
            map=map_value,
            fitness=fitness,
        )
    return size_metrics


def _ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tp.size == 0:
        unique_classes = np.unique(target_cls).astype(int)
        nc = len(unique_classes)
        return (
            np.zeros(nc),
            np.zeros(nc),
            np.zeros(nc),
            np.zeros(nc),
            np.zeros(nc),
            np.zeros((nc, tp.shape[1] if tp.ndim == 2 else len(DEFAULT_IOU_THRESHOLDS))),
            unique_classes,
        )

    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for class_idx, class_id in enumerate(unique_classes):
        pred_mask = pred_cls == class_id
        n_l = nt[class_idx]
        n_p = int(pred_mask.sum())
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[pred_mask]).cumsum(0)
        tpc = tp[pred_mask].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[class_idx] = np.interp(-x, -conf[pred_mask], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[class_idx] = np.interp(-x, -conf[pred_mask], precision[:, 0], left=1)
        for threshold_idx in range(tp.shape[1]):
            ap[class_idx, threshold_idx] = _compute_ap(recall[:, threshold_idx], precision[:, threshold_idx])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    best_idx = _smooth(f1_curve.mean(0), 0.1).argmax() if nc else 0
    p = p_curve[:, best_idx] if nc else np.array([])
    r = r_curve[:, best_idx] if nc else np.array([])
    f1 = f1_curve[:, best_idx] if nc else np.array([])
    tp_count = (r * nt).round() if nc else np.array([])
    fp_count = (tp_count / (p + eps) - tp_count).round() if nc else np.array([])
    return tp_count, fp_count, p, r, f1, ap, unique_classes.astype(int)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def _build_class_metrics(
    dataset: YoloDataset,
    *,
    selected: set[int] | None,
    excluded: set[int] | None,
    target_cls: np.ndarray,
    pred_count_by_class: dict[int, int],
    image_count_by_class: dict[int, int],
    p: np.ndarray,
    r: np.ndarray,
    f1: np.ndarray,
    ap: np.ndarray,
    ap_class: np.ndarray,
) -> list[ClassMetric]:
    target_count_by_class = {
        int(class_id): int(count)
        for class_id, count in zip(*np.unique(target_cls, return_counts=True))
    } if target_cls.size else {}
    if selected is not None:
        class_ids = sorted(selected)
    else:
        class_ids = sorted(
            (set(target_count_by_class) | set(pred_count_by_class))
            - (excluded or set())
        )
    result_by_class = {int(class_id): idx for idx, class_id in enumerate(ap_class)}
    rows: list[ClassMetric] = []
    for class_id in class_ids:
        metric_idx = result_by_class.get(class_id)
        ap_values = ap[metric_idx] if metric_idx is not None else np.zeros(ap.shape[1] if ap.ndim == 2 else 10)
        rows.append(
            ClassMetric(
                class_id=class_id,
                class_name=dataset.class_name(class_id),
                images=image_count_by_class.get(class_id, 0),
                labels=target_count_by_class.get(class_id, 0),
                predictions=pred_count_by_class.get(class_id, 0),
                precision=float(p[metric_idx]) if metric_idx is not None else 0.0,
                recall=float(r[metric_idx]) if metric_idx is not None else 0.0,
                f1=float(f1[metric_idx]) if metric_idx is not None else 0.0,
                ap50=float(ap_values[0]) if len(ap_values) else 0.0,
                ap75=float(ap_values[5]) if len(ap_values) > 5 else 0.0,
                map=float(ap_values.mean()) if len(ap_values) else 0.0,
            )
        )
    return rows


def _format_metric(value: float, precision: int) -> str:
    return f"{float(value):.{precision}g}"


def _confidence(annotation: YoloAnnotation) -> float:
    return 1.0 if annotation.confidence is None else float(annotation.confidence)


def _smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    nf = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(nf // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]), 0)
    return np.convolve(padded, np.ones(nf) / nf, mode="valid")
