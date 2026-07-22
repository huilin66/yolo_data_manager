from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from yolo_data_manager.core.models import YoloAnnotation, YoloDataset


DEFAULT_IOU_THRESHOLDS = tuple(float(v) for v in np.linspace(0.5, 0.95, 10))


@dataclass
class ClassMetric:
    class_id: int
    class_name: str
    labels: int
    predictions: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap75: float
    map: float


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
    iou_thresholds: list[float]
    size_filter: dict[str, float | str | None]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["report_type"] = "detection_metrics"
        return data


def compute_detection_metrics(
    gt: YoloDataset,
    pred: YoloDataset,
    *,
    class_ids: Iterable[int] | None = None,
    conf_thres: float = 0.0,
    min_width: float | None = None,
    min_height: float | None = None,
    min_area: float | None = None,
    min_size_logic: str = "or",
    min_pixels: float | None = None,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> DetectionMetrics:
    """Compute Ultralytics-style detection metrics from YOLO GT/prediction txt.

    Predictions without an explicit confidence are treated as confidence 1.0.
    If *class_ids* is provided, GT and predictions outside that class set are
    ignored before matching and averaging.
    """
    if min_size_logic not in {"or", "and"}:
        raise ValueError("min_size_logic must be 'or' or 'and'")
    selected = None if class_ids is None else set(int(class_id) for class_id in class_ids)
    iouv = np.array(iou_thresholds, dtype=np.float64)
    pred_by_stem = {image.stem: image for image in pred.images}

    tp_parts: list[np.ndarray] = []
    conf_values: list[float] = []
    pred_cls_values: list[int] = []
    target_cls_values: list[int] = []
    pred_count_by_class: dict[int, int] = {}

    seen_gt_stems: set[str] = set()
    for gt_image in gt.images:
        seen_gt_stems.add(gt_image.stem)
        pred_image = pred_by_stem.get(gt_image.stem)
        width = gt_image.width if gt_image.width is not None else (pred_image.width if pred_image is not None else None)
        height = gt_image.height if gt_image.height is not None else (pred_image.height if pred_image is not None else None)
        gt_anns = _filter_annotations(
            gt_image.annotations,
            selected,
            is_prediction=False,
            conf_thres=conf_thres,
            image_width=width,
            image_height=height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
        )
        pred_anns = _filter_annotations(
            pred_image.annotations if pred_image is not None else [],
            selected,
            is_prediction=True,
            conf_thres=conf_thres,
            image_width=width,
            image_height=height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
        )
        target_cls_values.extend(ann.class_id for ann in gt_anns)
        for ann in pred_anns:
            pred_count_by_class[ann.class_id] = pred_count_by_class.get(ann.class_id, 0) + 1

        correct = _match_predictions(gt_anns, pred_anns, iouv)
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
            is_prediction=True,
            conf_thres=conf_thres,
            image_width=pred_image.width,
            image_height=pred_image.height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
        )
        if not pred_anns:
            continue
        tp_parts.append(np.zeros((len(pred_anns), len(iouv)), dtype=bool))
        conf_values.extend(_confidence(ann) for ann in pred_anns)
        pred_cls_values.extend(ann.class_id for ann in pred_anns)
        for ann in pred_anns:
            pred_count_by_class[ann.class_id] = pred_count_by_class.get(ann.class_id, 0) + 1

    tp = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.array(conf_values, dtype=np.float64)
    pred_cls = np.array(pred_cls_values, dtype=np.int64)
    target_cls = np.array(target_cls_values, dtype=np.int64)

    tp_count, fp_count, p, r, f1, ap, ap_class = _ap_per_class(tp, conf, pred_cls, target_cls)
    class_metrics = _build_class_metrics(
        gt,
        selected=selected,
        target_cls=target_cls,
        pred_count_by_class=pred_count_by_class,
        p=p,
        r=r,
        f1=f1,
        ap=ap,
        ap_class=ap_class,
    )

    metric_rows = [row for row in class_metrics if row.labels > 0 or row.predictions > 0]
    if metric_rows:
        precision = float(np.mean([row.precision for row in metric_rows]))
        recall = float(np.mean([row.recall for row in metric_rows]))
        map50 = float(np.mean([row.ap50 for row in metric_rows]))
        map75 = float(np.mean([row.ap75 for row in metric_rows]))
        map_value = float(np.mean([row.map for row in metric_rows]))
    else:
        precision = recall = map50 = map75 = map_value = 0.0

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
        iou_thresholds=[float(v) for v in iouv],
        size_filter={
            "min_width": min_width,
            "min_height": min_height,
            "min_area": min_area,
            "min_size_logic": min_size_logic,
            "min_pixels": min_pixels,
        },
    )


def resolve_eval_class_ids(dataset: YoloDataset, values: Iterable[str] | None) -> list[int] | None:
    if values is None:
        return None
    return [dataset.class_id(value) for value in values]


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


def _filter_annotations(
    annotations: Sequence[YoloAnnotation],
    selected: set[int] | None,
    *,
    is_prediction: bool,
    conf_thres: float,
    image_width: int | None,
    image_height: int | None,
    min_width: float | None,
    min_height: float | None,
    min_area: float | None,
    min_size_logic: str,
    min_pixels: float | None,
) -> list[YoloAnnotation]:
    rows = [ann for ann in annotations if selected is None or ann.class_id in selected]
    if not is_prediction or conf_thres <= 0:
        return [
            ann
            for ann in rows
            if _keep_by_size(
                ann,
                image_width=image_width,
                image_height=image_height,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                min_size_logic=min_size_logic,
                min_pixels=min_pixels,
            )
        ]
    return [
        ann
        for ann in rows
        if (ann.confidence is None or ann.confidence >= conf_thres)
        and _keep_by_size(
            ann,
            image_width=image_width,
            image_height=image_height,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
        )
    ]


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
    correct = np.zeros((len(pred_anns), len(iouv)), dtype=bool)
    if not gt_anns or not pred_anns:
        return correct

    iou = _box_iou_matrix(
        np.array([_annotation_box_xyxy(ann) for ann in gt_anns], dtype=np.float64),
        np.array([_annotation_box_xyxy(ann) for ann in pred_anns], dtype=np.float64),
    )
    true_classes = np.array([ann.class_id for ann in gt_anns])
    pred_classes = np.array([ann.class_id for ann in pred_anns])
    iou *= true_classes[:, None] == pred_classes[None, :]

    for idx, threshold in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), idx] = True
    return correct


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
    target_cls: np.ndarray,
    pred_count_by_class: dict[int, int],
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
        class_ids = sorted(set(target_count_by_class) | set(pred_count_by_class))
    result_by_class = {int(class_id): idx for idx, class_id in enumerate(ap_class)}
    rows: list[ClassMetric] = []
    for class_id in class_ids:
        metric_idx = result_by_class.get(class_id)
        ap_values = ap[metric_idx] if metric_idx is not None else np.zeros(ap.shape[1] if ap.ndim == 2 else 10)
        rows.append(
            ClassMetric(
                class_id=class_id,
                class_name=dataset.class_name(class_id),
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


def _annotation_box_xyxy(annotation: YoloAnnotation) -> list[float]:
    box = annotation.geometry_box()
    if box is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        box.cx - box.width / 2.0,
        box.cy - box.height / 2.0,
        box.cx + box.width / 2.0,
        box.cy + box.height / 2.0,
    ]


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-16, None)


def _confidence(annotation: YoloAnnotation) -> float:
    return 1.0 if annotation.confidence is None else float(annotation.confidence)


def _smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    nf = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(nf // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]), 0)
    return np.convolve(padded, np.ones(nf) / nf, mode="valid")
