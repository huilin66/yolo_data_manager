from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from yolo_data_manager.core.geometry import XYXY, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloAnnotation, YoloDataset


@dataclass
class CompareRow:
    image: str
    class_id: int
    class_name: str
    status: str
    iou: float = 0.0
    gt_line: int | None = None
    pred_line: int | None = None
    confidence: float | None = None


def compare_datasets(
    gt: YoloDataset,
    pred: YoloDataset,
    iou_threshold: float = 0.5,
    confidence_threshold: float | None = None,
) -> tuple[list[CompareRow], dict[str, int]]:
    pred_by_stem = {image.stem: image for image in pred.images}
    rows: list[CompareRow] = []
    seen_gt_stems: set[str] = set()

    for gt_image in gt.images:
        seen_gt_stems.add(gt_image.stem)
        pred_image = pred_by_stem.get(gt_image.stem)
        gt_annotations = gt_image.annotations
        pred_annotations = pred_image.annotations if pred_image is not None else []
        if confidence_threshold is not None:
            pred_annotations = [
                annotation for annotation in pred_annotations
                if annotation.confidence is None or annotation.confidence >= confidence_threshold
            ]
        matched_pred: set[int] = set()

        for gt_ann in gt_annotations:
            best_idx = None
            best_iou = 0.0
            for pred_idx, pred_ann in enumerate(pred_annotations):
                if pred_idx in matched_pred or pred_ann.class_id != gt_ann.class_id:
                    continue
                iou = annotation_iou(gt_ann, pred_ann, gt_image.width or 1, gt_image.height or 1)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = pred_idx
            if best_idx is not None and best_iou >= iou_threshold:
                matched_pred.add(best_idx)
                pred_ann = pred_annotations[best_idx]
                rows.append(_row(gt, gt_image.file_name, gt_ann, "tp", best_iou, pred_ann))
            else:
                rows.append(_row(gt, gt_image.file_name, gt_ann, "fn", best_iou, None))

        for pred_idx, pred_ann in enumerate(pred_annotations):
            if pred_idx not in matched_pred:
                rows.append(_row(gt, gt_image.file_name, pred_ann, "fp", 0.0, pred_ann))

    for pred_image in pred.images:
        if pred_image.stem in seen_gt_stems:
            continue
        pred_annotations = pred_image.annotations
        if confidence_threshold is not None:
            pred_annotations = [
                annotation for annotation in pred_annotations
                if annotation.confidence is None or annotation.confidence >= confidence_threshold
            ]
        for pred_ann in pred_annotations:
            rows.append(_row(gt, pred_image.file_name, pred_ann, "fp", 0.0, pred_ann))

    summary = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        summary[row.status] = summary.get(row.status, 0) + 1
    return rows, summary


def annotation_iou(a: YoloAnnotation, b: YoloAnnotation, width: int, height: int) -> float:
    box_a = a.geometry_box()
    box_b = b.geometry_box()
    if box_a is None or box_b is None:
        return 0.0
    return box_iou(xywhn_to_xyxy(box_a.as_tuple(), width, height), xywhn_to_xyxy(box_b.as_tuple(), width, height))


def box_iou(a: XYXY, b: XYXY) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def write_compare_csv(rows: list[CompareRow], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["image", "class_id", "class_name", "status", "iou", "gt_line", "pred_line", "confidence"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _row(
    dataset: YoloDataset,
    image_name: str,
    annotation: YoloAnnotation,
    status: str,
    iou: float,
    pred_annotation: YoloAnnotation | None,
) -> CompareRow:
    return CompareRow(
        image=image_name,
        class_id=annotation.class_id,
        class_name=dataset.class_name(annotation.class_id),
        status=status,
        iou=round(iou, 6),
        gt_line=annotation.line_no if status != "fp" else None,
        pred_line=pred_annotation.line_no if pred_annotation is not None else None,
        confidence=pred_annotation.confidence if pred_annotation is not None else None,
    )
