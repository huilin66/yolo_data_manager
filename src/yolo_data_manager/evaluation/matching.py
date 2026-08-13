"""Shared detection matching helpers used by evaluation workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from yolo_data_manager.core.models import YoloAnnotation


def validate_nms_iou(nms_iou: float | None) -> None:
    """Validate an NMS IoU threshold; ``None`` disables NMS."""

    if nms_iou is not None and not 0.0 < float(nms_iou) <= 1.0:
        raise ValueError("nms_iou must be greater than 0 and at most 1, or None to disable NMS")


def annotation_box_xyxy(annotation: YoloAnnotation) -> list[float]:
    """Return an annotation box as normalised ``xyxy`` coordinates."""

    box = annotation.geometry_box()
    if box is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        box.cx - box.width / 2.0,
        box.cy - box.height / 2.0,
        box.cx + box.width / 2.0,
        box.cy + box.height / 2.0,
    ]


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return the pairwise IoU matrix for normalised ``xyxy`` boxes."""

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
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-16, None)


def non_max_suppress_annotations(
    annotations: Sequence[YoloAnnotation],
    nms_iou: float | None,
) -> list[YoloAnnotation]:
    """Keep highest-confidence same-class boxes using greedy NMS.

    ``None`` disables suppression.  The returned annotations retain their
    original label-file order so source line indices remain stable; confidence
    only determines which overlapping candidate wins.
    """

    validate_nms_iou(nms_iou)
    rows = list(annotations)
    if nms_iou is None or len(rows) < 2:
        return rows

    confidence_order = sorted(
        range(len(rows)),
        key=lambda index: (
            -(1.0 if rows[index].confidence is None else float(rows[index].confidence)),
            index,
        ),
    )
    kept_indices: list[int] = []
    kept_by_class: dict[int, list[int]] = {}
    for index in confidence_order:
        candidate = rows[index]
        candidate_box = np.array([annotation_box_xyxy(candidate)], dtype=np.float64)
        is_suppressed = False
        for kept_index in kept_by_class.get(candidate.class_id, []):
            kept_box = np.array([annotation_box_xyxy(rows[kept_index])], dtype=np.float64)
            if box_iou_matrix(candidate_box, kept_box)[0, 0] >= float(nms_iou):
                is_suppressed = True
                break
        if not is_suppressed:
            kept_indices.append(index)
            kept_by_class.setdefault(candidate.class_id, []).append(index)

    kept_set = set(kept_indices)
    return [row for index, row in enumerate(rows) if index in kept_set]


def greedy_match_indices(
    gt_annotations: Sequence[YoloAnnotation],
    pred_annotations: Sequence[YoloAnnotation],
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    """Match detections using the same one-to-one rule as metrics.

    Returns ``(iou, gt_index, pred_index)`` tuples.  Candidate pairs are
    sorted by IoU, then each prediction is claimed once, followed by each GT
    being claimed once.  Keeping this operation shared ensures that the
    detailed error report and aggregate metrics cannot disagree about TP
    assignment.
    """

    if not gt_annotations or not pred_annotations:
        return []

    iou = box_iou_matrix(
        np.array([annotation_box_xyxy(ann) for ann in gt_annotations], dtype=np.float64),
        np.array([annotation_box_xyxy(ann) for ann in pred_annotations], dtype=np.float64),
    )
    true_classes = np.array([ann.class_id for ann in gt_annotations])
    pred_classes = np.array([ann.class_id for ann in pred_annotations])
    iou *= true_classes[:, None] == pred_classes[None, :]

    matches = np.array(np.nonzero(iou >= iou_threshold)).T
    if matches.shape[0] > 1:
        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return [
        (float(iou[gt_idx, pred_idx]), int(gt_idx), int(pred_idx))
        for gt_idx, pred_idx in matches
    ]
