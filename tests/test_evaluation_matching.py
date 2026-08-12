from pathlib import Path

from yolo_data_manager.core.models import Box, ClassSchema, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.evaluation.error_analysis import analyze_errors
from yolo_data_manager.evaluation.metrics import compute_detection_metrics


def test_error_analysis_uses_metrics_one_to_one_matching():
    # The first prediction overlaps both GT boxes.  The second prediction also
    # overlaps both, creating a greedy-matching conflict.  The metrics matcher
    # claims predictions first and then GTs, so only one pair is a TP.
    gt = YoloDataset(
        root=Path("gt"),
        images=[
            YoloImage(
                Path("sample.jpg"),
                annotations=[
                    YoloAnnotation(0, Box(0.15, 0.5, 0.25, 0.4)),
                    YoloAnnotation(0, Box(0.30, 0.5, 0.45, 0.4)),
                ],
            )
        ],
        classes=ClassSchema(["object"]),
    )
    pred = YoloDataset(
        root=Path("pred"),
        images=[
            YoloImage(
                Path("sample.jpg"),
                annotations=[
                    YoloAnnotation(0, Box(0.20, 0.5, 0.35, 0.4), confidence=0.9),
                    YoloAnnotation(0, Box(0.15, 0.5, 0.45, 0.4), confidence=0.8),
                ],
            )
        ],
        classes=ClassSchema(["object"]),
    )

    rows, summary = analyze_errors(gt, pred, match_iou=0.5, low_iou=0.1)
    metrics = compute_detection_metrics(gt, pred, iou_thresholds=(0.5,))

    assert summary["tp"] == 1
    assert len([row for row in rows if row.status == "tp"]) == 1
    assert metrics.labels == 2
    assert metrics.predictions == 2
