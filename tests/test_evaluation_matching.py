from pathlib import Path

from yolo_data_manager.core.models import Box, ClassSchema, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.evaluation.error_analysis import analyze_errors
from yolo_data_manager.evaluation.metrics import compute_detection_metrics, format_metrics_table


def test_error_analysis_uses_metrics_one_to_one_matching():
    # The two predictions overlap each other.  NMS keeps the higher-confidence
    # first prediction before either evaluator performs GT matching.
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
    assert not any(row.error_type == "duplicate_prediction" for row in rows)
    assert metrics.labels == 2
    assert metrics.predictions == 1


def test_nms_is_class_aware_and_can_be_disabled():
    annotation_a = YoloAnnotation(0, Box(0.5, 0.5, 0.4, 0.4), confidence=0.9)
    annotation_b = YoloAnnotation(0, Box(0.5, 0.5, 0.4, 0.4), confidence=0.8)
    annotation_other_class = YoloAnnotation(1, Box(0.5, 0.5, 0.4, 0.4), confidence=0.7)
    pred = YoloDataset(
        root=Path("pred"),
        images=[YoloImage(Path("sample.jpg"), annotations=[annotation_a, annotation_b, annotation_other_class])],
        classes=ClassSchema(["first", "second"]),
    )
    gt = YoloDataset(
        root=Path("gt"),
        images=[YoloImage(Path("sample.jpg"), annotations=[])],
        classes=ClassSchema(["first", "second"]),
    )

    default_metrics = compute_detection_metrics(gt, pred, nms_iou=0.5)
    no_nms_metrics = compute_detection_metrics(gt, pred, nms_iou=None)

    assert default_metrics.predictions == 2
    assert no_nms_metrics.predictions == 3


def test_metrics_reports_small_medium_large_target_metrics():
    gt_annotations = [
        YoloAnnotation(0, Box(0.15, 0.2, 0.1, 0.1)),
        YoloAnnotation(0, Box(0.50, 0.5, 0.3, 0.3)),
        YoloAnnotation(0, Box(0.50, 0.5, 0.8, 0.8)),
    ]
    pred_annotations = [
        YoloAnnotation(0, Box(0.15, 0.2, 0.1, 0.1), confidence=0.9),
        YoloAnnotation(0, Box(0.50, 0.5, 0.3, 0.3), confidence=0.8),
        YoloAnnotation(0, Box(0.50, 0.5, 0.8, 0.8), confidence=0.7),
    ]
    gt = YoloDataset(
        root=Path("gt"),
        images=[YoloImage(Path("sample.jpg"), annotations=gt_annotations, width=200, height=200)],
        classes=ClassSchema(["object"]),
    )
    pred = YoloDataset(
        root=Path("pred"),
        images=[YoloImage(Path("sample.jpg"), annotations=pred_annotations, width=200, height=200)],
        classes=ClassSchema(["object"]),
    )

    metrics = compute_detection_metrics(gt, pred)

    assert metrics.target_size_thresholds == {
        "small_max_area": 1024,
        "medium_max_area": 9216,
    }
    assert [metrics.size_metrics[size].labels for size in ("small", "medium", "large")] == [1, 1, 1]
    assert [metrics.size_metrics[size].predictions for size in ("small", "medium", "large")] == [1, 1, 1]
    assert all(metrics.size_metrics[size].precision == 1.0 for size in ("small", "medium", "large"))
    assert "Target size metrics" in format_metrics_table(metrics)
