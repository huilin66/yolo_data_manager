from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import Box, ClassSchema, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.cli import main as cli_main
from yolo_data_manager.evaluation.error_analysis import (
    ATTRIBUTE_VALUE_MISMATCH,
    analyze_attribute_errors,
    analyze_errors,
    load_error_analysis_dataset,
    write_attribute_error_csv,
    write_attribute_error_review_pack,
)
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


def test_attribute_error_analysis_compares_only_matched_pairs(tmp_path):
    gt_root = tmp_path / "gt"
    (gt_root / "images").mkdir(parents=True)
    (gt_root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(gt_root / "images" / "sample.jpg")
    (gt_root / "class.txt").write_text("object\n", encoding="utf-8")
    (gt_root / "attribute.yaml").write_text(
        "attributes:\n  defect: [no, yes]\n",
        encoding="utf-8",
    )
    (gt_root / "labels" / "sample.txt").write_text(
        "0 1 1 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )

    pred_labels = tmp_path / "pred_labels"
    pred_labels.mkdir()
    # External prediction directories commonly contain txt files only.  The
    # GT attribute schema is shared so the attribute index can be decoded.
    (pred_labels / "sample.txt").write_text(
        "0 1 0 0.5 0.5 0.4 0.4 0.95\n",
        encoding="utf-8",
    )

    gt = load_error_analysis_dataset(gt_root, task="detect")
    pred = load_error_analysis_dataset(
        pred_labels,
        task="detect",
        class_file=gt_root / "class.txt",
        attributes=gt.attributes,
    )
    rows, summary = analyze_attribute_errors(gt, pred, match_iou=0.5)

    assert len(rows) == 1
    assert rows[0].attribute_name == "defect"
    assert rows[0].gt_value == "yes"
    assert rows[0].pred_value == "no"
    assert rows[0].error_type == ATTRIBUTE_VALUE_MISMATCH
    assert rows[0].pred_idx == 1
    assert rows[0].gt_idx == 1
    assert summary["attribute_error"] == 1
    assert summary["attribute_error:defect"] == 1

    out = tmp_path / "error_report"
    write_attribute_error_csv(rows, out)
    review_counts = write_attribute_error_review_pack(
        rows,
        gt,
        pred,
        out,
        workers=1,
    )
    group = "attribute_defect/gt_yes_pred_no"
    assert (out / "attribute_error.csv").exists()
    assert review_counts[group] == 1
    assert (out / "review" / "attribute_error" / group / "images").is_dir()
    assert (out / "review" / "attribute_error" / group / "crops" / "sample_pred1_gt1_defect.jpg").exists()


def test_cli_error_analysis_discovers_shared_attribute_schema(tmp_path):
    gt_root = tmp_path / "gt"
    (gt_root / "images").mkdir(parents=True)
    (gt_root / "labels").mkdir(parents=True)
    Image.new("RGB", (40, 40), color="white").save(gt_root / "images" / "sample.jpg")
    (gt_root / "class.txt").write_text("object\n", encoding="utf-8")
    (gt_root / "attribute.yaml").write_text(
        "attributes:\n  defect: [no, yes]\n",
        encoding="utf-8",
    )
    (gt_root / "labels" / "sample.txt").write_text(
        "0 1 1 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    pred_labels = tmp_path / "pred_labels"
    pred_labels.mkdir()
    (pred_labels / "sample.txt").write_text(
        "0 1 0 0.5 0.5 0.5 0.5 0.9\n",
        encoding="utf-8",
    )
    out = tmp_path / "cli_error_report"

    assert cli_main(
        [
            "eval",
            "error-analysis",
            "--gt-root",
            str(gt_root),
            "--pred-root",
            str(pred_labels),
            "--class-file",
            str(gt_root / "class.txt"),
            "--out",
            str(out),
            "--no-progress",
            "--review",
        ]
    ) == 0

    assert (out / "attribute_error.csv").exists()
    assert (out / "review" / "attribute_error" / "attribute_defect" / "gt_yes_pred_no" / "crops").is_dir()
