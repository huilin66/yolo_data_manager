import csv
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import Box, ClassSchema, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.evaluation.error_analysis import write_confidence_curve
from yolo_data_manager.scripting import build_task_argv


def test_confidence_curve_counts_review_groups_and_writes_outputs(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100), (30, 40, 50)).save(image_path)

    gt = YoloDataset(
        root=tmp_path / "gt",
        images=[
            YoloImage(
                image_path,
                annotations=[YoloAnnotation(0, Box(0.5, 0.5, 0.2, 0.2))],
            )
        ],
        classes=ClassSchema(["object"]),
    )
    pred = YoloDataset(
        root=tmp_path / "pred",
        images=[
            YoloImage(
                image_path,
                annotations=[
                    YoloAnnotation(0, Box(0.5, 0.5, 0.2, 0.2), confidence=0.9),
                    YoloAnnotation(0, Box(0.1, 0.1, 0.1, 0.1), confidence=0.2),
                ],
            )
        ],
        classes=ClassSchema(["object"]),
    )

    output = tmp_path / "error_report"
    result = write_confidence_curve(gt, pred, output, progress=False)

    assert result["thresholds"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert (output / "conf_curve.csv").exists()
    assert (output / "conf_curve_summary.csv").exists()
    assert (output / "conf_curve.png").exists()

    with (output / "conf_curve.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert [int(row["file_count"]) for row in rows] == [1, 1, 0, 0, 0]

    with (output / "conf_curve_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert [int(row["review_files"]) for row in summary] == [1, 1, 0, 0, 0]
    assert [int(row["fp_rows"]) for row in summary] == [1, 1, 0, 0, 0]


def test_eval_error_analysis_conf_curve_flag_is_available():
    argv = build_task_argv("eval.error_analysis", conf_curve=True)
    assert "--conf-curve" in argv
