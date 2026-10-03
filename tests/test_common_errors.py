import csv
from pathlib import Path

from PIL import Image

from yolo_data_manager.evaluation.common_errors import extract_common_error_analysis
from yolo_data_manager.evaluation.error_analysis import (
    BACKGROUND_FP,
    CLASS_ERROR_PRED,
    ErrorDetail,
    FN_NO_PRED,
    _review_crop_name,
    _review_group_name,
    write_error_csvs,
)
from yolo_data_manager.scripting import build_task_argv


def _row(
    image: str,
    *,
    status: str,
    error_type: str,
    pred_class_id: int | None = None,
    pred_class_name: str | None = None,
    gt_class_id: int | None = None,
    gt_class_name: str | None = None,
    pred_box: str | None = None,
    gt_box: str | None = None,
    pred_idx: int | None = None,
    gt_idx: int | None = None,
) -> ErrorDetail:
    return ErrorDetail(
        image=image,
        status=status,
        error_type=error_type,
        class_id=pred_class_id if status == "fp" and pred_class_id is not None else (gt_class_id or 0),
        class_name=pred_class_name if status == "fp" and pred_class_name else (gt_class_name or "cat"),
        pred_class_id=pred_class_id,
        pred_class_name=pred_class_name,
        pred_conf=0.9 if status == "fp" else None,
        gt_class_id=gt_class_id,
        gt_class_name=gt_class_name,
        best_iou=0.0,
        pred_box_xyxy=pred_box,
        gt_box_xyxy=gt_box,
        pred_idx=pred_idx,
        gt_idx=gt_idx,
    )


def _write_run(root: Path, rows: list[ErrorDetail]) -> None:
    write_error_csvs(rows, root)
    for row in rows:
        group = root / "review" / _review_group_name(row) / "crops"
        group.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), (20, 30, 40)).save(
            group / _review_crop_name(row, ".jpg")
        )


def test_extract_common_error_analysis_writes_intersection_and_summary(tmp_path: Path):
    fn_box = "[0.1, 0.1, 0.4, 0.4]"
    bg_box = "[0.5, 0.5, 0.8, 0.8]"
    class_box = "[0.2, 0.6, 0.5, 0.9]"

    common_fn = _row(
        "shared_fn",
        status="fn",
        error_type=FN_NO_PRED,
        gt_class_id=0,
        gt_class_name="cat",
        gt_box=fn_box,
        gt_idx=1,
    )
    common_background = _row(
        "shared_background",
        status="fp",
        error_type=BACKGROUND_FP,
        pred_class_id=1,
        pred_class_name="dog",
        pred_box=bg_box,
        pred_idx=1,
    )
    common_class_error = _row(
        "shared_class_error",
        status="fp",
        error_type=CLASS_ERROR_PRED,
        pred_class_id=1,
        pred_class_name="dog",
        gt_class_id=0,
        gt_class_name="cat",
        pred_box=class_box,
        gt_box="[0.2, 0.6, 0.5, 0.9]",
        pred_idx=2,
        gt_idx=2,
    )

    run_one = tmp_path / "run_one"
    run_two = tmp_path / "run_two"
    _write_run(
        run_one,
        [
            common_fn,
            common_background,
            common_class_error,
            _row(
                "only_one_fn",
                status="fn",
                error_type=FN_NO_PRED,
                gt_class_id=0,
                gt_class_name="cat",
                gt_box="[0.0, 0.0, 0.2, 0.2]",
                gt_idx=1,
            ),
            _row(
                "only_one_fp",
                status="fp",
                error_type=BACKGROUND_FP,
                pred_class_id=1,
                pred_class_name="dog",
                pred_box="[0.0, 0.0, 0.2, 0.2]",
                pred_idx=1,
            ),
        ],
    )
    _write_run(
        run_two,
        [
            common_fn,
            common_background,
            common_class_error,
            _row(
                "only_two_fn",
                status="fn",
                error_type=FN_NO_PRED,
                gt_class_id=0,
                gt_class_name="cat",
                gt_box="[0.7, 0.7, 0.9, 0.9]",
                gt_idx=1,
            ),
            _row(
                "only_two_fp",
                status="fp",
                error_type=BACKGROUND_FP,
                pred_class_id=1,
                pred_class_name="dog",
                pred_box="[0.7, 0.7, 0.9, 0.9]",
                pred_idx=1,
            ),
        ],
    )

    output = tmp_path / "common"
    result = extract_common_error_analysis([run_one, run_two], output, progress=False)

    assert result["common"] == {"fn": 1, "fp": 2, "total": 3}
    assert result["missing_crops"] == 0
    assert len(list((output / "common_crops").rglob("*.jpg"))) == 3

    with (output / "common_error_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert [row["original_fn"] for row in summary] == ["2", "2"]
    assert [row["common_fn"] for row in summary] == ["1", "1"]
    assert [row["remaining_fn"] for row in summary] == ["1", "1"]
    assert [row["original_fp"] for row in summary] == ["3", "3"]
    assert [row["common_fp"] for row in summary] == ["2", "2"]
    assert [row["remaining_fp"] for row in summary] == ["1", "1"]


def test_common_error_task_repeats_error_dir_argument():
    argv = build_task_argv(
        "eval.error_analysis_common",
        error_dirs=["run_one", "run_two"],
        out="common",
        iou=0.5,
        copy_crops=False,
        progress=False,
    )

    assert argv[:2] == ["eval", "error-analysis-common"]
    assert argv.count("--error-dir") == 2
    assert "--no-copy-crops" in argv
    assert "--no-progress" in argv


def test_common_error_analysis_reports_progress_stages(tmp_path: Path, capsys):
    row = _row(
        "shared",
        status="fn",
        error_type=FN_NO_PRED,
        gt_class_id=0,
        gt_class_name="cat",
        gt_box="[0.1, 0.1, 0.4, 0.4]",
        gt_idx=1,
    )
    run_one = tmp_path / "run_one"
    run_two = tmp_path / "run_two"
    _write_run(run_one, [row])
    _write_run(run_two, [row])

    extract_common_error_analysis(
        [run_one, run_two],
        tmp_path / "common",
        copy_crops=False,
        progress=True,
    )

    output = capsys.readouterr().err
    assert "common errors load reports" in output
    assert "common errors match fn" in output
    assert "common errors match fp" in output
    assert "common errors skip crop copy" in output
    assert "common errors write reports" in output
