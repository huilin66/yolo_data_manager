import csv
import importlib.util
import json
from pathlib import Path


def _load_dataset_template():
    path = Path(__file__).parents[1] / "example" / "dataset_template.py"
    spec = importlib.util.spec_from_file_location("ydm_dataset_template", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_metrics(path: Path, *, map50: float, fence: float) -> None:
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "images": 44,
                "labels": 171,
                "precision": 0.733,
                "recall": 0.657,
                "f1": 0.693,
                "map50": map50,
                "map": 0.671,
                "classes": [
                    {"class_name": "fence", "ap50": fence},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_metric_summary_overall_mode_writes_all_metrics_per_result(tmp_path):
    template = _load_dataset_template()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_metrics(first, map50=0.4, fence=0.3)
    _write_metrics(second, map50=0.8, fence=0.7)
    output = tmp_path / "summary.csv"

    template._write_metric_summary_csv(
        [first, second],
        output=output,
        metric="map50",
        summary_mode="overall",
    )

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["Class", "Images", "Instances", "Precision", "Recall", "F1", "mAP50", "mAP50-95"],
        ["all", "44", "171", "0.733", "0.657", "0.693", "0.400", "0.671"],
        ["all", "44", "171", "0.733", "0.657", "0.693", "0.800", "0.671"],
    ]


def test_metric_summary_per_result_mode_remains_unchanged(tmp_path):
    template = _load_dataset_template()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_metrics(first, map50=0.4, fence=0.3)
    _write_metrics(second, map50=0.8, fence=0.7)
    output = tmp_path / "summary.csv"

    template._write_metric_summary_csv(
        [first, second],
        output=output,
        metric="map50",
    )

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["map50", "fence"],
        ["0.400", "0.300"],
        ["0.800", "0.700"],
    ]
