"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller, not a reusable function module.  Keep the
dataset path and the operations for one dataset here; keep implementation
details in ``example/functions``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from yolo_data_manager import YoloManager, merge_modalities

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t_update_v6.yaml"
)

PRED_RUNS_DIR = Path(r"//localnvme/project/ultralytics/runs/detect")
PRED_NAMES = [
    # "predict-2",
    # "predict-3",
    # "predict-4",
    # "predict-5",
    # "predict-6",
    # "predict-13",
]
# Select any aggregate numeric metric written by eval_metrics. Common values
# are map50, map75, map, precision, recall, f1, and fitness. ``maps`` is an
# alias for ``map``.
METRIC_SUMMARY = {
    "metric": "map50",
    "csv": None,  # defaults to <dataset>/ydm_evaluation/metric_summary.csv
    "data": None,  # defaults to DATA_DIR.stem
    "model": "yolo8",
    "weight": "",
    "method": "ydm",
    "conf": 0.0,
}
LEAKAGE_ONLY_LIST = (
    # "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"
)
CROP_ROOT_LABEL = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_vis/crop_change"
CROP_ROOT_PRED = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_evaluation/error_analysis/predict-13/crop_change"

CROP_MAP_LABEL = {
    # os.path.join(CROP_ROOT_LABEL, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_LABEL, "2_none"): "none",
}
CROP_MAP_ATTR = {
    # os.path.join(CROP_ROOT_LABEL, "defect_no"): {
    #     "name": "defect",
    #     "value": "no",
    # },
}

CROP_MAP_PRED = {
    # os.path.join(CROP_ROOT_PRED, "2_at"): "Abnormal Temperature",
    # os.path.join(CROP_ROOT_PRED, "2_h"): "Hollow",
    # os.path.join(CROP_ROOT_PRED, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_PRED, "2_none"): "none",
}
MERGE_CLASS_MAP = {
    # "Hollow": [
    #     "Hollow Low Risk",
    #     "Hollow High Risk",
    # ],
    # "Temperature": [
    #     "Temperature Medium Risk",
    #     "Temperature High Risk",
    # ],
}

METRIC_CLASS_RULES = {
    "Leakage": {"width": 0.05, "height": 0.03, "logic": "and"},
}

UPDATE_CLASS_MAP = {
    # "merge": {
    #     "Hollow Confirmed": ["Hollow High Risk"],
    #     "Hollow Suspected": ["Hollow Low Risk"],
    #     "Leakage": ["Leakage High Risk"],
    # },
    # "drop": [
    #     "background",
    #     "Hollow High Risk Line",
    #     "Temperature Medium Risk",
    #     "Temperature High Risk",
    # ],
}
UPDATE_ATT_MAP = None
QUERY_CLASS = "Leakage"
SELECTION_FILE = None
# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    "merge",
    "rename",
    "split",
    "resize",
    "sta",
    "vis_draw",
    "vis_crop",
    "query",
    "copy_select",
    "metric",
    "error_ana",
    "update_class",
    "update_att",
    "filter_small",
    "mannual_draw",
    "update_class_from_anno",
    "update_class_from_pred",
    "update_att_from_anno",
    "update_att_from_pred",
]


def _resolve_metrics_file(path: str | Path) -> Path:
    """Resolve the file written by ``eval_metrics(out=...)``."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for item in (
        candidate / "metrics.json",
        candidate / "metrics",
        candidate.with_name(f"{candidate.name}.json"),
    ):
        if item.is_file():
            return item
    raise FileNotFoundError(
        f"metrics output not found for {candidate}; run eval_metrics first"
    )


def _metric_keys(metric: str) -> tuple[str, str]:
    """Return aggregate and per-class keys for a selected metric."""

    normalized = str(metric).strip().lower()
    aliases = {
        "maps": "map",
        "map50-95": "map",
        "map50_95": "map",
        "map50/95": "map",
    }
    aggregate_key = aliases.get(normalized, normalized)
    class_key = {
        "map50": "ap50",
        "map75": "ap75",
    }.get(aggregate_key, aggregate_key)
    return aggregate_key, class_key


def _format_metric_value(value: object) -> str:
    return "" if value is None else f"{float(value):.3f}"


def _write_metric_summary_csv(
    metric_outputs: list[Path],
    *,
    output: str | Path,
    data: str,
    model: str,
    weight: str,
    method: str,
    conf: object,
    metric: str,
) -> None:
    """Write aggregate and per-class values for a selected metric."""

    if not metric_outputs:
        return

    aggregate_key, class_key = _metric_keys(metric)
    payloads: list[dict[str, object]] = []
    class_names: list[str] = []
    class_values: list[dict[str, object]] = []
    for metrics_output in metric_outputs:
        metrics_path = _resolve_metrics_file(metrics_output)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"metrics output must be a JSON object: {metrics_path}")
        if aggregate_key not in payload:
            available = ", ".join(sorted(str(key) for key in payload))
            raise KeyError(
                f"metric {metric!r} is not available in {metrics_path}; "
                f"available aggregate fields: {available}"
            )
        payloads.append(payload)
        values: dict[str, object] = {}
        for item in payload.get("classes", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("class_name", item.get("class_id", "unknown")))
            values[name] = item.get(class_key)
            if name not in class_names:
                class_names.append(name)
        class_values.append(values)

    headers = [
        "data",
        "model",
        "weight",
        "method",
        "conf",
        aggregate_key,
        *class_names,
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for payload, values in zip(payloads, class_values):
            writer.writerow(
                [
                    data,
                    model,
                    weight,
                    method,
                    "" if conf is None else conf,
                    _format_metric_value(payload[aggregate_key]),
                    *[_format_metric_value(values.get(name)) for name in class_names],
                ]
            )
    print(f"Metric summary CSV: {output_path}")


def main() -> None:
    # init
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)

    if "merge" in RUN_LIST:
        merge_modalities(
            sources=[
                {"root": "/data/rgb", "name": "rgb"},
                {"root": "/data/thermal", "name": "thermal"},
                {"root": "/data/depth"},  # 默认 image3
            ],
            out_data="/data/multimodal",
            unmatched="keep",  # keep / drop / zero_fill
            merge_labels=True,  # 默认合并标签
            deduplicate=False,  # 默认不去重
            dedup_iou=0.9,
        )

    # preprocess
    if "rename" in RUN_LIST:
        ydm.remap_filenames(
            out=ydm.root + "_rename",
        )
    if "resize" in RUN_LIST:
        ydm.resize_images(
            width=640,
        )

    # pre-analysis
    if "split" in RUN_LIST:
        ydm.dataset_split(
            train=0.9,
            val=0.1,
            train_include_list=LEAKAGE_ONLY_LIST,
        )
    if "sta" in RUN_LIST:
        ydm.stats(
            stats_list=["all"],
            # only_val=True,
        )
    if "vis_draw" in RUN_LIST:
        ydm.vis_draw(
            only_val=True,
        )
    if "vis_crop" in RUN_LIST:
        ydm.vis_crop(
            only_val=True,
        )
    if "query" in RUN_LIST:
        ydm.query_class(
            class_=QUERY_CLASS,
            source="gt",
            only_val=True,
        )
    if "copy_select" in RUN_LIST:
        file_path = (
            Path(SELECTION_FILE)
            if SELECTION_FILE is not None
            else Path(ydm.root) / "val.txt"
        )
        out_path = Path(ydm.root) / "select"
        ydm.dataset_select(
            file=str(file_path),
            out=out_path,
            copy_images=True,
        )

    # post-analysis
    if "metric" in RUN_LIST:
        metric_outputs: list[Path] = []
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "metrics"
            ydm.eval_metrics(
                pred_root=pred_dir,
                out=output_dir,
                # class_rules=METRIC_CLASS_RULES,
                # merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=50,
                # only_val=True,
            )
            metric_outputs.append(output_dir)
        _write_metric_summary_csv(
            metric_outputs,
            output=(
                METRIC_SUMMARY["csv"]
                or ydm.output_evaluation / "metric_summary.csv"
            ),
            data=str(METRIC_SUMMARY["data"] or DATA_DIR.stem),
            model=str(METRIC_SUMMARY["model"]),
            weight=str(METRIC_SUMMARY["weight"]),
            method=str(METRIC_SUMMARY["method"]),
            conf=METRIC_SUMMARY["conf"],
            metric=str(METRIC_SUMMARY["metric"]),
        )
    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "error_analysis"
            ydm.eval_error_analysis(
                pred_root=pred_dir,
                out=output_dir,
                # attribute_file=ydm.root / "attribute.yaml",
                # only_val=True,
            )

    # data update
    if "update_class" in RUN_LIST:
        ydm.ann_update_from_map(UPDATE_CLASS_MAP)
    if "update_att" in RUN_LIST:
        ydm.ann_att_update_from_map(UPDATE_ATT_MAP)
    if "filter_small" in RUN_LIST:
        ydm.anno_update_by_size(
            min_pixels=50,
            logic="or",
            # class_rules=None,
        )

    if "mannual_draw" in RUN_LIST:
        ydm.vis_manual_box(
            "DJI_20260211161740_1654.png",
        )

    if "update_class_from_anno" in RUN_LIST:
        ydm.ann_correct_from_crops(
            crops_dir=CROP_MAP_LABEL,
        )

    if "update_class_from_pred" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            ydm.ann_correct_from_error_crops(
                crops_dir=CROP_MAP_PRED,
                pred_dir=pred_dir,
            )
    if "update_att_from_anno" in RUN_LIST and CROP_MAP_ATTR:
        ydm.ann_att_correct_from_crops(
            crops_dir=CROP_MAP_ATTR,
        )
    if "update_att_from_pred" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            ydm.ann_att_correct_from_error_crops(
                crops_dir=CROP_MAP_PRED,
                pred_dir=pred_dir,
            )


if __name__ == "__main__":
    main()
