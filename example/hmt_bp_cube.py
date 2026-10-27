"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from yolo_data_manager import YoloManager

DATA_DIR = Path(r"/srv/nas/home/dataset/hmt_cube_update_v6")

UPDATE_CLASS_MAP = {
    # "drop": [
    #     "background",
    #     "Cracked Tile",
    #     "Delaminated Tile Low Risk",
    #     "Efforescene Low Gray",
    #     "Broken Low Risk",
    # ],
    # "merge": {
    #     "Efflorescene": ["Efflorescene High Risk", "Efflorescene Low Risk"],
    #     "Corrosion": ["Corrosion", "Spalling"],
    #     "Broken": ["Delaminate Tile High Risk", "Broken High Risk"],
    # },
    "merge": {
        "Broken": ["Delaminate", "Broken"],
    },
}
METRIC_CLASS_MAP = {
    # "drop": [
    #     "Broken",
    # ],
    "merge": {
        "Delaminate": ["Delaminate", "Broken"],
    },
}
PRED_RUNS_DIR = Path(r"/root/huilin/projects/ultralytics/runs/detect")


PRED_NAMES = [
    "predict-13",
    "predict-14",
    "predict-15",
    "predict-16",
    "predict-17",
    "predict-18",
]
CROP_MAP_LABEL = {
    # os.path.join(CROP_ROOT_LABEL, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_LABEL, "2_none"): "none",
}


CROP_MAP_PRED_ROOT = os.path.join(DATA_DIR, "ydm_evaluation", "crop_change_common")
CROP_MAP_PRED = {
    os.path.join(CROP_MAP_PRED_ROOT, "gt2none"): "none",
    os.path.join(CROP_MAP_PRED_ROOT, "gt2broken"): ["Broken", False],
    os.path.join(CROP_MAP_PRED_ROOT, "gt2del"): ["Delaminate", False],
    # os.path.join(CROP_MAP_PRED_ROOT, "pred2broken"): "Broken",
    os.path.join(CROP_MAP_PRED_ROOT, "pred2corrosion"): "Corrosion",
    os.path.join(CROP_MAP_PRED_ROOT, "pred2de"): "Delaminate",
    os.path.join(CROP_MAP_PRED_ROOT, "pred2eff"): "Efflorescene",
}
# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "rename",
    # "split",
    # "resize",
    # "sta",
    # "vis_draw",
    # "vis_crop",
    # "query",
    # "copy_select",
    "metric",
    # "error_ana",
    # "error_ana_common",
    # "update_class",
    # "update_att",
    # "filter_small",
    # "mannual_draw",
    # "update_class_from_anno",
    # "update_class_from_pred",
    # "update_att_from_anno",
    # "update_att_from_pred",
]
METRIC_SUMMARY = {
    "metric": "map50",
    "csv": None,  # defaults to <dataset>/ydm_evaluation/metric_summary.csv
}


def main() -> None:
    ydm = YoloManager(
        DATA_DIR,
        layout="flat",
        init_check=False,
        init_layout=False,
    )
    # preprocess
    if "resize" in RUN_LIST:
        ydm.resize_images(
            width=640,
        )

    # pre-analysis
    if "split" in RUN_LIST:
        ydm.dataset_split(
            train=0.9,
            val=0.1,
            # train_include_list=LEAKAGE_ONLY_LIST,
        )
    if "sta" in RUN_LIST:
        ydm.stats(
            stats_list=["all"],
            # only_val=True,
        )
    if "vis_draw" in RUN_LIST:
        ydm.vis_draw(
            # only_val=True,
        )
    if "vis_crop" in RUN_LIST:
        ydm.vis_crop(
            # only_val=True,
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
        metric_outputs = []
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "metrics"
            ydm.eval_metrics(
                pred_root=pred_dir,
                out=output_dir,
                # class_rules=METRIC_CLASS_RULES,
                # merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=50,
                only_val=True,
                # class_map=METRIC_CLASS_MAP,
                overlap_metric="ios",
                conf=0.1,
            )
            metric_outputs.append(output_dir)

        from dataset_template import _write_metric_summary_csv

        _write_metric_summary_csv(
            metric_outputs,
            metric=str(METRIC_SUMMARY["metric"]),
            summary_mode="overall",
        )
    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "error_analysis"
            ydm.eval_error_analysis(
                pred_root=pred_dir,
                out=output_dir,
                conf_thres=0.1,
                class_rules={
                    "Efflorescene": {"min_area": 0.0025},
                },
                # only_val=True,
            )
    if "error_ana_common" in RUN_LIST:
        pred_list = [
            ydm.output_evaluation / pred_name / "error_analysis"
            for pred_name in PRED_NAMES
        ]
        output_dir = ydm.output_evaluation / "common_error_report" / "error_analysis"
        ydm.eval_error_analysis_common(
            pred_list,
            out=output_dir,
            iou=0.5,
        )

    # data update
    if "update_class" in RUN_LIST:
        ydm.ann_update_from_map(
            UPDATE_CLASS_MAP,
            out_data=r"/data/huilin/data/bdd_hmt/hmt_rgb_update_v7",
        )
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
        common_pred_txt = (
            ydm.output_evaluation
            / "common_error_report"
            / "error_analysis"
            / "common_pred_txt"
        )
        pred_txt = PRED_RUNS_DIR / PRED_NAMES[0] / "labels"
        ydm.ann_correct_from_error_crops(
            crops_dir=CROP_MAP_PRED,
            pred_dir=pred_txt,
            out_data=r"/data/huilin/data/bdd_hmt/hmt_rgb_update_v6",
        )


if __name__ == "__main__":
    main()
