"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

# HMT_V2_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v2"
HMT_V3_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v3"
HMT_V4_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v4"
# DATA_DIR = Path(
#     r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
# )
DATA_DIR = Path(r"E:\repository\isds_project\data\risk_data\data_risk_b_split\set_31")

PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    "predict-9",
    "predict-10",
    # "val-219",
    # "val-165",
    # "val-166",
    # "val-167",
    # "val-168",
    # "val-169",
]
LEAKAGE_ONLY_LIST = (
    "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"
)
PRED_DIR = "/localnvme/data/bdd_hmt/sua_t/ydm_evaluation/error_analysis/val-169/review/pred_txt"
CROP_ROOT_LABEL = "/localnvme/data/bdd_hmt/sua_t/ydm_vis/crop_change"
CROP_ROOT_PRED = (
    "/localnvme/data/bdd_hmt/sua_t/ydm_evaluation/error_analysis/val-169/crop_change"
)

CROP_MAP_LABEL = {
    Path(CROP_ROOT_LABEL) / "2_broken": "Broken",
}

CROP_MAP_PRED = {
    Path(CROP_ROOT_PRED) / "2_h_high": "Hollow High Risk",
    Path(CROP_ROOT_PRED) / "2_h_low": "Hollow Low Risk",
    Path(CROP_ROOT_PRED) / "none_2_h_high": "Hollow High Risk",
    Path(CROP_ROOT_PRED) / "none_2_h_low": "Hollow Low Risk",
}
MERGE_CLASS_MAP = {
    "Hollow": [
        "Hollow Low Risk",
        "Hollow High Risk",
    ],
    "Temperature": [
        "Temperature Medium Risk",
        "Temperature High Risk",
    ],
}

UPDATE_CLASS_MAP = {
    "merge": {
        "Hollow Confirmed": ["Hollow High Risk"],
        "Hollow Suspected": ["Hollow Low Risk"],
        "Leakage": ["Leakage High Risk"],
    },
    "drop": [
        "background",
        "Hollow High Risk Line",
        "Temperature Medium Risk",
        "Temperature High Risk",
    ],
}

# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    "vis",
    # "metric",
    # "error_ana",
    # "update",
    # "draw",
    # "resize",
    # "update_class",
    # "update_class_by_label",
    # "split"
]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)

    if "sta" in RUN_LIST:
        ydm.stats(stats_list=["all"])

    if "vis" in RUN_LIST:
        ydm.vis_draw(
            show_attrs=True,
            filter_level=[1],
            att_seperate=True,
        )
        ydm.vis_crop(filter_level=[1], att_seperate=True)
    if "update_class_by_label" in RUN_LIST:
        ydm.ann_correct_from_crops(crops_dir=CROP_MAP_LABEL)

    if "metric" in RUN_LIST:
        for pred_name in PRED_NAMES:
            ydm.eval_metrics(
                pred_root=PRED_RUNS_DIR / pred_name / "labels",
                out=ydm.output_evaluation / pred_name / "metrics",
                # merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=50,
            )

    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES:
            ydm.eval_error_analysis(
                pred_root=PRED_RUNS_DIR / pred_name / "labels",
                out=ydm.output_evaluation / pred_name / "error_analysis",
                only_val=True,
            )

    if "update" in RUN_LIST:
        ydm.ann_correct_from_error_crops(
            crops_dir=CROP_MAP_PRED,
            pred_dir=PRED_DIR,
        )

    if "draw" in RUN_LIST:
        ydm.vis_manual_box("DJI_20260211161740_1654.png")

    if "resize" in RUN_LIST:
        ydm.resize_images(width=640)
    if "update_class" in RUN_LIST:
        ydm.ann_update_from_map(UPDATE_CLASS_MAP)

    if "split" in RUN_LIST:
        ydm.dataset_split(train_include_list=LEAKAGE_ONLY_LIST)


if __name__ == "__main__":
    main()
