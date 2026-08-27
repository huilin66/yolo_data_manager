"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller, not a reusable function module.  Keep the
dataset path and the operations for one dataset here; keep implementation
details in ``example/functions``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Support both ``python example/my_dataset.py`` and
# ``python -m example.my_dataset`` from a repository checkout.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from example.functions import (
    yolo_draw,
    yolo_error_ana,
    yolo_metric,
    yolo_resize,
    yolo_split,
    yolo_sta,
    yolo_update_by_label,
    yolo_update_by_pred,
    yolo_update_class,
    yolo_vis,
)

# HMT_V2_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v2"
HMT_V3_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v3"
HMT_V4_DIR = r"/localnvme/data/bdd_hmt/hmt_t_update_v4"
# DATA_DIR = Path(
#     r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
# )
DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t_update_v6.yaml"
)

# PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
PRED_RUNS_DIR = Path(r"//localnvme/project/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    # "predict-2",
    # "predict-3",
    # "predict-4",
    # "predict-5",
    # "predict-6",
    "predict-13",
]
LEAKAGE_ONLY_LIST = (
    "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"
)
PRED_DIR = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_evaluation/error_analysis/predict-13/review/pred_txt"
CROP_ROOT_LABEL = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_vis/crop_change"
CROP_ROOT_PRED = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_evaluation/error_analysis/predict-13/crop_change"

CROP_MAP_LABEL = {
    os.path.join(CROP_ROOT_LABEL, "2_l"): "Leakage",
    os.path.join(CROP_ROOT_LABEL, "2_none"): "none",
}

CROP_MAP_PRED = {
    # os.path.join(CROP_ROOT_PRED, "2_at"): "Abnormal Temperature",
    os.path.join(CROP_ROOT_PRED, "2_h"): "Hollow",
    os.path.join(CROP_ROOT_PRED, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_PRED, "2_none"): "none",
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

# Per-class size filter applied in yolo_metric. `logic="and"` keeps a box
# unless BOTH normalized width and height are < 0.05 (i.e. long & wide small).
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


# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    # "vis",
    # "metric",
    # "error_ana",
    # "update",
    # "draw",
    # "resize",
    "update_class_by_pred",
    # "update_class_by_label",
    # "split"
]


def main() -> None:
    if "sta" in RUN_LIST:
        yolo_sta(
            DATA_DIR,
            stats_list=["all"],
            only_val=True,
        )

    if "vis" in RUN_LIST:
        yolo_vis(DATA_DIR, crop=True)
    if "update_class_by_label" in RUN_LIST:
        for crops_dir, target_class in CROP_MAP_LABEL.items():
            yolo_update_by_label(
                DATA_DIR,
                crops_dir=crops_dir,
                to=target_class,
            )

    if "metric" in RUN_LIST:
        for pred_name in PRED_NAMES[:]:
            yolo_metric(
                DATA_DIR,
                PRED_RUNS_DIR,
                pred_name,
                # class_rules=METRIC_CLASS_RULES,
                # merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=50,
            )

    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES[:]:
            yolo_error_ana(
                DATA_DIR,
                PRED_RUNS_DIR,
                pred_name,
                # only_val=True,
            )

    if "update_class_by_pred" in RUN_LIST:
        for crops_dir, target_class in CROP_MAP_PRED.items():
            yolo_update_by_pred(
                DATA_DIR,
                crops_dir=crops_dir,
                to=target_class,
                pred_dir=PRED_DIR,
            )

    if "draw" in RUN_LIST:
        yolo_draw(DATA_DIR, "DJI_20260211161740_1654.png")

    if "resize" in RUN_LIST:
        yolo_resize(
            DATA_DIR,
            width=640,
        )
    if "update_class" in RUN_LIST:
        yolo_update_class(DATA_DIR, class_map=UPDATE_CLASS_MAP)

    if "split" in RUN_LIST:
        yolo_split(HMT_V3_DIR, train_include_list=LEAKAGE_ONLY_LIST)


if __name__ == "__main__":
    main()
