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
    yolo_sta,
    yolo_update_by_pred,
    yolo_vis,
)

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
)


PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    "val-164",
    "val-165",
    "val-166",
    "val-167",
    "val-168",
    "val-169",
]

PRED_DIR = "/localnvme/data/bdd_hmt/sua_t/ydm_evaluation/error_analysis/val-169/review/pred_txt"
CROP_ROOT = (
    "/localnvme/data/bdd_hmt/sua_t/ydm_evaluation/error_analysis/val-169/crop_change"
)
CROP_MAP = {
    os.path.join(CROP_ROOT, "2_h_high"): "Hollow High Risk",
    os.path.join(CROP_ROOT, "2_h_low"): "Hollow Low Risk",
    os.path.join(CROP_ROOT, "none_2_h_high"): "Hollow High Risk",
    os.path.join(CROP_ROOT, "none_2_h_low"): "Hollow Low Risk",
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

# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    # "vis",
    # "metric",
    # "error_ana",
    # "update",
    # "draw",
    "resize",
]


def main() -> None:
    if "sta" in RUN_LIST:
        yolo_sta(
            DATA_DIR,
            stats_list=["all"],
        )

    if "vis" in RUN_LIST:
        yolo_vis(DATA_DIR, crop=True)

    if "metric" in RUN_LIST:
        for pred_name in PRED_NAMES[:]:
            yolo_metric(
                DATA_DIR,
                PRED_RUNS_DIR,
                pred_name,
                merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=20,
            )

    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES[:]:
            yolo_error_ana(
                DATA_DIR,
                PRED_RUNS_DIR,
                pred_name,
                only_val=True,
            )

    if "update" in RUN_LIST:
        for crops_dir, target_class in CROP_MAP.items():
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


if __name__ == "__main__":
    main()
