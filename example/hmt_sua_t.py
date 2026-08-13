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

from example.functions import yolo_error_ana

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
)

# Select the operations for this dataset by uncommenting the calls in main().
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


def main() -> None:
    # yolo_sta(
    #     DATA_DIR,
    #     stats_list=["all"],
    # )

    # yolo_vis(DATA_DIR, crop=True)

    # yolo_metric(
    #     DATA_DIR,
    #     PRED_RUNS_DIR,
    #     PRED_NAME,
    #     only_val=True,
    #     show_original=True,
    # )
    # for k, v in crops_map.items():
    #     yolo_update_from_crops(
    #         DATA_DIR,
    #         crops_dir=k,
    #         to=v,
    #     )
    # yolo_metric(
    #     DATA_DIR,
    #     PRED_RUNS_DIR,
    #     PRED_NAME,
    #     # merge_class_map=merge_class_map,
    #     # # exclude_class_=exclude_class_,
    #     min_pixels=20,
    #     # conf_thres=0.20,
    # )

    for PRED_NAME in PRED_NAMES[-1:]:
        # yolo_metric(
        #     DATA_DIR,
        #     PRED_RUNS_DIR,
        #     PRED_NAME,
        #     merge_class_map=MERGE_CLASS_MAP,
        #     # min_pixels=20,
        # )
        yolo_error_ana(
            DATA_DIR,
            PRED_RUNS_DIR,
            PRED_NAME,
            only_val=True,
        )

    # for k, v in CROP_MAP.items():
    #     yolo_update_by_pred(DATA_DIR, crops_dir=k, to=v, pred_dir=PRED_DIR)


if __name__ == "__main__":
    main()
