"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller, not a reusable function module.  Keep the
dataset path and the operations for one dataset here; keep implementation
details in ``example/functions``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support both ``python example/my_dataset.py`` and
# ``python -m example.my_dataset`` from a repository checkout.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from example.functions import yolo_metric

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
)

# Select the operations for this dataset by uncommenting the calls in main().
PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    "val-170",
    "val-171",
    "val-172",
    "val-173",
    "val-174",
    "val-175",
]

# crops_map = {
#     "/localnvme/data/bdd_hmt/bp_cube/ydm_vis/crop_change/2_b": "broken",
#     "/localnvme/data/bdd_hmt/bp_cube/ydm_vis/crop_change/b_2_none": None,
#     "/localnvme/data/bdd_hmt/bp_cube/ydm_vis/crop_change/e_2_none": None,
#     "/localnvme/data/bdd_hmt/bp_cube/ydm_vis/crop_change/p_2_none": None,
# }
MERGE_CLASS_MAP = (
    {
        "Broken high": [
            "Broken High Risk",
        ],
        "Delamination": [
            # "Broken High Risk",
            "Delaminated Tile Low Risk",
            "Delaminate Tile High Risk",
            "Cracked Tile",
        ],
        "Efforescene": [
            "Efforescene Low Gray",
            # "Efflorescene Low Risk",
            "Efflorescene High Risk",
            # "Broken Low Risk",
        ],
        # "Broken": [
        #     # "Broken Low Risk",
        #     "Efflorescene Low Risk",
        # ],
    },
)
DEL_CLASS = ["Broken Low Risk"]


def main() -> None:
    # yolo_sta(
    #     DATA_DIR,
    #     stats_list=["all"],
    #     only_val=False,
    # )

    # yolo_vis(DATA_DIR, crop=True, only_val=False)

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

    for PRED_NAME in PRED_NAMES:
        yolo_metric(
            DATA_DIR,
            PRED_RUNS_DIR,
            PRED_NAME,
            merge_class_map=MERGE_CLASS_MAP,
            exclude_class_=DEL_CLASS,
            min_pixels=20,
            conf_thres=0.20,
        )


if __name__ == "__main__":
    main()
