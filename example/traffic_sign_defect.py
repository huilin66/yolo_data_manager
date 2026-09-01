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
from example.functions import (
    yolo_sta,
    yolo_vis,
)
from yolo_data_manager import YoloManager

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_traffic_sign/tf_defect_3.yaml"
)

PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    "val-230",
]


# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    # "vis",
    # "metric",
    # "error_ana",
    # "update",
    # "draw",
    # "resize",
    # "update_class",
    # "update_class_by_label",
    # "split"
    "query",
]


def main() -> None:
    if "sta" in RUN_LIST:
        yolo_sta(
            DATA_DIR,
            stats_list=["all"],
        )

    if "vis" in RUN_LIST:
        yolo_vis(DATA_DIR, crop=True)

    if "query" in RUN_LIST:
        mgr = YoloManager(DATA_DIR, init_check=False, init_layout=False)
        mgr.query_class(
            source="pred",
            class_="occluded",
            pred_root="/localnvme/project/aic_mdet/models/ultralytics/runs/detect/predict-11/labels",
        )


if __name__ == "__main__":
    main()
