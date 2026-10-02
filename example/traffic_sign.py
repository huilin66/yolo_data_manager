"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

DATA_DIR = Path(r"D:\zhl\data\CCTSDB 2021\CCTSDB2021_yolo")

PRED_RUNS_DIR = Path(r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect")
# PRED_NAME = "val-161"
PRED_NAMES = [
    "val-230",
]


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
    # "query",
]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)

    if "sta" in RUN_LIST:
        ydm.stats(stats_list=["all"])

    if "vis" in RUN_LIST:
        ydm.vis_draw()
        ydm.vis_crop()

    if "query" in RUN_LIST:
        ydm.query_class(
            source="pred",
            class_="occluded",
            pred_root="/localnvme/project/aic_mdet/models/ultralytics/runs/detect/predict-11/labels",
        )


if __name__ == "__main__":
    main()
