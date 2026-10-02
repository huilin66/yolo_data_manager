"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

DATA_DIR = Path(
    r"E:\repository\road_project\outputs\stage1_detection_per_dataset\road_forniture_v1"
)


# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    # "vis",
    # "metric",
    # "error_ana",
    # "update",
    # "draw",
    # "resize",
    # "update_class_by_pred",
    # "update_class_by_label",
    "split"
]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)

    if "sta" in RUN_LIST:
        ydm.stats(stats_list=["all"])
    if "vis" in RUN_LIST:
        ydm.vis_draw(only_val=True)
        ydm.vis_crop(only_val=True)
    if "split" in RUN_LIST:
        ydm.dataset_split(ensure_class_presence=True)


if __name__ == "__main__":
    main()
