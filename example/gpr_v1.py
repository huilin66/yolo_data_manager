"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

DATA_DIR = Path(r"/srv/nas/home/dataset/gpr_dataset_v1")


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

    if "split" in RUN_LIST:
        ydm.dataset_split()


if __name__ == "__main__":
    main()
