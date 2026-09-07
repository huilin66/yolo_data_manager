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
from example.functions._manager import get_yolo_manager

DATA_DIR = Path(r"\\158.132.186.40\isds\huilin\mayolo\mayolo_v3")


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
    "split_vis"
]


def main() -> None:
    if "sta" in RUN_LIST:
        yolo_sta(
            DATA_DIR,
            stats_list=["all"],
        )

    if "vis" in RUN_LIST:
        yolo_vis(
            DATA_DIR,
            crop=True,
            show_attrs=True,
        )
    if "split_vis" in RUN_LIST:
        ydm = get_yolo_manager(
            DATA_DIR, layout="flat", init_check=False, init_layout=False
        )
        ydm.dataset_extract_split(
            train_include_list="train.txt",
            val_include_list="val.txt",
            test_include_list="test.txt",
        )


if __name__ == "__main__":
    main()
