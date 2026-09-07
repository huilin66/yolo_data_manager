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

DATA_DIR = Path(r"\\158.132.186.40\isds\huilin\mayolo\mayolo_v2")


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
        from yolo_data_manager.dataset.split import extract_splits
        from yolo_data_manager.io.loader import load_yolo_dataset

        dataset = load_yolo_dataset(DATA_DIR, workers=16, progress=True)
        result = extract_splits(
            dataset,
            train_include_list=DATA_DIR / "train.txt",
            val_include_list=DATA_DIR / "val.txt",
            test_include_list=DATA_DIR / "test.txt",
            out_root=DATA_DIR,
            workers=16,
            progress=True,
            progress_leave=True,
        )
        print(f"extracted splits: {result}")


if __name__ == "__main__":
    main()
