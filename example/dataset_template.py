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

from example.functions import yolo_metric, yolo_sta, yolo_vis


DATA_DIR = Path(r"/path/to/your/dataset.yaml")

# Select the operations for this dataset by uncommenting the calls in main().
PRED_RUNS_DIR = Path(r"/path/to/ultralytics/runs/detect")
PRED_NAME = "val-52"


def main() -> None:
    yolo_sta(
        DATA_DIR,
        stats_list=["all"],
        only_val=False,
    )

    # yolo_vis(DATA_DIR, crop=True, only_val=False)

    # yolo_metric(
    #     DATA_DIR,
    #     PRED_RUNS_DIR,
    #     PRED_NAME,
    #     only_val=True,
    #     show_original=True,
    # )


if __name__ == "__main__":
    main()
