"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller, not a reusable function module.  Keep the
dataset path and the operations for one dataset here; keep implementation
details in ``example/functions``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from example.functions import (
    yolo_error_ana,
    yolo_sta,
    yolo_vis,
)
from example.functions._manager import get_yolo_manager

DATA_DIR = Path(r"/localnvme/data/billboard/mayolo_v2")

# PRED_RUNS_DIR = Path(r"//localnvme/project/ultralytics/runs/mdetect")
PRED_RUNS_DIR = Path(r"/localnvme/project/isds_project/runs/mdetect")
PRED_NAMES = [
    "predict2",
]

ATT_CROP_PRED_DIR = os.path.join(
    DATA_DIR, "ydm_evaluation", "error_analysis", "predict2", "crop_changes"
)

ATT_CROP_PRED_DICT = {
    "add2no": ["added_billboard", "no"],
    "add2yes": ["added_billboard", "yes"],
    "frame_corroded2no": ["frame_corroded", "no"],
    "frame_corroded2yes": ["frame_corroded", "yes"],
    "peeling2no": ["surface_peeling", "no"],
    "surface_corroded2no": ["surface_corroded", "no"],
    "surface_corroded2yes": ["surface_corroded", "yes"],
}
# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "sta",
    # "vis",
    # "metric",
    # "error_ana",
    # "update_att",
    # "split_vis"
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
    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES[:]:
            yolo_error_ana(
                DATA_DIR,
                os.path.join(PRED_RUNS_DIR, pred_name, "labels"),
                pred_name,
                attribute_file=DATA_DIR / "attribute.yaml",
                abs_path=True,
                # only_val=True,
            )
    if "update_att" in RUN_LIST:
        mgr = get_yolo_manager(
            DATA_DIR,
            attribute_file=DATA_DIR / "attribute.yaml",
        )
        for att_dir_name, (att_name, att_value) in ATT_CROP_PRED_DICT.items():
            mgr.ann_correct_attr_from_error_crops(
                os.path.join(ATT_CROP_PRED_DIR, att_dir_name),
                name=att_name,
                value=att_value,
                # dry_run=True,
            )


if __name__ == "__main__":
    main()
