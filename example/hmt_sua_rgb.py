"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

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
MERGE_CLASS_MAP = {
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
}
DEL_CLASS = ["Broken Low Risk"]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)

    # ydm.stats(
    #     stats_list=["all"],
    #     only_val=False,
    # )

    # ydm.vis_draw(only_val=False)
    # ydm.vis_crop(only_val=False)

    # ydm.eval_metrics(
    #     pred_root=PRED_RUNS_DIR / PRED_NAMES[0] / "labels",
    #     only_val=True,
    #     show_original=True,
    # )
    # for k, v in crops_map.items():
    #     ydm.ann_correct_from_crops(crops_dir=k, to=v)
    # ydm.eval_metrics(
    #     pred_root=PRED_RUNS_DIR / PRED_NAMES[0] / "labels",
    #     merge_class_map=MERGE_CLASS_MAP,
    #     exclude_class_=DEL_CLASS,
    #     min_pixels=20,
    #     # conf_thres=0.20,
    # )

    for pred_name in PRED_NAMES:
        ydm.eval_metrics(
            pred_root=PRED_RUNS_DIR / pred_name / "labels",
            out=ydm.output_evaluation / pred_name / "metrics",
            merge_class_map=MERGE_CLASS_MAP,
            exclude_class_=DEL_CLASS,
            min_pixels=20,
            conf_thres=0.20,
        )


if __name__ == "__main__":
    main()
