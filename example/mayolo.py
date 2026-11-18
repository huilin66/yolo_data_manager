"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path
from yolo_data_manager import YoloManager

DATA_DIR = Path(r"E:\data\0417_signboard\mayolo_v3")

# PRED_VIS_DIR = Path(
#     r"E:\data\0417_signboard\select\predictions_seed0_rerun\YOLOv10x_native\labels"
# )
PRED_IMG_DIR = Path(r"E:\data\0417_signboard\select\predictions_seed0_robustness\IMAGE")
PRED_TXT_DIR = Path(
    r"E:\data\0417_signboard\select\predictions_seed0_robustness\MAYOLOx\labels"
)
# PRED_RUNS_DIR = Path(r"//localnvme/project/ultralytics/runs/mdetect")
PRED_RUNS_DIR = Path(r"/localnvme/project/isds_project/runs/mdetect")
PRED_NAMES = [
    "predict2",
]

ATT_CROP_PRED_DIR = (
    DATA_DIR / "ydm_evaluation" / "error_analysis" / "predict2" / "crop_changes"
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
    "pred_vis",
    # "metric",
    # "error_ana",
    # "update_att",
    # "split_vis"
]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="flat", init_layout=False, init_check=False)

    if "sta" in RUN_LIST:
        ydm.stats(stats_list=["all"])

    if "vis" in RUN_LIST:
        ydm.vis_draw(
            show_attrs=True,
            show_id=False,
            filter_level=[1],
        )
        ydm.vis_crop(filter_level=[1])
    if "pred_vis" in RUN_LIST:
        ydm.vis_draw(
            images_dir=PRED_IMG_DIR,
            labels_dir=PRED_TXT_DIR,
            out=PRED_TXT_DIR.with_name("pred_vis"),
            show_conf=True,
            conf=0.5,
            show_attrs=True,
            show_id=False,
        )
    if "split_vis" in RUN_LIST:
        ydm.dataset_extract_split(
            train_include_list="train.txt",
            val_include_list="val.txt",
            test_include_list="test.txt",
        )
    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES:
            ydm.eval_error_analysis(
                pred_root=PRED_RUNS_DIR / pred_name / "labels",
                out=ydm.output_evaluation / pred_name / "error_analysis",
                attribute_file=DATA_DIR / "attribute.yaml",
                copy_pred_txt=True,
                # only_val=True,
            )
    if "update_att" in RUN_LIST:
        ydm.ann_att_correct_from_error_crops(
            {
                ATT_CROP_PRED_DIR / att_dir_name: {
                    "name": att_name,
                    "value": att_value,
                }
                for att_dir_name, (att_name, att_value) in ATT_CROP_PRED_DICT.items()
            },
            # dry_run=True,
        )


if __name__ == "__main__":
    main()
