"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager

# DATA_DIR = Path(r"D:\zhl\data\road_forniture_v1")
DATA_DIR = Path(r"D:\zhl\data\road_forniture_v1_rename")
PRED_RUNS_DIR = Path(r"D:\zhl\project\ultralytics\runs\detect")


PRED_NAMES = [
    "predict-2",
]

QUERY_CLASS = None
SELECTION_FILE = None

# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    # "rename",
    # "split",
    # "resize",
    # "sta",
    # "vis_draw",
    # "vis_crop",
    # "query",
    # "copy_select",
    # "metric",
    "error_ana",
    # "update_class",
    # "update_att",
    # "filter_small",
    # "mannual_draw",
    # "update_class_from_anno",
    # "update_class_from_pred",
    # "update_att_from_anno",
    # "update_att_from_pred",
]


def main() -> None:
    ydm = YoloManager(DATA_DIR, layout="auto", init_check=False, init_layout=False)
    # preprocess
    if "rename" in RUN_LIST:
        ydm.remap_filenames(
            out=ydm.root + "_rename",
        )
    if "resize" in RUN_LIST:
        ydm.resize_images(
            width=640,
        )

    # pre-analysis
    if "split" in RUN_LIST:
        ydm.dataset_split(
            train=0.9,
            val=0.1,
        )
    if "sta" in RUN_LIST:
        ydm.stats(
            stats_list=["all"],
            # only_val=True,
        )
    if "vis_draw" in RUN_LIST:
        ydm.vis_draw(
            only_val=True,
        )
    if "vis_crop" in RUN_LIST:
        ydm.vis_crop(
            only_val=True,
        )
    if "query" in RUN_LIST:
        ydm.query_class(
            class_=QUERY_CLASS,
            source="gt",
            only_val=True,
        )
    if "copy_select" in RUN_LIST:
        file_path = (
            Path(SELECTION_FILE)
            if SELECTION_FILE is not None
            else Path(ydm.root) / "val.txt"
        )
        out_path = Path(ydm.root) / "select"
        ydm.dataset_select(
            file=str(file_path),
            out=out_path,
            copy_images=True,
        )

    # post-analysis
    if "metric" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "metrics"
            ydm.eval_metrics(
                pred_root=pred_dir,
                out=output_dir,
                # class_rules=METRIC_CLASS_RULES,
                # merge_class_map=MERGE_CLASS_MAP,
                # min_pixels=50,
                # only_val=True,
            )
    if "error_ana" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            output_dir = ydm.output_evaluation / pred_name / "error_analysis"
            ydm.eval_error_analysis(
                pred_root=pred_dir,
                out=output_dir,
                # attribute_file=ydm.root / "attribute.yaml",
                # only_val=True,
            )

    # data update
    if "update_class" in RUN_LIST:
        ydm.ann_update_from_map(UPDATE_CLASS_MAP)
    if "filter_small" in RUN_LIST:
        ydm.anno_update_by_size(
            min_pixels=50,
            logic="or",
            # class_rules=None,
        )

    if "mannual_draw" in RUN_LIST:
        ydm.vis_manual_box(
            "DJI_20260211161740_1654.png",
        )

    if "update_class_from_anno" in RUN_LIST:
        ydm.ann_correct_from_crops(
            crops_dir=CROP_MAP_LABEL,
        )

    if "update_class_from_pred" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            ydm.ann_correct_from_error_crops(
                crops_dir=CROP_MAP_PRED,
                pred_dir=pred_dir,
            )


if __name__ == "__main__":
    main()
