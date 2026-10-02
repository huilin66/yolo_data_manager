"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller, not a reusable function module.  Keep the
dataset path and the operations for one dataset here; keep implementation
details in ``example/functions``.
"""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager

DATA_DIR = Path(
    r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t_update_v6.yaml"
)

PRED_RUNS_DIR = Path(r"//localnvme/project/ultralytics/runs/detect")
PRED_NAMES = [
    # "predict-2",
    # "predict-3",
    # "predict-4",
    # "predict-5",
    # "predict-6",
    # "predict-13",
]
LEAKAGE_ONLY_LIST = (
    # "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"
)
CROP_ROOT_LABEL = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_vis/crop_change"
CROP_ROOT_PRED = "/localnvme/data/bdd_hmt/hmt_t_update_v6/ydm_evaluation/error_analysis/predict-13/crop_change"

CROP_MAP_LABEL = {
    # os.path.join(CROP_ROOT_LABEL, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_LABEL, "2_none"): "none",
}
CROP_MAP_ATTR = {
    # os.path.join(CROP_ROOT_LABEL, "defect_no"): {
    #     "name": "defect",
    #     "value": "no",
    # },
}

CROP_MAP_PRED = {
    # os.path.join(CROP_ROOT_PRED, "2_at"): "Abnormal Temperature",
    # os.path.join(CROP_ROOT_PRED, "2_h"): "Hollow",
    # os.path.join(CROP_ROOT_PRED, "2_l"): "Leakage",
    # os.path.join(CROP_ROOT_PRED, "2_none"): "none",
}
MERGE_CLASS_MAP = {
    # "Hollow": [
    #     "Hollow Low Risk",
    #     "Hollow High Risk",
    # ],
    # "Temperature": [
    #     "Temperature Medium Risk",
    #     "Temperature High Risk",
    # ],
}

METRIC_CLASS_RULES = {
    "Leakage": {"width": 0.05, "height": 0.03, "logic": "and"},
}

UPDATE_CLASS_MAP = {
    # "merge": {
    #     "Hollow Confirmed": ["Hollow High Risk"],
    #     "Hollow Suspected": ["Hollow Low Risk"],
    #     "Leakage": ["Leakage High Risk"],
    # },
    # "drop": [
    #     "background",
    #     "Hollow High Risk Line",
    #     "Temperature Medium Risk",
    #     "Temperature High Risk",
    # ],
}
QUERY_CLASS = "Leakage"
SELECTION_FILE = None
# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    "rename",
    "split",
    "resize",
    "sta",
    "vis_draw",
    "vis_crop",
    "query",
    "copy_select",
    "metric",
    "error_ana",
    "update_class",
    "update_att",
    "filter_small",
    "mannual_draw",
    "update_class_from_anno",
    "update_class_from_pred",
    "update_att_from_anno",
    "update_att_from_pred",
]


def main() -> None:
    # init
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
            train_include_list=LEAKAGE_ONLY_LIST,
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
    if "update_att" in RUN_LIST:
        ydm.ann_att_update_from_map(UPDATE_CLASS_MAP)
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
    if "update_att_from_anno" in RUN_LIST and CROP_MAP_ATTR:
        ydm.ann_att_correct_from_crops(
            crops_dir=CROP_MAP_ATTR,
        )
    if "update_att_from_pred" in RUN_LIST:
        for pred_name in PRED_NAMES:
            pred_dir = PRED_RUNS_DIR / pred_name / "labels"
            ydm.ann_att_correct_from_error_crops(
                crops_dir=CROP_MAP_PRED,
                pred_dir=pred_dir,
            )


if __name__ == "__main__":
    main()
