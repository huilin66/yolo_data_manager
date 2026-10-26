"""Copy this file to ``example/<dataset_name>.py`` and edit its parameters.

The file is a dataset-level caller. Keep the dataset path and operations for
one dataset here and call the public manager API directly.
"""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import MultiModalYoloManager, merge_modalities

# DATA_DIR = Path(r"/data/huilin/data/bdd_hmt/sua_rgb_update_v1")
# DATA_DIR = Path(r"/data/huilin/data/bdd_hmt/hmt_rgb_update_v2")
DATA_DIR = Path(r"/data/huilin/data/bdd_hmt/hmt_rgbt_update_v6")

# The merged dataset has one shared labels/ directory and one image directory
# per modality. Use one modality as the reference for single-image APIs.
REFERENCE_MODALITY = "rgb"
MODALITY_DIRS = ["rgb", "t"]

UPDATE_CLASS_MAP = {
    # "drop": [
    #     "background",
    #     "Cracked Tile",
    #     "Delaminated Tile Low Risk",
    #     "Efforescene Low Gray",
    #     "Broken Low Risk",
    # ],
    # "merge": {
    #     "Efflorescene": ["Efflorescene High Risk", "Efflorescene Low Risk"],
    #     "Corrosion": ["Corrosion", "Spalling"],
    #     "Broken": ["Delaminate Tile High Risk", "Broken High Risk"],
    # },
    "merge": {
        "Broken": ["Delaminate", "Broken"],
    },
}

PRED_RUNS_DIR = Path(r"/data/huilin/projects/ultralytics/runs/detect")


PRED_NAMES = [
    # "predict",
    # "predict-3",
    "predict-4",
    # "predict-5",
]
CROP_MAP_LABEL = {
    # Path(CROP_ROOT_LABEL) / "2_l": "Leakage",
    # Path(CROP_ROOT_LABEL) / "2_none": "none",
}


CROP_MAP_PRED_ROOT = DATA_DIR / "ydm_evaluation" / "crop_change_common"
CROP_MAP_PRED = {
    CROP_MAP_PRED_ROOT / "gt2none": "none",
    CROP_MAP_PRED_ROOT / "gt2broken": ["Broken", False],
    CROP_MAP_PRED_ROOT / "gt2del": ["Delaminate", False],
    # CROP_MAP_PRED_ROOT / "pred2broken": "Broken",
    CROP_MAP_PRED_ROOT / "pred2corrosion": "Corrosion",
    CROP_MAP_PRED_ROOT / "pred2de": "Delaminate",
    CROP_MAP_PRED_ROOT / "pred2eff": "Efflorescene",
}

# Optional operations use these values only when their names are added to
# RUN_LIST. Defining them here avoids a NameError when a switch is enabled.
QUERY_CLASS: str | None = None
SELECTION_FILE: Path | None = None
UPDATE_ATT_MAP: dict = {}


# Select operations by uncommenting names in RUN_LIST.
RUN_LIST = [
    "merge",
    # "rename",
    # "split",
    # "resize",
    # "sta",
    # "vis_draw",
    # "vis_crop",
    # "query",
    # "copy_select",
    # "metric",
    # "error_ana",
    # "error_ana_common",
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
    if "resize" in RUN_LIST:
        raise RuntimeError(
            "resize is not a post-merge operation; resize each source dataset "
            "before running this multimodal example"
        )

    if "merge" in RUN_LIST:
        merge_modalities(
            sources=[
                {"root": "/data/huilin/data/bdd_hmt/hmt_rgb_update_v6", "name": "rgb"},
                {"root": "/data/huilin/data/bdd_hmt/hmt_t_update_v6", "name": "t"},
            ],
            out_data="/data/huilin/data/bdd_hmt/hmt_rgbt_update_v6",
            unmatched="keep",  # keep / drop / zero_fill
            merge_labels=True,  # 默认合并标签
            deduplicate=False,  # 默认不去重
        )

    ydm = MultiModalYoloManager(
        DATA_DIR,
        image_dirs=MODALITY_DIRS,
        labels_dir="labels",
        reference_modality=REFERENCE_MODALITY,
        init_check=False,
    )

    # pre-analysis
    if "split" in RUN_LIST:
        ydm.dataset_split(
            train=0.9,
            val=0.1,
            test=0.0,
            require_labels=True,
        )
    if "sta" in RUN_LIST:
        ydm.stats(
            stats_list=["all"],
        )
    if "vis_draw" in RUN_LIST:
        ydm.vis_draw(
            modalities=MODALITY_DIRS,
        )
    if "vis_crop" in RUN_LIST:
        ydm.vis_crop(
            modalities=MODALITY_DIRS,
        )
    if "query" in RUN_LIST:
        if QUERY_CLASS is None:
            raise ValueError("set QUERY_CLASS before enabling 'query'")
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
                conf_thres=0.1,
                class_rules={
                    "Efflorescene": {"min_area": 0.0025},
                },
                # only_val=True,
            )
    if "error_ana_common" in RUN_LIST:
        pred_list = [
            ydm.output_evaluation / pred_name / "error_analysis"
            for pred_name in PRED_NAMES
        ]
        output_dir = ydm.output_evaluation / "common_error_report" / "error_analysis"
        ydm.eval_error_analysis_common(
            pred_list,
            out=output_dir,
            iou=0.5,
        )

    # data update
    if "update_class" in RUN_LIST:
        ydm.ann_update_from_map(
            UPDATE_CLASS_MAP,
            # Labels are shared by rgb and t. In-place editing preserves the
            # multimodal directory structure and creates labels_backup.
        )
    if "update_att" in RUN_LIST and UPDATE_ATT_MAP:
        ydm.ann_att_update_from_map(UPDATE_ATT_MAP)
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
        common_pred_txt = (
            ydm.output_evaluation
            / "common_error_report"
            / "error_analysis"
            / "common_pred_txt"
        )
        ydm.ann_correct_from_error_crops(
            crops_dir=CROP_MAP_PRED,
            pred_dir=common_pred_txt,
            # Do not pass out_data here: the current correction writer is
            # single-modal and would drop the other modality folders.
        )


if __name__ == "__main__":
    main()
