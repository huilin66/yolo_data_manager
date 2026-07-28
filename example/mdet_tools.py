import os

from yolo_data_manager import (
    compute_multimodal_stats,
    crop_multimodal_dataset,
    load_multimodal_yolo_dataset,
    render_multimodal_dataset,
    write_multimodal_stats_plots,
)
from yolo_data_manager.stats.report import write_json_report


IMAGE_DIR_NAMES = ["visible", "infrared", "depth"]

# Empty configuration means that matching uses the unchanged filename stem.
# For names such as 00000034_V.jpg / 00000034_T.png, configure suffixes here.
IMAGE_PARAMS = {
    # "visible": {"suffix": "_V"},
    # "infrared": {"suffix": "_T"},
    # "depth": {"suffix": "_D"},
}
LABEL_PARAMS = {
    # "suffix": "_gt",  # 00000034_gt.txt -> scene stem 00000034
}


def load_mdet_dataset(input_dir):
    return load_multimodal_yolo_dataset(
        input_dir,
        image_dirs=[os.path.join(input_dir, name) for name in IMAGE_DIR_NAMES],
        image_params=IMAGE_PARAMS,
        labels_dir=os.path.join(input_dir, "labels"),
        label_params=LABEL_PARAMS,
        class_file=os.path.join(input_dir, "class.txt"),
        task="detect",
        progress=True,
    )


def yolo_sta(dataset, input_dir):
    sta_dir = os.path.join(input_dir, "stats", "labels_sta")
    os.makedirs(sta_dir, exist_ok=True)

    stats = compute_multimodal_stats(dataset)
    write_json_report(stats, os.path.join(sta_dir, "multimodal_stats.json"))
    write_multimodal_stats_plots(dataset, sta_dir, stats_list=["all"])
    return stats


def yolo_vis(dataset, input_dir, crop=False):
    vis_dir = os.path.join(input_dir, "image_vis")
    os.makedirs(vis_dir, exist_ok=True)
    rendered = render_multimodal_dataset(dataset, vis_dir, workers=8, show_txt_id=True)
    if crop:
        crop_multimodal_dataset(dataset, os.path.join(vis_dir, "crops"), workers=8, progress=True)
    return rendered


if __name__ == "__main__":
    t_all_dir = r"\\158.132.186.40\isds\huilin\tp\aic_mdet\phase1\train"

    mdet_dataset = load_mdet_dataset(t_all_dir)
    # yolo_sta(mdet_dataset, t_all_dir)
    yolo_vis(mdet_dataset, t_all_dir, crop=False)
