"""Example workflow for a shared-label, visible/infrared/depth YOLO dataset.

Set ``DATASET_DIR`` and enable the operations wanted in ``__main__``.  The
manager is intentionally created once so check, stats and visualization reuse
the same multimodal association and parsed labels.
"""

from pathlib import Path

from yolo_data_manager import MultiModalYoloManager


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


def load_mdet_manager(
    input_dir: str | Path,
    *,
    image_dirs: list[str] | None = None,
) -> MultiModalYoloManager:
    """Create one cached multimodal manager for the selected image folders."""

    return MultiModalYoloManager(
        input_dir,
        image_dirs=image_dirs or IMAGE_DIR_NAMES,
        image_params=IMAGE_PARAMS or None,
        labels_dir="labels",
        label_params=LABEL_PARAMS or None,
        class_file="class.txt",
        task="detect",
        progress=True,
        progress_leave=False,
    )


def yolo_check(mgr: MultiModalYoloManager, input_dir: str | Path) -> dict[str, object]:
    """Check scene association and print image format/mode/dtype count groups."""

    return mgr.check(out=Path(input_dir) / "multimodal_check_result.json")


def yolo_sta(mgr: MultiModalYoloManager, input_dir: str | Path) -> dict[str, object]:
    """Write shared-label and per-modality statistics, including image types."""

    stats_dir = Path(input_dir) / "stats" / "labels_sta"
    return mgr.stats(
        out=stats_dir / "multimodal_stats.json",
        plots_dir=stats_dir,
        stats_list=["all"],
    )


def convert_depth_to_uint8(
    mgr: MultiModalYoloManager,
    input_dir: str | Path,
    *,
    value_range: tuple[float, float] = (0, 20000),
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy depth uint8 images and stretch non-uint8 depth maps into PNG files.

    This does not change ``depth/``. The converted modality is written to
    ``images_uint8/depth/`` and keeps zero-valued invalid depth pixels black.
    """

    return mgr.convert_to_uint8(
        Path(input_dir) / "images_uint8",
        modalities=["depth"],
        stretch=True,
        value_range=value_range,
        preserve_zero=True,
        overwrite=overwrite,
        workers=8,
    )


def yolo_vis(
    mgr: MultiModalYoloManager,
    input_dir: str | Path,
    *,
    crop: bool = False,
    output_name: str = "image_vis",
) -> dict[str, int]:
    """Render all modalities from the already-loaded manager."""

    vis_dir = Path(input_dir) / output_name
    rendered = mgr.vis_draw(vis_dir / "full", workers=8, show_id=True)
    if crop:
        mgr.vis_crop(vis_dir / "crops", workers=8)
    return rendered


if __name__ == "__main__":
    DATASET_DIR = Path(r"\\158.132.186.40\isds\huilin\tp\aic_mdet\phase1\train")

    # 1. Always inspect association and image dtype groups before rendering.
    manager = load_mdet_manager(DATASET_DIR)
    yolo_check(manager, DATASET_DIR)

    # 2. Optional: write depth images as display-safe uint8 PNG. The source
    #    depth folder remains unchanged. Afterwards create a manager whose
    #    depth folder is the converted output before visualizing it.
    USE_UINT8_DEPTH = True
    REBUILD_UINT8_DEPTH = False
    converted_depth_dir = DATASET_DIR / "images_uint8" / "depth"
    if USE_UINT8_DEPTH:
        if REBUILD_UINT8_DEPTH or not converted_depth_dir.exists():
            convert_depth_to_uint8(
                manager,
                DATASET_DIR,
                value_range=(0, 20000),
                overwrite=REBUILD_UINT8_DEPTH,
            )
        manager = load_mdet_manager(
            DATASET_DIR,
            image_dirs=["visible", "infrared", "images_uint8/depth"],
        )
        yolo_check(manager, DATASET_DIR)

    RUN_STATS = False
    RUN_VISUALIZATION = True
    if RUN_STATS:
        yolo_sta(manager, DATASET_DIR)
    if RUN_VISUALIZATION:
        yolo_vis(manager, DATASET_DIR, crop=True)
