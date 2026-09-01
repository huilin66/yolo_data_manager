"""Reusable helpers for datasets with multiple aligned image modalities.

Multimodality is configured on the manager; the surrounding operations are
the same check, stats, visualization, and conversion workflows used for a
single-modal dataset.
"""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import MultiModalYoloManager


IMAGE_DIR_NAMES = ["visible", "infrared", "depth"]

# Empty configuration means matching uses the unchanged filename stem.
IMAGE_PARAMS: dict[str, dict[str, str]] = {}
LABEL_PARAMS: dict[str, str] = {}


def load_mdet_manager(
    dataset_input: str | Path,
    *,
    image_dirs: list[str] | None = None,
    image_params: dict[str, dict[str, str]] | None = None,
    label_params: dict[str, str] | None = None,
    labels_dir: str = "labels",
    class_file: str = "class.txt",
) -> MultiModalYoloManager:
    """Create one cached multimodal manager for the selected image folders."""

    return MultiModalYoloManager(
        dataset_input,
        image_dirs=image_dirs or IMAGE_DIR_NAMES,
        image_params=image_params if image_params is not None else IMAGE_PARAMS or None,
        labels_dir=labels_dir,
        label_params=label_params if label_params is not None else LABEL_PARAMS or None,
        class_file=class_file,
        task="detect",
        progress=True,
        progress_leave=False,
    )


def yolo_check(
    mgr: MultiModalYoloManager,
    dataset_input: str | Path | None = None,
) -> dict[str, object]:
    """Check scene association and image format/mode/dtype groups."""

    return mgr.check()


def yolo_sta(
    mgr: MultiModalYoloManager,
    dataset_input: str | Path | None = None,
) -> dict[str, object]:
    """Write shared-label and per-modality statistics."""

    return mgr.stats(stats_list=["all"])


def convert_depth_to_uint8(
    mgr: MultiModalYoloManager,
    dataset_input: str | Path | None = None,
    *,
    value_range: tuple[float, float] = (0, 20000),
    overwrite: bool = False,
) -> dict[str, object]:
    """Write display-safe depth images without changing source images or labels."""

    return mgr.convert_to_uint8(
        None,
        modalities=["depth"],
        stretch=True,
        value_range=value_range,
        preserve_zero=True,
        overwrite=overwrite,
        workers=8,
    )


def yolo_vis(
    mgr: MultiModalYoloManager,
    dataset_input: str | Path | None = None,
    *,
    crop: bool = False,
    output_name: str | None = None,
    style: str = "cv2",
    show_attrs: bool = False,
    filter_no_attrs: bool = False,
    att_seperate: bool = False,
) -> dict[str, int]:
    """Render all configured modalities from the already-loaded manager."""

    separate_attributes = att_seperate and show_attrs
    if output_name is None:
        rendered = mgr.vis_draw(
            style=style,
            workers=8,
            show_id=True,
            show_attrs=show_attrs,
            filter_no_attrs=filter_no_attrs,
            att_seperate=separate_attributes,
        )
    else:
        vis_dir = Path(mgr.root) / output_name
        rendered = mgr.vis_draw(
            vis_dir / "full",
            style=style,
            workers=8,
            show_id=True,
            show_attrs=show_attrs,
            filter_no_attrs=filter_no_attrs,
            att_seperate=separate_attributes,
        )
    if crop:
        if output_name is None:
            mgr.vis_crop(
                style=style,
                workers=8,
                filter_no_attrs=filter_no_attrs,
                att_seperate=separate_attributes,
            )
        else:
            mgr.vis_crop(
                vis_dir / "crops",
                style=style,
                workers=8,
                filter_no_attrs=filter_no_attrs,
                att_seperate=separate_attributes,
            )
    return rendered
