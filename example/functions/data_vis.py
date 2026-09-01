"""Reusable visualization and crop example."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_vis(
    dataset_input: YoloManagerInput,
    crop: bool = True,
    *,
    draw_out: str | Path | None = None,
    crop_out: str | Path | None = None,
    only_val: bool = False,
    clean: bool = True,
    workers: int = 8,
    show_id: bool = True,
    show_attrs: bool = False,
    filter_no_attrs: bool = False,
    padding: float = 0,
    style: str = "cv2",
    att_seperate: bool = False,
    **kwargs,
) -> int:
    """Render boxes and optionally crops for all data by default.

    ``clean=True`` (default) clears the output directory before rendering /
    cropping, so stale files from previous runs are removed; pass
    ``clean=False`` to keep existing outputs.
    ``style`` accepts ``"cv2"`` (default, with ``"cv"`` as an alias) or
    ``"pil"``. Drawing and crop generation are separate operations; each uses
    the configured worker pool independently.
    """

    mgr = get_yolo_manager(
        dataset_input, layout="auto", init_check=False, init_layout=False
    )
    separate_attributes = att_seperate and show_attrs
    result = mgr.vis_draw(
        draw_out,
        style=style,
        workers=workers,
        show_id=show_id,
        show_attrs=show_attrs,
        filter_no_attrs=filter_no_attrs,
        att_seperate=separate_attributes,
        only_val=only_val,
        clean=clean,
        **kwargs,
    )
    if crop:
        result = mgr.vis_crop(
            crop_out,
            style=style,
            workers=workers,
            padding=padding,
            filter_no_attrs=filter_no_attrs,
            att_seperate=separate_attributes,
            only_val=only_val,
            clean=clean,
        )
    return result
