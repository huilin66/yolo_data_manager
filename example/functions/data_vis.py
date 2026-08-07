"""Reusable visualization and crop example."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_vis(
    input_dir: str | Path,
    crop: bool = True,
    *,
    draw_out: str | Path | None = None,
    crop_out: str | Path | None = None,
    only_val: bool = False,
    workers: int = 8,
    show_id: bool = True,
    padding: int | float = 0,
) -> int:
    """Render boxes and optionally crops for all data by default."""

    mgr = YoloManager(input_dir, layout="auto", init_check=False, init_layout=False)
    result = mgr.vis_draw(
        draw_out,
        workers=workers,
        show_id=show_id,
        only_val=only_val,
    )
    if crop:
        result = mgr.vis_crop(
            crop_out,
            workers=workers,
            padding=padding,
            only_val=only_val,
        )
    return result

