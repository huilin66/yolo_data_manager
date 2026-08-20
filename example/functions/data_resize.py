"""Reusable dataset image-resize example."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_resize(
    dataset_input: YoloManagerInput,
    width: int | None = 640,
    height: int | None = None,
    scale: float | None = None,
    *,
    out: str | Path | None = None,
    keep_ratio: bool = True,
    interpolation: str = "lanczos",
    fill_color: int | Sequence[int] = (114, 114, 114),
    keep_empty_labels: bool = True,
    only_val: bool = False,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    dry_run: bool = False,
) -> int:
    """Resize dataset images and keep YOLO labels synchronized.

    The default width is 640 pixels.  Specify only ``width`` or only ``height``
    to preserve the original aspect ratio.  When both are specified,
    ``keep_ratio=True`` uses letterboxing and
    transforms detection boxes and segmentation polygons automatically.
    If *out* is omitted, the manager writes to ``ydm_conversion/resize``.
    """

    mgr = get_yolo_manager(
        dataset_input,
        layout="auto",
        init_check=False,
        init_layout=False,
    )
    return mgr.resize_images(
        out=out,
        width=width,
        height=height,
        scale=scale,
        keep_ratio=keep_ratio,
        interpolation=interpolation,
        fill_color=fill_color,
        keep_empty_labels=keep_empty_labels,
        only_val=only_val,
        workers=workers,
        progress=progress,
        progress_leave=progress_leave,
        dry_run=dry_run,
    )


# Keep compatibility with the initial example filename implementation.
yolo_vis = yolo_resize
