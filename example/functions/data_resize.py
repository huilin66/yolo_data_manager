"""Reusable visualization and crop example."""

from __future__ import annotations

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_vis(dataset_input: YoloManagerInput, width=640) -> int:
    """Render boxes and optionally crops for all data by default."""

    mgr = get_yolo_manager(
        dataset_input, layout="auto", init_check=False, init_layout=False
    )
    mgr.resize_images(
        out="resized",
        width=640,
    )
