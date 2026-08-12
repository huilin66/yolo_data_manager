"""Reusable manual-box visualization example."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_draw(
    dataset_input: YoloManagerInput,
    image_name: str | Path,
    *,
    label: str | Path | None = None,
    class_id: int | None = None,
    show_existing: bool = True,
    out: str | Path | None = None,
) -> int:
    """Draw one temporary box and save its coordinates without editing labels."""

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    return mgr.vis_manual_box(
        image_name,
        label=label,
        class_id=class_id,
        show_existing=show_existing,
        out=out,
    )
