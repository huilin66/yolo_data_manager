"""Reusable manual-box visualization example."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_draw(
    input_dir: str | Path,
    image_name: str | Path,
    *,
    label: str | Path | None = None,
    class_id: int | None = None,
    show_existing: bool = True,
    out: str | Path | None = None,
) -> int:
    """Draw one temporary box and save its coordinates without editing labels."""

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    return mgr.vis_manual_box(
        image_name,
        label=label,
        class_id=class_id,
        show_existing=show_existing,
        out=out,
    )

