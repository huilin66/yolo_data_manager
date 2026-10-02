"""Reusable dataset filename-remapping example."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_filename_remap(
    dataset_input: YoloManagerInput,
    *,
    out: str | Path | None = None,
    digits: int | None = None,
    start: int = 0,
    mapping_file: str | Path | None = None,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    dry_run: bool = False,
) -> int:
    """Copy a dataset with numeric image/label names and a mapping JSON.

    If ``digits`` is omitted, the width is calculated from image_count * 10.
    For example, 8,951 images use six-digit names such as ``000000.jpg``.
    The default output is ``ydm_conversion/filename_remap``.
    """

    manager = get_yolo_manager(
        dataset_input,
        layout="auto",
        init_check=False,
        init_layout=False,
    )
    return manager.remap_filenames(
        out=out,
        digits=digits,
        start=start,
        mapping_file=mapping_file,
        workers=workers,
        progress=progress,
        progress_leave=progress_leave,
        dry_run=dry_run,
    )
