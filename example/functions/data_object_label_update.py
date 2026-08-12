"""Reusable corrections based on ordinary visualization crops."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_update(
    dataset_input: YoloManagerInput,
    crops_dir: str | Path,
    to: str | int | None,
    *,
    report: str | Path | None = None,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    only_val: bool = False,
) -> int:
    """Update or delete the GT instance referenced by crop filename ``..._gty``."""

    mgr = get_yolo_manager(dataset_input, layout="auto", init_check=False, init_layout=False)
    return mgr.ann_correct_from_crops(
        crops_dir=crops_dir,
        to=to,
        report=report,
        backup_dir=backup_dir,
        dry_run=dry_run,
        only_val=only_val,
    )
