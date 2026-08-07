"""Reusable corrections based on ordinary visualization crops."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_update(
    input_dir: str | Path,
    crops_dir: str | Path,
    to: str | int | None,
    *,
    report: str | Path | None = None,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    only_val: bool = False,
) -> int:
    """Update or delete the GT instance referenced by crop filename ``..._gty``."""

    mgr = YoloManager(input_dir, layout="auto", init_check=False, init_layout=False)
    return mgr.ann_correct_from_crops(
        crops_dir=crops_dir,
        to=to,
        report=report,
        backup_dir=backup_dir,
        dry_run=dry_run,
        only_val=only_val,
    )

