"""Reusable attribute corrections based on error-analysis crops."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_update_attr_by_pred(
    dataset_input: YoloManagerInput,
    crops_dir: str | Path,
    attribute_name: str,
    attribute_value: str | int | float,
    *,
    report: str | Path | None = None,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    only_val: bool = False,
) -> int:
    """Set one GT attribute on boxes selected by error-analysis crop names."""

    mgr = get_yolo_manager(
        dataset_input, layout="auto", init_check=False, init_layout=False
    )
    return mgr.ann_correct_attr_from_error_crops(
        crops_dir=crops_dir,
        name=attribute_name,
        value=attribute_value,
        report=report,
        backup_dir=backup_dir,
        dry_run=dry_run,
        only_val=only_val,
    )
