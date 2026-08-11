"""Reusable corrections based on error-analysis crops."""

from __future__ import annotations

from pathlib import Path

from ._manager import YoloManagerInput, get_yolo_manager


def yolo_update(
    dataset_input: YoloManagerInput,
    crops_dir: str | Path,
    to: str | int | None,
    *,
    pred_dir: str | Path | None = None,
    dedup_iou: float | None = 0.5,
    delete_pred_none: bool = False,
    replace_gt_from_pred: bool = False,
    report: str | Path | None = None,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    only_val: bool = False,
) -> int:
    """Correct GT, delete GT, or add/replace predictions referenced by crops."""

    mgr = get_yolo_manager(dataset_input, layout="auto", init_check=False, init_layout=False)
    return mgr.ann_correct_from_error_crops(
        crops_dir=crops_dir,
        to=to,
        pred_dir=pred_dir,
        dedup_iou=dedup_iou,
        delete_pred_none=delete_pred_none,
        replace_gt_from_pred=replace_gt_from_pred,
        report=report,
        backup_dir=backup_dir,
        dry_run=dry_run,
        only_val=only_val,
    )
