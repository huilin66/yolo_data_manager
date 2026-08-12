"""Reusable class-merging example."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_merge_class(
    dataset_input: YoloManagerInput,
    output_dir: str | Path | None,
    merge_dict: Mapping[str | int, str | int | Sequence[str | int]],
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Merge source classes into target classes in a new dataset by default."""

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    return mgr.ann_merge_class(
        merge_dict,
        out=output_dir,
        backup_dir=backup_dir,
        dry_run=dry_run,
    )
