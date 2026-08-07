"""Reusable class-merging example."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from yolo_data_manager import YoloManager


def yolo_merge_class(
    input_dir: str | Path,
    output_dir: str | Path | None,
    merge_dict: Mapping[str | int, str | int | Sequence[str | int]],
    *,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Merge source classes into target classes in a new dataset by default."""

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    return mgr.ann_merge_class(
        merge_dict,
        out=output_dir,
        backup_dir=backup_dir,
        dry_run=dry_run,
    )

