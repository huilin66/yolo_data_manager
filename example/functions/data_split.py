"""Reusable dataset split example."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_split(
    dataset_input: YoloManagerInput,
    train: float = 0.9,
    val: float = 0.1,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = True,
    *,
    out: str | Path | None = None,
    train_include_list: str | Path | Sequence[str] | None = None,
    val_include_list: str | Path | Sequence[str] | None = None,
) -> int:
    """Write train/val/test lists with optional forced train/val images.

    Include values can be image names/paths, a comma-separated string, or a
    txt file containing one image name/path per line. Included images are
    removed from the random pool before the requested ratios are applied.
    """

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    return mgr.dataset_split(
        train=train,
        val=val,
        test=test,
        seed=seed,
        absolute_paths=absolute_paths,
        out=out,
        train_include_list=train_include_list,
        val_include_list=val_include_list,
    )
