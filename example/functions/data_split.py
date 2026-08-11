"""Reusable dataset split example."""

from __future__ import annotations

from pathlib import Path

from ._manager import YoloManagerInput, get_yolo_manager


def yolo_split(
    dataset_input: YoloManagerInput,
    train: float = 0.9,
    val: float = 0.1,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = True,
    *,
    out: str | Path | None = None,
) -> int:
    """Write train/val/test lists; omitted output stays at the dataset root."""

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    return mgr.dataset_split(
        train=train,
        val=val,
        test=test,
        seed=seed,
        absolute_paths=absolute_paths,
        out=out,
    )
