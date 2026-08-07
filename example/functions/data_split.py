"""Reusable dataset split example."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_split(
    input_dir: str | Path,
    train: float = 0.9,
    val: float = 0.1,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = True,
    *,
    out: str | Path | None = None,
) -> int:
    """Write train/val/test lists; omitted output stays at the dataset root."""

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    return mgr.dataset_split(
        train=train,
        val=val,
        test=test,
        seed=seed,
        absolute_paths=absolute_paths,
        out=out,
    )

