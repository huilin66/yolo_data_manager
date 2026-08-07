"""Reusable statistics example."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_sta(
    input_dir: str | Path,
    *,
    stats_list: str | list[str] | None = "all",
    only_val: bool = False,
    out: str | Path | None = None,
    class_csv: str | Path | None = None,
    ann_csv: str | Path | None = None,
    attr_csv: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> int:
    """Compute statistics for all input data unless ``only_val`` is enabled."""

    mgr = YoloManager(input_dir, layout="flat", init_check=False)
    return mgr.stats(
        stats_list=stats_list,
        only_val=only_val,
        out=out,
        class_csv=class_csv,
        ann_csv=ann_csv,
        attr_csv=attr_csv,
        plots_dir=plots_dir,
    )

