"""Reusable prediction metrics example."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ._manager import YoloManagerInput, get_yolo_manager


def yolo_metric(
    dataset_input: YoloManagerInput,
    pred_dir: str | Path,
    pred_name: str | None = None,
    *,
    abs_path: bool = False,
    workers: int = 8,
    class_: str | list[str] | None = None,
    exclude_class_: str | list[str] | None = None,
    merge_class_map: Mapping[str | int, str | int | Sequence[str | int]]
    | str
    | Path
    | None = None,
    min_width: float | None = None,
    min_height: float | None = None,
    min_area: float | None = None,
    min_size_logic: str = "or",
    min_pixels: float | None = None,
    conf_thres: float = 0.001,
    only_val: bool = True,
    show_original: bool = False,
    out: str | Path | None = None,
    csv: str | Path | None = None,
) -> int:
    """Compute metrics for all data by default, or only the validation split."""

    resolved_pred_dir = Path(pred_dir)
    if not abs_path:
        if not pred_name:
            raise ValueError("pred_name is required when abs_path=False")
        resolved_pred_dir = resolved_pred_dir / pred_name / "labels"

    mgr = get_yolo_manager(
        dataset_input, layout="auto", init_check=False, init_layout=False
    )
    return mgr.eval_metrics(
        pred_root=resolved_pred_dir,
        class_=class_,
        exclude_class_=exclude_class_,
        merge_class_map=merge_class_map,
        min_width=min_width,
        min_height=min_height,
        min_area=min_area,
        min_size_logic=min_size_logic,
        min_pixels=min_pixels,
        conf_thres=conf_thres,
        only_val=only_val,
        show_original=show_original,
        out=out,
        csv=csv,
        print_table=True,
        workers=workers,
    )
