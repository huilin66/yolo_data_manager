"""Reusable prediction error-analysis example."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from yolo_data_manager import YoloManager


def yolo_error_ana(
    input_dir: str | Path,
    pred_dir: str | Path,
    pred_name: str | None = None,
    *,
    abs_path: bool = False,
    only_val: bool = False,
    workers: int = 8,
    conf_thres: float = 0.001,
    class_: str | list[str] | None = None,
    exclude_class_: str | list[str] | None = None,
    min_width: float | None = None,
    min_height: float | None = None,
    min_area: float | None = None,
    min_size_logic: str = "or",
    min_pixels: float | None = None,
    class_rules: Mapping[str | int, Mapping[str, Any]] | str | Path | None = None,
    out: str | Path | None = None,
    **kwargs: Any,
) -> int:
    """Run error analysis for an external prediction directory.

    With ``abs_path=False`` and a ``pred_name``, ``pred_dir`` is interpreted
    as an Ultralytics run root and ``<pred_dir>/<pred_name>/labels`` is used.
    """

    resolved_pred_dir = Path(pred_dir)
    if not abs_path:
        if not pred_name:
            raise ValueError("pred_name is required when abs_path=False")
        resolved_pred_dir = resolved_pred_dir / pred_name / "labels"

    mgr = YoloManager(input_dir, layout="auto", init_check=False, init_layout=False)
    return mgr.eval_error_analysis(
        pred_root=resolved_pred_dir,
        out=out,
        conf_thres=conf_thres,
        crop_padding=12,
        review_workers=workers,
        only_val=only_val,
        class_=class_,
        exclude_class_=exclude_class_,
        min_width=min_width,
        min_height=min_height,
        min_area=min_area,
        min_size_logic=min_size_logic,
        min_pixels=min_pixels,
        class_rules=class_rules,
        workers=workers,
        **kwargs,
    )
