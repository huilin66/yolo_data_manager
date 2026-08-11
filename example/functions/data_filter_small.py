"""Reusable dataset geometry filtering example."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._manager import YoloManagerInput, get_yolo_manager


def yolo_filter_small(
    dataset_input: YoloManagerInput,
    filter_ratio: float = 0.01,
    logic: str = "or",
    class_rules: Mapping[str, Mapping[str, Any]] | str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    class_: str | list[str] | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    min_conf: float | None = None,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Filter small boxes globally or with per-class width/height rules.

    ``class_rules`` uses normalized YOLO width/height values, for example::

        {
            "Efflorescene Low Risk": {
                "width": 0.03,
                "height": 0.03,
                "logic": "or",
            },
        }

    Classes without a rule use the global ``filter_ratio`` and ``logic``.
    The manager decides the canonical default output when ``out_dir`` is
    omitted.
    """

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    return mgr.dataset_filter(
        out=out_dir,
        class_=class_,
        min_width=None if class_rules is not None else filter_ratio,
        min_height=None if class_rules is not None else filter_ratio,
        min_size_logic=logic,
        min_area=min_area,
        max_area=max_area,
        min_conf=min_conf,
        class_rules=class_rules,
        backup_dir=backup_dir,
        dry_run=dry_run,
    )
