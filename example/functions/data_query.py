"""Reusable class-query example for GT or prediction labels."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_query_class(
    dataset_input: YoloManagerInput,
    class_: str | int | list[str | int],
    *,
    source: str = "gt",
    pred_root: str | Path | None = None,
    class_file: str | Path | None = None,
    out: str | Path | None = None,
    copy_images: str | Path | None = None,
    copy_labels: str | Path | None = None,
    filtered_labels: bool = False,
    only_val: bool | None = None,
) -> int:
    """List files containing a class in GT or a prediction label source.

    For prediction queries, ``pred_root`` may be a full YOLO prediction
    dataset or its ``labels`` directory.  ``class_file`` can point to the
    shared class names file when the prediction directory does not contain
    one.
    """

    mgr = get_yolo_manager(
        dataset_input, layout="flat", init_check=False, init_layout=False
    )
    return mgr.query_class(
        class_=class_,
        source=source,
        pred_root=pred_root,
        class_file=class_file,
        out=out,
        copy_images=copy_images,
        copy_labels=copy_labels,
        filtered_labels=filtered_labels,
        only_val=only_val,
    )
