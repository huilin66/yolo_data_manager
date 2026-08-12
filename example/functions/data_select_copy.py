"""Reusable subset-selection example."""

from __future__ import annotations

from pathlib import Path

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


def yolo_select_val(
    dataset_input: YoloManagerInput,
    copy_images: bool = True,
    *,
    selection_file: str | Path | None = None,
    out: str | Path | None = None,
) -> int:
    """Copy the validation list, or another supplied image-list file."""

    mgr = get_yolo_manager(dataset_input, layout="flat", init_check=False, init_layout=False)
    file_path = Path(selection_file) if selection_file is not None else Path(mgr.root) / "val.txt"
    return mgr.dataset_select(
        file=str(file_path),
        out=out,
        copy_images=copy_images,
    )
