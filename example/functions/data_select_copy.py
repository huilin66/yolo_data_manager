"""Reusable subset-selection example."""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager import YoloManager


def yolo_select_val(
    input_dir: str | Path,
    copy_images: bool = True,
    *,
    selection_file: str | Path | None = None,
    out: str | Path | None = None,
) -> int:
    """Copy the validation list, or another supplied image-list file."""

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    file_path = Path(selection_file) if selection_file is not None else Path(mgr.root) / "val.txt"
    return mgr.dataset_select(
        file=str(file_path),
        out=out,
        copy_images=copy_images,
    )

