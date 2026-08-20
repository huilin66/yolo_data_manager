"""Reusable class-remapping example."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from ._manager import YoloManagerInput, get_yolo_manager
except ImportError:  # Support direct execution of this module.
    from _manager import YoloManagerInput, get_yolo_manager


# The insertion order of these mappings is intentional.  With compact=True,
# it produces the final class order:
# 0 Hollow Confirmed, 1 Hollow Suspected, 2 Leakage.
DEFAULT_CLASS_MAP: dict[str, Any] = {
    "merge": {
        "Hollow Confirmed": ["Hollow High Risk"],
        "Hollow Suspected": ["Hollow Low Risk"],
        "Leakage": ["Leakage High Risk"],
    },
    "drop": [
        "background",
        "Hollow High Risk Line",
        "Temperature Medium Risk",
        "Temperature High Risk",
    ],
}


def yolo_update_class(
    dataset_input: YoloManagerInput,
    output_dir: str | Path | None = None,
    *,
    map_file: str | Path | None = None,
    class_map: Mapping[str, Any] | None = None,
    compact: bool = True,
    copy_images: bool = True,
    keep_empty_labels: bool = True,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Remap dataset classes and retain images with empty labels.

    By default, this applies the HMT class mapping in ``DEFAULT_CLASS_MAP``:

    * ``Hollow High Risk`` -> ``Hollow Confirmed``
    * ``Hollow Low Risk`` -> ``Hollow Suspected``
    * ``Leakage High Risk`` -> ``Leakage``
    * background, line, and temperature classes have their boxes removed

    Images whose annotations become empty are retained by default, making
    them suitable as hard-negative samples.  Pass ``map_file`` to use a
    custom apply-map YAML, or pass ``class_map`` to provide the YAML content
    directly.
    """

    if map_file is not None and class_map is not None:
        raise ValueError("map_file and class_map cannot be used together")

    temporary_map: Path | None = None
    if map_file is None:
        fd, temporary_name = tempfile.mkstemp(
            prefix="ydm_class_map_",
            suffix=".yaml",
        )
        os.close(fd)
        temporary_map = Path(temporary_name)
        temporary_map.write_text(
            yaml.safe_dump(
                dict(DEFAULT_CLASS_MAP if class_map is None else class_map),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        resolved_map_file = temporary_map
    else:
        resolved_map_file = Path(map_file)

    try:
        mgr = get_yolo_manager(
            dataset_input,
            layout="flat",
            init_check=False,
            init_layout=False,
        )
        return mgr.ann_apply_map(
            map_file=str(resolved_map_file),
            out=output_dir,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )
    finally:
        if temporary_map is not None:
            temporary_map.unlink(missing_ok=True)
