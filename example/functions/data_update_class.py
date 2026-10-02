"""Reusable class-remapping example."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    *,
    class_map: Mapping[str, Any] | None = None,
    compact: bool = True,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    report: str | Path | None = None,
) -> int:
    """Update classes in place from a Python mapping.

    By default, this applies the HMT class mapping in ``DEFAULT_CLASS_MAP``:

    * ``Hollow High Risk`` -> ``Hollow Confirmed``
    * ``Hollow Low Risk`` -> ``Hollow Suspected``
    * ``Leakage High Risk`` -> ``Leakage``
    * background, line, and temperature classes have their boxes removed

    The source labels and class schema are backed up in one timestamped
    snapshot before the update. Images whose labels become empty remain in the
    dataset as hard-negative samples.
    """

    mgr = get_yolo_manager(
        dataset_input, layout="auto", init_check=False, init_layout=False
    )
    return mgr.ann_update_from_map(
        class_map=dict(DEFAULT_CLASS_MAP if class_map is None else class_map),
        compact=compact,
        backup_dir=backup_dir,
        dry_run=dry_run,
        report=report,
    )


def yolo_update_from_map(
    dataset_input: YoloManagerInput,
    class_map: Mapping[str, Any],
    *,
    compact: bool = True,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
    report: str | Path | None = None,
) -> int:
    """Explicitly named alias for :func:`yolo_update_class`."""

    return yolo_update_class(
        dataset_input,
        class_map=class_map,
        compact=compact,
        backup_dir=backup_dir,
        dry_run=dry_run,
        report=report,
    )
