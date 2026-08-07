"""Compatibility wrapper for :mod:`example.functions.data_select_copy`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_select_copy import yolo_select_val
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_select_copy.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_select_copy import yolo_select_val
    from example.datasets.run_dataset import main

__all__ = ["yolo_select_val"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="select-val"))
