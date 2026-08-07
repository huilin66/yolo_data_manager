"""Compatibility wrapper for :mod:`example.functions.data_merge_class`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_merge_class import yolo_merge_class
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_merge_class.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_merge_class import yolo_merge_class
    from example.datasets.run_dataset import main

__all__ = ["yolo_merge_class"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="merge-class"))
