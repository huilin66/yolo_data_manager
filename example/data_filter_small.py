"""Compatibility wrapper for :mod:`example.functions.data_filter_small`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_filter_small import yolo_filter_small
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_filter_small.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_filter_small import yolo_filter_small
    from example.datasets.run_dataset import main

__all__ = ["yolo_filter_small"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="filter-small"))
