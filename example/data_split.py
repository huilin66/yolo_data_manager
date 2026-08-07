"""Compatibility wrapper for :mod:`example.functions.data_split`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_split import yolo_split
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_split.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_split import yolo_split
    from example.datasets.run_dataset import main

__all__ = ["yolo_split"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="split"))
