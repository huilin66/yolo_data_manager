"""Compatibility wrapper for :mod:`example.functions.data_sta`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_sta import yolo_sta
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_sta.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_sta import yolo_sta
    from example.datasets.run_dataset import main

__all__ = ["yolo_sta"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="stats"))
