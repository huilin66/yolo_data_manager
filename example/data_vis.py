"""Compatibility wrapper for :mod:`example.functions.data_vis`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_vis import yolo_vis
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_vis.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_vis import yolo_vis
    from example.datasets.run_dataset import main

__all__ = ["yolo_vis"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="vis"))
