"""Compatibility wrapper for :mod:`example.functions.data_object_label_update`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_object_label_update import yolo_update
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_object_label_update.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_object_label_update import yolo_update
    from example.datasets.run_dataset import main

__all__ = ["yolo_update"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="correct-crops"))
