"""Compatibility wrapper for :mod:`example.functions.data_manual_draw_box`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.data_manual_draw_box import yolo_draw
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/data_manual_draw_box.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.data_manual_draw_box import yolo_draw
    from example.datasets.run_dataset import main

__all__ = ["yolo_draw"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="manual-box"))
