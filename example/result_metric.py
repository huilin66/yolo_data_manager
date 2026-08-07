"""Compatibility wrapper for :mod:`example.functions.result_metric`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.result_metric import yolo_metric
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/result_metric.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.result_metric import yolo_metric
    from example.datasets.run_dataset import main

__all__ = ["yolo_metric"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="metric"))
