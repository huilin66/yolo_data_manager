"""Compatibility wrapper for :mod:`example.functions.result_error_ana`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.result_error_ana import yolo_error_ana
    from .datasets.run_dataset import main
except ImportError:  # supports ``python example/result_error_ana.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.result_error_ana import yolo_error_ana
    from example.datasets.run_dataset import main

__all__ = ["yolo_error_ana"]


if __name__ == "__main__":
    raise SystemExit(main(default_task="error-analysis"))
