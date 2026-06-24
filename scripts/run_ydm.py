"""Editable Python entry point for YOLO Data Manager.

Change TASK and PARAMS, then run:
    python scripts/run_ydm.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yolo_data_manager.scripting import run_task  # noqa: E402


# Select one task from docs/PYTHON_USAGE.md and edit its parameters here.
TASK = "stats"
PARAMS = {
    "root": Path(r"E:\datasets\my_yolo"),
    "layout": "auto",
    "out": Path(r"E:\datasets\reports\stats.json"),
    "class_csv": Path(r"E:\datasets\reports\class_counts.csv"),
    "attr_csv": Path(r"E:\datasets\reports\attributes.csv"),
}


def main() -> int:
    print(f"Running YOLO Data Manager task: {TASK}")
    return run_task(TASK, **PARAMS)


if __name__ == "__main__":
    raise SystemExit(main())
