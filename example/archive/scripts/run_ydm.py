"""Archived editable ``run_task`` entry point.

New dataset work should use ``example.datasets.run_dataset`` so paths are
provided at runtime instead of edited in this file.
"""

from __future__ import annotations

from pathlib import Path

from yolo_data_manager.scripting import run_task


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

