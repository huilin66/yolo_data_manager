"""Shared input handling for the reusable example functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_data_manager import YoloManager


YoloManagerInput = str | Path | YoloManager


def get_yolo_manager(dataset_input: YoloManagerInput, **kwargs: Any) -> YoloManager:
    """Reuse an existing manager, or initialize one from a dataset path."""

    if isinstance(dataset_input, YoloManager):
        return dataset_input
    return YoloManager(dataset_input, **kwargs)
