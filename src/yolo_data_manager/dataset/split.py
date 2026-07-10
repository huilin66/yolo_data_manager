from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.models import YoloDataset


def split_dataset(
    dataset: YoloDataset,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = False,
) -> dict[str, list[str]]:
    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = {"train": train / total, "val": val / total, "test": test / total}
    names = [
        str(image.path.resolve()) if absolute_paths else image.file_name
        for image in dataset.images
    ]
    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }


def class_counts_for_images(
    dataset: YoloDataset,
    image_names: Iterable[str] | None = None,
) -> dict[str, int]:
    selected = _image_key_set(image_names) if image_names is not None else None
    counts = {name: 0 for name in dataset.classes.names}

    for image in dataset.images:
        if selected is not None and not (_image_keys(image) & selected):
            continue
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def _image_key_set(values: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        text = str(value)
        path = Path(text)
        keys.update({text, path.name, path.stem})
    return keys


def _image_keys(image) -> set[str]:
    return {
        image.file_name,
        image.stem,
        str(image.path),
        str(image.path.resolve()),
        image.path.name,
        image.path.stem,
    }
