from __future__ import annotations

import random

from yolo_data_manager.core.models import YoloDataset


def split_dataset(
    dataset: YoloDataset,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 233,
) -> dict[str, list[str]]:
    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = {"train": train / total, "val": val / total, "test": test / total}
    names = [image.file_name for image in dataset.images]
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

