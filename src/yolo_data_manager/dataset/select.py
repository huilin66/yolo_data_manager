from __future__ import annotations

import copy
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset


def select_by_stems(dataset: YoloDataset, stems: set[str]) -> YoloDataset:
    result = copy.deepcopy(dataset)
    result.images = [image for image in result.images if image.stem in stems or image.file_name in stems]
    return result


def read_selection_file(path: str | Path) -> set[str]:
    selected: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        selected.add(text)
        selected.add(Path(text).name)
        selected.add(Path(text).stem)
    return selected


def select_from_file(dataset: YoloDataset, path: str | Path) -> YoloDataset:
    return select_by_stems(dataset, read_selection_file(path))

