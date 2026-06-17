from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset


@dataclass
class DuplicateImageGroup:
    digest: str
    images: list[str]


def find_duplicate_images(dataset: YoloDataset, algorithm: str = "sha256") -> list[DuplicateImageGroup]:
    by_hash: dict[str, list[str]] = {}
    for image in dataset.images:
        if not image.path.exists():
            continue
        digest = hash_file(image.path, algorithm=algorithm)
        by_hash.setdefault(digest, []).append(image.file_name)
    return [
        DuplicateImageGroup(digest=digest, images=images)
        for digest, images in by_hash.items()
        if len(images) > 1
    ]


def hash_file(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_duplicate_image_csv(groups: list[DuplicateImageGroup], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["digest", "image"])
        writer.writeheader()
        for group in groups:
            for image in group.images:
                writer.writerow({"digest": group.digest, "image": image})
