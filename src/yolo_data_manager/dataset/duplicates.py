from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.runtime import iter_progress, normalize_workers


@dataclass
class DuplicateImageGroup:
    digest: str
    images: list[str]


def find_duplicate_images(
    dataset: YoloDataset,
    algorithm: str = "sha256",
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> list[DuplicateImageGroup]:
    by_hash: dict[str, list[str]] = {}
    for digest, image_name in _hash_images(dataset.images, algorithm=algorithm, workers=workers, progress=progress, progress_leave=progress_leave):
        if digest is not None:
            by_hash.setdefault(digest, []).append(image_name)
    return [
        DuplicateImageGroup(digest=digest, images=images)
        for digest, images in by_hash.items()
        if len(images) > 1
    ]


def _hash_images(
    images: list[YoloImage],
    *,
    algorithm: str,
    workers: int,
    progress: bool,
    progress_leave: bool,
) -> list[tuple[str | None, str]]:
    worker_count = normalize_workers(workers)
    if worker_count == 1:
        return [
            _hash_image(image, algorithm)
            for image in iter_progress(images, enabled=progress, total=len(images), desc="duplicate hash", leave=progress_leave)
        ]

    indexed: list[tuple[int, tuple[str | None, str]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(_hash_image, image, algorithm): idx
            for idx, image in enumerate(images)
        }
        for future in iter_progress(as_completed(future_to_idx), enabled=progress, total=len(future_to_idx), desc="duplicate hash", leave=progress_leave):
            indexed.append((future_to_idx[future], future.result()))
    return [result for _, result in sorted(indexed, key=lambda item: item[0])]


def _hash_image(image: YoloImage, algorithm: str) -> tuple[str | None, str]:
    if not image.path.exists():
        return None, image.file_name
    return hash_file(image.path, algorithm=algorithm), image.file_name


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
