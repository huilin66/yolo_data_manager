from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.runtime import iter_progress, normalize_workers


@dataclass
class ImageQualityIssue:
    image: str
    path: str
    code: str
    message: str


def find_bad_images(
    dataset: YoloDataset,
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> list[ImageQualityIssue]:
    worker_count = normalize_workers(workers)
    if worker_count == 1:
        return [
            issue
            for image in iter_progress(dataset.images, enabled=progress, total=len(dataset.images), desc="bad images", leave=progress_leave)
            for issue in [_check_image(image)]
            if issue is not None
        ]

    indexed: list[tuple[int, ImageQualityIssue | None]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(_check_image, image): idx
            for idx, image in enumerate(dataset.images)
        }
        for future in iter_progress(as_completed(future_to_idx), enabled=progress, total=len(future_to_idx), desc="bad images", leave=progress_leave):
            indexed.append((future_to_idx[future], future.result()))
    return [issue for _, issue in sorted(indexed, key=lambda item: item[0]) if issue is not None]


def _check_image(image: YoloImage) -> ImageQualityIssue | None:
    if not image.path.exists():
        return ImageQualityIssue(image.file_name, str(image.path), "missing_image", "image file does not exist")
    try:
        with Image.open(image.path) as img:
            img.verify()
    except Exception as exc:
        return ImageQualityIssue(image.file_name, str(image.path), "bad_image", str(exc))
    return None


def write_image_quality_csv(issues: list[ImageQualityIssue], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "path", "code", "message"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.__dict__)
