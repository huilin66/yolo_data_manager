from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import YoloDataset


@dataclass
class ImageQualityIssue:
    image: str
    path: str
    code: str
    message: str


def find_bad_images(dataset: YoloDataset) -> list[ImageQualityIssue]:
    issues: list[ImageQualityIssue] = []
    for image in dataset.images:
        if not image.path.exists():
            issues.append(ImageQualityIssue(image.file_name, str(image.path), "missing_image", "image file does not exist"))
            continue
        try:
            with Image.open(image.path) as img:
                img.verify()
        except Exception as exc:
            issues.append(ImageQualityIssue(image.file_name, str(image.path), "bad_image", str(exc)))
    return issues


def write_image_quality_csv(issues: list[ImageQualityIssue], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "path", "code", "message"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.__dict__)

