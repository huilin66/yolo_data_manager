from __future__ import annotations

import shutil
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset
from yolo_data_manager.core.schema import write_attribute_schema, write_class_schema, write_dataset_yaml


def write_yolo_dataset(
    dataset: YoloDataset,
    out_root: str | Path,
    copy_images: bool = True,
    keep_empty_labels: bool = True,
    include_confidence: bool = False,
    overwrite_images: bool = True,
) -> None:
    out_path = Path(out_root)
    image_dir = out_path / "images"
    label_dir = out_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    write_class_schema(dataset.classes, out_path / "class.txt")
    write_dataset_yaml(dataset.classes, out_path / "dataset.yaml", train="images", val="images")
    write_attribute_schema(dataset.attributes, out_path / "attribute.yaml")

    for image in dataset.images:
        dst_image = image_dir / image.file_name
        if copy_images and image.path.exists() and (overwrite_images or not dst_image.exists()):
            shutil.copy2(image.path, dst_image)

        if not image.annotations and not keep_empty_labels:
            continue
        dst_label = label_dir / f"{image.stem}.txt"
        lines = [ann.to_yolo_line(include_confidence=include_confidence) for ann in image.annotations]
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_split_file(image_names: list[str], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(image_names) + ("\n" if image_names else ""), encoding="utf-8")
