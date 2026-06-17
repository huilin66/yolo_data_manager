from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.geometry import xyxy_to_xywhn
from yolo_data_manager.core.models import Box, ClassSchema, YoloAnnotation, YoloDataset, YoloImage, is_image_file
from yolo_data_manager.io.writer import write_yolo_dataset


def import_voc_dir(
    annotations_dir: str | Path,
    images_dir: str | Path,
    out_root: str | Path | None = None,
    class_names: list[str] | None = None,
    skip_difficult: bool = True,
) -> YoloDataset:
    ann_root = Path(annotations_dir)
    img_root = Path(images_dir)
    classes = ClassSchema(list(class_names or []))
    image_by_stem = {path.stem: path for path in img_root.rglob("*") if path.is_file() and is_image_file(path)}
    images: list[YoloImage] = []

    for xml_path in sorted(ann_root.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        file_name = _text(root, "filename") or f"{xml_path.stem}.jpg"
        image_path = image_by_stem.get(Path(file_name).stem) or img_root / file_name
        width, height = _read_size(root, image_path)
        annotations: list[YoloAnnotation] = []

        for line_no, obj in enumerate(root.findall("object"), start=1):
            difficult = _text(obj, "difficult")
            if skip_difficult and difficult is not None and int(difficult) == 1:
                continue
            name = _text(obj, "name")
            if not name:
                continue
            box_node = obj.find("bndbox")
            if box_node is None:
                continue
            x1 = float(_text(box_node, "xmin") or 0)
            y1 = float(_text(box_node, "ymin") or 0)
            x2 = float(_text(box_node, "xmax") or 0)
            y2 = float(_text(box_node, "ymax") or 0)
            cx, cy, bw, bh = xyxy_to_xywhn(x1, y1, x2, y2, width, height)
            annotations.append(
                YoloAnnotation(
                    class_id=classes.ensure(name),
                    box=Box(cx, cy, bw, bh),
                    line_no=line_no,
                )
            )

        images.append(YoloImage(path=image_path, width=width, height=height, annotations=annotations))

    dataset = YoloDataset(root=Path(out_root or ann_root.parent), images=images, classes=classes, task="detect")
    if out_root is not None:
        write_yolo_dataset(dataset, out_root, copy_images=True)
    return dataset


def _text(root: ET.Element, tag: str) -> str | None:
    node = root.find(tag)
    return node.text.strip() if node is not None and node.text is not None else None


def _read_size(root: ET.Element, image_path: Path) -> tuple[int, int]:
    size = root.find("size")
    if size is not None:
        width = _text(size, "width")
        height = _text(size, "height")
        if width and height:
            return int(float(width)), int(float(height))
    with Image.open(image_path) as image:
        return image.size
