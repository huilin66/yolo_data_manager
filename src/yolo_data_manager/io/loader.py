from __future__ import annotations

from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import (
    TASK_AUTO,
    TASK_DETECT,
    TASK_SEGMENT,
    AttributeSchema,
    Box,
    ClassSchema,
    Polygon,
    YoloAnnotation,
    YoloDataset,
    YoloImage,
    is_image_file,
)
from yolo_data_manager.core.schema import (
    find_attribute_file,
    read_attribute_schema,
    read_dataset_class_schema,
)
from yolo_data_manager.io.layout import infer_label_path_from_image, read_image_list, resolve_layout


def load_yolo_dataset(
    root: str | Path,
    images_dir: str | Path = "images",
    labels_dir: str | Path = "labels",
    class_file: str | Path | None = None,
    attribute_file: str | Path | None = None,
    task: str = TASK_AUTO,
    split_file: str | Path | None = None,
    layout: str = "flat",
    read_image_size: bool = True,
) -> YoloDataset:
    root_path = Path(root)
    layout_info = resolve_layout(root_path, layout=layout, images_dir=images_dir, labels_dir=labels_dir)
    image_root = layout_info.images_dir or _resolve_under(root_path, images_dir)
    label_root = layout_info.labels_dir or _resolve_under(root_path, labels_dir)

    classes = read_dataset_class_schema(root_path) if class_file is None else read_dataset_class_schema(Path(class_file).parent)
    if class_file is not None:
        from yolo_data_manager.core.schema import read_class_schema

        classes = read_class_schema(class_file)

    attr_path = Path(attribute_file) if attribute_file is not None else find_attribute_file(root_path)
    attributes = read_attribute_schema(attr_path)

    if layout_info.layout == "image_list":
        source_lists = [Path(split_file)] if split_file is not None else layout_info.split_files
        image_paths = read_image_list(source_lists, root_path)
    else:
        image_paths = sorted([path for path in image_root.rglob("*") if path.is_file() and is_image_file(path)])
    if split_file is not None:
        allowed_stems = _read_split_stems(split_file)
        image_paths = [path for path in image_paths if path.stem in allowed_stems or path.name in allowed_stems]

    label_paths = sorted([path for path in label_root.rglob("*.txt")]) if label_root.exists() else []
    labels_by_stem = {path.stem: path for path in label_paths}

    images: list[YoloImage] = []
    for image_path in image_paths:
        label_path = infer_label_path_from_image(image_path) if layout_info.layout == "image_list" else labels_by_stem.get(image_path.stem)
        if label_path is not None and not label_path.exists():
            label_path = None
        width, height = _read_image_size(image_path) if read_image_size else (None, None)
        annotations = parse_label_file(label_path, task=task, attributes=attributes) if label_path else []
        images.append(
            YoloImage(
                path=image_path,
                label_path=label_path,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    image_stems = {path.stem for path in image_paths}
    orphan_labels = [path for path in label_paths if path.stem not in image_stems]
    return YoloDataset(
        root=root_path,
        images=images,
        classes=classes,
        attributes=attributes,
        task=task,
        orphan_labels=orphan_labels,
    )


def parse_label_file(
    label_path: str | Path | None,
    task: str = TASK_AUTO,
    attributes: AttributeSchema | None = None,
) -> list[YoloAnnotation]:
    if label_path is None:
        return []
    path = Path(label_path)
    if not path.exists():
        return []
    annotations: list[YoloAnnotation] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        annotations.append(parse_yolo_line(stripped, line_no=line_no, task=task, attributes=attributes))
    return annotations


def parse_yolo_line(
    line: str,
    line_no: int | None = None,
    task: str = TASK_AUTO,
    attributes: AttributeSchema | None = None,
) -> YoloAnnotation:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"line {line_no}: expected at least 5 values, got {len(parts)}")

    class_id = int(float(parts[0]))
    cursor = 1
    attr_values: list[float] = []
    if attributes is not None and len(parts) > 2 and _looks_int(parts[1]):
        attr_len = int(float(parts[1]))
        if attr_len >= 0 and len(parts) >= 2 + attr_len + 4:
            cursor = 2
            attr_values = [float(x) for x in parts[cursor : cursor + attr_len]]
            cursor += attr_len

    geometry = [float(x) for x in parts[cursor:]]
    if not geometry:
        raise ValueError(f"line {line_no}: missing geometry values")

    confidence: float | None = None
    if task == TASK_DETECT:
        if len(geometry) == 5:
            confidence = geometry[-1]
            geometry = geometry[:4]
        if len(geometry) != 4:
            raise ValueError(f"line {line_no}: detection labels need 4 geometry values")
        annotation = YoloAnnotation(class_id=class_id, box=Box(*geometry), attributes=attr_values)
    elif task == TASK_SEGMENT:
        if len(geometry) % 2 == 1:
            confidence = geometry[-1]
            geometry = geometry[:-1]
        if len(geometry) < 6 or len(geometry) % 2 != 0:
            raise ValueError(f"line {line_no}: segmentation labels need even polygon values")
        annotation = YoloAnnotation(
            class_id=class_id,
            polygon=Polygon(_pairs(geometry)),
            attributes=attr_values,
        )
    else:
        annotation = _parse_auto_geometry(class_id, geometry, attr_values, line_no)
        confidence = annotation.confidence

    annotation.confidence = confidence
    annotation.line_no = line_no
    annotation.source_line = line
    return annotation


def _parse_auto_geometry(
    class_id: int,
    geometry: list[float],
    attr_values: list[float],
    line_no: int | None,
) -> YoloAnnotation:
    confidence: float | None = None
    if len(geometry) in (4, 5):
        if len(geometry) == 5:
            confidence = geometry[-1]
            geometry = geometry[:4]
        return YoloAnnotation(
            class_id=class_id,
            box=Box(*geometry),
            attributes=attr_values,
            confidence=confidence,
            line_no=line_no,
        )
    if len(geometry) >= 6:
        if len(geometry) % 2 == 1:
            confidence = geometry[-1]
            geometry = geometry[:-1]
        if len(geometry) % 2 == 0:
            return YoloAnnotation(
                class_id=class_id,
                polygon=Polygon(_pairs(geometry)),
                attributes=attr_values,
                confidence=confidence,
                line_no=line_no,
            )
    raise ValueError(f"line {line_no}: cannot infer YOLO geometry")


def _pairs(values: list[float]) -> list[tuple[float, float]]:
    return [(values[idx], values[idx + 1]) for idx in range(0, len(values), 2)]


def _looks_int(value: str) -> bool:
    try:
        return float(value).is_integer()
    except ValueError:
        return False


def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def _resolve_under(root: Path, child: str | Path) -> Path:
    child_path = Path(child)
    return child_path if child_path.is_absolute() else root / child_path


def _read_split_stems(path: str | Path) -> set[str]:
    values: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        values.add(text)
        values.add(Path(text).stem)
        values.add(Path(text).name)
    return values
