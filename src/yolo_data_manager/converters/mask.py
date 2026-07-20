from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from yolo_data_manager.core.models import ClassSchema, Polygon, YoloAnnotation, YoloDataset, YoloImage, is_image_file
from yolo_data_manager.io.writer import write_yolo_dataset


def import_semantic_mask_dir(
    images_dir: str | Path,
    masks_dir: str | Path,
    out_root: str | Path | None = None,
    class_map: dict[Any, str] | None = None,
    background: int | str | tuple[int, int, int] = 0,
    min_area: int = 1,
    copy_images: bool = True,
) -> YoloDataset:
    """Import semantic segmentation masks as YOLO segmentation polygons.

    Single-channel masks use pixel values as class ids. RGB masks can use
    class-map keys such as ``"#ff0000"`` or ``"255,0,0"``.
    """
    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    image_paths = sorted(path for path in image_root.rglob("*") if path.is_file() and is_image_file(path))
    masks_by_stem = {path.stem: path for path in mask_root.rglob("*") if path.is_file() and is_image_file(path)}

    images: list[YoloImage] = []
    for image_path in image_paths:
        mask_path = masks_by_stem.get(image_path.stem)
        width, height = _image_size(image_path)
        annotations: list[YoloAnnotation] = []
        if mask_path is not None:
            annotations = _mask_to_annotations(
                mask_path,
                width=width,
                height=height,
                class_map=class_map,
                background=background,
                min_area=min_area,
            )
        images.append(YoloImage(path=image_path, width=width, height=height, annotations=annotations))

    classes = _classes_from_annotations(class_map, images, background)
    _remap_annotations_to_class_schema(images, classes, class_map, background)
    dataset = YoloDataset(root=Path(out_root or image_root.parent), images=images, classes=classes, task="segment")
    if out_root is not None:
        write_yolo_dataset(dataset, out_root, copy_images=copy_images)
    return dataset


def _mask_to_annotations(
    mask_path: Path,
    *,
    width: int,
    height: int,
    class_map: dict[Any, str] | None,
    background: int | str | tuple[int, int, int],
    min_area: int,
) -> list[YoloAnnotation]:
    with Image.open(mask_path) as mask:
        array = np.array(mask)

    if array.ndim == 3 and array.shape[2] >= 3:
        array = array[:, :, :3]
        values = _rgb_values(array, class_map, background)
    else:
        values = _scalar_values(array, class_map, background)

    annotations: list[YoloAnnotation] = []
    for raw_value in values:
        class_mask = _value_mask(array, raw_value)
        for component in _connected_components(class_mask, min_area=min_area):
            polygon = _component_polygon(component, width=width, height=height)
            if polygon is None:
                continue
            annotations.append(YoloAnnotation(class_id=_raw_value_key(raw_value), polygon=polygon))
    return annotations


def _scalar_values(array: np.ndarray, class_map: dict[Any, str] | None, background: Any) -> list[int]:
    bg = int(background)
    if class_map:
        return [int(value) for value in class_map.keys() if int(value) != bg]
    return [int(value) for value in sorted(np.unique(array).tolist()) if int(value) != bg]


def _rgb_values(
    array: np.ndarray,
    class_map: dict[Any, str] | None,
    background: int | str | tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    bg = _parse_color(background)
    if class_map:
        return [_parse_color(value) for value in class_map.keys() if _parse_color(value) != bg]
    flat = array.reshape(-1, 3)
    values = sorted({tuple(int(v) for v in row) for row in flat})
    return [value for value in values if value != bg]


def _value_mask(array: np.ndarray, raw_value: int | tuple[int, int, int]) -> np.ndarray:
    if isinstance(raw_value, tuple):
        return np.all(array[:, :, :3] == np.array(raw_value, dtype=array.dtype), axis=2)
    return array == raw_value


def _connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= min_area:
            components.append(np.array(pixels, dtype=np.int32))
    return components


def _component_polygon(component: np.ndarray, *, width: int, height: int) -> Polygon | None:
    points = _component_contour_with_cv2(component)
    if points is None:
        points = _component_bbox_points(component)
    if len(points) < 3:
        return None
    normalised = [
        (
            min(1.0, max(0.0, float(x) / max(1, width))),
            min(1.0, max(0.0, float(y) / max(1, height))),
        )
        for x, y in points
    ]
    return Polygon(normalised)


def _component_contour_with_cv2(component: np.ndarray) -> list[tuple[float, float]] | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    ys = component[:, 0]
    xs = component[:, 1]
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    crop = np.zeros((y2 - y1 + 3, x2 - x1 + 3), dtype=np.uint8)
    crop[ys - y1 + 1, xs - x1 + 1] = 255
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        return None
    return [(float(x + x1 - 1), float(y + y1 - 1)) for x, y in points]


def _component_bbox_points(component: np.ndarray) -> list[tuple[float, float]]:
    ys = component[:, 0]
    xs = component[:, 1]
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _classes_from_annotations(
    class_map: dict[Any, str] | None,
    images: list[YoloImage],
    background: Any,
) -> ClassSchema:
    if class_map:
        return ClassSchema(
            list(
                OrderedDict(
                    (str(name), None)
                    for raw_value, name in class_map.items()
                    if not _same_raw_value(raw_value, background)
                ).keys()
            )
        )
    raw_values = sorted({annotation.class_id for image in images for annotation in image.annotations}, key=str)
    return ClassSchema([str(value) for value in raw_values])


def _remap_annotations_to_class_schema(
    images: list[YoloImage],
    classes: ClassSchema,
    class_map: dict[Any, str] | None,
    background: Any,
) -> None:
    value_to_class_id: dict[Any, int] = {}
    if class_map:
        for raw_value, name in class_map.items():
            if _same_raw_value(raw_value, background):
                continue
            value_to_class_id[_raw_value_key(_parse_raw_value(raw_value))] = classes.ensure(str(name))
    else:
        for idx, name in enumerate(classes.names):
            value_to_class_id[name] = idx
            try:
                value_to_class_id[int(name)] = idx
            except ValueError:
                pass
    for image in images:
        for annotation in image.annotations:
            annotation.class_id = value_to_class_id.get(annotation.class_id, classes.ensure(str(annotation.class_id)))


def _raw_value_key(value: int | tuple[int, int, int]) -> int | str:
    if isinstance(value, tuple):
        return ",".join(str(part) for part in value)
    return int(value)


def _parse_raw_value(value: Any) -> int | tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    text = str(value).strip()
    if "," in text or text.startswith("#"):
        return _parse_color(text)
    return int(text)


def _same_raw_value(left: Any, right: Any) -> bool:
    try:
        return _parse_raw_value(left) == _parse_raw_value(right)
    except ValueError:
        return str(left) == str(right)


def _parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return tuple(int(part) for part in value[:3])
    if isinstance(value, int):
        return (value, value, value)
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))
    parts = [part.strip() for part in text.strip("()[]").split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB mask class value must be r,g,b or #rrggbb: {value}")
    return tuple(int(part) for part in parts)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
