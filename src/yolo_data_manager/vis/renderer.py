from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from PIL import Image, ImageDraw

from yolo_data_manager.core.geometry import normalized_points_to_pixels, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloDataset, YoloImage

COLORS = [
    (255, 42, 4),
    (235, 219, 11),
    (183, 223, 0),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (255, 0, 189),
    (255, 180, 0),
    (0, 192, 38),
]

T = TypeVar("T")


def render_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    limit: int | None = None,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    filter_no_attributes: bool = False,
    workers: int = 1,
    progress: bool = False,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images = dataset.images[:limit] if limit is not None else dataset.images
    worker_count = max(1, int(workers))

    def save_image(image: YoloImage) -> None:
        rendered = render_image(
            dataset,
            image,
            show_confidence=show_confidence,
            confidence_threshold=confidence_threshold,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attributes=show_attributes,
            filter_no_attributes=filter_no_attributes,
        )
        save_path = out_path / image.file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(save_path)

    if worker_count == 1:
        for image in _progress(images, enabled=progress, total=len(images), desc="vis draw"):
            save_image(image)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(save_image, image) for image in images]
        for future in _progress(as_completed(futures), enabled=progress, total=len(futures), desc="vis draw"):
            future.result()


def crop_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    keep_shape: bool = False,
    min_size: int = 1,
    confidence_threshold: float | None = None,
    by_attribute: bool = False,
    filter_no_attributes: bool = True,
    workers: int = 1,
    progress: bool = False,
) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, int(workers))

    def crop_image(image: YoloImage) -> int:
        return _crop_image(
            dataset,
            image,
            out_path,
            keep_shape=keep_shape,
            min_size=min_size,
            confidence_threshold=confidence_threshold,
            by_attribute=by_attribute,
            filter_no_attributes=filter_no_attributes,
        )

    if worker_count == 1:
        return sum(
            crop_image(image)
            for image in _progress(dataset.images, enabled=progress, total=len(dataset.images), desc="vis crop")
        )

    saved = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(crop_image, image) for image in dataset.images]
        for future in _progress(as_completed(futures), enabled=progress, total=len(futures), desc="vis crop"):
            saved += future.result()
    return saved


def render_image(
    dataset: YoloDataset,
    image: YoloImage,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    filter_no_attributes: bool = False,
) -> Image.Image:
    with Image.open(image.path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for annotation in image.annotations:
        if confidence_threshold is not None and annotation.confidence is not None and annotation.confidence < confidence_threshold:
            continue
        color = COLORS[annotation.class_id % len(COLORS)]
        label = dataset.class_name(annotation.class_id)
        if show_confidence and annotation.confidence is not None:
            label = f"{label} {annotation.confidence:.2f}"
        attr_lines = _attribute_lines(dataset, annotation, filter_no=filter_no_attributes) if show_attributes else []
        if annotation.polygon is not None:
            points = normalized_points_to_pixels(annotation.polygon.points, width, height)
            fill = (*color, mask_alpha) if fill_mask else None
            draw.polygon(points, fill=fill, outline=(*color, 255))
            if points:
                _draw_label(draw, points[0][0], points[0][1], "\n".join([label] + attr_lines), color)
        else:
            box = annotation.geometry_box()
            if box is None:
                continue
            xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
            draw.rectangle([xyxy.x1, xyxy.y1, xyxy.x2, xyxy.y2], outline=(*color, 255), width=2)
            _draw_label(draw, xyxy.x1, xyxy.y1, "\n".join([label] + attr_lines), color)
    return canvas


def _crop_image(
    dataset: YoloDataset,
    image: YoloImage,
    out_path: Path,
    *,
    keep_shape: bool,
    min_size: int,
    confidence_threshold: float | None,
    by_attribute: bool,
    filter_no_attributes: bool,
) -> int:
    saved = 0
    with Image.open(image.path) as source:
        canvas = source.convert("RGB")
    width, height = canvas.size
    for idx, annotation in enumerate(image.annotations):
        if confidence_threshold is not None and annotation.confidence is not None and annotation.confidence < confidence_threshold:
            continue
        box = annotation.geometry_box()
        if box is None:
            continue
        xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
        left = max(0, int(round(xyxy.x1)))
        top = max(0, int(round(xyxy.y1)))
        right = min(width, int(round(xyxy.x2)))
        bottom = min(height, int(round(xyxy.y2)))
        if right - left < min_size or bottom - top < min_size:
            continue
        crop = Image.new("RGB", canvas.size, color=(0, 0, 0)) if keep_shape else canvas.crop((left, top, right, bottom))
        if keep_shape:
            crop.paste(canvas.crop((left, top, right, bottom)), (left, top))
        class_name = dataset.class_name(annotation.class_id)
        save_dirs = [out_path / class_name]
        if by_attribute:
            for attr_name, attr_value in dataset.annotation_attributes(annotation).items():
                if filter_no_attributes and dataset.attributes is not None and dataset.attributes.is_no_value(attr_value):
                    continue
                save_dirs.append(out_path / class_name / f"{_safe_name(attr_name)}-{_safe_name(str(attr_value))}")
        for save_dir in save_dirs:
            save_dir.mkdir(parents=True, exist_ok=True)
            crop.save(save_dir / f"{image.stem}_{idx}{image.path.suffix}")
            saved += 1
    return saved


def _draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: tuple[int, int, int]) -> None:
    if not text:
        return
    x = max(0, float(x))
    y = max(0, float(y))
    text_box = draw.textbbox((x, y), text)
    pad = 2
    rect = [text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad]
    draw.rectangle(rect, fill=(*color, 220))
    draw.text((x, y), text, fill=(0, 0, 0, 255))


def _attribute_lines(dataset: YoloDataset, annotation, filter_no: bool = False) -> list[str]:
    lines: list[str] = []
    if dataset.attributes is None:
        return lines
    for name, value in dataset.annotation_attributes(annotation).items():
        if filter_no and dataset.attributes.is_no_value(value):
            continue
        lines.append(f"{name}: {value}")
    return lines


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _progress(items: Iterable[T], *, enabled: bool, total: int, desc: str) -> Iterable[T]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return _simple_progress(items, total=total, desc=desc)
    return tqdm(items, total=total, desc=desc)


def _simple_progress(items: Iterable[T], *, total: int, desc: str) -> Iterator[T]:
    step = max(1, total // 20) if total else 1
    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx == total or idx % step == 0:
            print(f"{desc}: {idx}/{total}")
        yield item
