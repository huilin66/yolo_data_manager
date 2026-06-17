from __future__ import annotations

from pathlib import Path

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


def render_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    limit: int | None = None,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images = dataset.images[:limit] if limit is not None else dataset.images
    for image in images:
        rendered = render_image(
            dataset,
            image,
            show_confidence=show_confidence,
            confidence_threshold=confidence_threshold,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
        )
        rendered.save(out_path / image.file_name)


def crop_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    keep_shape: bool = False,
    min_size: int = 1,
    confidence_threshold: float | None = None,
) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved = 0
    for image in dataset.images:
        canvas = Image.open(image.path).convert("RGB")
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
            class_dir = out_path / dataset.class_name(annotation.class_id)
            class_dir.mkdir(parents=True, exist_ok=True)
            crop.save(class_dir / f"{image.stem}_{idx}{image.path.suffix}")
            saved += 1
    return saved


def render_image(
    dataset: YoloDataset,
    image: YoloImage,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
) -> Image.Image:
    canvas = Image.open(image.path).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for annotation in image.annotations:
        if confidence_threshold is not None and annotation.confidence is not None and annotation.confidence < confidence_threshold:
            continue
        color = COLORS[annotation.class_id % len(COLORS)]
        label = dataset.class_name(annotation.class_id)
        if show_confidence and annotation.confidence is not None:
            label = f"{label} {annotation.confidence:.2f}"
        if annotation.polygon is not None:
            points = normalized_points_to_pixels(annotation.polygon.points, width, height)
            fill = (*color, mask_alpha) if fill_mask else None
            draw.polygon(points, fill=fill, outline=(*color, 255))
            if points:
                _draw_label(draw, points[0][0], points[0][1], label, color)
        else:
            box = annotation.geometry_box()
            if box is None:
                continue
            xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
            draw.rectangle([xyxy.x1, xyxy.y1, xyxy.x2, xyxy.y2], outline=(*color, 255), width=2)
            _draw_label(draw, xyxy.x1, xyxy.y1, label, color)
    return canvas


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
