from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in environments without OpenCV
    cv2 = None

from yolo_data_manager.core.geometry import normalized_points_to_pixels, xywhn_to_xyxy
from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.runtime import iter_progress, normalize_workers

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

# Keep the PIL palette stable for existing users.  The OpenCV palette follows
# the palette used by the project's previous cv2 visualizer.
CV2_COLORS = [
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (255, 255, 0),
    (0, 192, 38),
    (179, 255, 1),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (11, 255, 162),
]

_VISUAL_STYLE_ALIASES = {
    "pil": "pil",
    "cv": "cv2",
    "cv2": "cv2",
}


def normalize_visual_style(style: str) -> str:
    """Return the canonical visualization style name.

    ``cv`` is accepted as a short alias for ``cv2`` so existing example code
    can use the shorter spelling while the documented choices remain ``pil``
    and ``cv2``.
    """

    normalized = str(style).strip().lower()
    try:
        return _VISUAL_STYLE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError("style must be one of: pil, cv2") from exc


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "style='cv2' requires OpenCV; install the package dependency "
            "'opencv-python-headless'"
        )


def render_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    limit: int | None = None,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    show_txt_id: bool = False,
    filter_no_attributes: bool = False,
    clean: bool = True,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    style: str = "cv2",
) -> None:
    visual_style = normalize_visual_style(style)
    if visual_style == "cv2":
        _require_cv2()
    out_path = Path(out_dir)
    _prepare_vis_output_dir(dataset, out_path, clean=clean)
    images = dataset.images[:limit] if limit is not None else dataset.images
    worker_count = normalize_workers(workers)

    def save_image(image: YoloImage) -> None:
        save_path = out_path / image.file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if visual_style == "pil":
            rendered = render_image(
                dataset,
                image,
                show_confidence=show_confidence,
                confidence_threshold=confidence_threshold,
                mask_alpha=mask_alpha,
                fill_mask=fill_mask,
                show_attributes=show_attributes,
                show_txt_id=show_txt_id,
                filter_no_attributes=filter_no_attributes,
            )
            rendered.save(save_path)
        else:
            rendered = render_image_cv2(
                dataset,
                image,
                show_confidence=show_confidence,
                confidence_threshold=confidence_threshold,
                mask_alpha=mask_alpha,
                fill_mask=fill_mask,
                show_attributes=show_attributes,
                show_txt_id=show_txt_id,
                filter_no_attributes=filter_no_attributes,
            )
            _write_cv2_image(save_path, rendered)

    if worker_count == 1:
        for image in iter_progress(images, enabled=progress, total=len(images), desc="vis draw", leave=progress_leave):
            save_image(image)
        return

    executor = ThreadPoolExecutor(max_workers=worker_count)
    progress_items = None
    try:
        futures = [executor.submit(save_image, image) for image in images]
        progress_items = iter_progress(
            as_completed(futures),
            enabled=progress,
            total=len(futures),
            desc="vis draw",
            leave=progress_leave,
        )
        for future in progress_items:
            future.result()
    except KeyboardInterrupt:
        _close_progress(progress_items)
        _cancel_parallel_work(executor, operation="vis draw")
        raise
    except BaseException:
        _close_progress(progress_items)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _prepare_vis_output_dir(
    dataset: YoloDataset, out_path: Path, *, clean: bool
) -> None:
    """Create ``out_path``, optionally clearing any existing contents first.

    With ``clean=True`` (the default), stale files from previous runs are
    removed before rendering.  For safety the output is never allowed to be,
    or contain, the dataset root or a source image/label directory.
    """
    if not clean:
        out_path.mkdir(parents=True, exist_ok=True)
        return
    resolved = out_path.resolve()
    root = dataset.root.resolve()
    if root.is_relative_to(resolved):
        raise ValueError(
            f"refusing to clear {out_path!s}: it is the dataset root or one of "
            "its parent directories"
        )
    source_dirs = {
        path.parent.resolve()
        for image in dataset.images
        for path in (image.path, image.label_path)
        if path is not None
    }
    if resolved in source_dirs:
        raise ValueError(
            f"refusing to clear {out_path!s}: it contains source images or labels"
        )
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def crop_dataset(
    dataset: YoloDataset,
    out_dir: str | Path,
    keep_shape: bool = False,
    min_size: int = 1,
    padding: int | float = 0,
    confidence_threshold: float | None = None,
    by_attribute: bool = False,
    filter_no_attributes: bool = True,
    clean: bool = True,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    style: str = "cv2",
) -> int:
    visual_style = normalize_visual_style(style)
    if visual_style == "cv2":
        _require_cv2()
    _validate_crop_padding(padding)
    out_path = Path(out_dir)
    _prepare_vis_output_dir(dataset, out_path, clean=clean)
    worker_count = normalize_workers(workers)

    def crop_image(image: YoloImage) -> int:
        crop_kwargs = dict(
            keep_shape=keep_shape,
            min_size=min_size,
            padding=padding,
            confidence_threshold=confidence_threshold,
            by_attribute=by_attribute,
            filter_no_attributes=filter_no_attributes,
        )
        if visual_style == "pil":
            return _crop_image(dataset, image, out_path, **crop_kwargs)
        return _crop_image_cv2(dataset, image, out_path, **crop_kwargs)

    if worker_count == 1:
        return sum(
            crop_image(image)
            for image in iter_progress(dataset.images, enabled=progress, total=len(dataset.images), desc="vis crop", leave=progress_leave)
        )

    saved = 0
    executor = ThreadPoolExecutor(max_workers=worker_count)
    progress_items = None
    try:
        futures = [executor.submit(crop_image, image) for image in dataset.images]
        progress_items = iter_progress(
            as_completed(futures),
            enabled=progress,
            total=len(futures),
            desc="vis crop",
            leave=progress_leave,
        )
        for future in progress_items:
            saved += future.result()
    except KeyboardInterrupt:
        _close_progress(progress_items)
        _cancel_parallel_work(executor, operation="vis crop")
        raise
    except BaseException:
        _close_progress(progress_items)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return saved


def _close_progress(progress_items: object | None) -> None:
    close = getattr(progress_items, "close", None)
    if callable(close):
        close()


def _cancel_parallel_work(executor: ThreadPoolExecutor, *, operation: str) -> None:
    """Stop queued visualization jobs after Ctrl+C without waiting for all work."""

    executor.shutdown(wait=False, cancel_futures=True)
    print(
        f"\n{operation} cancelled; pending work was stopped. "
        "Completed files remain in the output directory.",
        file=sys.stderr,
        flush=True,
    )


def render_image(
    dataset: YoloDataset,
    image: YoloImage,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    show_txt_id: bool = False,
    filter_no_attributes: bool = False,
) -> Image.Image:
    with Image.open(image.path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for annotation_idx, annotation in enumerate(image.annotations):
        if confidence_threshold is not None and annotation.confidence is not None and annotation.confidence < confidence_threshold:
            continue
        color = COLORS[annotation.class_id % len(COLORS)]
        label = _annotation_label(dataset, annotation, show_txt_id=show_txt_id, annotation_idx=annotation_idx)
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


def render_image_cv2(
    dataset: YoloDataset,
    image: YoloImage,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    show_txt_id: bool = False,
    filter_no_attributes: bool = False,
) -> np.ndarray:
    """Render one image with OpenCV drawing primitives.

    The returned array is in OpenCV's BGR format and is intended to be saved
    with :func:`_write_cv2_image` or further processed by OpenCV callers.
    """

    _require_cv2()
    canvas = _read_cv2_image(image.path)
    height, width = canvas.shape[:2]
    line_width = _cv2_line_width(canvas)
    alpha = max(0, min(255, int(mask_alpha))) / 255.0

    for annotation_idx, annotation in enumerate(image.annotations):
        if (
            confidence_threshold is not None
            and annotation.confidence is not None
            and annotation.confidence < confidence_threshold
        ):
            continue

        color = CV2_COLORS[annotation.class_id % len(CV2_COLORS)]
        label = _annotation_label(
            dataset,
            annotation,
            show_txt_id=show_txt_id,
            annotation_idx=annotation_idx,
        )
        if show_confidence and annotation.confidence is not None:
            label = f"{label} {annotation.confidence:.2f}"

        if annotation.polygon is not None:
            points = normalized_points_to_pixels(annotation.polygon.points, width, height)
            if len(points) < 2:
                continue
            polygon = np.asarray(
                [(int(round(x)), int(round(y))) for x, y in points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            if fill_mask and alpha > 0:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [polygon], color)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
            cv2.polylines(
                canvas,
                [polygon],
                isClosed=True,
                color=color,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
            anchor_x, anchor_y = points[0]
        else:
            box = annotation.geometry_box()
            if box is None:
                continue
            xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
            x1, x2 = sorted((int(round(xyxy.x1)), int(round(xyxy.x2))))
            y1, y2 = sorted((int(round(xyxy.y1)), int(round(xyxy.y2))))
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
            anchor_x, anchor_y = xyxy.x1, xyxy.y1

        label_info = _draw_cv2_label(
            canvas,
            anchor_x,
            anchor_y,
            label,
            color,
            line_width,
        )
        if show_attributes:
            _draw_cv2_attributes(
                canvas,
                _attribute_values(dataset, annotation, filter_no=filter_no_attributes),
                label_info,
            )
    return canvas


def _crop_image(
    dataset: YoloDataset,
    image: YoloImage,
    out_path: Path,
    *,
    keep_shape: bool,
    min_size: int,
    padding: int | float,
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
        box_left = max(0, int(round(xyxy.x1)))
        box_top = max(0, int(round(xyxy.y1)))
        box_right = min(width, int(round(xyxy.x2)))
        box_bottom = min(height, int(round(xyxy.y2)))
        if box_right - box_left < min_size or box_bottom - box_top < min_size:
            continue
        padding_x, padding_y = _crop_padding_pixels(
            padding,
            xyxy.x2 - xyxy.x1,
            xyxy.y2 - xyxy.y1,
        )
        left = max(0, int(round(xyxy.x1 - padding_x)))
        top = max(0, int(round(xyxy.y1 - padding_y)))
        right = min(width, int(round(xyxy.x2 + padding_x)))
        bottom = min(height, int(round(xyxy.y2 + padding_y)))
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
            crop.save(save_dir / f"{image.stem}_{idx + 1}{image.path.suffix}")
            saved += 1
    return saved


def _crop_image_cv2(
    dataset: YoloDataset,
    image: YoloImage,
    out_path: Path,
    *,
    keep_shape: bool,
    min_size: int,
    padding: int | float,
    confidence_threshold: float | None,
    by_attribute: bool,
    filter_no_attributes: bool,
) -> int:
    """Write object crops using OpenCV I/O; called independently per image."""

    _require_cv2()
    canvas = _read_cv2_image(image.path)
    height, width = canvas.shape[:2]
    saved = 0
    for idx, annotation in enumerate(image.annotations):
        if (
            confidence_threshold is not None
            and annotation.confidence is not None
            and annotation.confidence < confidence_threshold
        ):
            continue
        box = annotation.geometry_box()
        if box is None:
            continue
        xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
        box_left = max(0, int(round(xyxy.x1)))
        box_top = max(0, int(round(xyxy.y1)))
        box_right = min(width, int(round(xyxy.x2)))
        box_bottom = min(height, int(round(xyxy.y2)))
        if box_right - box_left < min_size or box_bottom - box_top < min_size:
            continue
        padding_x, padding_y = _crop_padding_pixels(
            padding,
            xyxy.x2 - xyxy.x1,
            xyxy.y2 - xyxy.y1,
        )
        left = max(0, int(round(xyxy.x1 - padding_x)))
        top = max(0, int(round(xyxy.y1 - padding_y)))
        right = min(width, int(round(xyxy.x2 + padding_x)))
        bottom = min(height, int(round(xyxy.y2 + padding_y)))
        if right <= left or bottom <= top:
            continue

        if keep_shape:
            crop = np.zeros_like(canvas)
            crop[top:bottom, left:right] = canvas[top:bottom, left:right]
        else:
            crop = canvas[top:bottom, left:right].copy()

        class_name = dataset.class_name(annotation.class_id)
        save_dirs = [out_path / class_name]
        if by_attribute:
            for attr_name, attr_value in dataset.annotation_attributes(annotation).items():
                if (
                    filter_no_attributes
                    and dataset.attributes is not None
                    and dataset.attributes.is_no_value(attr_value)
                ):
                    continue
                save_dirs.append(
                    out_path
                    / class_name
                    / f"{_safe_name(attr_name)}-{_safe_name(str(attr_value))}"
                )
        for save_dir in save_dirs:
            save_dir.mkdir(parents=True, exist_ok=True)
            _write_cv2_image(save_dir / f"{image.stem}_{idx + 1}{image.path.suffix}", crop)
            saved += 1
    return saved


def _read_cv2_image(path: str | Path) -> np.ndarray:
    _require_cv2()
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unable to read image with OpenCV: {path}")
    return image


def _write_cv2_image(path: str | Path, image: np.ndarray) -> None:
    _require_cv2()
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    extension = ".jpg" if suffix == ".jpeg" else suffix
    if extension not in {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        raise ValueError(f"OpenCV cannot encode output format: {output_path.suffix or '<none>'}")
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"OpenCV failed to encode image: {output_path}")
    encoded.tofile(str(output_path))


def _cv2_line_width(image: np.ndarray) -> int:
    return max(round(sum(image.shape) / 2 * 0.003), 2)


def _draw_cv2_label(
    image: np.ndarray,
    x: float,
    y: float,
    text: str,
    color: tuple[int, int, int],
    line_width: int,
) -> dict[str, int | float | bool] | None:
    if not text:
        return None
    _require_cv2()
    height, width = image.shape[:2]
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    pad = 2
    rect_width = text_width + pad * 2
    rect_height = text_height + baseline + pad * 2
    anchor_x = max(0, min(int(round(x)), max(width - 1, 0)))
    anchor_y = max(0, min(int(round(y)), max(height - 1, 0)))
    outside = anchor_y >= rect_height
    if outside and anchor_y - rect_height < 0:
        outside = False
    elif not outside and anchor_y + rect_height >= height:
        outside = True

    left = max(0, min(anchor_x, max(width - rect_width, 0)))
    if outside:
        top = max(0, anchor_y - rect_height)
        bottom = min(height - 1, anchor_y)
        text_baseline = max(top + text_height + pad, bottom - baseline - pad)
    else:
        top = min(max(anchor_y, 0), max(height - rect_height, 0))
        bottom = min(height - 1, top + rect_height)
        text_baseline = min(bottom - baseline - pad, top + text_height + pad)
    right = min(width - 1, left + rect_width)

    cv2.rectangle(image, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        str(text),
        (left + pad, int(text_baseline)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        _cv2_text_color(color),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )
    return {
        "left": left,
        "bottom": bottom,
        "outside": outside,
        "font_scale": font_scale,
        "font_thickness": font_thickness,
        "line_height": max(text_height + baseline, 1),
    }


def _draw_cv2_attributes(
    image: np.ndarray,
    attributes: list[tuple[str, object]],
    label_info: dict[str, int | float | bool] | None,
) -> None:
    if not attributes or label_info is None:
        return
    _require_cv2()
    height, width = image.shape[:2]
    font_scale = float(label_info["font_scale"])
    font_thickness = int(label_info["font_thickness"])
    line_height = max(int(float(label_info["line_height"]) * 0.85), 12)
    texts = [f"{name}-{value}" for name, value in attributes]
    sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        for text in texts
    ]
    max_text_width = max(size[0][0] for size in sizes)
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    x = max(0, min(int(label_info["left"]) + 2, max(width - max_text_width - 4, 0)))
    start_y = int(label_info["bottom"]) + line_height
    top = start_y - text_height - 2
    bottom = start_y + line_height * (len(texts) - 1) + baseline + 2
    if bottom >= height:
        shift = bottom - height + 1
        start_y -= shift
        top -= shift
        bottom -= shift
    if top < 0:
        start_y -= top
        bottom -= top
        top = 0
    right = min(width - 1, x + max_text_width + 5)
    bottom = min(height - 1, max(bottom, top))

    overlay = image.copy()
    cv2.rectangle(overlay, (x, top), (right, bottom), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    for idx, ((_, value), text) in enumerate(zip(attributes, texts)):
        text_y = min(max(start_y + line_height * idx, top + text_height), height - 1)
        text_color = (255, 0, 0) if value is not False else (0, 0, 0)
        cv2.putText(
            image,
            text,
            (x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def _attribute_values(
    dataset: YoloDataset,
    annotation,
    *,
    filter_no: bool,
) -> list[tuple[str, object]]:
    if dataset.attributes is None:
        return []
    values: list[tuple[str, object]] = []
    for name, value in dataset.annotation_attributes(annotation).items():
        if filter_no and dataset.attributes.is_no_value(value):
            continue
        values.append((name, value))
    return values


def _cv2_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    dark_colors = {
        (235, 219, 11),
        (243, 243, 243),
        (183, 223, 0),
        (221, 111, 255),
        (0, 237, 204),
        (68, 243, 0),
        (255, 255, 0),
        (179, 255, 1),
        (11, 255, 162),
    }
    if color in dark_colors:
        return (104, 31, 17)
    return (255, 255, 255)


def _validate_crop_padding(padding: int | float) -> None:
    if isinstance(padding, bool) or not isinstance(padding, (int, float)):
        raise TypeError("padding must be an integer pixel value or a floating-point ratio")
    if not math.isfinite(float(padding)) or padding < 0:
        raise ValueError("padding must be a finite non-negative value")


def _crop_padding_pixels(
    padding: int | float,
    box_width_pixels: float,
    box_height_pixels: float,
) -> tuple[float, float]:
    _validate_crop_padding(padding)
    if isinstance(padding, int) and not isinstance(padding, bool):
        return float(padding), float(padding)
    return box_width_pixels * float(padding), box_height_pixels * float(padding)


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


def _annotation_label(dataset: YoloDataset, annotation, *, show_txt_id: bool, annotation_idx: int | None = None) -> str:
    class_name = dataset.class_name(annotation.class_id)
    if show_txt_id:
        txt_id = annotation.line_no if annotation.line_no is not None else (annotation_idx + 1 if annotation_idx is not None else None)
        return f"{txt_id} {class_name}"
    return class_name


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
