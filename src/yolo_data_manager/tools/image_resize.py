"""Image resizing helpers for YOLO datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from PIL import Image

from yolo_data_manager.core.models import Box, Polygon, YoloAnnotation, YoloDataset
from yolo_data_manager.core.schema import (
    write_attribute_schema,
    write_class_schema,
    write_dataset_yaml,
)
from yolo_data_manager.runtime import iter_progress, normalize_workers


_RESAMPLING_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class ResizeResult:
    """Summary returned after resizing a YOLO dataset."""

    images: int
    annotations: int
    out: Path
    keep_ratio: bool
    letterboxed_images: int

    def to_dict(self) -> dict[str, object]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "out": str(self.out),
            "keep_ratio": self.keep_ratio,
            "letterboxed_images": self.letterboxed_images,
        }


@dataclass(frozen=True)
class _ResizeTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def box(self, value: Box) -> Box:
        left = value.cx - value.width / 2
        top = value.cy - value.height / 2
        right = value.cx + value.width / 2
        bottom = value.cy + value.height / 2
        new_left = self._x(left)
        new_top = self._y(top)
        new_right = self._x(right)
        new_bottom = self._y(bottom)
        return Box(
            cx=(new_left + new_right) / 2,
            cy=(new_top + new_bottom) / 2,
            width=new_right - new_left,
            height=new_bottom - new_top,
        )

    def polygon(self, value: Polygon) -> Polygon:
        return Polygon([(self._x(x), self._y(y)) for x, y in value.points])

    def _x(self, normalized_x: float) -> float:
        pixel_x = normalized_x * self.source_width * self.scale_x + self.offset_x
        return pixel_x / self.target_width

    def _y(self, normalized_y: float) -> float:
        pixel_y = normalized_y * self.source_height * self.scale_y + self.offset_y
        return pixel_y / self.target_height


def resize_image(
    source: str | Path,
    destination: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    keep_ratio: bool = True,
    interpolation: str = "lanczos",
    fill_color: int | Sequence[int] = (114, 114, 114),
) -> tuple[int, int]:
    """Resize one image and return its output ``(width, height)``.

    If both target dimensions are given and ``keep_ratio`` is true, the image
    is letterboxed onto the requested canvas.  If ``keep_ratio`` is false,
    the image is stretched to the requested dimensions.  Supplying only one
    dimension preserves the source aspect ratio; ``scale`` scales both
    dimensions directly.
    """

    validate_resize_options(
        width=width,
        height=height,
        scale=scale,
        interpolation=interpolation,
    )
    with Image.open(source) as image:
        resized, _transform, _letterboxed = _resize_pil_image(
            image,
            width=width,
            height=height,
            scale=scale,
            keep_ratio=keep_ratio,
            interpolation=interpolation,
            fill_color=fill_color,
        )
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(resized, destination_path)
        output_size = resized.size
        if resized is not image:
            resized.close()
    return output_size


def resize_yolo_dataset(
    dataset: YoloDataset,
    out_root: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    keep_ratio: bool = True,
    interpolation: str = "lanczos",
    fill_color: int | Sequence[int] = (114, 114, 114),
    keep_empty_labels: bool = True,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> ResizeResult:
    """Resize all images in a YOLO dataset and write a new flat dataset.

    YOLO coordinates are normalized, so direct stretching keeps their values
    unchanged.  For letterboxed output, boxes and segmentation polygons are
    transformed to account for the scale and padding.
    """

    validate_resize_options(
        width=width,
        height=height,
        scale=scale,
        interpolation=interpolation,
    )
    out_path = Path(out_root)
    source_root = Path(dataset.root)
    if out_path.resolve() == source_root.resolve():
        raise ValueError("resize output must be different from the source dataset root")

    image_dir = out_path / "images"
    label_dir = out_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    write_class_schema(dataset.classes, out_path / "class.txt")
    write_dataset_yaml(dataset.classes, out_path / "dataset.yaml", train="images", val="images")
    write_attribute_schema(dataset.attributes, out_path / "attribute.yaml")

    output_names = [image.file_name for image in dataset.images]
    name_counts: dict[str, int] = {}
    for name in output_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "resize output requires unique image file names; duplicates: "
            + ", ".join(duplicates[:5])
        )
    label_stems = [image.stem for image in dataset.images if image.annotations or keep_empty_labels]
    stem_counts: dict[str, int] = {}
    for stem in label_stems:
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicate_stems:
        raise ValueError(
            "resize output requires unique label stems; duplicates: "
            + ", ".join(duplicate_stems[:5])
        )

    def process(image) -> tuple[int, int, bool]:
        destination = image_dir / image.file_name
        with Image.open(image.path) as source_image:
            resized, transform, letterboxed = _resize_pil_image(
                source_image,
                width=width,
                height=height,
                scale=scale,
                keep_ratio=keep_ratio,
                interpolation=interpolation,
                fill_color=fill_color,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save_image(resized, destination)
            annotations = [
                _transform_annotation(annotation, transform)
                for annotation in image.annotations
            ]
            if annotations or keep_empty_labels:
                label_path = label_dir / f"{image.stem}.txt"
                lines = [annotation.to_yolo_line() for annotation in annotations]
                label_path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
            output_size = resized.size
            if resized is not source_image:
                resized.close()
        return output_size[0], output_size[1], letterboxed

    worker_count = normalize_workers(workers)
    results: list[tuple[int, int, bool]] = []
    if worker_count == 1:
        for image in iter_progress(
            dataset.images,
            enabled=progress,
            total=len(dataset.images),
            desc="resize images",
            leave=progress_leave,
        ):
            results.append(process(image))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process, image) for image in dataset.images]
            for future in iter_progress(
                as_completed(futures),
                enabled=progress,
                total=len(futures),
                desc="resize images",
                leave=progress_leave,
            ):
                results.append(future.result())

    return ResizeResult(
        images=len(dataset.images),
        annotations=dataset.annotation_count(),
        out=out_path,
        keep_ratio=keep_ratio,
        letterboxed_images=sum(1 for _width, _height, letterboxed in results if letterboxed),
    )


def validate_resize_options(
    *,
    width: int | None,
    height: int | None,
    scale: float | None,
    interpolation: str,
) -> None:
    if width is None and height is None and scale is None:
        raise ValueError("resize requires width, height, or scale")
    if scale is not None and (width is not None or height is not None):
        raise ValueError("scale cannot be combined with width or height")
    if width is not None and width <= 0:
        raise ValueError("width must be greater than zero")
    if height is not None and height <= 0:
        raise ValueError("height must be greater than zero")
    if scale is not None and scale <= 0:
        raise ValueError("scale must be greater than zero")
    if interpolation not in _RESAMPLING_FILTERS:
        choices = ", ".join(sorted(_RESAMPLING_FILTERS))
        raise ValueError(f"interpolation must be one of: {choices}")


def _resize_pil_image(
    image: Image.Image,
    *,
    width: int | None,
    height: int | None,
    scale: float | None,
    keep_ratio: bool,
    interpolation: str,
    fill_color: int | Sequence[int],
) -> tuple[Image.Image, _ResizeTransform, bool]:
    source_width, source_height = image.size
    target_width, target_height = _target_size(
        source_width,
        source_height,
        width=width,
        height=height,
        scale=scale,
    )
    resampling = _RESAMPLING_FILTERS[interpolation]

    if keep_ratio and width is not None and height is not None and scale is None:
        ratio = min(target_width / source_width, target_height / source_height)
        content_width = max(1, round(source_width * ratio))
        content_height = max(1, round(source_height * ratio))
        content = image.resize((content_width, content_height), resample=resampling)
        canvas = _new_canvas(content.mode, (target_width, target_height), fill_color)
        if content.mode != canvas.mode:
            converted_content = content.convert(canvas.mode)
            content.close()
            content = converted_content
        offset_x = (target_width - content_width) // 2
        offset_y = (target_height - content_height) // 2
        canvas.paste(content, (offset_x, offset_y))
        content.close()
        transform = _ResizeTransform(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            scale_x=content_width / source_width,
            scale_y=content_height / source_height,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        return canvas, transform, True

    resized = image.resize((target_width, target_height), resample=resampling)
    transform = _ResizeTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        scale_x=target_width / source_width,
        scale_y=target_height / source_height,
        offset_x=0.0,
        offset_y=0.0,
    )
    return resized, transform, False


def _target_size(
    source_width: int,
    source_height: int,
    *,
    width: int | None,
    height: int | None,
    scale: float | None,
) -> tuple[int, int]:
    if scale is not None:
        return (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )
    if width is None:
        width = max(1, round(source_width * height / source_height))
    if height is None:
        height = max(1, round(source_height * width / source_width))
    return int(width), int(height)


def _new_canvas(
    mode: str,
    size: tuple[int, int],
    fill_color: int | Sequence[int],
) -> Image.Image:
    if mode in {"1", "L", "I", "F"}:
        if isinstance(fill_color, Sequence) and not isinstance(fill_color, (str, bytes)):
            fill_color = fill_color[0]
        return Image.new(mode, size, fill_color)
    if mode == "RGBA":
        if not isinstance(fill_color, Sequence) or isinstance(fill_color, (str, bytes)):
            fill_color = (int(fill_color), int(fill_color), int(fill_color), 255)
        else:
            values = tuple(fill_color)
            if len(values) == 3:
                values = (*values, 255)
            elif len(values) != 4:
                raise ValueError("RGBA fill color must have 3 or 4 values")
            fill_color = values
        return Image.new(mode, size, fill_color)
    if mode != "RGB":
        mode = "RGBA" if "A" in mode else "RGB"
    if not isinstance(fill_color, Sequence) or isinstance(fill_color, (str, bytes)):
        fill_color = (int(fill_color), int(fill_color), int(fill_color))
    else:
        values = tuple(fill_color)
        if len(values) == 4 and mode == "RGB":
            values = values[:3]
        elif len(values) != 3:
            raise ValueError("RGB fill color must have 3 values")
        fill_color = values
    return Image.new(mode, size, fill_color)


def _save_image(image: Image.Image, destination: Path) -> None:
    output = image
    converted = False
    if destination.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
        output = image.convert("RGB")
        converted = True
    try:
        output.save(destination)
    finally:
        if converted:
            output.close()


def _transform_annotation(
    annotation: YoloAnnotation,
    transform: _ResizeTransform,
) -> YoloAnnotation:
    box = transform.box(annotation.box) if annotation.box is not None else None
    polygon = (
        transform.polygon(annotation.polygon)
        if annotation.polygon is not None
        else None
    )
    return replace(annotation, box=box, polygon=polygon, attributes=list(annotation.attributes))
