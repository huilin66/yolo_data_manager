"""Convert the TT100K custom JSON annotations to YOLO detection format.

TT100K does not use the COCO/VOC/YOLO annotation schemas.  Its annotation file
contains a ``types`` list and an ``imgs`` mapping; each object has a category
name and an ``xmin/ymin/xmax/ymax`` bounding box.  This module deliberately
keeps the converter independent of the rest of the dataset manager so it can
also be used from the small command-line wrapper in ``tools/``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


class TT100KFormatError(ValueError):
    """Raised when a TT100K annotation file cannot be converted safely."""


@dataclass
class TT100KConversionStats:
    """Summary returned by :func:`convert_tt100k`."""

    images: int = 0
    instances: int = 0
    split_images: dict[str, int] = field(default_factory=dict)
    split_instances: dict[str, int] = field(default_factory=dict)
    clipped_boxes: int = 0


def convert_tt100k(
    source_root: str | Path,
    out_root: str | Path,
    *,
    annotation_file: str | Path | None = None,
    splits: Iterable[str] | None = None,
    copy_images: bool = True,
    workers: int = 8,
    progress: bool = True,
) -> TT100KConversionStats:
    """Convert a TT100K directory to a YOLO detection dataset.

    The output layout is a flat YOLO layout so the caller can create its own
    train/validation/test split::

        output/
          images/...
          labels/...txt
          classes.txt

    Only images listed in the TT100K JSON are written.  Bounding boxes are
    clipped to the actual image dimensions and normalized to the YOLO range
    ``[0, 1]``.  A malformed annotation raises an error instead of silently
    producing a bad label file.
    """

    source = Path(source_root)
    output = Path(out_root)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"TT100K source directory does not exist: {source}")
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise TT100KFormatError(
            "Output directory must be outside the source directory to avoid modifying the source"
        )

    annotation_path = Path(annotation_file) if annotation_file is not None else source / "annotations_all.json"
    if not annotation_path.is_absolute():
        annotation_path = source / annotation_path
    if not annotation_path.exists():
        raise FileNotFoundError(f"TT100K annotation file does not exist: {annotation_path}")

    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TT100KFormatError(f"Invalid JSON in {annotation_path}: {exc}") from exc

    class_names = _read_class_names(data)
    image_items = data.get("imgs")
    if not isinstance(image_items, dict):
        raise TT100KFormatError("Expected the TT100K JSON field 'imgs' to be an object")

    if workers < 1:
        raise TT100KFormatError("workers must be at least 1")
    requested_splits = {str(split) for split in splits} if splits is not None else None
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    stats = TT100KConversionStats()
    split_instance_counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    destination_sources: dict[str, str] = {}

    # Validate paths and detect flattening collisions before starting worker
    # threads.  This prevents two workers from writing the same destination.
    selected_entries: list[tuple[str, dict[str, Any], Path, str]] = []
    for image_key, image_item in sorted(image_items.items()):
        if not isinstance(image_item, dict):
            raise TT100KFormatError(f"Image entry {image_key!r} is not an object")

        relative_image = _safe_relative_image_path(image_item.get("path"), image_key)
        split = relative_image.parts[0] if len(relative_image.parts) > 1 else "unspecified"
        if requested_splits is not None and split not in requested_splits:
            continue

        destination_name = relative_image.name
        previous_source = destination_sources.get(destination_name)
        if previous_source is not None:
            raise TT100KFormatError(
                "Cannot flatten TT100K splits because image name "
                f"{destination_name!r} occurs in both {previous_source!r} and "
                f"{str(relative_image)!r}"
            )
        destination_sources[destination_name] = str(relative_image)
        selected_entries.append((str(image_key), image_item, relative_image, split))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _convert_entry,
                source=source,
                output=output,
                image_key=image_key,
                image_item=image_item,
                relative_image=relative_image,
                split=split,
                class_names=class_names,
                copy_images=copy_images,
            )
            for image_key, image_item, relative_image, split in selected_entries
        ]
        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="convert TT100K",
            unit="image",
            disable=not progress,
        ) as progress_items:
            for future in progress_items:
                result = future.result()
                stats.images += 1
                stats.instances += result.instances
                stats.clipped_boxes += result.clipped_boxes
                split_image_counts[result.split] += 1
                split_instance_counts[result.split] += result.instances

    stats.split_images = dict(sorted(split_image_counts.items()))
    stats.split_instances = dict(sorted(split_instance_counts.items()))
    _write_class_files(output, class_names)
    return stats


@dataclass(frozen=True)
class _ConvertedEntry:
    split: str
    instances: int
    clipped_boxes: int


def _convert_entry(
    *,
    source: Path,
    output: Path,
    image_key: str,
    image_item: dict[str, Any],
    relative_image: Path,
    split: str,
    class_names: list[str],
    copy_images: bool,
) -> _ConvertedEntry:
    source_image = source.joinpath(*relative_image.parts)
    if not source_image.exists():
        raise FileNotFoundError(f"Image listed in annotations is missing: {source_image}")

    try:
        with Image.open(source_image) as image:
            width, height = image.size
    except Exception as exc:  # Pillow uses several exception types for bad files.
        raise TT100KFormatError(f"Cannot read image dimensions for {source_image}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise TT100KFormatError(f"Image has invalid dimensions ({width}x{height}): {source_image}")

    objects = image_item.get("objects", [])
    if not isinstance(objects, list):
        raise TT100KFormatError(f"Image entry {image_key!r} has a non-list 'objects' field")
    label_lines: list[str] = []
    clipped_boxes = 0
    for object_index, obj in enumerate(objects, start=1):
        line, was_clipped = _object_to_yolo_line(
            obj,
            class_names=class_names,
            width=width,
            height=height,
            image_key=image_key,
            object_index=object_index,
        )
        label_lines.append(line)
        if was_clipped:
            clipped_boxes += 1

    destination_name = relative_image.name
    destination_image = output / "images" / destination_name
    destination_label = output / "labels" / Path(destination_name).with_suffix(".txt")
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source_image, destination_image)
    destination_label.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )
    return _ConvertedEntry(split=split, instances=len(label_lines), clipped_boxes=clipped_boxes)


def _read_class_names(data: dict[str, Any]) -> list[str]:
    raw_names = data.get("types")
    if not isinstance(raw_names, list) or not raw_names:
        raise TT100KFormatError("Expected a non-empty TT100K JSON field 'types'")
    class_names = [str(name).strip() for name in raw_names]
    if any(not name for name in class_names):
        raise TT100KFormatError("TT100K class names cannot be empty")
    if len(set(class_names)) != len(class_names):
        raise TT100KFormatError("TT100K class names contain duplicates")
    return class_names


def _safe_relative_image_path(raw_path: object, image_key: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TT100KFormatError(f"Image entry {image_key!r} has no valid 'path'")
    # TT100K uses POSIX separators even on Windows.  Reject absolute paths and
    # traversal so a malformed JSON cannot write outside the output directory.
    posix_path = PurePosixPath(raw_path.replace("\\", "/"))
    if posix_path.is_absolute() or ".." in posix_path.parts or not posix_path.parts:
        raise TT100KFormatError(f"Unsafe image path for entry {image_key!r}: {raw_path!r}")
    return Path(*posix_path.parts)


def _object_to_yolo_line(
    obj: object,
    *,
    class_names: list[str],
    width: int,
    height: int,
    image_key: str,
    object_index: int,
) -> tuple[str, bool]:
    if not isinstance(obj, dict):
        raise TT100KFormatError(f"Image {image_key!r} object {object_index} is not an object")
    category = obj.get("category")
    if not isinstance(category, str) or category not in class_names:
        raise TT100KFormatError(
            f"Image {image_key!r} object {object_index} has unknown category: {category!r}"
        )
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        raise TT100KFormatError(f"Image {image_key!r} object {object_index} has no bbox object")
    try:
        x1 = float(bbox["xmin"])
        y1 = float(bbox["ymin"])
        x2 = float(bbox["xmax"])
        y2 = float(bbox["ymax"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TT100KFormatError(
            f"Image {image_key!r} object {object_index} has an invalid bbox: {bbox!r}"
        ) from exc
    values = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values):
        raise TT100KFormatError(f"Image {image_key!r} object {object_index} bbox is not finite")

    clipped = (max(0.0, min(x1, float(width))), max(0.0, min(y1, float(height))))
    clipped_end = (max(0.0, min(x2, float(width))), max(0.0, min(y2, float(height))))
    x1, y1 = clipped
    x2, y2 = clipped_end
    if x2 <= x1 or y2 <= y1:
        raise TT100KFormatError(
            f"Image {image_key!r} object {object_index} has an empty bbox after clipping: {bbox!r}"
        )

    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    class_id = class_names.index(category)
    line = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
    return line, values != (x1, y1, x2, y2)


def _write_class_files(output: Path, class_names: list[str]) -> None:
    (output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert TT100K custom annotations_all.json to YOLO detection format"
    )
    parser.add_argument("--src", required=True, help="TT100K source directory")
    parser.add_argument("--out", required=True, help="output YOLO dataset directory")
    parser.add_argument(
        "--annotation",
        default=None,
        help="annotation JSON path (default: <src>/annotations_all.json)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="only convert these split names, e.g. train test (default: all entries)",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="write labels and metadata without copying image files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of worker threads for image reading/copying (default: 8)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stats = convert_tt100k(
            args.src,
            args.out,
            annotation_file=args.annotation,
            splits=args.splits,
            copy_images=not args.no_copy_images,
            workers=args.workers,
            progress=not args.quiet,
        )
    except (OSError, TT100KFormatError) as exc:
        parser.error(str(exc))
        return 2

    print(f"Converted {stats.images} images and {stats.instances} instances to {Path(args.out)}")
    for split, count in stats.split_images.items():
        print(f"  {split}: {count} images, {stats.split_instances[split]} instances")
    if stats.clipped_boxes:
        print(f"  clipped boxes: {stats.clipped_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
