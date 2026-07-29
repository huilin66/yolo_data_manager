from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from yolo_data_manager.core.multimodal import MultimodalImage, MultimodalYoloDataset
from yolo_data_manager.io.image_types import image_mode_details
from yolo_data_manager.runtime import iter_progress, normalize_workers


@dataclass(frozen=True)
class _Uint8ConversionJob:
    source: Path
    destination: Path
    convert: bool


def convert_multimodal_images_to_uint8(
    dataset: MultimodalYoloDataset,
    out_dir: str | Path,
    *,
    modalities: Iterable[str] | None = None,
    stretch: bool = True,
    value_range: tuple[float, float] | None = None,
    preserve_zero: bool = True,
    overwrite: bool = False,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
) -> dict[str, object]:
    """Copy uint8 images and convert other selected modality images to uint8 PNG.

    Non-uint8 images are written as PNG so their output type is unambiguous.
    With ``stretch=True``, each image is linearly mapped to the display range;
    zero values remain zero by default, which preserves common depth invalid
    pixels. ``value_range`` supplies a fixed source range when comparable
    brightness across images is required.
    """

    selected = _selected_modalities(dataset, modalities)
    output = Path(out_dir)
    normalized_range = _validate_value_range(value_range)
    worker_count = normalize_workers(workers)
    result: dict[str, dict[str, object]] = {}

    for modality in selected:
        sources = _source_images(dataset, modality)
        jobs = _prepare_jobs(
            sources,
            output / modality,
            progress=progress,
            progress_leave=progress_leave,
            modality=modality,
        )
        _validate_destinations(jobs, overwrite=overwrite)
        converted_count = _run_jobs(
            jobs,
            stretch=stretch,
            value_range=normalized_range,
            preserve_zero=preserve_zero,
            workers=worker_count,
            progress=progress,
            progress_leave=progress_leave,
            modality=modality,
        )
        result[modality] = {
            "image_count": len(jobs),
            "converted_count": converted_count,
            "copied_uint8_count": len(jobs) - converted_count,
            "out_dir": str(output / modality),
        }

    return {
        "report_type": "multimodal_uint8_conversion",
        "out": str(output),
        "stretch": stretch,
        "value_range": list(normalized_range) if normalized_range is not None else None,
        "preserve_zero": preserve_zero,
        "modalities": result,
    }


def _selected_modalities(dataset: MultimodalYoloDataset, values: Iterable[str] | None) -> list[str]:
    selected = list(dataset.modalities) if values is None else list(values)
    unknown = [name for name in selected if name not in dataset.modalities]
    if unknown:
        raise ValueError(f"unknown modality type(s): {', '.join(unknown)}")
    return selected


def _source_images(dataset: MultimodalYoloDataset, modality: str) -> list[MultimodalImage]:
    if modality in dataset.source_images:
        return dataset.source_images[modality]
    return [scene.images[modality] for scene in dataset.scenes.values() if modality in scene.images]


def _prepare_jobs(
    images: list[MultimodalImage],
    output: Path,
    *,
    progress: bool,
    progress_leave: bool,
    modality: str,
) -> list[_Uint8ConversionJob]:
    jobs: list[_Uint8ConversionJob] = []
    for image in iter_progress(
        images,
        enabled=progress,
        total=len(images),
        desc=f"uint8 prepare {modality}",
        leave=progress_leave,
    ):
        with Image.open(image.path) as source:
            dtype, _ = image_mode_details(source.mode)
        convert = dtype != "uint8"
        relative_path = image.relative_path.with_suffix(".png") if convert else image.relative_path
        jobs.append(
            _Uint8ConversionJob(
                source=image.path,
                destination=output / relative_path,
                convert=convert,
            )
        )
    return jobs


def _validate_destinations(jobs: list[_Uint8ConversionJob], *, overwrite: bool) -> None:
    destinations = [job.destination for job in jobs]
    destination_keys = [str(path.absolute()).casefold() for path in destinations]
    if len(set(destination_keys)) != len(destination_keys):
        raise ValueError("uint8 conversion would create duplicate output image paths")
    if not overwrite:
        existing = next((path for path in destinations if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"output image already exists: {existing}; choose a new out directory or set overwrite=True")


def _run_jobs(
    jobs: list[_Uint8ConversionJob],
    *,
    stretch: bool,
    value_range: tuple[float, float] | None,
    preserve_zero: bool,
    workers: int,
    progress: bool,
    progress_leave: bool,
    modality: str,
) -> int:
    def write(job: _Uint8ConversionJob) -> bool:
        job.destination.parent.mkdir(parents=True, exist_ok=True)
        if not job.convert:
            shutil.copy2(job.source, job.destination)
            return False
        with Image.open(job.source) as source:
            pixels = np.asarray(source)
        converted = _to_uint8(
            pixels,
            stretch=stretch,
            value_range=value_range,
            preserve_zero=preserve_zero,
        )
        _image_from_uint8(converted).save(job.destination, format="PNG")
        return True

    if workers == 1:
        return sum(
            write(job)
            for job in iter_progress(
                jobs,
                enabled=progress,
                total=len(jobs),
                desc=f"uint8 convert {modality}",
                leave=progress_leave,
            )
        )

    executor = ThreadPoolExecutor(max_workers=workers)
    progress_items = None
    try:
        futures = [executor.submit(write, job) for job in jobs]
        progress_items = iter_progress(
            as_completed(futures),
            enabled=progress,
            total=len(futures),
            desc=f"uint8 convert {modality}",
            leave=progress_leave,
        )
        converted_count = sum(future.result() for future in progress_items)
    except KeyboardInterrupt:
        _close_progress(progress_items)
        executor.shutdown(wait=False, cancel_futures=True)
        print(
            f"\nuint8 conversion for {modality} cancelled; pending work was stopped. "
            "Completed files remain in the output directory.",
            file=sys.stderr,
            flush=True,
        )
        raise
    except BaseException:
        _close_progress(progress_items)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
        return converted_count


def _to_uint8(
    values: np.ndarray,
    *,
    stretch: bool,
    value_range: tuple[float, float] | None,
    preserve_zero: bool,
) -> np.ndarray:
    if values.dtype == np.uint8:
        return values

    source = values.astype(np.float64, copy=False)
    finite = np.isfinite(source)
    result = np.zeros(source.shape, dtype=np.uint8)
    if not finite.any():
        return result

    if not stretch:
        result[finite] = np.rint(np.clip(source[finite], 0, 255)).astype(np.uint8)
        return result

    valid = finite & (source != 0) if preserve_zero else finite
    if not valid.any():
        return result
    low, high = value_range if value_range is not None else (float(source[valid].min()), float(source[valid].max()))
    if high == low:
        result[valid] = 255
        return result

    scaled = np.clip((source[valid] - low) / (high - low), 0.0, 1.0)
    upper = 254.0 if preserve_zero else 255.0
    offset = 1.0 if preserve_zero else 0.0
    result[valid] = np.rint(offset + scaled * upper).astype(np.uint8)
    return result


def _image_from_uint8(values: np.ndarray) -> Image.Image:
    if values.ndim == 2:
        return Image.fromarray(values, mode="L")
    if values.ndim != 3:
        raise ValueError(f"unsupported image array shape for uint8 conversion: {values.shape}")
    if values.shape[2] == 1:
        return Image.fromarray(values[:, :, 0], mode="L")
    modes = {2: "LA", 3: "RGB", 4: "RGBA"}
    mode = modes.get(values.shape[2])
    if mode is None:
        raise ValueError(f"unsupported channel count for uint8 conversion: {values.shape[2]}")
    return Image.fromarray(values, mode=mode)


def _validate_value_range(value_range: tuple[float, float] | None) -> tuple[float, float] | None:
    if value_range is None:
        return None
    if len(value_range) != 2:
        raise ValueError("value_range must be a (low, high) pair")
    low, high = (float(value) for value in value_range)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("value_range must contain two finite values where high > low")
    return low, high


def _close_progress(progress_items: object | None) -> None:
    close = getattr(progress_items, "close", None)
    if callable(close):
        close()
