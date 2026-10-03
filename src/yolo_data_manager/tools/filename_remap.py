"""Numeric filename remapping helpers for YOLO datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from yolo_data_manager.core.models import ClassSchema, YoloDataset, YoloImage
from yolo_data_manager.core.schema import (
    write_attribute_schema,
    write_class_schema,
    write_dataset_yaml,
)
from yolo_data_manager.io.output_paths import ydm_dir
from yolo_data_manager.runtime import iter_progress, normalize_workers


@dataclass(frozen=True)
class FilenameRemapItem:
    """One image/label filename remapping entry."""

    index: int
    number: int
    old_image: str
    new_image: str
    old_label: str | None
    new_label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "number": self.number,
            "old_image": self.old_image,
            "new_image": self.new_image,
            "old_label": self.old_label,
            "new_label": self.new_label,
        }


@dataclass(frozen=True)
class FilenameRemapResult:
    """Summary returned by :func:`remap_yolo_dataset_filenames`."""

    images: int
    labels: int
    digits: int
    start: int
    out: Path
    mapping: Path
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "images": self.images,
            "labels": self.labels,
            "digits": self.digits,
            "start": self.start,
            "out": str(self.out),
            "mapping": str(self.mapping),
            "split_counts": dict(self.split_counts),
        }


@dataclass(frozen=True)
class _FilenameRemapJob:
    item: FilenameRemapItem
    source_image: Path
    target_image: Path
    source_label: Path | None
    target_label: Path | None


def filename_digits_for_count(count: int) -> int:
    """Return the automatic numeric width for *count* images.

    The width is based on the next power of ten at or above ``count * 10``.
    For example, 8,951 images require ``100,000`` as the rounded-up capacity,
    so the generated names use six digits: ``000000`` through ``008950``.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return 1

    capacity = 1
    target = count * 10
    while capacity < target:
        capacity *= 10
    return len(str(capacity))


def remap_yolo_dataset_filenames(
    dataset: YoloDataset,
    out_root: str | Path,
    *,
    digits: int | None = None,
    start: int = 0,
    mapping_file: str | Path | None = None,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
    dry_run: bool = False,
) -> FilenameRemapResult:
    """Copy a YOLO dataset with numeric image and label filenames.

    The source dataset is not modified. Relative image/label directory
    structure is preserved, while each image receives a numeric stem and its
    matching label receives the same stem. Existing root split txt files are
    rewritten with the new image names. The mapping JSON contains both a
    convenient image mapping and full image/label entries.
    """

    source_root = Path(dataset.root).resolve()
    out_path = Path(out_root)
    if out_path.resolve() == source_root:
        raise ValueError("filename remap output must be different from the source dataset root")
    if start < 0:
        raise ValueError("start must be non-negative")

    count = len(dataset.images)
    resolved_digits = filename_digits_for_count(count) if digits is None else int(digits)
    if resolved_digits <= 0:
        raise ValueError("digits must be greater than zero")
    last_number = start + count - 1
    if count and len(str(last_number)) > resolved_digits:
        raise ValueError(
            f"digits={resolved_digits} cannot represent filename number {last_number}; "
            f"use at least {len(str(last_number))} digits"
        )

    jobs: list[_FilenameRemapJob] = []
    for index, image in enumerate(dataset.images):
        number = start + index
        new_stem = f"{number:0{resolved_digits}d}"
        source_image = Path(image.path)
        source_image_rel = _relative_path(source_root, source_image)
        new_image_rel = _new_image_relative_path(
            source_image_rel,
            new_stem,
            source_image.suffix,
        )

        source_label = _existing_label_path(image)
        source_label_rel = (
            _relative_path(source_root, source_label)
            if source_label is not None
            else None
        )
        new_label_rel = (
            _new_label_relative_path(source_label_rel, new_stem)
            if source_label_rel is not None
            else None
        )

        item = FilenameRemapItem(
            index=index,
            number=number,
            old_image=_display_path(source_root, source_image),
            new_image=new_image_rel.as_posix(),
            old_label=(
                _display_path(source_root, source_label)
                if source_label is not None
                else None
            ),
            new_label=new_label_rel.as_posix() if new_label_rel is not None else None,
        )
        jobs.append(
            _FilenameRemapJob(
                item=item,
                source_image=source_image,
                target_image=out_path / new_image_rel,
                source_label=source_label,
                target_label=(out_path / new_label_rel if new_label_rel is not None else None),
            )
        )

    source_split_files = {
        split_name: source_root / f"{split_name}.txt"
        for split_name in ("train", "val", "test")
        if (source_root / f"{split_name}.txt").is_file()
    }
    split_lines = _remapped_split_lines(source_root, source_split_files, jobs)
    mapping_path = _resolve_mapping_path(source_root, out_path, mapping_file)

    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)
        write_class_schema(dataset.classes, out_path / "class.txt")
        write_attribute_schema(dataset.attributes, out_path / "attribute.yaml")
        _write_dataset_metadata(
            out_path,
            jobs,
            source_split_files=source_split_files,
            classes=dataset.classes,
        )

        worker_count = normalize_workers(workers)
        if worker_count == 1:
            for job in iter_progress(
                jobs,
                enabled=progress,
                total=len(jobs),
                desc="remap filenames",
                leave=progress_leave,
            ):
                _copy_remap_job(job)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_copy_remap_job, job) for job in jobs]
                for future in iter_progress(
                    as_completed(futures),
                    enabled=progress,
                    total=len(futures),
                    desc="remap filenames",
                    leave=progress_leave,
                ):
                    future.result()

        for split_name, lines in split_lines.items():
            (out_path / f"{split_name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

        _write_mapping_json(
            mapping_path,
            source_root=source_root,
            out_path=out_path,
            digits=resolved_digits,
            start=start,
            jobs=jobs,
            split_counts={name: len(lines) for name, lines in split_lines.items()},
        )

    return FilenameRemapResult(
        images=count,
        labels=sum(1 for job in jobs if job.source_label is not None),
        digits=resolved_digits,
        start=start,
        out=out_path,
        mapping=mapping_path,
        split_counts={name: len(lines) for name, lines in split_lines.items()},
    )


def _existing_label_path(image: YoloImage) -> Path | None:
    if image.label_path is None:
        return None
    label_path = Path(image.label_path)
    return label_path if label_path.exists() else None


def _relative_path(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return None


def _display_path(root: Path, path: Path) -> str:
    relative = _relative_path(root, path)
    return relative.as_posix() if relative is not None else str(path)


def _new_image_relative_path(
    source_relative: Path | None,
    new_stem: str,
    suffix: str,
) -> Path:
    if source_relative is None:
        return Path("images") / f"{new_stem}{suffix}"
    return source_relative.parent / f"{new_stem}{suffix}"


def _new_label_relative_path(source_relative: Path, new_stem: str) -> Path:
    return source_relative.parent / f"{new_stem}.txt"


def _resolve_mapping_path(
    source_root: Path,
    out_path: Path,
    mapping_file: str | Path | None,
) -> Path:
    if mapping_file is None:
        return ydm_dir(source_root, "conversion") / "filename_mapping.json"
    path = Path(mapping_file)
    return path if path.is_absolute() else out_path / path


def _copy_remap_job(job: _FilenameRemapJob) -> None:
    if not job.source_image.exists():
        raise FileNotFoundError(f"source image not found: {job.source_image}")
    job.target_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job.source_image, job.target_image)
    if job.source_label is not None and job.target_label is not None:
        job.target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job.source_label, job.target_label)


def _write_dataset_metadata(
    out_path: Path,
    jobs: list[_FilenameRemapJob],
    *,
    source_split_files: dict[str, Path],
    classes: ClassSchema,
) -> None:
    if source_split_files:
        train = "train.txt" if "train" in source_split_files else "images"
        val = "val.txt" if "val" in source_split_files else "images"
        test = "test.txt" if "test" in source_split_files else None
    else:
        split_dirs = {
            part
            for job in jobs
            for part in job.item.new_image.split("/")[:-1]
            if part in {"train", "val", "test"}
        }
        train = "images/train" if "train" in split_dirs else "images"
        val = "images/val" if "val" in split_dirs else "images"
        test = "images/test" if "test" in split_dirs else None
    write_dataset_yaml(
        classes,
        out_path / "dataset.yaml",
        train=train,
        val=val,
        test=test,
    )


def _remapped_split_lines(
    source_root: Path,
    source_split_files: dict[str, Path],
    jobs: list[_FilenameRemapJob],
) -> dict[str, list[str]]:
    if not source_split_files:
        return {}

    lookup: dict[str, set[str]] = {}
    for job in jobs:
        new_name = job.item.new_image
        source_relative = _relative_path(source_root, job.source_image)
        candidates = {
            _normalise_key(job.source_image),
            _normalise_key(job.source_image.name),
            _normalise_key(job.source_image.stem),
        }
        if source_relative is not None:
            candidates.add(_normalise_key(source_relative))
        for candidate in candidates:
            if candidate:
                lookup.setdefault(candidate, set()).add(new_name)

    result: dict[str, list[str]] = {}
    for split_name, split_file in source_split_files.items():
        lines: list[str] = []
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            text = raw_line.strip()
            if not text or text.startswith("#"):
                continue
            candidates = {
                _normalise_key(text),
                _normalise_key(Path(text).name),
                _normalise_key(Path(text).stem),
            }
            raw_path = Path(text).expanduser()
            if not raw_path.is_absolute():
                candidates.add(_normalise_key(source_root / raw_path))
            matches = set().union(*(lookup.get(key, set()) for key in candidates))
            if len(matches) == 1:
                lines.append(next(iter(matches)))
        result[split_name] = lines
    return result


def _normalise_key(value: str | Path) -> str:
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    return str(Path(text)).replace("\\", "/").casefold()


def _write_mapping_json(
    path: Path,
    *,
    source_root: Path,
    out_path: Path,
    digits: int,
    start: int,
    jobs: list[_FilenameRemapJob],
    split_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_mapping: dict[str, str] = {}
    label_mapping: dict[str, str] = {}
    for job in jobs:
        item = job.item
        image_mapping[item.old_image] = item.new_image
        if item.old_label is not None and item.new_label is not None:
            label_mapping[item.old_label] = item.new_label

    payload = {
        "schema_version": 1,
        "source_root": str(source_root),
        "output_root": str(out_path),
        "images": len(jobs),
        "labels": sum(1 for job in jobs if job.source_label is not None),
        "digits": digits,
        "start": start,
        "split_counts": split_counts,
        "image_mapping": image_mapping,
        "label_mapping": label_mapping,
        "items": [job.item.to_dict() for job in jobs],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

