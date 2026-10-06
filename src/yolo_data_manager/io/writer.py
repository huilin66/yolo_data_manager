from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any

from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.core.schema import write_attribute_schema, write_class_schema, write_dataset_yaml
from yolo_data_manager.io.backup import (
    LabelBackup,
    ensure_source_labels_backup,
    write_snapshot_metadata,
)
from yolo_data_manager.logging_utils import current_operation
from yolo_data_manager.runtime import iter_progress, normalize_workers


def write_yolo_dataset(
    dataset: YoloDataset,
    out_root: str | Path,
    copy_images: bool = True,
    keep_empty_labels: bool = True,
    include_confidence: bool = False,
    overwrite_images: bool = True,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
    backup_dir: str | Path | None = None,
    backup: bool = True,
    operation: str | None = None,
    backup_result: Mapping[str, Any] | None = None,
) -> LabelBackup | None:
    out_path = Path(out_root)
    image_dir = out_path / "images"
    label_dir = out_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    backup_obj: LabelBackup | None = None
    if backup:
        ensure_source_labels_backup(
            dataset.root,
            backup_dir,
            source_paths=(
                image.label_path
                for image in dataset.images
                if image.label_path is not None
            ),
            extra_paths=(
                Path(dataset.root) / name
                for name in ("class.txt", "dataset.yaml", "attribute.yaml")
            ),
        )
        backup_obj = LabelBackup(
            dataset.root,
            backup_dir,
            method=operation or current_operation() or "write_yolo_dataset",
        )
        for image in dataset.images:
            if image.label_path is not None:
                backup_obj.backup(image.label_path)

    write_class_schema(dataset.classes, out_path / "class.txt")
    write_dataset_yaml(dataset.classes, out_path / "dataset.yaml", train="images", val="images")
    write_attribute_schema(dataset.attributes, out_path / "attribute.yaml")

    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for image in iter_progress(dataset.images, enabled=progress, total=len(dataset.images), desc="write dataset", leave=progress_leave):
            _write_image_item(
                image,
                image_dir=image_dir,
                label_dir=label_dir,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
                include_confidence=include_confidence,
                overwrite_images=overwrite_images,
            )
        _finalize_dataset_backup(
            backup_obj,
            output_root=out_path,
            dataset=dataset,
            copy_images=copy_images,
            operation=operation,
            backup_result=backup_result,
        )
        return backup_obj

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _write_image_item,
                image,
                image_dir=image_dir,
                label_dir=label_dir,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
                include_confidence=include_confidence,
                overwrite_images=overwrite_images,
            )
            for image in dataset.images
        ]
        for future in iter_progress(as_completed(futures), enabled=progress, total=len(futures), desc="write dataset", leave=progress_leave):
            future.result()
    _finalize_dataset_backup(
        backup_obj,
        output_root=out_path,
        dataset=dataset,
        copy_images=copy_images,
        operation=operation,
        backup_result=backup_result,
    )
    return backup_obj


def clone_yolo_dataset_for_output(
    dataset: YoloDataset,
    out_root: str | Path,
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> YoloDataset:
    """Materialize a dataset copy and retarget it for annotation edits.

    The returned dataset has label paths under ``out_root/labels`` and image
    paths under ``out_root/images``.  The source dataset is never written or
    backed up.  This is used by ``out_data`` on crop-based correction APIs so
    the existing in-place correction implementation can safely operate on a
    new dataset.
    """

    output_root = Path(out_root).expanduser()
    source_root = Path(dataset.root).expanduser()
    if output_root.resolve() == source_root.resolve():
        raise ValueError(
            "out_data must point to a different dataset root; "
            "omit out_data to edit the source dataset in place"
        )

    write_yolo_dataset(
        dataset,
        output_root,
        copy_images=True,
        keep_empty_labels=True,
        workers=workers,
        progress=progress,
        progress_leave=progress_leave,
        backup=False,
        operation="annotation.out_data",
    )

    copied = deepcopy(dataset)
    copied.root = output_root
    for image in copied.images:
        file_name = image.file_name
        stem = Path(file_name).stem
        image.path = output_root / "images" / file_name
        # write_yolo_dataset keeps empty labels by default.  Giving every
        # image a target label path also lets error-crop correction append a
        # prediction to an image that originally had no txt file.
        image.label_path = output_root / "labels" / f"{stem}.txt"
    copied.orphan_labels = [
        output_root / "labels" / Path(path).name
        for path in dataset.orphan_labels
    ]
    return copied


def write_yolo_labels_in_place(
    dataset: YoloDataset,
    *,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
    backup_dir: str | Path | None = None,
    operation: str | None = None,
    backup_result: Mapping[str, Any] | None = None,
) -> LabelBackup:
    """Rewrite the loaded dataset's label files in place.

    The source labels are backed up before any write.  Images, split files,
    and dataset layout are left untouched; this is the in-place counterpart
    of :func:`write_yolo_dataset` for annotation-only operations.
    """

    ensure_source_labels_backup(
        dataset.root,
        backup_dir,
        source_paths=(
            image.label_path
            for image in dataset.images
            if image.label_path is not None
        ),
        extra_paths=(
            Path(dataset.root) / name
            for name in ("class.txt", "dataset.yaml", "attribute.yaml")
        ),
    )
    backup_obj = LabelBackup(
        dataset.root,
        backup_dir,
        method=operation or current_operation() or "write_yolo_labels_in_place",
    )
    for image in dataset.images:
        if image.label_path is not None:
            backup_obj.backup(image.label_path)

    def write_one(image: YoloImage) -> None:
        if image.label_path is None:
            return
        label_path = Path(image.label_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            annotation.to_yolo_line(include_confidence=False)
            for annotation in image.annotations
        ]
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for image in iter_progress(
            dataset.images,
            enabled=progress,
            total=len(dataset.images),
            desc="write labels",
            leave=progress_leave,
        ):
            write_one(image)
        _finalize_in_place_backup(
            backup_obj,
            dataset=dataset,
            operation=operation,
            backup_result=backup_result,
        )
        return backup_obj

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(write_one, image) for image in dataset.images]
        for future in iter_progress(
            as_completed(futures),
            enabled=progress,
            total=len(futures),
            desc="write labels",
            leave=progress_leave,
        ):
            future.result()
    _finalize_in_place_backup(
        backup_obj,
        dataset=dataset,
        operation=operation,
        backup_result=backup_result,
    )
    return backup_obj


def _finalize_dataset_backup(
    backup: LabelBackup | None,
    *,
    output_root: Path,
    dataset: YoloDataset,
    copy_images: bool,
    operation: str | None,
    backup_result: Mapping[str, Any] | None,
) -> None:
    if backup is None or backup.count == 0:
        return
    result: dict[str, Any] = {
        "action": "write_yolo_dataset",
        "output_root": str(output_root),
        "images": len(dataset.images),
        "labels": sum(image.label_path is not None for image in dataset.images),
        "annotations": dataset.annotation_count(),
        "copy_images": copy_images,
    }
    if backup_result:
        result.update(dict(backup_result))
    backup.write_metadata(method=operation, result=result)


def _finalize_in_place_backup(
    backup: LabelBackup,
    *,
    dataset: YoloDataset,
    operation: str | None,
    backup_result: Mapping[str, Any] | None,
) -> None:
    if backup.count == 0:
        return
    result: dict[str, Any] = {
        "action": "write_yolo_labels_in_place",
        "labels": sum(image.label_path is not None for image in dataset.images),
        "annotations": dataset.annotation_count(),
    }
    if backup_result:
        result.update(dict(backup_result))
    backup.write_metadata(method=operation, result=result)


def _write_image_item(
    image: YoloImage,
    *,
    image_dir: Path,
    label_dir: Path,
    copy_images: bool,
    keep_empty_labels: bool,
    include_confidence: bool,
    overwrite_images: bool,
) -> None:
    dst_image = image_dir / image.file_name
    if copy_images and image.path.exists() and (overwrite_images or not dst_image.exists()):
        shutil.copy2(image.path, dst_image)

    if not image.annotations and not keep_empty_labels:
        return
    dst_label = label_dir / f"{image.stem}.txt"
    lines = [ann.to_yolo_line(include_confidence=include_confidence) for ann in image.annotations]
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_split_file(image_names: list[str], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(image_names) + ("\n" if image_names else ""), encoding="utf-8")


def move_existing_split_files_to_backup(
    split_root: str | Path,
    backup_dir: str | Path,
    *,
    method: str | None = None,
    result: Mapping[str, Any] | None = None,
) -> Path | None:
    """Move existing train/val/test lists into one timestamped snapshot.

    The snapshot directory is created only when at least one split file
    already exists.  This is intentionally a move because split files are
    regenerated immediately after the backup is made.
    """

    source_root = Path(split_root)
    existing = [
        source_root / f"{split_name}.txt"
        for split_name in ("train", "val", "test")
        if (source_root / f"{split_name}.txt").is_file()
    ]
    if not existing:
        return None

    backup_root = Path(backup_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot = backup_root / timestamp
    counter = 1
    while snapshot.exists():
        snapshot = backup_root / f"{timestamp}_{counter}"
        counter += 1
    snapshot.mkdir(parents=True, exist_ok=False)

    for source in existing:
        shutil.move(str(source), str(snapshot / source.name))
    metadata_result: dict[str, Any] = {
        "action": "move_existing_split_files",
        "split_root": str(source_root.resolve()),
        "moved_files": [source.name for source in existing],
    }
    if result:
        metadata_result.update(dict(result))
    write_snapshot_metadata(
        snapshot,
        dataset_root=source_root,
        method=method or current_operation() or "dataset.split",
        created_at=datetime.now().astimezone().isoformat(timespec="microseconds"),
        files=[source.name for source in existing],
        result=metadata_result,
    )
    return snapshot
