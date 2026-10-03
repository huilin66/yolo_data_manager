"""VLM-assisted automatic YOLO annotation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Any

from yolo_data_manager.core.models import Box, YoloAnnotation, YoloDataset, YoloImage
from yolo_data_manager.io.writer import write_yolo_dataset
from yolo_data_manager.runtime import create_progress_bar, normalize_workers
from yolo_data_manager.vlm.prompts import auto_label_prompt
from yolo_data_manager.vlm.providers import VLMProvider, VLMResponse
from yolo_data_manager.vlm.schemas import VLMBox, VLMOutputError, parse_detection_response


@dataclass
class AutoLabelSummary:
    images: int
    succeeded: int
    failed: int
    annotations: int
    out: str | None
    results: str | None
    errors: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_yolo_labels(
    dataset: YoloDataset,
    out_root: str | Path,
    provider: VLMProvider,
    *,
    workers: int = 4,
    limit: int | None = None,
    prompt: str | None = None,
    allow_new_classes: bool = False,
    copy_images: bool = True,
    dry_run: bool = False,
    progress: bool = True,
    progress_leave: bool = False,
) -> AutoLabelSummary:
    """Generate a new YOLO dataset from images and VLM detections."""

    images = dataset.images[:limit] if limit is not None else list(dataset.images)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    output_dataset = deepcopy(dataset)
    output_dataset.images = []
    class_lock = Lock()
    result_rows: list[dict[str, Any]] = []

    def annotate(image: YoloImage) -> tuple[YoloImage, dict[str, Any]]:
        request_prompt = prompt or auto_label_prompt(
            dataset.classes.names,
            attribute_schema=dataset.attributes.attributes if dataset.attributes else None,
            image_name=image.file_name,
        )
        response = provider.generate(
            request_prompt,
            images=[image.path],
            json_mode=True,
        )
        response_text = response.text if isinstance(response, VLMResponse) else str(response)
        boxes = parse_detection_response(response_text, image_size=(image.width, image.height) if image.width and image.height else None)
        with class_lock:
            annotations = [
                _to_annotation(
                    box,
                    output_dataset,
                    allow_new_classes=allow_new_classes,
                )
                for box in boxes
            ]
        output_image = deepcopy(image)
        output_image.annotations = annotations
        return output_image, {
            "image": str(image.path),
            "file_name": image.file_name,
            "annotations": len(annotations),
            "usage": response.usage if isinstance(response, VLMResponse) else None,
        }

    completed: list[tuple[YoloImage, dict[str, Any]] | Exception | None] = [None] * len(images)
    worker_count = normalize_workers(workers)
    if worker_count == 1:
        iterator = enumerate(images)
        progress_bar = create_progress_bar(
            total=len(images),
            desc="vlm auto-label",
            enabled=progress,
            leave=progress_leave,
        )
        try:
            for index, image in iterator:
                try:
                    completed[index] = annotate(image)
                except Exception as exc:
                    completed[index] = exc
                progress_bar.update()
        finally:
            progress_bar.close()
    else:
        progress_bar = create_progress_bar(
            total=len(images),
            desc="vlm auto-label",
            enabled=progress,
            leave=progress_leave,
        )
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(annotate, image): index for index, image in enumerate(images)}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        completed[index] = future.result()
                    except Exception as exc:
                        completed[index] = exc
                    progress_bar.update()
        finally:
            progress_bar.close()

    errors: list[dict[str, str]] = []
    succeeded = 0
    annotation_count = 0
    for image, result in zip(images, completed):
        if isinstance(result, Exception) or result is None:
            error = result if isinstance(result, Exception) else RuntimeError("unknown VLM error")
            errors.append({"image": str(image.path), "error": str(error)})
            result_rows.append({"image": str(image.path), "status": "error", "error": str(error)})
            continue
        output_image, row = result
        output_dataset.images.append(output_image)
        result_rows.append({"status": "ok", **row})
        succeeded += 1
        annotation_count += len(output_image.annotations)

    out_path = Path(out_root)
    result_path = out_path / "vlm_results.json"
    if not dry_run:
        write_yolo_dataset(
            output_dataset,
            out_path,
            copy_images=copy_images,
            keep_empty_labels=True,
            include_confidence=False,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            backup=False,
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "provider": getattr(provider, "name", type(provider).__name__),
                    "images": len(images),
                    "succeeded": succeeded,
                    "failed": len(errors),
                    "annotations": annotation_count,
                    "results": result_rows,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    else:
        result_path = None

    return AutoLabelSummary(
        images=len(images),
        succeeded=succeeded,
        failed=len(errors),
        annotations=annotation_count,
        out=None if dry_run else str(out_path),
        results=None if result_path is None else str(result_path),
        errors=errors,
    )


def _to_annotation(
    box: VLMBox,
    dataset: YoloDataset,
    *,
    allow_new_classes: bool,
) -> YoloAnnotation:
    class_id = box.class_id
    if class_id is None and box.class_name is not None:
        try:
            class_id = dataset.classes.id(box.class_name)
        except Exception:
            if not allow_new_classes:
                raise VLMOutputError(f"unknown VLM class: {box.class_name}")
            class_id = dataset.classes.ensure(box.class_name)
    if class_id is None:
        raise VLMOutputError("VLM detection has no class")
    dataset.classes.id(class_id)

    attributes: list[float] = []
    if box.attributes:
        if dataset.attributes is None:
            raise VLMOutputError("VLM returned attributes but dataset has no attribute schema")
        class_name = dataset.classes.name(class_id)
        for name in dataset.attributes.names_for_class(class_name):
            if name not in box.attributes:
                continue
            attributes.append(
                dataset.attributes.value_to_raw(
                    name,
                    box.attributes[name],
                    class_name=class_name,
                )
            )
    return YoloAnnotation(
        class_id=class_id,
        box=Box(*box.bbox),
        attributes=attributes,
        confidence=box.confidence,
    )


__all__ = ["AutoLabelSummary", "generate_yolo_labels"]
