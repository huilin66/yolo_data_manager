from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from yolo_data_manager.core.models import AttributeSchema, ClassSchema, YoloDataset, YoloImage


@dataclass
class MergeReport:
    image_count: int = 0
    annotation_count: int = 0
    class_names: list[str] = field(default_factory=list)
    renamed_images: dict[str, str] = field(default_factory=dict)
    class_maps: list[dict[int, int]] = field(default_factory=list)


def merge_datasets(
    datasets: list[YoloDataset],
    root: str | Path,
    rename_duplicates: bool = True,
    source_prefix: bool = True,
) -> tuple[YoloDataset, MergeReport]:
    if not datasets:
        raise ValueError("at least one dataset is required")

    classes = ClassSchema([])
    class_maps: list[dict[int, int]] = []
    for dataset in datasets:
        mapping: dict[int, int] = {}
        for old_id, name in enumerate(dataset.classes.names):
            mapping[old_id] = classes.ensure(name)
        class_maps.append(mapping)

    images: list[YoloImage] = []
    used_names: set[str] = set()
    renamed_images: dict[str, str] = {}

    for dataset_idx, dataset in enumerate(datasets):
        mapping = class_maps[dataset_idx]
        for image in dataset.images:
            new_image = copy.deepcopy(image)
            original_name = new_image.file_name
            output_name = _unique_name(original_name, used_names, dataset_idx, rename_duplicates, source_prefix)
            if output_name != original_name:
                renamed_images[f"{dataset_idx}:{original_name}"] = output_name
            new_image.output_name = output_name
            used_names.add(output_name)
            for annotation in new_image.annotations:
                annotation.class_id = mapping.get(annotation.class_id, annotation.class_id)
            images.append(new_image)

    attributes = _merge_attributes([dataset.attributes for dataset in datasets])
    merged = YoloDataset(
        root=Path(root),
        images=images,
        classes=classes,
        attributes=attributes,
        task=datasets[0].task,
    )
    report = MergeReport(
        image_count=len(images),
        annotation_count=merged.annotation_count(),
        class_names=classes.names,
        renamed_images=renamed_images,
        class_maps=class_maps,
    )
    return merged, report


def _unique_name(
    name: str,
    used_names: set[str],
    dataset_idx: int,
    rename_duplicates: bool,
    source_prefix: bool,
) -> str:
    if name not in used_names:
        return f"d{dataset_idx}_{name}" if source_prefix else name
    if not rename_duplicates:
        raise ValueError(f"duplicate image name: {name}")
    path = Path(name)
    base = f"d{dataset_idx}_{path.stem}" if source_prefix else path.stem
    candidate = f"{base}{path.suffix}"
    counter = 1
    while candidate in used_names:
        candidate = f"{base}_{counter}{path.suffix}"
        counter += 1
    return candidate


def _merge_attributes(schemas: list[AttributeSchema | None]) -> AttributeSchema | None:
    merged: dict[str, object] = {}
    for schema in schemas:
        if schema is None:
            continue
        for name, value in schema.attributes.items():
            merged.setdefault(name, value)
    return AttributeSchema(merged) if merged else None
