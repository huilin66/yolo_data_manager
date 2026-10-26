"""Build a shared-label multimodal dataset from aligned single-modality datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import yaml

from yolo_data_manager.core.models import ClassSchema
from yolo_data_manager.core.schema import write_attribute_schema, write_class_schema
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.writer import write_split_file
from yolo_data_manager.runtime import iter_progress, normalize_workers, progress_stage
from yolo_data_manager.stats.report import infer_image_splits, _split_key_variants, _format_table
from yolo_data_manager.logging_utils import operation_scope, console_event


def _parallel(function, items, workers, progress, leave, desc):
    if workers == 1:
        return [function(item) for item in iter_progress(items, total=len(items), enabled=progress, leave=leave, desc=desc)]
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(function, item): index for index, item in enumerate(items)}
        results = [None] * len(items)
        for future in iter_progress(as_completed(futures), total=len(futures), enabled=progress, leave=leave, desc=desc):
            results[futures[future]] = future.result()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _image_spec(path):
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"cannot read image: {path}")
    return (array.shape[1], array.shape[0], 1 if array.ndim == 2 else array.shape[2], str(array.dtype))


def _zero_spec(source, images, workers, progress, leave):
    explicit = source.get("zero_fill")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError("zero_fill must specify width, height, channels and dtype")
        spec = tuple(explicit[key] for key in ("width", "height", "channels", "dtype"))
    else:
        specs = set(_parallel(_image_spec, [image.path for image in images], workers, progress, leave, "merge modalities inspect zero-fill type"))
        if len(specs) != 1:
            raise ValueError("missing modality has no uniform image specification; supply source zero_fill={width, height, channels, dtype}")
        spec = next(iter(specs))
    width, height, channels, dtype = spec
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height, channels)):
        raise ValueError("zero_fill dimensions and channels must be positive integers")
    if channels not in (1, 3, 4) or dtype not in ("uint8", "uint16", "float32", "int32"):
        raise ValueError("zero_fill supports 1/3/4 channels and uint8/uint16/float32/int32")
    if dtype in ("float32", "int32") and channels != 1:
        raise ValueError("float32/int32 zero_fill currently requires one channel")
    return spec


def _write_image(job):
    destination, source, spec = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copy2(source, destination)
        return
    width, height, channels, dtype = spec
    shape = (height, width) if channels == 1 else (height, width, channels)
    array = np.zeros(shape, dtype=dtype)
    if dtype in ("float32", "int32"):
        Image.fromarray(array).save(destination)
    else:
        success, encoded = cv2.imencode(".png", array)
        if not success:
            raise ValueError(f"cannot encode zero image: {destination}")
        encoded.tofile(str(destination))


def _merge_attributes(datasets):
    result = None
    for dataset in datasets:
        schema = dataset.attributes
        if schema is None:
            if any(ann.attributes for image in dataset.images for ann in image.annotations):
                raise ValueError("attributes require an attribute schema")
            continue
        if result is None:
            result = deepcopy(schema)
        elif result.class_scoped and schema.class_scoped:
            for name, attributes in schema.attributes.items():
                if name in result.attributes and result.attributes[name] != attributes:
                    raise ValueError(f"conflicting attribute schema for class: {name}")
                result.attributes[name] = deepcopy(attributes)
        elif result.attributes != schema.attributes:
            raise ValueError("conflicting attribute schemas across modalities")
    return result


def _split_membership(dataset):
    # Reject ambiguous memberships rather than silently losing split provenance.
    assignments = {}
    for split in ("train", "val", "test"):
        path = Path(dataset.root) / f"{split}.txt"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                for key in _split_key_variants(Path(dataset.root), line):
                    assignments.setdefault(key, set()).add(split)
    for image in dataset.images:
        candidates = set().union(*(assignments.get(key, set()) for key in _split_key_variants(Path(dataset.root), image.path)))
        if len(candidates) > 1:
            raise ValueError(f"conflicting split membership: {image.path}")
    return infer_image_splits(dataset)


def merge_modalities(
    sources: Sequence[Mapping[str, Any]],
    out_data: str | Path,
    *,
    unmatched: str = "keep",
    merge_labels: bool = True,
    inherit_splits: bool = False,
    deduplicate: bool = False,
    dedup_iou: float = 0.9,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge equal-status sources without requiring a manager instance.

    Operation logs live in out_data/ydm_log, including dry runs. Validate
    source/output separation before logging so input datasets stay read-only.
    """
    output = Path(out_data).expanduser().resolve()
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or len(sources) < 2:
        raise ValueError("sources requires at least two source mappings")
    for source in sources:
        if not isinstance(source, Mapping) or "root" not in source:
            raise ValueError("each source requires root")
        root = Path(source["root"]).expanduser().resolve()
        if root.is_file() and root.suffix.lower() in (".yaml", ".yml"):
            from yolo_data_manager.scripting import _resolve_manager_root
            root, _, _ = _resolve_manager_root(root)
            root = root.resolve()
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("output must be outside the source dataset directories")
    _check_output_directory(output)
    options = dict(
        unmatched=unmatched, merge_labels=merge_labels, inherit_splits=inherit_splits, deduplicate=deduplicate,
        dedup_iou=dedup_iou, workers=workers, progress=progress,
        progress_leave=progress_leave, dry_run=dry_run,
    )
    with operation_scope(output, "dataset.merge_modalities", {"sources": sources, **options}):
        report = _merge_modalities(sources, output, **options)
        console_event("INFO", format_merge_summary(report), stream=sys.stdout)
        return report


def format_merge_summary(report: Mapping[str, Any]) -> str:
    """Show physical image counts separately from shared-label scene counts."""
    summary = report["summary"]
    rows = [
        [f"input:{item['name']}", item["images"], item["scenes"], item["classes"],
         item["used_classes"], item["annotations"], item["label_files"], item["empty_labels"]]
        for item in summary["inputs"]
    ]
    item = summary["output"]
    rows.append(["output", item["images"], item["scenes"], item["classes"],
                 item["used_classes"], item["annotations"], item["label_files"], item["empty_labels"]])
    table = _format_table(
        ["dataset", "images", "scenes", "classes", "used_classes", "annotations", "labels", "empty_labels"], rows,
    )
    modality_table = _format_table(
        ["modality", "input_images", "copied", "zero_filled", "discarded", "output_images"],
        [[item[key] for key in ("name", "input_images", "copied", "zero_filled", "discarded", "output_images")]
         for item in summary["modalities"]],
    )
    return (
        f"Merge modalities summary{' (dry run / planned output)' if report['dry_run'] else ''}\n{table}\n\n"
        f"Image changes by modality\n{modality_table}\n"
        f"Common scenes: {report['common_scenes']}; discarded scenes: {report['discarded_scenes']}; "
        f"deduplicated annotations: {report['deduplicated']}\n"
        "Output annotations are shared across modalities and counted once; classes includes unused schema entries.\n"
        f"Output: {report['out']}"
    )


def _check_output_directory(output: Path) -> None:
    if output.exists() and (
        not output.is_dir()
        or any(path.name != "ydm_log" or not path.is_dir() for path in output.iterdir())
    ):
        raise FileExistsError(f"output must be new or contain only ydm_log: {output}")


def _merge_modalities(
    sources: Sequence[Mapping[str, Any]],
    out_data: str | Path,
    *,
    unmatched: str = "keep",
    merge_labels: bool = True,
    inherit_splits: bool = False,
    deduplicate: bool = False,
    dedup_iou: float = 0.9,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Align images by stem; coordinates must already refer to aligned views.

    Each source defines root, optional name (image1, image2, ...), loader
    images_dir/labels_dir/layout/class_file/attribute_file/task options, and
    optional zero_fill={width,height,channels,dtype} for heterogeneous images.
    Existing non-empty output directories are rejected to avoid stale data.
    """
    if isinstance(sources, (str, bytes, Mapping)) or len(sources) < 2:
        raise ValueError("sources requires at least two source mappings")
    if unmatched not in ("keep", "drop", "zero_fill"):
        raise ValueError("unmatched must be keep, drop or zero_fill")
    if not 0 < dedup_iou <= 1:
        raise ValueError("dedup_iou must be in (0, 1]")
    worker_count = normalize_workers(workers)
    output = Path(out_data).expanduser().resolve()
    configs, names = [], []
    reserved = {"labels", "ydm_log", "dataset.yaml", "multimodal.yaml", "merge_report.json", "class.txt", "attribute.yaml", "train.txt", "val.txt", "test.txt"}
    for index, source in enumerate(sources, 1):
        if not isinstance(source, Mapping) or "root" not in source:
            raise ValueError("each source requires root")
        config = dict(source)
        root = Path(config["root"]).expanduser().resolve()
        if root.is_file() and root.suffix.lower() in (".yaml", ".yml"):
            from yolo_data_manager.scripting import _resolve_manager_root
            root, class_file, _ = _resolve_manager_root(root)
            config.setdefault("class_file", class_file)
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"source dataset not found: {root}")
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("output must be outside the source dataset directories")
        name = str(config.get("name", f"image{index}"))
        if not name or name in (".", "..") or any(char in name for char in '/\\<>:"|?*') or name.endswith((".", " ")) or name.casefold() in reserved or name.casefold() in {value.casefold() for value in names}:
            raise ValueError(f"invalid, reserved or duplicate modality folder name: {name}")
        names.append(name)
        config["root"] = root
        configs.append(config)
    _check_output_directory(output)

    progress_stage("merge modalities load sources", enabled=progress)
    datasets = []
    for config in configs:
        params = {key: config[key] for key in ("images_dir", "labels_dir", "layout", "class_file", "attribute_file", "task") if key in config}
        for key in ("class_file", "attribute_file"):
            if params.get(key) is not None and not Path(params[key]).is_absolute():
                params[key] = config["root"] / params[key]
        datasets.append(load_yolo_dataset(config["root"], workers=worker_count, progress=progress, progress_leave=progress_leave, **params))
    progress_stage("merge modalities align scenes and labels", enabled=progress)
    indexes, memberships = [], []
    classes = ClassSchema([])
    for dataset in datasets:
        index = {}
        for image in dataset.images:
            key = image.stem.casefold()
            if key in index:
                raise ValueError(f"duplicate image stem in source: {image.stem}, {dataset.root}")
            index[key] = image
        indexes.append(index)
        memberships.append(_split_membership(dataset) if inherit_splits else {})
        for name in dataset.classes.names:
            classes.ensure(name)
    attributes = _merge_attributes(datasets if merge_labels else datasets[:1])
    all_stems = set().union(*(set(index) for index in indexes))
    common = set.intersection(*(set(index) for index in indexes))
    selected = sorted(common if unmatched == "drop" else all_stems)
    missing = {name: sorted(set(selected) - set(index)) for name, index in zip(names, indexes)}
    zero_specs = {}
    if unmatched == "zero_fill":
        for name, config, dataset in zip(names, configs, datasets):
            if missing[name]:
                zero_specs[name] = _zero_spec(config, dataset.images, worker_count, progress, progress_leave)
    tasks, labels, scene_splits, reference_paths, scene_names = [], {}, {}, {}, {}
    removed, input_annotations = 0, 0
    primary_labels_missing = []
    for stem in iter_progress(selected, total=len(selected), enabled=progress, leave=progress_leave, desc="merge modalities prepare scenes"):
        available = [(idx, index[stem]) for idx, index in enumerate(indexes) if stem in index]
        scene_name = available[0][1].stem
        scene_names[stem] = scene_name
        if not merge_labels:
            primary = indexes[0].get(stem)
            if primary is None or primary.label_path is None or not primary.label_path.is_file():
                primary_labels_missing.append(scene_name)
        splits = {memberships[idx][id(image)] for idx, image in available if id(image) in memberships[idx]}
        if len(splits) > 1:
            raise ValueError(f"conflicting train/val/test assignments for scene: {scene_name}")
        scene_splits[stem] = next(iter(splits), None)
        annotations = []
        for idx, image in available:
            name = names[idx]
            destination = output / name / f"{scene_name}{image.path.suffix}"
            tasks.append((destination, image.path, None))
            reference_paths.setdefault(stem, destination)
            if not merge_labels and idx != 0:
                continue
            dataset = datasets[idx]
            for annotation in image.annotations:
                ann = deepcopy(annotation)
                old_name = dataset.class_name(ann.class_id)
                if ann.class_id < 0 or ann.class_id >= len(dataset.classes.names):
                    raise ValueError(f"class id outside source schema: {image.label_path}, {ann.class_id}")
                ann.class_id = classes.id(old_name)
                if attributes is not None and ann.attributes:
                    if dataset.attributes is None:
                        raise ValueError(f"missing attribute schema: {dataset.root}")
                    source_names = dataset.attributes.names_for_class(old_name)
                    target_names = attributes.names_for_class(old_name)
                    values = dict(zip(source_names, ann.attributes))
                    if len(values) != len(ann.attributes) or set(values) != set(target_names):
                        raise ValueError(f"attribute values do not match schema: {image.label_path}")
                    ann.attributes = [values[key] for key in target_names]
                input_annotations += 1
                if deduplicate and any(_duplicate(ann, previous, dedup_iou) for previous in annotations):
                    removed += 1
                    continue
                annotations.append(ann)
        if unmatched == "zero_fill":
            for name, index in zip(names, indexes):
                if stem not in index:
                    spec = zero_specs[name]
                    extension = ".tif" if spec[3] in ("float32", "int32") else ".png"
                    tasks.append((output / name / f"{scene_name}{extension}", None, spec))
        labels[scene_name] = [ann.to_yolo_line(include_confidence=False) for ann in annotations]

    has_splits = inherit_splits and (any(scene_splits.values()) or any((Path(dataset.root) / f"{split}.txt").is_file() for dataset in datasets for split in ("train", "val", "test")))
    report = {
        "out": str(output), "image_dirs": names, "scenes": len(selected),
        "common_scenes": len(common), "discarded_scenes": len(all_stems) - len(selected),
        "unmatched": unmatched, "missing_modalities": {name: len(stems) for name, stems in missing.items()},
        "missing_scenes": {name: [scene_names[stem] for stem in stems] for name, stems in missing.items()},
        "zero_filled": sum(len(stems) for stems in missing.values()) if unmatched == "zero_fill" else 0,
        "zero_specs": zero_specs, "merge_labels": merge_labels, "inherit_splits": inherit_splits, "deduplicate": deduplicate,
        "dedup_iou": dedup_iou, "input_annotations": input_annotations,
        "deduplicated": removed, "annotations": sum(len(lines) for lines in labels.values()),
        "empty_labels": sum(not lines for lines in labels.values()),
        "primary_labels_missing": primary_labels_missing,
        "splits": {split: sum(value == split for value in scene_splits.values()) for split in ("train", "val", "test")},
        "unassigned_scenes": sum(value is None for value in scene_splits.values()),
        "sources": [{"root": str(config["root"]), "name": name} for name, config in zip(names, configs)],
        "dry_run": dry_run,
    }
    input_summaries = []
    modality_summaries = []
    for name, dataset, index in zip(names, datasets, indexes):
        copied = len(set(selected) & set(index))
        filled = len(missing[name]) if unmatched == "zero_fill" else 0
        input_summaries.append({
            "name": name, "root": str(dataset.root), "images": len(dataset.images),
            "scenes": len(index), "classes": len(dataset.classes.names),
            "used_classes": len({ann.class_id for image in dataset.images for ann in image.annotations}),
            "annotations": dataset.annotation_count(),
            "label_files": sum(image.label_path is not None for image in dataset.images),
            "empty_labels": sum(image.label_path is not None and not image.annotations for image in dataset.images),
        })
        modality_summaries.append({
            "name": name, "input_images": len(dataset.images), "copied": copied,
            "zero_filled": filled, "discarded": len(dataset.images) - copied,
            "output_images": copied + filled,
        })
    report["summary"] = {
        "inputs": input_summaries,
        "output": {
            "root": str(output), "images": len(tasks), "scenes": len(selected),
            "classes": len(classes.names),
            "used_classes": len({line.split()[0] for lines in labels.values() for line in lines}),
            "annotations": report["annotations"], "label_files": len(labels),
            "empty_labels": report["empty_labels"],
        },
        "modalities": modality_summaries,
    }
    if dry_run:
        return report
    progress_stage("merge modalities write output", enabled=progress)
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name).mkdir(exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    _parallel(_write_image, tasks, worker_count, progress, progress_leave, "merge modalities copy and zero-fill")
    def write_label(item):
        stem, lines = item
        (output / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    _parallel(write_label, list(labels.items()), worker_count, progress, progress_leave, "merge modalities write labels")
    write_class_schema(classes, output / "class.txt")
    write_attribute_schema(attributes, output / "attribute.yaml")
    configuration = {"path": str(output), "image_dirs": names, "labels_dir": "labels", "names": classes.names, "nc": len(classes.names)}
    if has_splits:
        for split in ("train", "val", "test"):
            write_split_file([str(reference_paths[stem]) for stem in selected if scene_splits[stem] == split], output / f"{split}.txt")
            configuration[split] = f"{split}.txt"
    text = yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True)
    (output / "dataset.yaml").write_text(text, encoding="utf-8")
    (output / "multimodal.yaml").write_text(text, encoding="utf-8")
    (output / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _duplicate(first, second, threshold):
    if first.class_id != second.class_id or first.attributes != second.attributes:
        return False
    # Polygon deduplication only removes identical polygons, avoiding false
    # suppression of distinct shapes with the same enclosing rectangle.
    if first.polygon is not None or second.polygon is not None:
        return first.polygon is not None and second.polygon is not None and first.polygon.points == second.polygon.points
    a, b = first.geometry_box(), second.geometry_box()
    if a is None or b is None:
        return False
    width = max(0, min(a.cx + a.width / 2, b.cx + b.width / 2) - max(a.cx - a.width / 2, b.cx - b.width / 2))
    height = max(0, min(a.cy + a.height / 2, b.cy + b.height / 2) - max(a.cy - a.height / 2, b.cy - b.height / 2))
    intersection = width * height
    union = a.width * a.height + b.width * b.height - intersection
    return union > 0 and intersection / union >= threshold
