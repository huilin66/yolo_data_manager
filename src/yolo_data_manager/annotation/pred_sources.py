"""Combine reviewed prediction sources into a single correction session."""

from collections.abc import Mapping
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from yolo_data_manager.core.models import is_image_file
from yolo_data_manager.runtime import iter_progress


def correct_prediction_sources(dataset, sources, *, progress=True, **options):
    # Keep the original GT indexing for the whole batch. Only prediction
    # indices are remapped, since pred1 in two runs denotes different rows.
    from yolo_data_manager.annotation.crop_correction import (
        _parse_error_crop_rule,
        _parse_error_crop_name,
        _prediction_at_index,
        correct_gt_labels_from_error_crops,
    )
    from yolo_data_manager.io.loader import parse_label_file

    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("pred_sources must be a non-empty list")
    prepared = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError("each pred_sources entry must be a mapping")
        if "pred_dir" not in source or "crops_dir" not in source:
            raise ValueError("each pred_sources entry requires pred_dir and crops_dir")
        pred_root = Path(source["pred_dir"])
        crops = source["crops_dir"]
        if not pred_root.is_dir():
            raise FileNotFoundError(f"prediction directory not found: {pred_root}")
        if not isinstance(crops, Mapping) or not crops:
            raise ValueError("each source crops_dir must be a non-empty target-class mapping")
        for directory, target in crops.items():
            if not Path(directory).is_dir():
                raise FileNotFoundError(f"crop directory not found: {directory}")
            value, _ = _parse_error_crop_rule(target, options.get("replace_gt_from_pred", False))
            if value is not None:
                dataset.class_id(value)
        prepared.append((pred_root, crops))

    conflicts: dict[tuple[str, int], Any] = {}
    summaries = []
    with TemporaryDirectory(prefix="ydm_pred_sources_") as temporary:
        temporary_root = Path(temporary)
        combined_predictions = temporary_root / "predictions"
        combined_predictions.mkdir()
        combined_crops = {}
        lines_by_stem: dict[str, list[str]] = {}
        indices = {}
        cached = {}
        for source_index, (pred_root, crops) in enumerate(iter_progress(
            prepared, enabled=progress, total=len(prepared), desc="prepare prediction sources", leave=False,
        )):
            files = {}
            for path in pred_root.rglob("*.txt"):
                files.setdefault(path.stem, []).append(path)
            count = 0
            for crop_index, (directory, target) in enumerate(crops.items()):
                target_value, local_replace = _parse_error_crop_rule(target, options.get("replace_gt_from_pred", False))
                destination = temporary_root / f"source_{source_index}" / str(crop_index)
                destination.mkdir(parents=True)
                combined_crops[destination] = target
                for crop in sorted(Path(directory).rglob("*")):
                    if not crop.is_file() or not is_image_file(crop):
                        continue
                    count += 1
                    parsed = _parse_error_crop_name(crop)
                    if parsed is None:
                        raise ValueError(f"invalid error crop filename: {crop}")
                    stem, pred_index, gt_index = parsed
                    prediction = None
                    needs_prediction = pred_index is not None and (
                        gt_index is None or local_replace
                    )
                    if needs_prediction:
                        candidates = files.get(stem, [])
                        if len(candidates) != 1:
                            raise ValueError(f"missing or ambiguous prediction txt: {pred_root}, image={stem}")
                        path = candidates[0]
                        if path not in cached:
                            cached[path] = parse_label_file(path, task=dataset.task, attributes=dataset.attributes)
                        prediction = _prediction_at_index(cached[path], pred_index)
                        if prediction is None:
                            raise ValueError(f"invalid prediction index: {crop}, source={path}")
                        key = (pred_root.resolve(), stem, pred_index)
                        if key not in indices:
                            lines = lines_by_stem.setdefault(stem, [])
                            lines.append(prediction.to_yolo_line(include_confidence=True))
                            indices[key] = len(lines)
                        pred_index = indices[key]
                    if gt_index is not None:
                        value = target_value
                        deleting = pred_index is None and (
                            options.get("delete_pred_none", False) or local_replace
                        )
                        if prediction is not None:
                            action = ("replace", prediction.to_yolo_line(include_confidence=False))
                        else:
                            action = ("class", None if deleting or value is None else dataset.class_id(value))
                        key = (stem, gt_index)
                        if key in conflicts and conflicts[key] != action:
                            raise ValueError(f"conflicting corrections for {stem}, GT {gt_index}: {crop}")
                        conflicts[key] = action
                    name = f"{stem}_pred{pred_index if pred_index is not None else 'none'}_gt{gt_index if gt_index is not None else 'none'}.jpg"
                    # Correction only reads crop filenames, never image bytes.
                    # Lightweight markers avoid copying the reviewed images.
                    (destination / name).touch()
            summaries.append({"source_index": source_index, "pred_dir": str(pred_root), "crop_files": count,
                              "crops_dir": {str(path): target for path, target in crops.items()}})
        for stem, lines in lines_by_stem.items():
            (combined_predictions / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result, report = correct_gt_labels_from_error_crops(
            dataset, combined_crops, pred_labels_dir=combined_predictions, **options,
        )
        result.pred_sources = summaries
        # Report the user's crop directories, not temporary staging paths.
        result.target_classes = {
            str(directory): {
                "target_class_id": dataset.class_id(value) if value is not None else None,
                "target_class_name": dataset.class_name(dataset.class_id(value)) if value is not None else None,
                "replace_gt_from_pred": local_replace,
            }
            for _, crops in prepared for directory, target in crops.items()
            for value, local_replace in [_parse_error_crop_rule(target, options.get("replace_gt_from_pred", False))]
        }
        if result.backup_dir is not None:
            from yolo_data_manager.io.backup import write_snapshot_metadata
            from yolo_data_manager.annotation.crop_correction import _backup_result_summary

            metadata_path = Path(result.backup_dir) / "backup_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            write_snapshot_metadata(
                result.backup_dir,
                dataset_root=metadata["dataset_root"],
                method=metadata["method"],
                created_at=metadata["created_at"],
                files=metadata["files"],
                result={
                    **_backup_result_summary(result.to_dict()),
                    "pred_sources": summaries,
                    "target_classes": result.target_classes,
                },
            )
        return result, report
