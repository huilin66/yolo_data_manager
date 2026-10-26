from __future__ import annotations

from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.multimodal import MultimodalYoloDataset
from yolo_data_manager.stats.compute import compute_stats
from yolo_data_manager.stats.export import write_stats_plots
from yolo_data_manager.runtime import progress_stage


def compute_multimodal_stats(
    dataset: MultimodalYoloDataset,
    *,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, object]:
    """Compute shared annotation statistics once and image statistics per modality."""

    complete = dataset.complete_scenes
    primary = _primary_modality(dataset)
    annotation_dataset = dataset.to_yolo_dataset(primary) if primary is not None else None
    if annotation_dataset is not None:
        progress_stage("stats compute shared annotations", enabled=progress)
        annotation_stats = compute_stats(
            annotation_dataset,
            progress=progress,
            progress_leave=progress_leave,
        )
    else:
        annotation_stats = _empty_annotation_stats(dataset)

    modality_stats: dict[str, dict[str, object]] = {}
    for modality in dataset.modalities:
        view = dataset.to_yolo_dataset(modality)
        progress_stage(f"stats compute modality {modality}", enabled=progress)
        modality_stats[modality] = {
            "image_count": len(view.images),
            "missing_scene_count": len(complete) - len(view.images),
            "image_type_summary": dataset.image_type_summary.get(modality, {}),
            "stats": compute_stats(
                view,
                progress=progress,
                progress_leave=progress_leave,
            ),
        }

    return {
        "report_type": "multimodal_stats",
        "scene_count": len(complete),
        "candidate_scene_count": len(dataset.scenes),
        "annotation_stats": annotation_stats,
        "modalities": modality_stats,
        "alignment": dataset.alignment_report.to_dict(),
    }


def write_multimodal_stats_plots(
    dataset: MultimodalYoloDataset,
    out_dir: str | Path,
    *,
    stats_list: str | Iterable[str] | None = None,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, str]:
    """Write one plot set per modality from an already-associated dataset."""

    output = Path(out_dir)
    written: dict[str, str] = {}
    for modality in dataset.modalities:
        modality_dir = output / modality
        progress_stage(f"stats plots modality {modality}", enabled=progress)
        write_stats_plots(
            dataset.to_yolo_dataset(modality),
            modality_dir,
            stats_list=stats_list,
            progress=progress,
            progress_leave=progress_leave,
        )
        written[modality] = str(modality_dir)
    return written


def _primary_modality(dataset: MultimodalYoloDataset) -> str | None:
    for modality in dataset.required_modalities:
        return modality
    return next(iter(dataset.modalities), None)


def _empty_annotation_stats(dataset: MultimodalYoloDataset) -> dict[str, object]:
    return {
        "image_count": 0,
        "label_count": 0,
        "orphan_label_count": 0,
        "annotation_count": 0,
        "empty_image_count": 0,
        "class_counts": {name: 0 for name in dataset.classes.names},
        "class_id_counts": {},
        "objects_per_image": {"count": 0, "min": None, "max": None, "mean": None},
    }
