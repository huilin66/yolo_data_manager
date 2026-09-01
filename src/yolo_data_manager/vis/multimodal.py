from __future__ import annotations

from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.multimodal import MultimodalYoloDataset
from yolo_data_manager.vis.renderer import crop_dataset, render_dataset


def render_multimodal_dataset(
    dataset: MultimodalYoloDataset,
    out_dir: str | Path,
    *,
    style: str = "cv2",
    modalities: Iterable[str] | None = None,
    limit: int | None = None,
    show_confidence: bool = False,
    confidence_threshold: float | None = None,
    mask_alpha: int = 64,
    fill_mask: bool = True,
    show_attributes: bool = False,
    show_txt_id: bool = False,
    filter_no_attributes: bool = False,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
    att_seperate: bool = False,
) -> dict[str, int]:
    """Render every selected modality using the same already-parsed annotations."""

    selected = _selected_modalities(dataset, modalities)
    output = Path(out_dir)
    counts: dict[str, int] = {}
    for modality in selected:
        view = dataset.to_yolo_dataset(modality)
        render_dataset(
            view,
            output / modality,
            style=style,
            limit=limit,
            show_confidence=show_confidence,
            confidence_threshold=confidence_threshold,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attributes=show_attributes,
            show_txt_id=show_txt_id,
            filter_no_attributes=filter_no_attributes,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            att_seperate=att_seperate,
            att_seperate_dir=output.parent / "att_seperate" / modality,
        )
        counts[modality] = len(view.images[:limit] if limit is not None else view.images)
    return counts


def crop_multimodal_dataset(
    dataset: MultimodalYoloDataset,
    out_dir: str | Path,
    *,
    style: str = "cv2",
    modalities: Iterable[str] | None = None,
    keep_shape: bool = False,
    min_size: int = 1,
    padding: int | float = 0,
    confidence_threshold: float | None = None,
    by_attribute: bool = False,
    filter_no_attributes: bool = True,
    workers: int = 8,
    progress: bool = True,
    progress_leave: bool = False,
) -> dict[str, int]:
    """Crop objects for each selected modality without reparsing label files."""

    selected = _selected_modalities(dataset, modalities)
    output = Path(out_dir)
    counts: dict[str, int] = {}
    for modality in selected:
        counts[modality] = crop_dataset(
            dataset.to_yolo_dataset(modality),
            output / modality,
            style=style,
            keep_shape=keep_shape,
            min_size=min_size,
            padding=padding,
            confidence_threshold=confidence_threshold,
            by_attribute=by_attribute,
            filter_no_attributes=filter_no_attributes,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
    return counts


def _selected_modalities(dataset: MultimodalYoloDataset, values: Iterable[str] | None) -> list[str]:
    selected = list(dataset.modalities) if values is None else list(values)
    unknown = [name for name in selected if name not in dataset.modalities]
    if unknown:
        raise ValueError(f"unknown modality type(s): {', '.join(unknown)}")
    return selected
