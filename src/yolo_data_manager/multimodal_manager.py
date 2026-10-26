from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from yolo_data_manager.core.models import TASK_AUTO, YoloDataset, YoloImage
from yolo_data_manager.core.multimodal import (
    AlignmentReport,
    MultimodalScene,
    MultimodalYoloDataset,
)
from yolo_data_manager.core.schema import write_attribute_schema, write_class_schema
from yolo_data_manager.dataset.select import read_selection_file
from yolo_data_manager.dataset.split import (
    SplitIncludeList,
    class_counts_for_images,
    split_dataset,
)
from yolo_data_manager.io.backup import ensure_source_labels_backup
from yolo_data_manager.io.image_conversion import convert_multimodal_images_to_uint8
from yolo_data_manager.io.loader import parse_label_file
from yolo_data_manager.io.multimodal import load_multimodal_yolo_dataset
from yolo_data_manager.io.output_paths import default_dataset_output, ydm_dir
from yolo_data_manager.io.writer import move_existing_split_files_to_backup, write_split_file
from yolo_data_manager.logging_utils import operation_scope
from yolo_data_manager.runtime import iter_progress, normalize_workers
from yolo_data_manager.scripting import YoloManager
from yolo_data_manager.stats.multimodal import (
    compute_multimodal_stats,
    write_multimodal_stats_plots,
)
from yolo_data_manager.stats.report import write_json_report
from yolo_data_manager.vis.multimodal import (
    crop_multimodal_dataset,
    render_multimodal_dataset,
)


def _logged_operation(operation: str):
    def decorator(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with operation_scope(
                self.root,
                operation,
                {"positional": args, "parameters": kwargs},
            ):
                return function(self, *args, **kwargs)

        return wrapped

    return decorator


def _reference_modality_name(
    image_dirs: Sequence[str | Path],
    image_params: Mapping[str, Mapping[str, object]] | None,
) -> str:
    if image_params:
        return str(next(iter(image_params)))
    if not image_dirs:
        raise ValueError("image_dirs must contain at least one image folder")
    return Path(image_dirs[0]).name or str(image_dirs[0])


def _resolve_reference_image_dir(
    image_dirs: Sequence[str | Path],
    image_params: Mapping[str, Mapping[str, object]] | None,
    reference_modality: str | Path | None,
) -> str | Path:
    if not image_dirs:
        raise ValueError("image_dirs must contain at least one image folder")

    reference = (
        str(reference_modality)
        if reference_modality is not None
        else _reference_modality_name(image_dirs, image_params)
    )
    if image_params and reference in image_params:
        config = image_params[reference]
        if not isinstance(config, Mapping):
            raise TypeError(f"image_params[{reference!r}] must be a mapping")
        return config.get("dir", config.get("path", reference))  # type: ignore[return-value]

    for image_dir in image_dirs:
        if str(image_dir) == reference or Path(image_dir).name == reference:
            return image_dir
    raise ValueError(
        f"reference_modality {reference!r} does not identify an image directory"
    )


class MultiModalYoloManager(YoloManager):
    """Modality-aware loader/cache for a shared-label YOLO dataset.

    Multimodality is a property of the dataset, not a separate business
    workflow. This adapter associates several image folders with one label set
    and reuses the same label/query/evaluation APIs as :class:`YoloManager`.
    Operations that write or inspect every modality are implemented here so
    they do not silently fall back to a single image directory.
    """

    def __init__(
        self,
        root: str | Path,
        image_dirs: Sequence[str | Path],
        *,
        image_params: Mapping[str, Mapping[str, object]] | None = None,
        labels_dir: str | Path = "labels",
        label_params: Mapping[str, object] | None = None,
        class_file: str | Path | None = None,
        attribute_file: str | Path | None = None,
        task: str = TASK_AUTO,
        reference_modality: str | Path | None = None,
        read_image_size: bool = True,
        init_load: bool = False,
        init_check: bool | str | Path = False,
        progress: bool = True,
        progress_leave: bool = False,
    ) -> None:
        reference_dir = _resolve_reference_image_dir(
            image_dirs,
            image_params,
            reference_modality,
        )
        super().__init__(
            root,
            layout="flat",
            task=task,
            images_dir=reference_dir,
            labels_dir=labels_dir,
            class_file=class_file,
            attribute_file=attribute_file,
            init_layout=False,
            init_check=False,
        )
        self.image_dirs = tuple(image_dirs)
        self.image_params = dict(image_params or {})
        self.labels_dir = labels_dir
        self.label_params = dict(label_params or {})
        if class_file is not None:
            self.class_file = class_file
        if attribute_file is not None:
            self.attribute_file = attribute_file
        self.task = task
        self.reference_modality = (
            str(reference_modality)
            if reference_modality is not None
            else _reference_modality_name(image_dirs, image_params)
        )
        self.reference_images_dir = reference_dir
        self.read_image_size = read_image_size
        self.progress = progress
        self.progress_leave = progress_leave

        if init_load:
            self.load()
        if init_check:
            self.check(out=init_check if isinstance(init_check, (str, Path)) else None)

    @property
    def dataset(self) -> MultimodalYoloDataset:
        return self.load()

    @property
    def alignment_report(self) -> AlignmentReport:
        return self.load().alignment_report

    @property
    def output_log(self) -> Path:
        """Default daily operation-log directory."""

        return Path(self.root) / "ydm_log"

    _SINGLE_MODAL_OUTPUT_TASKS = frozenset(
        {
            "dataset.normalize",
            "dataset.extract_split",
            "convert.seg2det",
            "convert.pseudo",
            "convert.resize",
            "convert.filename_remap",
        }
    )
    _LABEL_OUTPUT_TASKS = frozenset(
        {
            "ann.delete_class",
            "ann.replace_class",
            "ann.merge_class",
            "ann.rename_class",
            "ann.apply_map",
            "ann.correct_from_crops",
            "ann.correct_from_error_crops",
            "ann.att_correct_from_crops",
            "ann.att_correct_from_error_crops",
            "ann.set_attr",
            "ann.delete_attr",
        }
    )

    def _run(self, task: str, **params: Any) -> int:
        """Reuse the base task runner without allowing unsafe flat outputs."""

        if task in self._SINGLE_MODAL_OUTPUT_TASKS:
            raise NotImplementedError(
                f"{task} is not available on MultiModalYoloManager because its "
                "default writer produces a single-modality dataset; run it on "
                "each source before merge or add a modality-aware conversion"
            )
        if task == "dataset.filter" and (
            params.get("out") is not None or params.get("out_data") is not None
        ):
            raise NotImplementedError(
                "dataset.filter with out/out_data is not available on "
                "MultiModalYoloManager; omit the output argument to update the "
                "shared labels in place, or filter before merge"
            )
        if task in self._LABEL_OUTPUT_TASKS and (
            params.get("out") is not None or params.get("out_data") is not None
        ):
            raise NotImplementedError(
                f"{task} output mode is not available on MultiModalYoloManager "
                "because it would copy only the reference modality; omit the "
                "output argument for a backed-up shared-label update"
            )
        return super()._run(task, **params)

    def ann_update_from_map(self, class_map: Mapping[str, Any], *args: Any, **kwargs: Any) -> int:
        """Apply a class map to the shared labels in place.

        The inherited implementation is safe for the shared-label in-place
        operation.  Its flat ``out_data`` writer is intentionally rejected so
        it cannot create a dataset containing only the reference modality.
        """

        if kwargs.get("out_data") is not None:
            raise NotImplementedError(
                "ann_update_from_map(out_data=...) is not available on "
                "MultiModalYoloManager; update shared labels in place or run "
                "the operation before merge"
            )
        return super().ann_update_from_map(class_map, *args, **kwargs)

    def ann_att_update_from_map(
        self,
        attribute_map: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Apply an attribute map to the shared labels in place."""

        if kwargs.get("out_data") is not None:
            raise NotImplementedError(
                "ann_att_update_from_map(out_data=...) is not available on "
                "MultiModalYoloManager; update shared labels in place or run "
                "the operation before merge"
            )
        return super().ann_att_update_from_map(attribute_map, *args, **kwargs)

    def load(
        self,
        *,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> MultimodalYoloDataset:
        """Load once, or explicitly rebuild the cached multimodal association."""

        if self._dataset is None or reload:
            self._dataset = load_multimodal_yolo_dataset(
                self.root,
                self.image_dirs,
                image_params=self.image_params,
                labels_dir=self.labels_dir,
                label_params=self.label_params,
                class_file=self.class_file,
                attribute_file=self.attribute_file,
                task=self.task,
                read_image_size=self.read_image_size,
                progress=self.progress if progress is None else progress,
                progress_leave=self.progress_leave
                if progress_leave is None
                else progress_leave,
            )
        return self._dataset

    @_logged_operation("multimodal_check")
    def check(
        self,
        *,
        out: str | Path | None = None,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> dict[str, object]:
        """Return the multimodal association report without rereading cached data."""

        dataset = self.load(
            reload=reload, progress=progress, progress_leave=progress_leave
        )
        report = dataset.alignment_report
        payload = {
            "report_type": "multimodal_check",
            "ok": not any(issue.level == "error" for issue in report.issues),
            "scene_count": len(dataset.complete_scenes),
            "image_type_summary": dataset.image_type_summary,
            **report.to_dict(),
        }
        report_path = Path(out) if out is not None else ydm_dir(self.root, "quality") / "multimodal_check.json"
        write_json_report(payload, report_path)
        _print_check_summary(payload, report_path)
        return payload

    @_logged_operation("multimodal_split")
    def dataset_split(
        self,
        *,
        train: float = 0.8,
        val: float = 0.2,
        test: float = 0.0,
        seed: int = 233,
        out: str | Path | None = None,
        out_data: str | Path | None = None,
        backup_dir: str | Path | None = None,
        absolute_paths: bool = True,
        train_include_list: SplitIncludeList = None,
        val_include_list: SplitIncludeList = None,
        val_source: SplitIncludeList = None,
        ensure_class_presence: bool = True,
        require_labels: bool = False,
        workers: int = 8,
        progress: bool | None = None,
        progress_leave: bool | None = None,
        reload: bool = False,
    ) -> dict[str, object]:
        """Split scenes once and write the same membership for every modality.

        The split algorithm is shared with :class:`YoloManager`, but its input
        is a scene-level single view.  A materialized ``out_data`` keeps all
        configured image directories and the shared labels directory.
        """

        if out is not None and out_data is not None:
            raise ValueError("out and out_data are mutually exclusive")
        progress_enabled = self.progress if progress is None else progress
        progress_leave_enabled = (
            self.progress_leave if progress_leave is None else progress_leave
        )
        dataset = self.load(
            reload=reload,
            progress=progress_enabled,
            progress_leave=progress_leave_enabled,
        )
        split_view = _multimodal_split_view(
            dataset,
            reference_modality=self.reference_modality,
        )
        splits = split_dataset(
            split_view,
            train=train,
            val=val,
            test=test,
            seed=seed,
            absolute_paths=absolute_paths,
            train_include_list=train_include_list,
            val_include_list=val_include_list,
            val_source=val_source,
            ensure_class_presence=ensure_class_presence,
            require_labels=require_labels,
        )
        split_scenes = {
            split_name: _scenes_for_split_names(split_view, names)
            for split_name, names in splits.items()
        }
        output_root = Path(out_data) if out_data is not None else Path(out or self.root)
        source_root = Path(self.root).resolve()

        if out_data is not None:
            _write_multimodal_split_dataset(
                dataset,
                split_scenes,
                output_root,
                labels_dir=self.labels_dir,
                reference_modality=self.reference_modality,
                absolute_paths=absolute_paths,
                include_test=test > 0,
                copy_images=True,
                workers=workers,
                progress=progress_enabled,
                progress_leave=progress_leave_enabled,
            )
            backup_snapshot = None
            output_splits = _output_split_paths(
                dataset,
                split_scenes,
                output_root,
                reference_modality=self.reference_modality,
                absolute_paths=absolute_paths,
            )
        else:
            backup_snapshot = None
            existing_split_files = [
                output_root / f"{split_name}.txt"
                for split_name in ("train", "val", "test")
                if (output_root / f"{split_name}.txt").is_file()
            ]
            if output_root.resolve() == source_root and existing_split_files:
                resolved_backup_dir = Path(backup_dir) if backup_dir is not None else Path(self.root) / "labels_backup"
                ensure_source_labels_backup(
                    self.root,
                    resolved_backup_dir,
                    source_paths=(
                        scene.label_path
                        for scene in dataset.scenes.values()
                        if scene.label_path is not None
                    ),
                )
                backup_snapshot = move_existing_split_files_to_backup(
                    output_root,
                    resolved_backup_dir,
                    method="multimodal.split",
                    result={
                        "action": "regenerate_multimodal_split_files",
                        "splits": {name: len(values) for name, values in splits.items()},
                    },
                )
            output_splits = _output_split_paths(
                dataset,
                split_scenes,
                output_root,
                reference_modality=self.reference_modality,
                absolute_paths=absolute_paths,
            )
            for split_name, names in output_splits.items():
                write_split_file(names, output_root / f"{split_name}.txt")

        if output_root.resolve() == source_root:
            self._invalidate_dataset()
        payload: dict[str, object] = {
            "report_type": "multimodal_split",
            "out": str(output_root),
            "out_data": str(output_root) if out_data is not None else None,
            "splits": {name: len(values) for name, values in output_splits.items()},
            "backup_dir": str(backup_snapshot) if backup_snapshot is not None else None,
            "backup_metadata": (
                str(backup_snapshot / "backup_metadata.json")
                if backup_snapshot is not None
                else None
            ),
            "total_class_counts": class_counts_for_images(split_view),
            "val_class_counts": class_counts_for_images(
                split_view,
                splits.get("val", []),
            ),
            "modalities": list(dataset.modalities),
            "scene_count": len(split_view.images),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    @_logged_operation("multimodal_select")
    def dataset_select(
        self,
        file: str | Path,
        out: str | Path | None = None,
        *,
        copy_images: bool = True,
        workers: int = 8,
        progress: bool | None = None,
        progress_leave: bool | None = None,
        reload: bool = False,
    ) -> int:
        """Select scenes from a list and materialize every modality.

        ``YoloManager.dataset_select`` is intentionally not reused here:
        its writer receives one ``YoloDataset`` and would copy only the
        reference modality.  A multimodal selection is scene-based, so one
        selected stem always carries all available modality images and its
        shared label file.
        """

        progress_enabled = self.progress if progress is None else progress
        progress_leave_enabled = (
            self.progress_leave if progress_leave is None else progress_leave
        )
        dataset = self.load(
            reload=reload,
            progress=progress_enabled,
            progress_leave=progress_leave_enabled,
        )
        split_view = _multimodal_split_view(
            dataset,
            reference_modality=self.reference_modality,
        )
        selected_values = read_selection_file(file)
        selected_stems = _scenes_for_split_names(
            split_view,
            sorted(selected_values),
        )
        selected_set = set(selected_stems)
        split_scenes = {"train": [], "val": [], "test": []}
        for stem in selected_stems:
            source_split = dataset.scenes[stem].split
            split_name = source_split if source_split in split_scenes else "train"
            split_scenes[split_name].append(stem)

        output_root = (
            Path(out).expanduser()
            if out is not None
            else default_dataset_output(self.root, "select")
        )
        _write_multimodal_split_dataset(
            dataset,
            split_scenes,
            output_root,
            labels_dir=self.labels_dir,
            reference_modality=self.reference_modality,
            absolute_paths=False,
            include_test=bool(split_scenes["test"]),
            copy_images=copy_images,
            workers=workers,
            progress=progress_enabled,
            progress_leave=progress_leave_enabled,
        )
        print(
            json.dumps(
                {
                    "images": len(selected_stems),
                    "scenes": len(selected_set),
                    "out": str(output_root),
                    "modalities": list(dataset.modalities),
                    "splits": {
                        name: len(values) for name, values in split_scenes.items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    @_logged_operation("multimodal_stats")
    def stats(
        self,
        *,
        out: str | Path | None = None,
        plots_dir: str | Path | None = None,
        stats_list: str | Sequence[str] | None = None,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> dict[str, object]:
        """Compute shared annotation statistics plus per-modality image statistics."""

        dataset = self.load(
            reload=reload, progress=progress, progress_leave=progress_leave
        )
        payload = compute_multimodal_stats(dataset)
        stats_dir = ydm_dir(self.root, "stats")
        report_path = Path(out) if out is not None else stats_dir / "multimodal_stats.json"
        plots_path = Path(plots_dir) if plots_dir is not None else stats_dir / "plots"
        write_json_report(payload, report_path)
        write_multimodal_stats_plots(dataset, plots_path, stats_list=stats_list)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    @_logged_operation("multimodal_convert_to_uint8")
    def convert_to_uint8(
        self,
        out: str | Path | None = None,
        *,
        modalities: Sequence[str] | None = None,
        stretch: bool = True,
        value_range: tuple[float, float] | None = None,
        preserve_zero: bool = True,
        overwrite: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, object]:
        """Write selected modalities as uint8 images without changing the source data.

        Existing uint8 files are copied unchanged. Other dtypes are written as
        PNG after optional linear stretching; use ``value_range`` for one fixed
        mapping across images, such as ``(0, 20000)`` for depth values.
        """

        output_path = Path(out) if out is not None else ydm_dir(self.root, "conversion") / "uint8"
        payload = convert_multimodal_images_to_uint8(
            self.load(
                reload=reload, progress=progress, progress_leave=progress_leave
            ),
            output_path,
            modalities=modalities,
            stretch=stretch,
            value_range=value_range,
            preserve_zero=preserve_zero,
            overwrite=overwrite,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    @_logged_operation("multimodal_vis_draw")
    def vis_draw(
        self,
        out: str | Path | None = None,
        *,
        style: str = "cv2",
        modalities: Sequence[str] | None = None,
        limit: int | None = None,
        show_conf: bool = False,
        conf: float | None = None,
        mask_alpha: int = 64,
        fill_mask: bool = True,
        show_attrs: bool = False,
        show_id: bool = False,
        filter_level: Sequence[int | str] | int | str | None = None,
        att_seperate: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Render every selected modality from the shared cached annotations."""

        output_path = Path(out) if out is not None else ydm_dir(self.root, "vis") / "draw"
        counts = render_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            output_path,
            style=style,
            modalities=modalities,
            limit=limit,
            show_confidence=show_conf,
            confidence_threshold=conf,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attributes=show_attrs,
            show_txt_id=show_id,
            filter_level=filter_level,
            att_seperate=att_seperate,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(
            json.dumps(
                {"out": str(output_path), "modalities": counts}, indent=2, ensure_ascii=False
            )
        )
        return counts

    @_logged_operation("multimodal_vis_crop")
    def vis_crop(
        self,
        out: str | Path | None = None,
        *,
        style: str = "cv2",
        modalities: Sequence[str] | None = None,
        keep_shape: bool = False,
        min_size: int = 1,
        padding: int | float = 0,
        conf: float | None = None,
        by_attr: bool = False,
        filter_level: Sequence[int | str] | int | str | None = (1,),
        att_seperate: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Write object crops grouped by modality without reparsing labels."""

        output_path = Path(out) if out is not None else ydm_dir(self.root, "vis") / "crop"
        counts = crop_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            output_path,
            style=style,
            modalities=modalities,
            keep_shape=keep_shape,
            min_size=min_size,
            padding=padding,
            confidence_threshold=conf,
            by_attribute=by_attr,
            filter_level=filter_level,
            att_seperate=att_seperate,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(
            json.dumps({"out": str(output_path), "crops": counts}, indent=2, ensure_ascii=False)
        )
        return counts


def _multimodal_split_view(
    dataset: MultimodalYoloDataset,
    *,
    reference_modality: str,
) -> YoloDataset:
    images: list[YoloImage] = []
    for scene in sorted(dataset.scenes.values(), key=lambda item: item.stem):
        representative = _representative_scene_image(
            dataset,
            scene,
            reference_modality,
        )
        if representative is None:
            continue
        _, source = representative
        annotations = (
            parse_label_file(
                scene.label_path,
                task=dataset.task,
                attributes=dataset.attributes,
            )
            if scene.label_path is not None and Path(scene.label_path).is_file()
            else []
        )
        images.append(
            YoloImage(
                path=source.path,
                label_path=scene.label_path,
                width=source.width,
                height=source.height,
                annotations=annotations,
                # Keep the scene stem in the split view so include lists and
                # split output remain independent of modality suffixes.
                output_name=f"{reference_modality}/{scene.stem}{source.path.suffix}",
                split=scene.split,
            )
        )
    return YoloDataset(
        root=dataset.root,
        images=images,
        classes=dataset.classes,
        attributes=dataset.attributes,
        task=dataset.task,
    )


def _scenes_for_split_names(
    split_view: YoloDataset,
    names: Sequence[str],
) -> list[str]:
    lookup: dict[str, list[str]] = {}
    for image in split_view.images:
        scene_stem = image.stem
        for key in (
            str(image.path.resolve()),
            image.file_name,
            Path(image.file_name).name,
            image.path.name,
            image.stem,
        ):
            lookup.setdefault(str(key).replace("\\", "/").casefold(), []).append(scene_stem)

    result: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        text = str(raw_name)
        path = Path(text)
        keys = (
            str(path.resolve()).replace("\\", "/").casefold(),
            text.replace("\\", "/").casefold(),
            path.name.casefold(),
            path.stem.casefold(),
        )
        matches: set[str] = set()
        for key in keys:
            matches.update(lookup.get(key, []))
        if not matches:
            raise ValueError(f"split item does not match a multimodal scene: {text!r}")
        if len(matches) > 1:
            raise ValueError(f"split item matches multiple multimodal scenes: {text!r}")
        scene_stem = next(iter(matches))
        if scene_stem not in seen:
            result.append(scene_stem)
            seen.add(scene_stem)
    return result


def _representative_scene_image(
    dataset: MultimodalYoloDataset,
    scene: MultimodalScene,
    reference_modality: str,
):
    source = scene.images.get(reference_modality)
    if source is not None:
        return reference_modality, source
    for modality in dataset.modalities:
        source = scene.images.get(modality)
        if source is not None:
            return modality, source
    return None


def _output_split_paths(
    dataset: MultimodalYoloDataset,
    split_scenes: Mapping[str, Sequence[str]],
    output_root: Path,
    *,
    reference_modality: str,
    absolute_paths: bool,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for split_name, stems in split_scenes.items():
        values: list[str] = []
        for stem in stems:
            scene = dataset.scenes[stem]
            representative = _representative_scene_image(
                dataset,
                scene,
                reference_modality,
            )
            if representative is None:
                continue
            modality, source = representative
            path = output_root / modality / source.relative_path
            values.append(str(path.resolve()) if absolute_paths else path.relative_to(output_root).as_posix())
        output[split_name] = values
    return output


def _write_multimodal_split_dataset(
    dataset: MultimodalYoloDataset,
    split_scenes: Mapping[str, Sequence[str]],
    output_root: Path,
    *,
    labels_dir: str | Path,
    reference_modality: str,
    absolute_paths: bool,
    include_test: bool,
    copy_images: bool,
    workers: int,
    progress: bool,
    progress_leave: bool,
) -> None:
    if output_root.exists():
        existing = [path for path in output_root.iterdir() if path.name != "ydm_log"]
        if existing:
            raise ValueError(
                f"multimodal split output must be new or contain only ydm_log: {output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    label_folder = Path(labels_dir)
    if label_folder.is_absolute():
        label_folder = Path(label_folder.name)
    label_root = Path(dataset.root) / label_folder
    selected_stems: list[str] = []
    seen: set[str] = set()
    for stems in split_scenes.values():
        for stem in stems:
            if stem not in seen:
                selected_stems.append(stem)
                seen.add(stem)

    copy_jobs: list[tuple[Path, Path]] = []
    for stem in selected_stems:
        scene = dataset.scenes[stem]
        if copy_images:
            for modality, source in scene.images.items():
                copy_jobs.append(
                    (
                        source.path,
                        output_root / modality / source.relative_path,
                    )
                )
        if scene.label_path is not None and Path(scene.label_path).is_file():
            try:
                label_relative = Path(scene.label_path).resolve().relative_to(label_root.resolve())
            except ValueError:
                label_relative = Path(scene.label_path).name
            copy_jobs.append(
                (
                    Path(scene.label_path),
                    output_root / label_folder / label_relative,
                )
            )
        else:
            label_path = output_root / label_folder / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")

    def copy_one(job: tuple[Path, Path]) -> None:
        source, destination = job
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for job in iter_progress(
            copy_jobs,
            enabled=progress,
            total=len(copy_jobs),
            desc="multimodal split copy files",
            leave=progress_leave,
        ):
            copy_one(job)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(copy_one, job) for job in copy_jobs]
            for future in iter_progress(
                as_completed(futures),
                enabled=progress,
                total=len(futures),
                desc="multimodal split copy files",
                leave=progress_leave,
            ):
                future.result()

    write_class_schema(dataset.classes, output_root / "class.txt")
    write_attribute_schema(dataset.attributes, output_root / "attribute.yaml")
    output_splits = _output_split_paths(
        dataset,
        split_scenes,
        output_root,
        reference_modality=reference_modality,
        absolute_paths=absolute_paths,
    )
    for split_name, names in output_splits.items():
        write_split_file(names, output_root / f"{split_name}.txt")

    configuration: dict[str, object] = {
        "path": str(output_root),
        "image_dirs": list(dataset.modalities),
        "labels_dir": label_folder.as_posix(),
        "names": dataset.classes.names,
        "nc": len(dataset.classes.names),
        "train": "train.txt",
        "val": "val.txt",
    }
    if include_test:
        configuration["test"] = "test.txt"
    text = yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True)
    (output_root / "dataset.yaml").write_text(text, encoding="utf-8")
    (output_root / "multimodal.yaml").write_text(text, encoding="utf-8")
    (output_root / "split_report.json").write_text(
        json.dumps(
            {
                "report_type": "multimodal_split",
                "splits": {name: len(values) for name, values in output_splits.items()},
                "images": sum(len(scene.images) for scene in dataset.scenes.values() if scene.stem in seen),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_check_summary(payload: dict[str, object], report_path: Path) -> None:
    summary = payload.get("summary", {})
    counts = summary if isinstance(summary, dict) else {}
    error_count = sum(
        count for key, count in counts.items() if str(key).startswith("error:")
    )
    warning_count = sum(
        count for key, count in counts.items() if str(key).startswith("warning:")
    )
    scene_count = payload.get("scene_count", 0)
    _print_image_type_summary(payload.get("image_type_summary"))
    if error_count or warning_count:
        color = "\033[31m"
        reset = "\033[0m"
        print(
            f"{color}[MULTIMODAL CHECK WARNING] complete_scenes={scene_count}, "
            f"errors={error_count}, warnings={warning_count}. Full report: {report_path}{reset}",
            file=sys.stderr,
        )
        for key, count in sorted(counts.items()):
            print(f"{color}  {key}: {count}{reset}", file=sys.stderr)
        return
    print(
        f"\033[32m[MULTIMODAL CHECK OK] complete_scenes={scene_count}. "
        f"Full report: {report_path}\033[0m",
        file=sys.stderr,
    )


def _print_image_type_summary(value: object) -> None:
    if not isinstance(value, dict):
        return
    print("[MULTIMODAL IMAGE TYPES]", file=sys.stderr)
    for modality, raw_summary in value.items():
        if not isinstance(raw_summary, dict):
            continue
        image_count = raw_summary.get("image_count", 0)
        type_count = raw_summary.get("type_count", 0)
        print(
            f"  {modality}: {image_count} source image(s), {type_count} type(s)",
            file=sys.stderr,
        )
        raw_types = raw_summary.get("types", [])
        if not isinstance(raw_types, list):
            continue
        for raw_type in raw_types:
            if not isinstance(raw_type, dict):
                continue
            print(
                "    "
                f"{raw_type.get('count', 0)} x "
                f"{raw_type.get('format', 'unknown')}/"
                f"{raw_type.get('mode', 'unknown')}/"
                f"{raw_type.get('dtype', 'unknown')}/"
                f"{raw_type.get('channels', '?')}ch "
                f"{raw_type.get('width', '?')}x{raw_type.get('height', '?')}",
                file=sys.stderr,
            )
