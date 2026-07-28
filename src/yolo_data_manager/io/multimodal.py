from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image

from yolo_data_manager.core.models import TASK_AUTO, AttributeSchema, ClassSchema, is_image_file
from yolo_data_manager.core.multimodal import (
    AlignmentReport,
    ModalityConfig,
    MultimodalImage,
    MultimodalScene,
    MultimodalYoloDataset,
)
from yolo_data_manager.core.schema import find_attribute_file, read_attribute_schema, read_class_schema, read_dataset_class_schema
from yolo_data_manager.io.loader import parse_label_file
from yolo_data_manager.runtime import iter_progress, progress_stage, scan_matching_files


def load_multimodal_yolo_dataset(
    root: str | Path,
    image_dirs: Sequence[str | Path],
    *,
    image_params: Mapping[str, Mapping[str, object]] | None = None,
    labels_dir: str | Path = "labels",
    label_params: Mapping[str, object] | None = None,
    class_file: str | Path | None = None,
    attribute_file: str | Path | None = None,
    task: str = TASK_AUTO,
    read_image_size: bool = True,
    progress: bool = False,
    progress_leave: bool = False,
) -> MultimodalYoloDataset:
    """Load one shared YOLO label set and associate images from multiple modalities.

    ``image_dirs`` supplies the image folders. ``image_params`` is keyed by
    modality type and optionally provides ``dir``, ``suffix`` and ``required``.
    If it is empty, each folder name becomes the modality type and all source
    stems are used unchanged. ``label_params`` accepts the label ``suffix`` and
    ``extension``; without it labels use the standard ``<stem>.txt`` rule.
    """

    progress_stage("load resolve multimodal configuration", enabled=progress)
    root_path = Path(root)
    modalities = _resolve_modalities(root_path, image_dirs, image_params)
    label_root = _resolve_under(root_path, labels_dir)
    label_config = _resolve_label_config(label_params)
    report = AlignmentReport()

    progress_stage("load read dataset schema", enabled=progress)
    classes = _read_classes(root_path, class_file)
    attr_path = _resolve_under(root_path, attribute_file) if attribute_file is not None else find_attribute_file(root_path)
    attributes = read_attribute_schema(attr_path)

    image_indexes: dict[str, dict[str, list[MultimodalImage]]] = {}
    for modality, config in modalities.items():
        image_indexes[modality] = _scan_images(
            config,
            report,
            read_image_size=read_image_size,
            progress=progress,
            progress_leave=progress_leave,
        )

    label_index = _scan_labels(
        label_root,
        suffix=label_config["suffix"],
        extension=label_config["extension"],
        report=report,
        progress=progress,
        progress_leave=progress_leave,
    )

    all_stems = set(label_index)
    for index in image_indexes.values():
        all_stems.update(index)

    progress_stage("load prepare scene association", enabled=progress)
    scene_stems = sorted(all_stems)
    scenes: dict[str, MultimodalScene] = {}
    required_modalities = {name for name, config in modalities.items() if config.required}
    for stem in iter_progress(
        scene_stems,
        enabled=progress,
        total=len(scene_stems),
        desc="load associate scenes and parse labels",
        leave=progress_leave,
    ):
        label_paths = label_index.get(stem, [])
        images_by_modality = {name: index.get(stem, []) for name, index in image_indexes.items()}
        duplicate_modalities = {name for name, paths in images_by_modality.items() if len(paths) > 1}

        for modality in sorted(duplicate_modalities):
            report.add(
                "error",
                "duplicate_scene_image",
                f"multiple {modality} images resolve to scene stem: {stem}",
                scene_stem=stem,
                modality=modality,
            )
        if len(label_paths) > 1:
            report.add(
                "error",
                "duplicate_scene_label",
                f"multiple label files resolve to scene stem: {stem}",
                scene_stem=stem,
                modality="label",
            )

        label_path = label_paths[0] if len(label_paths) == 1 else None
        scene = MultimodalScene(
            stem=stem,
            label_path=label_path,
            images={name: paths[0] for name, paths in images_by_modality.items() if len(paths) == 1},
            duplicate_modalities=duplicate_modalities,
        )

        has_images = bool(scene.images) or bool(duplicate_modalities)
        if label_path is None and has_images:
            for modality, paths in images_by_modality.items():
                for image in paths:
                    report.add(
                        "warning",
                        "orphan_image",
                        "image has no matching label",
                        scene_stem=stem,
                        modality=modality,
                        path=image.path,
                    )
        elif label_path is not None and not has_images:
            report.add(
                "warning",
                "orphan_label",
                "label has no matching modality image",
                scene_stem=stem,
                modality="label",
                path=label_path,
            )
        elif label_path is not None:
            for modality in sorted(required_modalities):
                if modality not in scene.images and modality not in duplicate_modalities:
                    report.add(
                        "warning",
                        "missing_modality",
                        f"scene is missing required modality: {modality}",
                        scene_stem=stem,
                        modality=modality,
                    )

        complete = (
            label_path is not None
            and not (duplicate_modalities & required_modalities)
            and all(modality in scene.images for modality in required_modalities)
        )
        if complete:
            scene.annotations = parse_label_file(label_path, task=task, attributes=attributes)
        scenes[stem] = scene

    return MultimodalYoloDataset(
        root=root_path,
        classes=classes,
        attributes=attributes,
        task=task,
        modalities=modalities,
        scenes=scenes,
        alignment_report=report,
    )


def _resolve_modalities(
    root: Path,
    image_dirs: Sequence[str | Path],
    image_params: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, ModalityConfig]:
    if not image_dirs:
        raise ValueError("image_dirs must contain at least one image folder")

    folders = [_resolve_under(root, path) for path in image_dirs]
    resolved_folders = [_path_key(path) for path in folders]
    if len(set(resolved_folders)) != len(resolved_folders):
        raise ValueError("image_dirs contains the same folder more than once")

    pending = set(range(len(folders)))
    modalities: dict[str, ModalityConfig] = {}
    for modality, raw_config in (image_params or {}).items():
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"image_params[{modality!r}] must be a mapping")
        name = str(modality).strip()
        if not name:
            raise ValueError("modality type must not be empty")
        folder_index = _find_folder_index(root, folders, name, raw_config, pending)
        pending.remove(folder_index)
        modalities[name] = ModalityConfig(
            type=name,
            path=folders[folder_index],
            suffix=_clean_suffix(raw_config.get("suffix", ""), extensions=None),
            required=bool(raw_config.get("required", True)),
        )

    for folder_index in sorted(pending):
        folder = folders[folder_index]
        name = folder.name
        if not name:
            raise ValueError(f"cannot infer modality type from image folder: {folder}")
        if name in modalities:
            raise ValueError(f"duplicate modality type: {name}; set image_params[*].dir explicitly")
        modalities[name] = ModalityConfig(type=name, path=folder)

    return modalities


def _find_folder_index(
    root: Path,
    folders: list[Path],
    modality: str,
    config: Mapping[str, object],
    pending: set[int],
) -> int:
    folder_value = config.get("dir", config.get("path"))
    if folder_value is not None:
        expected = _path_key(_resolve_under(root, str(folder_value)))
        matches = [idx for idx in pending if _path_key(folders[idx]) == expected]
    else:
        matches = [idx for idx in pending if folders[idx].name == modality]
    if not matches:
        raise ValueError(f"image_params[{modality!r}] does not identify a folder in image_dirs")
    if len(matches) > 1:
        raise ValueError(f"image_params[{modality!r}] matches more than one folder; set its dir field")
    return matches[0]


def _resolve_label_config(params: Mapping[str, object] | None) -> dict[str, str]:
    config = params or {}
    extension = str(config.get("extension", ".txt")).strip().lower() or ".txt"
    if not extension.startswith("."):
        extension = f".{extension}"
    return {
        "suffix": _clean_suffix(config.get("suffix", ""), extensions=(extension,)),
        "extension": extension,
    }


def _read_classes(root: Path, class_file: str | Path | None) -> ClassSchema:
    if class_file is None:
        return read_dataset_class_schema(root)
    return read_class_schema(_resolve_under(root, class_file))


def _scan_images(
    config: ModalityConfig,
    report: AlignmentReport,
    *,
    read_image_size: bool,
    progress: bool,
    progress_leave: bool,
) -> dict[str, list[MultimodalImage]]:
    paths = scan_matching_files(
        config.path,
        is_image_file,
        progress=progress,
        progress_leave=progress_leave,
        desc=f"load scan {config.type}",
    )
    if not paths:
        report.add("warning", "empty_modality", "modality folder contains no supported images", modality=config.type, path=config.path)
    indexed: dict[str, list[MultimodalImage]] = {}
    for path in iter_progress(
        paths,
        enabled=progress,
        total=len(paths),
        desc=f"load read {config.type} metadata",
        leave=progress_leave,
    ):
        source_stem = path.stem
        scene_stem, suffix_matched = _scene_stem(source_stem, config.suffix)
        if config.suffix and not suffix_matched:
            report.add(
                "warning",
                "suffix_not_matched",
                f"image stem does not end with configured suffix: {config.suffix}",
                scene_stem=scene_stem,
                modality=config.type,
                path=path,
            )
        width, height = _read_image_size(path) if read_image_size else (None, None)
        relative_path = path.relative_to(config.path)
        indexed.setdefault(scene_stem, []).append(
            MultimodalImage(
                path=path,
                relative_path=relative_path,
                source_stem=source_stem,
                scene_stem=scene_stem,
                width=width,
                height=height,
            )
        )
    return indexed


def _scan_labels(
    label_root: Path,
    *,
    suffix: str,
    extension: str,
    report: AlignmentReport,
    progress: bool,
    progress_leave: bool,
) -> dict[str, list[Path]]:
    paths = scan_matching_files(
        label_root,
        lambda path: path.suffix.lower() == extension,
        progress=progress,
        progress_leave=progress_leave,
        desc="load scan labels",
    )
    if not paths:
        report.add("warning", "empty_labels", "label folder contains no matching label files", modality="label", path=label_root)
    indexed: dict[str, list[Path]] = {}
    for path in iter_progress(
        paths,
        enabled=progress,
        total=len(paths),
        desc="load index labels",
        leave=progress_leave,
    ):
        scene_stem, suffix_matched = _scene_stem(path.stem, suffix)
        if suffix and not suffix_matched:
            report.add(
                "warning",
                "suffix_not_matched",
                f"label stem does not end with configured suffix: {suffix}",
                scene_stem=scene_stem,
                modality="label",
                path=path,
            )
        indexed.setdefault(scene_stem, []).append(path)
    return indexed


def _scene_stem(source_stem: str, suffix: str) -> tuple[str, bool]:
    if suffix and source_stem.endswith(suffix):
        return source_stem[: -len(suffix)], True
    return source_stem, not suffix


def _clean_suffix(value: object, *, extensions: tuple[str, ...] | None) -> str:
    suffix = str(value or "").strip()
    candidates = extensions or tuple(sorted((".jpeg", ".tiff", ".jpg", ".png", ".bmp", ".tif", ".webp"), key=len, reverse=True))
    for extension in candidates:
        if suffix.lower().endswith(extension.lower()):
            return suffix[: -len(extension)]
    return suffix


def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()
