from __future__ import annotations

import os
import random
from collections.abc import Iterable
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset, is_image_file


SplitIncludeList = str | Path | Iterable[str] | None


def split_dataset(
    dataset: YoloDataset,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = False,
    train_include_list: SplitIncludeList = None,
    val_include_list: SplitIncludeList = None,
) -> dict[str, list[str]]:
    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = {"train": train / total, "val": val / total, "test": test / total}
    names = [
        str(image.path.resolve()) if absolute_paths else image.file_name
        for image in dataset.images
    ]
    train_indices = _resolve_include_indices(
        dataset,
        train_include_list,
        parameter="train_include_list",
    )
    val_indices = _resolve_include_indices(
        dataset,
        val_include_list,
        parameter="val_include_list",
    )
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        overlap_names = ", ".join(dataset.images[index].file_name for index in sorted(overlap))
        raise ValueError(
            "train_include_list and val_include_list overlap: "
            f"{overlap_names}"
        )

    forced = set(train_indices) | set(val_indices)
    remaining_indices = [index for index in range(len(names)) if index not in forced]
    rng = random.Random(seed)
    rng.shuffle(remaining_indices)

    n = len(remaining_indices)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    def output_names(indices: Iterable[int]) -> list[str]:
        return [names[index] for index in indices]

    return {
        "train": output_names(train_indices + remaining_indices[:n_train]),
        "val": output_names(val_indices + remaining_indices[n_train : n_train + n_val]),
        "test": output_names(remaining_indices[n_train + n_val :]),
    }


def _resolve_include_indices(
    dataset: YoloDataset,
    include_list: SplitIncludeList,
    *,
    parameter: str,
) -> list[int]:
    values = _read_include_values(include_list, dataset.root)
    if not values:
        return []

    exact_lookup: dict[str, set[int]] = {}
    stem_lookup: dict[str, set[int]] = {}
    for index, image in enumerate(dataset.images):
        for key in _image_exact_keys(dataset.root, image):
            exact_lookup.setdefault(key, set()).add(index)
        stem_lookup.setdefault(_normalise_key(image.stem), set()).add(index)

    resolved: list[int] = []
    seen: set[int] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        matches: set[int] = set()
        for key in _include_exact_keys(dataset.root, text):
            matches.update(exact_lookup.get(key, set()))
        if not matches:
            matches.update(stem_lookup.get(_normalise_key(Path(text).stem), set()))

        if not matches:
            raise ValueError(
                f"{parameter} item {text!r} does not match any dataset image"
            )
        if len(matches) > 1:
            choices = ", ".join(
                dataset.images[index].file_name for index in sorted(matches)[:5]
            )
            suffix = "..." if len(matches) > 5 else ""
            raise ValueError(
                f"{parameter} item {text!r} matches multiple dataset images: "
                f"{choices}{suffix}; use a relative or absolute path"
            )

        index = next(iter(matches))
        if index not in seen:
            resolved.append(index)
            seen.add(index)
    return resolved


def _read_include_values(
    include_list: SplitIncludeList,
    dataset_root: Path,
) -> list[str]:
    if include_list is None:
        return []

    if isinstance(include_list, (str, Path)):
        text = str(include_list).strip()
        if not text:
            return []
        candidate = Path(text).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = dataset_root / candidate
        if candidate.is_file() and not is_image_file(candidate):
            return _read_include_file(candidate)
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]

    values: list[str] = []
    for value in include_list:
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def _read_include_file(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values.append(text)
    return values


def _image_exact_keys(root: Path, image) -> set[str]:
    keys = {
        _normalise_key(image.file_name),
        _normalise_key(image.path.name),
        _normalise_key(image.path),
        _normalise_key(image.path.resolve()),
    }
    try:
        relative = image.path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = None
    if relative is not None:
        keys.add(_normalise_key(relative))
    return {key for key in keys if key}


def _include_exact_keys(root: Path, value: str) -> set[str]:
    path = Path(value).expanduser()
    keys = {_normalise_key(value)}
    if path.is_absolute():
        keys.add(_normalise_key(path.resolve()))
    else:
        keys.add(_normalise_key(root / path))
        keys.add(_normalise_key((root / path).resolve()))
    return {key for key in keys if key}


def _normalise_key(value: str | Path) -> str:
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text)).replace("\\", "/")


def class_counts_for_images(
    dataset: YoloDataset,
    image_names: Iterable[str] | None = None,
) -> dict[str, int]:
    selected = _image_key_set(image_names) if image_names is not None else None
    counts = {name: 0 for name in dataset.classes.names}

    for image in dataset.images:
        if selected is not None and not (_image_keys(image) & selected):
            continue
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def _image_key_set(values: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        text = str(value)
        path = Path(text)
        keys.update({text, path.name, path.stem})
    return keys


def _image_keys(image) -> set[str]:
    return {
        image.file_name,
        image.stem,
        str(image.path),
        str(image.path.resolve()),
        image.path.name,
        image.path.stem,
    }
