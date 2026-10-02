from __future__ import annotations

import copy
import math
import os
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset, is_image_file


SplitIncludeList = str | Path | Iterable[str] | None
_SPLIT_PRIORITY = {"train": 2, "test": 1, "val": 0}


def split_dataset(
    dataset: YoloDataset,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = False,
    train_include_list: SplitIncludeList = None,
    val_include_list: SplitIncludeList = None,
    ensure_class_presence: bool = True,
) -> dict[str, list[str]]:
    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    if not all(math.isfinite(value) for value in ratios.values()):
        raise ValueError("split ratios must be finite numbers")
    if min(ratios.values()) < 0:
        raise ValueError("split ratios must be non-negative")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train + val + test split ratios must sum to 1.0")
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

    if ensure_class_presence:
        split_sizes = _allocate_split_sizes(len(remaining_indices), ratios)
        split_indices = _assign_balanced_indices(
            dataset,
            remaining_indices,
            split_sizes,
            {
                "train": train_indices,
                "val": val_indices,
                "test": [],
            },
            rng,
            ratios,
        )
    else:
        rng.shuffle(remaining_indices)
        split_sizes = _allocate_split_sizes(len(remaining_indices), ratios)
        n_train = split_sizes["train"]
        n_val = split_sizes["val"]
        split_indices = {
            "train": train_indices + remaining_indices[:n_train],
            "val": val_indices + remaining_indices[n_train : n_train + n_val],
            "test": remaining_indices[
                n_train + n_val : n_train + n_val + split_sizes["test"]
            ],
        }

    def output_names(indices: Iterable[int]) -> list[str]:
        return [names[index] for index in indices]

    return {
        split_name: output_names(split_indices[split_name])
        for split_name in ("train", "val", "test")
    }


def _allocate_split_sizes(
    count: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """Allocate every image to a split without assigning data to zero ratios."""

    if ratios.get("test", 0.0) == 0.0:
        # With no test split, train is the only requested allocation and all
        # rounding remainder belongs to val. This makes the zero-test rule
        # explicit and prevents a third split from receiving leftovers.
        train_size = int(count * ratios.get("train", 0.0))
        return {
            "train": train_size,
            "val": count - train_size,
            "test": 0,
        }

    raw_sizes = {name: count * ratio for name, ratio in ratios.items()}
    sizes = {name: int(value) for name, value in raw_sizes.items()}
    remainder = count - sum(sizes.values())
    fractional = sorted(
        (name for name, ratio in ratios.items() if ratio > 0),
        key=lambda name: (
            -(raw_sizes[name] - sizes[name]),
            -_SPLIT_PRIORITY[name],
        ),
    )
    for name in fractional[:remainder]:
        sizes[name] += 1
    return sizes


def _assign_balanced_indices(
    dataset: YoloDataset,
    remaining_indices: list[int],
    split_sizes: dict[str, int],
    initial: dict[str, list[int]],
    rng: random.Random,
    ratios: dict[str, float],
) -> dict[str, list[int]]:
    """Assign images with weighted, image-level multi-label stratification.

    An image cannot be split between datasets, so exact box-level ratios are
    not always possible. The heuristic therefore uses two objectives:

    * first, spread classes that are missing from a split; if there are not
      enough images for every split, the deterministic priority is
      ``train > test > val``;
    * once a split already contains a class, assign the image to the split
      that most reduces the squared deviation between its current box count
      and the requested class-level target.

    The image order and every tie are controlled by ``rng``. Include-list
    images remain fixed and contribute to the initial presence and box-count
    state, so the remaining images compensate for them where possible.
    """

    split_names = ("train", "val", "test")
    considered_indices = set(remaining_indices).union(*initial.values())
    image_box_counts = {
        index: Counter(
            annotation.class_id
            for annotation in dataset.images[index].annotations
        )
        for index in considered_indices
    }
    class_box_totals: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    for counts in image_box_counts.values():
        class_box_totals.update(counts)
        class_image_counts.update(counts.keys())

    # Rare classes receive a stronger presence score. Box balancing itself is
    # normalized by each class target below, so common and rare classes still
    # get comparable ratio treatment after their first occurrence is placed.
    class_presence_weights = {
        class_id: 1.0 / count
        for class_id, count in class_image_counts.items()
        if count > 0
    }
    class_targets = {
        split_name: {
            class_id: total * ratio
            for class_id, total in class_box_totals.items()
        }
        for split_name, ratio in ratios.items()
    }

    split_indices = {
        name: list(initial.get(name, [])) for name in split_names
    }
    split_presence = {name: set() for name in split_names}
    split_box_counts: dict[str, Counter[int]] = {
        name: Counter() for name in split_names
    }
    for split_name, indices in split_indices.items():
        for index in indices:
            counts = image_box_counts.get(index, Counter())
            split_presence[split_name].update(counts.keys())
            split_box_counts[split_name].update(counts)

    remaining_capacity = dict(split_sizes)
    rng.shuffle(remaining_indices)
    remaining_indices.sort(
        key=lambda index: (
            -sum(
                class_presence_weights.get(class_id, 0.0)
                for class_id in image_box_counts.get(index, {})
            ),
            -len(image_box_counts.get(index, {})),
        )
    )
    assigned_remaining = {name: 0 for name in split_names}

    for index in remaining_indices:
        available = [
            name for name in split_names if remaining_capacity.get(name, 0) > 0
        ]
        if not available:
            raise RuntimeError("split allocation did not leave a destination for every image")
        rng.shuffle(available)
        counts = image_box_counts.get(index, Counter())

        def score(split_name: str) -> tuple[float, ...]:
            missing_classes = [
                class_id
                for class_id in counts
                if class_id not in split_presence[split_name]
            ]
            missing_weight = sum(
                class_presence_weights.get(class_id, 0.0)
                for class_id in missing_classes
            )
            missing_count = len(missing_classes)
            box_improvement = sum(
                _squared_target_improvement(
                    split_box_counts[split_name].get(class_id, 0),
                    class_targets[split_name].get(class_id, 0.0),
                    box_count,
                )
                for class_id, box_count in counts.items()
            )
            target_images = max(1, split_sizes.get(split_name, 0))
            current_images = assigned_remaining[split_name]
            image_improvement = _squared_target_improvement(
                current_images,
                split_sizes.get(split_name, 0),
                1,
            )
            fill_ratio = current_images / target_images

            if missing_count:
                # Presence is a coverage constraint. The priority before the
                # box score intentionally keeps the historical fallback for
                # too-few examples: train, then test, then val.
                return (
                    1.0,
                    missing_weight,
                    float(missing_count),
                    float(_SPLIT_PRIORITY[split_name]),
                    box_improvement,
                    image_improvement,
                    -fill_ratio,
                )
            # Once coverage is satisfied, box-count deviation is the primary
            # objective; this is what prevents one split from getting only a
            # single instance of a class and then stopping.
            return (
                0.0,
                box_improvement,
                image_improvement,
                -fill_ratio,
                float(_SPLIT_PRIORITY[split_name]),
            )

        split_name = max(available, key=score)
        split_indices[split_name].append(index)
        split_presence[split_name].update(counts.keys())
        split_box_counts[split_name].update(counts)
        assigned_remaining[split_name] += 1
        remaining_capacity[split_name] -= 1

    return split_indices


def _squared_target_improvement(
    current: int,
    target: float,
    delta: int,
) -> float:
    """Return the reduction in normalized squared target error."""

    scale = max(abs(target), 1.0)
    before = ((current - target) / scale) ** 2
    after = ((current + delta - target) / scale) ** 2
    return before - after


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
        if not _looks_absolute(candidate) and not candidate.exists():
            candidate = dataset_root / candidate
        if candidate.is_file() and not is_image_file(candidate):
            return _read_include_file(candidate)
        if candidate.suffix.lower() == ".txt":
            # A .txt suffix marks an include-list file. Reading it must not
            # rely on is_file()/is_absolute() being reliable for the path
            # (e.g. UNC shares); surface a clear error instead of treating the
            # path itself as an image name.
            if not candidate.is_file():
                raise FileNotFoundError(f"include list file not found: {candidate}")
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


def _looks_absolute(path: Path) -> bool:
    """Return True for drive/UNC/POSIX-root paths.

    ``Path.is_absolute()`` treats ``\\server\\share`` as relative on some
    runtimes, so include-list resolution checks these forms explicitly.
    """
    text = str(path)
    return (
        text.startswith("\\\\")
        or text.startswith("/")
        or text.startswith("\\")
        or (len(text) >= 2 and text[1] == ":")
    )


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


def extract_splits(
    dataset: YoloDataset,
    *,
    train_include_list: SplitIncludeList = None,
    val_include_list: SplitIncludeList = None,
    test_include_list: SplitIncludeList = None,
    out_root: str | Path | None = None,
    copy_images: bool = True,
    keep_empty_labels: bool = True,
    include_confidence: bool = False,
    dry_run: bool = False,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
    backup_dir: str | Path | None = None,
    backup: bool = False,
) -> dict[str, dict[str, object]]:
    """Materialize each split (given by include lists) into its own dataset dir.

    Each non-empty include list (a txt file path, a comma-separated string, or
    an iterable of image names/paths) selects the images for that split, which
    are written to ``<out_root>/<split>`` as a standalone flat YOLO dataset
    (``images/`` + ``labels/``). ``out_root`` defaults to
    ``<dataset.root>/ydm_subsets``. Pass ``dry_run`` to report the counts and
    output paths without writing anything.

    This is a copy operation: source labels are not modified, so no label
    backup is created by default. Set ``backup=True`` (and optionally
    ``backup_dir``) to snapshot source labels before writing.
    """
    from yolo_data_manager.io.writer import write_yolo_dataset

    splits = {
        "train": train_include_list,
        "val": val_include_list,
        "test": test_include_list,
    }
    out = Path(out_root) if out_root is not None else Path(dataset.root) / "ydm_subsets"
    result: dict[str, dict[str, object]] = {}
    for split_name, include_list in splits.items():
        if include_list is None:
            continue
        indices = _resolve_include_indices(
            dataset, include_list, parameter=f"{split_name}_include_list"
        )
        selected = _select_by_indices(dataset, indices)
        if not selected.images:
            result[split_name] = {"out": None, "images": 0, "annotations": 0}
            continue
        split_out = out / split_name
        result[split_name] = {
            "out": split_out,
            "images": len(selected.images),
            "annotations": selected.annotation_count(),
        }
        if dry_run:
            continue
        write_yolo_dataset(
            selected,
            split_out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            include_confidence=include_confidence,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            backup_dir=backup_dir,
            backup=backup,
        )
    return result


def _select_by_indices(dataset: YoloDataset, indices: list[int]) -> YoloDataset:
    result = copy.deepcopy(dataset)
    result.images = [result.images[index] for index in indices]
    return result
