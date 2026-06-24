"""Python-friendly task runner backed by the canonical CLI handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_COMMANDS: Mapping[str, tuple[str, ...]] = {
    "check": ("check",),
    "stats": ("stats",),
    "layout.detect": ("layout", "detect"),
    "query.class": ("query", "class"),
    "query.attr": ("query", "attr"),
    "dataset.select": ("dataset", "select"),
    "dataset.normalize": ("dataset", "normalize"),
    "dataset.split": ("dataset", "split"),
    "dataset.yaml": ("dataset", "yaml"),
    "dataset.filter": ("dataset", "filter"),
    "dataset.merge": ("dataset", "merge"),
    "dataset.duplicates": ("dataset", "duplicates"),
    "dataset.bad_images": ("dataset", "bad-images"),
    "ann.delete_class": ("ann", "delete-class"),
    "ann.replace_class": ("ann", "replace-class"),
    "ann.merge_class": ("ann", "merge-class"),
    "ann.rename_class": ("ann", "rename-class"),
    "ann.apply_map": ("ann", "apply-map"),
    "ann.set_attr": ("ann", "set-attr"),
    "ann.delete_attr": ("ann", "delete-attr"),
    "vis.draw": ("vis", "draw"),
    "vis.crop": ("vis", "crop"),
    "export.coco": ("export", "coco"),
    "export.xany": ("export", "xany"),
    "import.labelme": ("import", "labelme"),
    "import.coco": ("import", "coco"),
    "import.voc": ("import", "voc"),
    "convert.seg2det": ("convert", "seg2det"),
    "convert.pseudo": ("convert", "pseudo"),
    "eval.compare": ("eval", "compare"),
    "eval.review_pack": ("eval", "review-pack"),
}

_PARAMETER_ALIASES = {
    "class_": "class",
    "from_": "from",
    "map_file": "map",
    "json_path": "json",
}

_FALSE_FLAGS = {
    "copy_images": "--no-copy-images",
    "keep_empty_labels": "--drop-empty-labels",
    "source_prefix": "--no-source-prefix",
    "rename_duplicates": "--no-rename-duplicates",
    "fill_mask": "--no-fill-mask",
    "drop_confidence": "--keep-conf",
    "skip_difficult": "--keep-difficult",
}


def build_task_argv(task: str, **params: Any) -> list[str]:
    """Convert a Python task call into the argument list accepted by ``ydm``.

    Lists, tuples, and sets become comma-separated values. ``None`` values are
    omitted. Python keyword collisions use aliases such as ``class_`` and
    ``from_``.
    """

    if task not in TASK_COMMANDS:
        available = ", ".join(sorted(TASK_COMMANDS))
        raise ValueError(f"unknown task {task!r}; available tasks: {available}")

    argv = list(TASK_COMMANDS[task])
    for python_name, value in params.items():
        if value is None:
            continue
        option_name = _PARAMETER_ALIASES.get(python_name, python_name).replace("_", "-")
        flag = f"--{option_name}"

        if isinstance(value, bool):
            bool_flag = _boolean_flag(task, python_name, value, flag)
            if bool_flag:
                argv.append(bool_flag)
            continue

        argv.extend((flag, _stringify(value)))
    return argv


def run_task(task: str, **params: Any) -> int:
    """Run any YOLO Data Manager task from Python and return its exit code."""

    from yolo_data_manager.cli import main

    return main(build_task_argv(task, **params))


def _boolean_flag(task: str, name: str, value: bool, default_flag: str) -> str | None:
    if name == "compact":
        if task in {"ann.merge_class", "ann.apply_map"}:
            return None if value else "--no-compact"
        return "--compact" if value else None
    if name == "filter_no_attrs" and task == "vis.crop":
        return None if value else "--keep-no-attrs"
    if name in _FALSE_FLAGS:
        return None if value else _FALSE_FLAGS[name]
    return default_flag if value else None


def _stringify(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return ",".join(str(item) for item in sorted(value, key=str))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ",".join(str(item) for item in value)
    return str(value)
