"""Python-friendly task runner backed by the canonical CLI handlers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from yolo_data_manager.core.models import AttributeSchema, ClassSchema, YoloDataset
from yolo_data_manager.core.schema import read_dataset_yaml
from yolo_data_manager.io.output_paths import ydm_dir
from yolo_data_manager.logging_utils import operation_scope

TASK_COMMANDS: Mapping[str, tuple[str, ...]] = {
    "check": ("check",),
    "stats": ("stats",),
    "layout.detect": ("layout", "detect"),
    "query.class": ("query", "class"),
    "query.attr": ("query", "attr"),
    "dataset.select": ("dataset", "select"),
    "dataset.normalize": ("dataset", "normalize"),
    "dataset.split": ("dataset", "split"),
    "dataset.extract_split": ("dataset", "extract-split"),
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
    "ann.restore_backup": ("ann", "restore-backup"),
    "ann.correct_from_crops": ("ann", "correct-from-crops"),
    "ann.correct_from_error_crops": ("ann", "correct-from-error-crops"),
    "ann.att_correct_from_crops": ("ann", "correct-attr-from-crops"),
    "ann.att_correct_from_error_crops": ("ann", "correct-attr-from-error-crops"),
    "ann.set_attr": ("ann", "set-attr"),
    "ann.delete_attr": ("ann", "delete-attr"),
    "vis.draw": ("vis", "draw"),
    "vis.crop": ("vis", "crop"),
    "vis.manual_box": ("vis", "manual-box"),
    "export.coco": ("export", "coco"),
    "export.xany": ("export", "xany"),
    "import.labelme": ("import", "labelme"),
    "import.coco": ("import", "coco"),
    "import.voc": ("import", "voc"),
    "import.mask": ("import", "mask"),
    "convert.seg2det": ("convert", "seg2det"),
    "convert.pseudo": ("convert", "pseudo"),
    "convert.resize": ("convert", "resize"),
    "convert.filename_remap": ("convert", "filename-remap"),
    "vlm.auto_label": ("vlm", "auto-label"),
    "vlm.verify_errors": ("vlm", "verify-errors"),
    "vlm.assistant": ("vlm", "assistant"),
    "eval.compare": ("eval", "compare"),
    "eval.review_pack": ("eval", "review-pack"),
    "eval.error_analysis": ("eval", "error-analysis"),
    "eval.error_analysis_common": ("eval", "error-analysis-common"),
    "eval.metrics": ("eval", "metrics"),
}

_PARAMETER_ALIASES = {
    "class_": "class",
    "exclude_class_": "exclude-class",
    "from_": "from",
    "map_file": "map",
    "json_path": "json",
    "class_map": "class-map",
}

_FALSE_FLAGS = {
    "copy_images": "--no-copy-images",
    "ensure_class_presence": "--no-ensure-class-presence",
    "keep_empty_labels": "--drop-empty-labels",
    "source_prefix": "--no-source-prefix",
    "rename_duplicates": "--no-rename-duplicates",
    "fill_mask": "--no-fill-mask",
    "drop_confidence": "--keep-conf",
    "keep_ratio": "--no-keep-ratio",
    "skip_difficult": "--keep-difficult",
    "ignore_empty_classes": "--include-empty-classes",
    "copy_crops": "--no-copy-crops",
    "backup_current": "--no-backup-current",
}


def _check_output_aliases(
    out: str | Path | None,
    out_data: str | Path | None,
) -> None:
    """Reject ambiguous output arguments while keeping both APIs compatible."""

    if out is not None and out_data is not None:
        raise ValueError("out and out_data are mutually exclusive")


def build_task_argv(command: str, **params: Any) -> list[str]:
    """Convert a Python task call into the argument list accepted by ``ydm``.

    Lists, tuples, and sets become comma-separated values. ``None`` values are
    omitted. Python keyword collisions use aliases such as ``class_`` and
    ``from_``.
    """

    if command not in TASK_COMMANDS:
        available = ", ".join(sorted(TASK_COMMANDS))
        raise ValueError(f"unknown task {command!r}; available tasks: {available}")

    argv = list(TASK_COMMANDS[command])
    for python_name, value in params.items():
        if value is None:
            continue
        if command == "eval.error_analysis_common" and python_name == "error_dirs":
            values = (
                value
                if isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                else [value]
            )
            argv.extend(
                item
                for path in values
                for item in ("--error-dir", str(path))
            )
            continue
        option_name = _PARAMETER_ALIASES.get(python_name, python_name).replace("_", "-")
        flag = f"--{option_name}"

        if isinstance(value, bool):
            bool_flag = _boolean_flag(command, python_name, value, flag)
            if bool_flag:
                argv.append(bool_flag)
            continue

        argv.extend((flag, _stringify(value)))
    return argv


def run_task(command: str, **params: Any) -> int:
    """Run any YOLO Data Manager task from Python and return its exit code."""

    from yolo_data_manager.cli import main

    return main(build_task_argv(command, **params))


def _boolean_flag(task: str, name: str, value: bool, default_flag: str) -> str | None:
    if name == "progress":
        return None if value else "--no-progress"
    if name == "show_existing" and task == "vis.manual_box":
        return None if value else "--hide-existing"
    if name == "progress_leave":
        return "--progress-leave" if value else None
    if name == "compact":
        if task in {"ann.merge_class", "ann.apply_map"}:
            return None if value else "--no-compact"
        return "--compact" if value else None
    if name == "clean" and task in {"vis.draw", "vis.crop"}:
        return None if value else "--no-clean"
    if name in _FALSE_FLAGS:
        return None if value else _FALSE_FLAGS[name]
    return default_flag if value else None


def _stringify(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(
            {str(key): item for key, item in value.items()},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    if isinstance(value, set):
        return ",".join(str(item) for item in sorted(value, key=str))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ",".join(str(item) for item in value)
    return str(value)


def _class_values(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _write_updated_class_source(path: Path, class_names: Sequence[str]) -> None:
    """Write updated names to a class text file or dataset YAML source."""

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = read_dataset_yaml(path) if path.is_file() else {}
        data["names"] = list(class_names)
        data["nc"] = len(class_names)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return

    from yolo_data_manager.core.models import ClassSchema
    from yolo_data_manager.core.schema import write_class_schema

    write_class_schema(ClassSchema(list(class_names)), path)


def _default_existing_path(root: str, name: str) -> str | None:
    path = Path(root) / name
    return str(path) if path.exists() else None


def _resolve_manager_root(root: str | Path) -> tuple[Path, str | None, str | None]:
    root_path = Path(root)
    if root_path.suffix.lower() not in {".yaml", ".yml"} or not root_path.is_file():
        return root_path, None, None

    yaml_path = root_path
    data = read_dataset_yaml(yaml_path)
    dataset_root = _resolve_yaml_dataset_root(yaml_path, data.get("path"))
    split_file = _resolve_yaml_split_file(yaml_path, dataset_root, data.get("val"))
    return dataset_root, str(yaml_path), split_file


def _resolve_yaml_dataset_root(yaml_path: Path, path_value: Any) -> Path:
    if path_value is None:
        return yaml_path.parent
    text = os.path.expandvars(str(path_value)).strip()
    if not text:
        return yaml_path.parent
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (yaml_path.parent / path).resolve()


def _resolve_yaml_split_file(
    yaml_path: Path, dataset_root: Path, val_value: Any
) -> str | None:
    if isinstance(val_value, Sequence) and not isinstance(
        val_value, (str, bytes, bytearray)
    ):
        values = [item for item in val_value if item is not None]
        val_value = values[0] if values else None
    if val_value is None:
        return None
    path = _resolve_yaml_data_path(yaml_path, dataset_root, val_value)
    if path.suffix.lower() == ".txt" or path.exists():
        return str(path)
    return None


def _resolve_yaml_data_path(yaml_path: Path, dataset_root: Path, value: Any) -> Path:
    text = os.path.expandvars(str(value)).strip()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    candidate = dataset_root / path
    if candidate.exists():
        return candidate
    return (yaml_path.parent / path).resolve()


# ---------------------------------------------------------------------------
# Tasks that accept a --root argument (root is auto-filled by YoloManager)
# ---------------------------------------------------------------------------
_ROOT_TASKS: frozenset[str] = frozenset(
    {
        "check",
        "stats",
        "query.class",
        "query.attr",
        "dataset.select",
        "dataset.normalize",
        "dataset.split",
        "dataset.extract_split",
        "dataset.yaml",
        "dataset.filter",
        "dataset.duplicates",
        "dataset.bad_images",
        "ann.delete_class",
        "ann.replace_class",
        "ann.merge_class",
        "ann.rename_class",
        "ann.apply_map",
        "ann.restore_backup",
        "ann.correct_from_crops",
        "ann.correct_from_error_crops",
        "ann.att_correct_from_crops",
        "ann.att_correct_from_error_crops",
        "ann.set_attr",
        "ann.delete_attr",
        "vis.draw",
        "vis.crop",
        "vis.manual_box",
        "export.coco",
        "export.xany",
        "convert.seg2det",
        "convert.pseudo",
        "convert.resize",
        "convert.filename_remap",
        "vlm.auto_label",
        "vlm.verify_errors",
        "vlm.assistant",
    }
)

_DATASET_MUTATING_TASKS: frozenset[str] = frozenset(
    {
        "check",
        "dataset.select",
        "dataset.normalize",
        "dataset.split",
        "dataset.extract_split",
        "dataset.yaml",
        "dataset.filter",
        "dataset.merge",
        "ann.delete_class",
        "ann.replace_class",
        "ann.merge_class",
        "ann.rename_class",
        "ann.apply_map",
        "ann.restore_backup",
        "ann.correct_from_crops",
        "ann.correct_from_error_crops",
        "ann.att_correct_from_crops",
        "ann.att_correct_from_error_crops",
        "ann.set_attr",
        "ann.delete_attr",
    }
)


def _logged_direct_operation(operation: str, *, invalidate_dataset: bool = False):
    """Add the same operation logging used by CLI-backed manager methods."""

    def decorator(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            try:
                with operation_scope(
                    self.root,
                    operation,
                    {"positional": args, "parameters": kwargs},
                ):
                    return function(self, *args, **kwargs)
            finally:
                if invalidate_dataset:
                    self._invalidate_dataset()

        return wrapped

    return decorator


class YoloManager:
    """Stateful manager that remembers the dataset root and common settings.

    Usage::

        from yolo_data_manager import YoloManager

        mgr = YoloManager(r"E:\\datasets\\my_yolo", layout="auto")

        mgr.check()
        mgr.stats(out="stats.json")
        mgr.query_class(class_=["car", "truck"], out="vehicles.csv")
        mgr.ann_merge_class(from_=["crack", "break"], to="defect",
                            out="merged_yolo", compact=True)
        mgr.vis_draw(out="vis_output", show_conf=True)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        layout: str = "auto",
        task: str = "auto",
        images_dir: str = "images",
        labels_dir: str = "labels",
        class_file: str | None = None,
        attribute_file: str | None = None,
        split_file: str | None = None,
        only_val: bool = False,
        init_layout: bool = True,
        init_layout_progress: bool = True,
        init_layout_progress_leave: bool = False,
        init_check: bool | str | Path = True,
        init_check_fill_missing_txt: bool = False,
        init_check_workers: int = 8,
        init_check_progress: bool = True,
        init_check_progress_leave: bool = False,
    ) -> None:
        resolved_root, yaml_class_file, yaml_split_file = _resolve_manager_root(root)
        self.root = str(resolved_root)
        self.layout = layout
        self.task = task
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_file = class_file or yaml_class_file
        self.attribute_file = attribute_file
        self.split_file = split_file or yaml_split_file
        self._explicit_split_file = split_file
        self.only_val = only_val
        self.init_layout = init_layout
        self.init_layout_progress = init_layout_progress
        self.init_layout_progress_leave = init_layout_progress_leave
        self.init_check = init_check
        self.init_check_fill_missing_txt = init_check_fill_missing_txt
        self.init_check_workers = init_check_workers
        self.init_check_progress = init_check_progress
        self.init_check_progress_leave = init_check_progress_leave
        self._dataset: YoloDataset | None = None

        self._warmup_()

    # -- default output paths ----------------------------------------------

    @property
    def output_quality(self) -> Path:
        """Default data-quality output directory."""

        return ydm_dir(self.root, "quality")

    @property
    def output_stats(self) -> Path:
        """Default statistics output directory."""

        return ydm_dir(self.root, "stats")

    @property
    def output_basic_info(self) -> Path:
        """Default compact image/box/attribute statistics CSV."""

        return self.output_stats / "basic_info.csv"

    @property
    def output_vis(self) -> Path:
        """Default visualization output directory."""

        return ydm_dir(self.root, "vis")

    @property
    def output_evaluation(self) -> Path:
        """Default evaluation output directory."""

        return ydm_dir(self.root, "evaluation")

    @property
    def output_dataset(self) -> Path:
        """Default dataset-producing output directory."""

        return ydm_dir(self.root, "dataset")

    @property
    def output_annotation(self) -> Path:
        """Default annotation-edit output directory."""

        return ydm_dir(self.root, "annotation")

    @property
    def output_conversion(self) -> Path:
        """Default conversion output directory."""

        return ydm_dir(self.root, "conversion")

    @property
    def output_vlm(self) -> Path:
        """Default VLM output directory."""

        return ydm_dir(self.root, "vlm")

    @property
    def output_vlm_auto_label(self) -> Path:
        """Default VLM automatic-label output dataset."""

        return self.output_vlm / "auto_label"

    @property
    def output_vlm_error_verify(self) -> Path:
        """Default VLM error-verification correction package."""

        return self.output_vlm / "error_verify"

    @property
    def output_labels_backup(self) -> Path:
        """Default timestamped label-backup directory."""

        return Path(self.root) / "labels_backup"

    @property
    def output_log(self) -> Path:
        """Default daily operation-log directory."""

        return Path(self.root) / "ydm_log"

    @property
    def output_train(self) -> Path:
        """Default training split file."""

        return Path(self.root) / "train.txt"

    @property
    def output_val(self) -> Path:
        """Default validation split file."""

        return Path(self.root) / "val.txt"

    @property
    def output_test(self) -> Path:
        """Default test split file."""

        return Path(self.root) / "test.txt"

    @property
    def output_dataset_yaml(self) -> Path:
        """Default dataset YAML file."""

        return Path(self.root) / "dataset.yaml"

    @property
    def dataset(self) -> YoloDataset:
        """Lazily load and cache the current YOLO dataset."""

        return self.load()

    @property
    def classes(self) -> ClassSchema:
        """Class schema from the lazily loaded dataset."""

        return self.load().classes

    @property
    def attributes(self) -> AttributeSchema | None:
        """Attribute schema from the lazily loaded dataset, if available."""

        return self.load().attributes

    def load(
        self,
        *,
        reload: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
    ) -> YoloDataset:
        """Load the dataset on demand and reuse it until explicitly reloaded.

        ``reload=True`` rebuilds the cached object from the current files. The
        cache is also invalidated automatically after manager operations that
        can modify labels, class/attribute schemas, or split files.
        """

        if self._dataset is None or reload:
            from yolo_data_manager.io.loader import load_yolo_dataset

            requested_split_file = (
                self.split_file if self.only_val else self._explicit_split_file
            )
            self._dataset = load_yolo_dataset(
                self.root,
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                class_file=self.class_file,
                attribute_file=self.attribute_file,
                task=self.task,
                split_file=requested_split_file,
                only_val=self.only_val,
                layout=self.layout,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
            )
        return self._dataset

    def _invalidate_dataset(self) -> None:
        """Drop the lazy dataset cache after a source-changing operation."""

        self._dataset = None

    def _warmup_(self) -> None:
        if self.init_layout:
            self.layout_detect(
                progress=self.init_layout_progress,
                progress_leave=self.init_layout_progress_leave,
            )
        check_kwargs = {
            "fill_missing_txt": self.init_check_fill_missing_txt,
            "workers": self.init_check_workers,
            "progress": self.init_check_progress,
            "progress_leave": self.init_check_progress_leave,
        }
        if isinstance(self.init_check, (str, Path)):
            self.check(out=str(self.init_check), **check_kwargs)
        elif self.init_check:
            self.check(**check_kwargs)

    # -- helpers ------------------------------------------------------------

    def _run(self, task: str, **params: Any) -> int:
        """Invoke *task* via ``run_task``, auto-filling common parameters."""
        if task in _ROOT_TASKS:
            requested_only_val = params.pop("only_val", None)
            only_val = (
                self.only_val if requested_only_val is None else requested_only_val
            )
            params.setdefault("root", self.root)
            params.setdefault("layout", self.layout)
            params.setdefault("images_dir", self.images_dir)
            params.setdefault("labels_dir", self.labels_dir)
            if self.class_file is not None:
                params.setdefault("class_file", self.class_file)
            if self.attribute_file is not None:
                params.setdefault("attribute_file", self.attribute_file)
            if only_val:
                params["only_val"] = True
            if only_val and self.split_file is not None:
                params.setdefault("split_file", self.split_file)
            elif not only_val and self._explicit_split_file is not None:
                params.setdefault("split_file", self._explicit_split_file)
        if task in _DATASET_MUTATING_TASKS:
            try:
                return run_task(task, **params)
            finally:
                self._invalidate_dataset()
        return run_task(task, **params)

    # -- check & stats -----------------------------------------------------

    def check(
        self,
        *,
        out: str | None = None,
        fill_missing_txt: bool = False,
        only_val: bool | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Validate the dataset (``ydm check``).

        The full JSON report is written to ``out``. If ``out`` is omitted,
        the CLI writes ``ydm_quality/check.json`` under the dataset root and prints
        only a compact terminal summary. Progress is enabled by default with
        multiple validation workers and ``leave=False``.
        """
        return self._run(
            "check",
            out=out,
            fill_missing_txt=fill_missing_txt,
            only_val=only_val,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def stats(
        self,
        *,
        out: str | None = None,
        class_csv: str | None = None,
        ann_csv: str | None = None,
        basic_info_csv: str | None = None,
        plots_dir: str | None = None,
        stats_list: str | list[str] | None = None,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Compute dataset statistics (``ydm stats``)."""
        return self._run(
            "stats",
            out=out,
            class_csv=class_csv,
            ann_csv=ann_csv,
            basic_info_csv=basic_info_csv,
            plots_dir=plots_dir,
            stats_list=stats_list,
            only_val=only_val,
            **kwargs,
        )

    # -- layout -------------------------------------------------------------

    def layout_detect(
        self, *, progress: bool = True, progress_leave: bool = False
    ) -> int:
        """Detect the YOLO layout under the manager root."""
        return run_task(
            "layout.detect",
            root=self.root,
            progress=progress,
            progress_leave=progress_leave,
        )

    # -- query --------------------------------------------------------------

    def query_class(
        self,
        class_: str | int | list[str | int],
        *,
        source: str = "gt",
        pred_root: str | Path | None = None,
        class_file: str | Path | None = None,
        out: str | None = None,
        copy_images: str | None = None,
        copy_labels: str | None = None,
        filtered_labels: bool = False,
        **kwargs: Any,
    ) -> int:
        """Query annotations by class (``ydm query class``)."""
        params: dict[str, Any] = {
            "class_": class_,
            "source": source,
            "pred_root": pred_root,
            "out": out,
            "copy_images": copy_images,
            "copy_labels": copy_labels,
            "filtered_labels": filtered_labels,
            **kwargs,
        }
        if class_file is not None:
            params["class_file"] = class_file
        return self._run(
            "query.class",
            **params,
        )

    def query_attr(
        self,
        name: str,
        *,
        value: str | list[str] | None = None,
        nonzero: bool = False,
        out: str | None = None,
        copy_images: str | None = None,
        copy_labels: str | None = None,
        filtered_labels: bool = False,
        **kwargs: Any,
    ) -> int:
        """Query annotations by attribute (``ydm query attr``)."""
        return self._run(
            "query.attr",
            name=name,
            value=value,
            nonzero=nonzero,
            out=out,
            copy_images=copy_images,
            copy_labels=copy_labels,
            filtered_labels=filtered_labels,
            **kwargs,
        )

    # -- dataset ------------------------------------------------------------

    def dataset_select(
        self,
        file: str,
        out: str | None = None,
        *,
        copy_images: bool = True,
        backup_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> int:
        """Select a subset via a txt file (``ydm dataset select``)."""
        return self._run(
            "dataset.select",
            file=file,
            out=out,
            copy_images=copy_images,
            backup_dir=backup_dir,
            **kwargs,
        )

    def dataset_normalize(
        self,
        out: str | None = None,
        *,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Normalize layout into flat images/labels (``ydm dataset normalize``)."""
        return self._run(
            "dataset.normalize",
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            **kwargs,
        )

    def dataset_split(
        self,
        *,
        train: float = 0.8,
        val: float = 0.2,
        test: float = 0.0,
        seed: int = 233,
        out: str | None = None,
        backup_dir: str | Path | None = None,
        absolute_paths: bool = True,
        train_include_list: str | Path | Sequence[str] | None = None,
        val_include_list: str | Path | Sequence[str] | None = None,
        val_source: str | Path | Sequence[str] | None = None,
        ensure_class_presence: bool = True,
        require_labels: bool = False,
        **kwargs: Any,
    ) -> int:
        """Write train/val/test split files (``ydm dataset split``)."""
        return self._run(
            "dataset.split",
            train=train,
            val=val,
            test=test,
            seed=seed,
            out=out,
            backup_dir=backup_dir,
            absolute_paths=absolute_paths,
            train_include_list=train_include_list,
            val_include_list=val_include_list,
            val_source=val_source,
            ensure_class_presence=ensure_class_presence,
            require_labels=require_labels,
            **kwargs,
        )

    @_logged_direct_operation("generate_attribute_com")
    def generate_attribute_com(
        self,
        *,
        labels_dir: str | Path | None = None,
        attribute_file: str | Path | None = None,
        output: str | Path | None = None,
        summary: str | Path | None = None,
        split: str = "train",
        mode: str = "cross",
        smoothing: float = 0.0,
        attribute_start: int = 2,
        num_attributes: int | None = None,
        filter_background: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Generate the GCA attribute co-occurrence matrix.

        This uses the same parser, matrix formulas, output names, and summary
        fields as the standalone ``generate_com.py`` utility.
        """
        from types import SimpleNamespace

        from yolo_data_manager.stats.attribute_com import generate

        args = SimpleNamespace(
            data_root=self.root,
            labels_dir=labels_dir,
            attribute_file=attribute_file,
            output=output,
            summary=summary,
            split=split,
            mode=mode,
            smoothing=smoothing,
            attribute_start=attribute_start,
            num_attributes=num_attributes,
            filter_background=filter_background,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        return generate(args)

    def dataset_extract_split(
        self,
        *,
        train_include_list: str | Path | Sequence[str] | None = None,
        val_include_list: str | Path | Sequence[str] | None = None,
        test_include_list: str | Path | Sequence[str] | None = None,
        out: str | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        backup: bool = False,
        backup_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> int:
        """Materialize train/val/test subsets into separate dataset dirs (``ydm dataset extract-split``)."""
        return self._run(
            "dataset.extract_split",
            train_include_list=train_include_list,
            val_include_list=val_include_list,
            test_include_list=test_include_list,
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            backup=backup,
            backup_dir=backup_dir,
            **kwargs,
        )

    @_logged_direct_operation("merge_manual_groups")
    def merge_manual_groups(
        self,
        *,
        group_src: str | Path | None = None,
        group_dir: str | Path | None = None,
        images_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
        link: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> Path:
        """Merge manually reviewed groups using the standalone-script logic."""
        from types import SimpleNamespace

        from yolo_data_manager.dataset.manual_group import merge_groups

        root = Path(self.root)
        args = SimpleNamespace(
            group_src=str(group_src if group_src is not None else root / "group_src"),
            group_dir=str(group_dir if group_dir is not None else root / "group"),
            images_dir=str(
                images_dir if images_dir is not None else root / self.images_dir
            ),
            out_dir=str(out_dir if out_dir is not None else root / "group_merged"),
            link=link,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        return merge_groups(args)

    @_logged_direct_operation("split_by_manual_group")
    def split_by_manual_group(
        self,
        *,
        images_dir: str | Path | None = None,
        labels_dir: str | Path | None = None,
        groups_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
        ratios: str | Sequence[float] = "0.80,0.10,0.10",
        seed: int = 42,
        attempts: int = 32,
        manual_groups_split: str = "train",
        attribute_file: str | Path | None = None,
        attribute_names: str | Sequence[str] | None = None,
        num_attributes: int | None = None,
        attribute_start: int = 2,
        nal: int = 2,
        class_file: str | Path | None = None,
        make_yolo: bool = False,
        link: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Split whole manual groups into train/val/test.

        The assignment objective, attribute coverage priority, reports, and
        optional YOLO materialization are identical to
        ``split_by_manual_group.py``.
        """
        from types import SimpleNamespace

        from yolo_data_manager.dataset.manual_group import split_dataset

        root = Path(self.root)
        ratio_text = (
            ratios
            if isinstance(ratios, str)
            else ",".join(str(value) for value in ratios)
        )
        attribute_name_text = (
            attribute_names
            if attribute_names is None or isinstance(attribute_names, str)
            else ",".join(str(value) for value in attribute_names)
        )
        args = SimpleNamespace(
            images_dir=str(
                images_dir if images_dir is not None else root / self.images_dir
            ),
            labels_dir=str(
                labels_dir if labels_dir is not None else root / self.labels_dir
            ),
            groups_dir=str(
                groups_dir if groups_dir is not None else root / "group_merged"
            ),
            out_dir=str(
                out_dir if out_dir is not None else root / "manual_group_split"
            ),
            ratios=ratio_text,
            seed=seed,
            attempts=attempts,
            manual_groups_split=manual_groups_split,
            attribute_file=(
                str(attribute_file)
                if attribute_file is not None
                else self.attribute_file
            ),
            attribute_names=attribute_name_text,
            num_attributes=num_attributes,
            attribute_start=attribute_start,
            nal=nal,
            class_file=(str(class_file) if class_file is not None else self.class_file),
            make_yolo=make_yolo,
            link=link,
            overwrite=overwrite,
        )
        return split_dataset(args)

    def dataset_yaml(
        self,
        *,
        out: str | None = None,
        train: str = "images/train",
        val: str = "images/val",
        test: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Write dataset.yaml (``ydm dataset yaml``)."""
        return self._run(
            "dataset.yaml",
            out=out,
            train=train,
            val=val,
            test=test,
            **kwargs,
        )

    def anno_update_by_size(
        self,
        out: str | None = None,
        *,
        out_data: str | Path | None = None,
        class_: str | list[str] | None = None,
        min_width: float | None = None,
        min_height: float | None = None,
        min_size_logic: str = "or",
        min_area: float | None = None,
        max_area: float | None = None,
        min_conf: float | None = None,
        class_rules: str | Path | Mapping[str, Mapping[str, Any]] | None = None,
        copy_images: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Update annotations by geometry/confidence (``ydm dataset filter``).

        When ``out`` is omitted, source label files are filtered in place and
        backed up under ``<dataset-root>/labels_backup``.  Supplying ``out``
        writes a separate filtered dataset instead.
        """
        _check_output_aliases(out, out_data)
        if isinstance(class_rules, Mapping):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                yaml.safe_dump(
                    dict(class_rules), f, allow_unicode=True, sort_keys=False
                )
                class_rules_path = f.name
            try:
                return self.anno_update_by_size(
                    out,
                    out_data=out_data,
                    class_=class_,
                    min_width=min_width,
                    min_height=min_height,
                    min_size_logic=min_size_logic,
                    min_area=min_area,
                    max_area=max_area,
                    min_conf=min_conf,
                    class_rules=class_rules_path,
                    copy_images=copy_images,
                    backup_dir=backup_dir,
                    dry_run=dry_run,
                    **kwargs,
                )
            finally:
                Path(class_rules_path).unlink(missing_ok=True)
        return self._run(
            "dataset.filter",
            out=out,
            out_data=out_data,
            class_=class_,
            min_width=min_width,
            min_height=min_height,
            min_size_logic=min_size_logic,
            min_area=min_area,
            max_area=max_area,
            min_conf=min_conf,
            class_rules=class_rules,
            copy_images=copy_images,
            backup_dir=backup_dir,
            dry_run=dry_run,
            **kwargs,
        )

    def dataset_filter(self, *args: Any, **kwargs: Any) -> int:
        """Backward-compatible alias for :meth:`anno_update_by_size`."""

        return self.anno_update_by_size(*args, **kwargs)

    def dataset_merge(
        self,
        roots: str | list[str],
        out: str | None = None,
        *,
        source_prefix: bool = True,
        rename_duplicates: bool = True,
        copy_images: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Merge multiple datasets (``ydm dataset merge``).

        *roots* may be a comma-separated string or a list of paths.
        """
        return run_task(
            "dataset.merge",
            roots=roots,
            out=out,
            task=self.task,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            source_prefix=source_prefix,
            rename_duplicates=rename_duplicates,
            copy_images=copy_images,
            backup_dir=backup_dir,
            dry_run=dry_run,
            **kwargs,
        )

    def dataset_duplicates(
        self,
        *,
        out: str | None = None,
        algorithm: str = "sha256",
        **kwargs: Any,
    ) -> int:
        """Find duplicate images by content hash (``ydm dataset duplicates``)."""
        return self._run(
            "dataset.duplicates",
            out=out,
            algorithm=algorithm,
            **kwargs,
        )

    def dataset_bad_images(
        self,
        *,
        out: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Find missing or corrupt images (``ydm dataset bad-images``)."""
        return self._run("dataset.bad_images", out=out, **kwargs)

    # -- annotation ---------------------------------------------------------

    def ann_delete_class(
        self,
        class_: str | list[str],
        out: str | None = None,
        *,
        out_data: str | Path | None = None,
        compact: bool = False,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Delete annotations of given classes (``ydm ann delete-class``)."""
        _check_output_aliases(out, out_data)
        return self._run(
            "ann.delete_class",
            class_=class_,
            out=out,
            out_data=out_data,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def ann_replace_class(
        self,
        from_: str | list[str],
        to: str,
        out: str | None = None,
        *,
        out_data: str | Path | None = None,
        compact: bool = False,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Replace source classes with a target class (``ydm ann replace-class``)."""
        _check_output_aliases(out, out_data)
        return self._run(
            "ann.replace_class",
            from_=from_,
            to=to,
            out=out,
            out_data=out_data,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_merge_class(
        self,
        from_: str | list[str] | Mapping[str, str | list[str]],
        to: str | None = None,
        *,
        out: str | None = None,
        out_data: str | Path | None = None,
        compact: bool = True,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Merge source classes into one (``ydm ann merge-class``)."""
        _check_output_aliases(out, out_data)
        if isinstance(from_, Mapping):
            requested_only_val = kwargs.pop("only_val", None)
            with operation_scope(
                self.root,
                "ann_merge_class",
                {
                    "from": from_,
                    "out": out,
                    "out_data": out_data,
                    "dry_run": dry_run,
                    "only_val": requested_only_val,
                },
            ):
                try:
                    return self._ann_merge_class_map(
                        from_,
                        out=out_data if out_data is not None else out,
                        compact=compact,
                        copy_images=copy_images,
                        keep_empty_labels=keep_empty_labels,
                        backup_dir=backup_dir,
                        dry_run=dry_run,
                        report=report,
                        workers=workers,
                        progress=progress,
                        progress_leave=progress_leave,
                        only_val=requested_only_val,
                        backup_source=out_data is None,
                    )
                finally:
                    self._invalidate_dataset()
        if to is None:
            raise ValueError("to is required when from_ is not a merge mapping")
        return self._run(
            "ann.merge_class",
            from_=from_,
            to=to,
            out=out,
            out_data=out_data,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def _ann_merge_class_map(
        self,
        merge_map: Mapping[str, str | list[str]],
        *,
        out: str | None,
        compact: bool,
        copy_images: bool,
        keep_empty_labels: bool,
        dry_run: bool,
        report: str | None,
        backup_dir: str | Path | None,
        workers: int,
        progress: bool,
        progress_leave: bool,
        only_val: bool | None,
        backup_source: bool = True,
    ) -> int:
        import json

        from yolo_data_manager.annotation.edit import EditReport, merge_classes
        from yolo_data_manager.io.loader import load_yolo_dataset
        from yolo_data_manager.io.output_paths import default_annotation_output
        from yolo_data_manager.io.writer import write_yolo_dataset

        requested_only_val = self.only_val if only_val is None else only_val
        split_file = (
            self.split_file if requested_only_val else self._explicit_split_file
        )
        dataset = load_yolo_dataset(
            self.root,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            class_file=self.class_file,
            attribute_file=self.attribute_file,
            task=self.task,
            split_file=split_file,
            only_val=requested_only_val,
            layout=self.layout,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        current = dataset
        reports: list[EditReport] = []
        for target, sources in merge_map.items():
            current, merge_report = merge_classes(
                current,
                _class_values(sources),
                target,
                compact=compact,
                add_missing=True,
            )
            reports.append(merge_report)

        rows = []
        for merge_report in reports:
            rows.extend(merge_report.rows)
        combined_report = EditReport(rows=rows)
        resolved_out = out or str(default_annotation_output(self.root, "merge_class"))
        resolved_report = report or str(
            default_annotation_output(self.root, "merge_class") / "edit_report.csv"
        )

        if not dry_run:
            write_yolo_dataset(
                current,
                resolved_out,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
                backup_dir=backup_dir,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
                backup=backup_source,
                operation="ann.merge_class",
                backup_result={
                    "action": "merge_class",
                    "changed": len(combined_report.rows),
                },
            )
        combined_report.write_csv(resolved_report)
        print(
            json.dumps(
                {
                    "changed": len(combined_report.rows),
                    "out": None if dry_run else resolved_out,
                    "report": resolved_report,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    def ann_rename_class(
        self,
        from_: str,
        to: str,
        out: str | None = None,
        *,
        out_data: str | Path | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Rename a class without changing IDs (``ydm ann rename-class``)."""
        _check_output_aliases(out, out_data)
        return self._run(
            "ann.rename_class",
            from_=from_,
            to=to,
            out=out,
            out_data=out_data,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_apply_map(
        self,
        map_file: str,
        out: str | None = None,
        *,
        out_data: str | Path | None = None,
        compact: bool = True,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Apply a class remap YAML (``ydm ann apply-map``)."""
        _check_output_aliases(out, out_data)
        return self._run(
            "ann.apply_map",
            map_file=map_file,
            out=out,
            out_data=out_data,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_restore_backup(
        self,
        timestamp: str | Path,
        *,
        backup_dir: str | Path | None = None,
        backup_current: bool = True,
        dry_run: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Restore files from a timestamped backup snapshot.

        ``source_labels`` restores the immutable first-source baseline;
        timestamped operation snapshots roll back later backups in reverse
        order. The current versions of the files being restored are backed up
        first under ``labels_backup`` unless ``backup_current=False``.
        """

        return self._run(
            "ann.restore_backup",
            timestamp=timestamp,
            backup_dir=backup_dir,
            backup_current=backup_current,
            dry_run=dry_run,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    @_logged_direct_operation("ann_update_from_map", invalidate_dataset=True)
    def ann_update_from_map(
        self,
        class_map: Mapping[str, Any],
        *,
        out_data: str | Path | None = None,
        compact: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | Path | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
    ) -> int:
        """Update the current dataset from a Python class-operation mapping.

        Unlike :meth:`ann_apply_map`, this API does not read a YAML map and
        updates the source labels and class schema in place by default.
        Supplying ``out_data`` writes a complete edited dataset to that new
        root and leaves the source untouched. In-place writes copy all source
        labels and the class schema into one timestamped backup under
        ``<dataset-root>/labels_backup`` by default.
        """

        if not isinstance(class_map, Mapping):
            raise TypeError("class_map must be a mapping")

        from yolo_data_manager.annotation.edit import EditReport
        from yolo_data_manager.annotation.remap import apply_class_map_data
        from yolo_data_manager.core.models import YoloImage
        from yolo_data_manager.core.schema import find_class_source
        from yolo_data_manager.io.backup import (
            LabelBackup,
            ensure_source_labels_backup,
        )
        from yolo_data_manager.io.loader import load_yolo_dataset, parse_label_file
        from yolo_data_manager.io.writer import write_yolo_dataset

        dataset = load_yolo_dataset(
            self.root,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            class_file=self.class_file,
            attribute_file=self.attribute_file,
            task=self.task,
            split_file=None,
            only_val=False,
            layout=self.layout,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        for orphan_label in dataset.orphan_labels:
            dataset.images.append(
                YoloImage(
                    path=orphan_label,
                    label_path=orphan_label,
                    annotations=parse_label_file(
                        orphan_label,
                        task=dataset.task,
                        attributes=dataset.attributes,
                    ),
                )
            )

        current, reports = apply_class_map_data(
            dataset,
            class_map,
            compact=compact,
        )
        rows = []
        for edit_report in reports:
            rows.extend(edit_report.rows)
        combined_report = EditReport(rows=rows)

        if self.class_file is not None:
            class_source = Path(self.class_file)
            if not class_source.is_absolute() and not class_source.is_file():
                class_source = Path(self.root) / class_source
        else:
            class_source = find_class_source(Path(self.root))
        if class_source is None:
            class_source = Path(self.root) / "class.txt"

        backup = None
        if not dry_run:
            if out_data is not None:
                output_root = Path(out_data).expanduser()
                if output_root.resolve() == Path(self.root).resolve():
                    raise ValueError(
                        "out_data must point to a different dataset root; "
                        "omit out_data to edit the source dataset in place"
                    )
                write_yolo_dataset(
                    current,
                    output_root,
                    copy_images=True,
                    keep_empty_labels=True,
                    workers=workers,
                    progress=progress,
                    progress_leave=progress_leave,
                    backup=False,
                    operation="ann.update_from_map",
                )
            else:
                resolved_backup_dir = (
                    self.output_labels_backup if backup_dir is None else backup_dir
                )
                ensure_source_labels_backup(
                    self.root,
                    resolved_backup_dir,
                    source_paths=(
                        image.label_path
                        for image in dataset.images
                        if image.label_path is not None
                    ),
                    extra_paths=[class_source],
                )
                backup = LabelBackup(
                    self.root,
                    resolved_backup_dir,
                    method="ann_update_from_map",
                )
                for image in dataset.images:
                    if image.label_path is not None:
                        backup.backup(image.label_path)
                if class_source.is_file():
                    backup.backup(class_source)

                for image in current.images:
                    if image.label_path is None:
                        continue
                    lines = [
                        annotation.to_yolo_line(include_confidence=False)
                        for annotation in image.annotations
                    ]
                    image.label_path.parent.mkdir(parents=True, exist_ok=True)
                    image.label_path.write_text(
                        "\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8",
                    )
                _write_updated_class_source(class_source, current.classes.names)

                backup.write_metadata(
                    method="ann_update_from_map",
                    result={
                        "action": "update_annotation_class_map",
                        "changed": len(combined_report.rows),
                        "deleted": sum(
                            1 for row in combined_report.rows if row.action == "delete"
                        ),
                        "classes": current.classes.names,
                    },
                )

        report_path = (
            Path(report)
            if report is not None
            else self.output_annotation / "update_from_map" / "edit_report.csv"
        )
        combined_report.write_csv(report_path)
        print(
            json.dumps(
                {
                    "changed": len(combined_report.rows),
                    "deleted": sum(
                        1 for row in combined_report.rows if row.action == "delete"
                    ),
                    "classes": current.classes.names,
                    "dry_run": dry_run,
                    "out_data": (
                        str(out_data) if out_data is not None and not dry_run else None
                    ),
                    "backup_dir": (
                        str(backup.snapshot_dir)
                        if backup is not None and backup.count
                        else None
                    ),
                    "backup_files": backup.count if backup is not None else 0,
                    "report": str(report_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    def ann_correct_from_crops(
        self,
        crops_dir: str | Path | Mapping[str | Path, str | int | None],
        to: str | int | None = None,
        *,
        out_data: str | Path | None = None,
        report: str | None = None,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Correct classes from crop filenames.

        ``crops_dir`` may also be a mapping of ``crop_directory`` to target
        class. In mapping mode, all directories are processed in one task and
        one label-backup snapshot is used for the complete operation.
        """
        crop_map_input = isinstance(crops_dir, Mapping)
        if crop_map_input:
            if to is not None:
                raise ValueError("to must be omitted when crops_dir is a mapping")
            crops_dir = {str(path): target for path, target in crops_dir.items()}
        cli_target = None if crop_map_input else ("none" if to is None else to)
        resolved_backup_dir = (
            backup_dir
            if out_data is not None
            else self.output_labels_backup if backup_dir is None else backup_dir
        )
        return self._run(
            "ann.correct_from_crops",
            crops_dir=crops_dir,
            to=cli_target,
            out_data=out_data,
            report=report,
            backup_dir=resolved_backup_dir,
            dry_run=dry_run,
            only_val=only_val,
            **kwargs,
        )

    def ann_correct_from_error_crops(
        self,
        crops_dir: str | Path | Mapping[str | Path, str | int | None],
        to: str | int | None = None,
        *,
        out_data: str | Path | None = None,
        pred_dir: str | Path | None = None,
        dedup_iou: float | None = 0.5,
        delete_pred_none: bool = False,
        replace_gt_from_pred: bool = False,
        report: str | None = None,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Correct GT classes, replace GT rows, and optionally add predictions from error crops.

        ``crops_dir`` may also be a mapping of ``error_crop_directory`` to
        target class. Mapping mode processes all directories in one task and
        uses one label-backup snapshot.
        """
        crop_map_input = isinstance(crops_dir, Mapping)
        if crop_map_input:
            if to is not None:
                raise ValueError("to must be omitted when crops_dir is a mapping")
            crops_dir = {str(path): target for path, target in crops_dir.items()}
        cli_target = None if crop_map_input else ("none" if to is None else to)
        resolved_backup_dir = (
            backup_dir
            if out_data is not None
            else self.output_labels_backup if backup_dir is None else backup_dir
        )
        return self._run(
            "ann.correct_from_error_crops",
            crops_dir=crops_dir,
            out_data=out_data,
            pred_dir=pred_dir,
            dedup_iou=dedup_iou,
            delete_pred_none=delete_pred_none,
            replace_gt_from_pred=replace_gt_from_pred,
            to=cli_target,
            report=report,
            backup_dir=resolved_backup_dir,
            dry_run=dry_run,
            only_val=only_val,
            **kwargs,
        )

    def ann_att_correct_from_crops(
        self,
        crops_dir: str | Path | Mapping[str | Path, Any],
        name: str | None = None,
        value: str | float | None = None,
        *,
        out_data: str | Path | None = None,
        attribute_name: str | None = None,
        attribute_value: str | float | None = None,
        attribute: str | None = None,
        to: str | float | None = None,
        report: str | None = None,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Correct attributes from standard ``vis_crop`` directories.

        A mapping can assign a different attribute rule to each crop
        directory, for example::

            {
                "crops/defect": {"name": "defect", "value": "yes"},
                "crops/material": {"material": "metal"},
            }
        """

        resolved_name = (
            attribute_name
            if attribute_name is not None
            else attribute
            if attribute is not None
            else name
        )
        resolved_value = (
            attribute_value
            if attribute_value is not None
            else to
            if to is not None
            else value
        )
        if not isinstance(crops_dir, Mapping) and (
            resolved_name is None or resolved_value is None
        ):
            raise ValueError("name and value are required")
        resolved_backup_dir = (
            backup_dir
            if out_data is not None
            else self.output_labels_backup if backup_dir is None else backup_dir
        )
        return self._run(
            "ann.att_correct_from_crops",
            crops_dir=crops_dir,
            out_data=out_data,
            name=resolved_name,
            value=resolved_value,
            report=report,
            backup_dir=resolved_backup_dir,
            dry_run=dry_run,
            only_val=only_val,
            **kwargs,
        )

    def ann_att_correct_from_error_crops(
        self,
        crops_dir: str | Path | Mapping[str | Path, Any],
        name: str | None = None,
        value: str | float | None = None,
        *,
        out_data: str | Path | None = None,
        attribute_name: str | None = None,
        attribute_value: str | float | None = None,
        attribute: str | None = None,
        to: str | float | None = None,
        report: str | None = None,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Correct attributes from one or more error-crop directories.

        A mapping may be used to assign different attribute rules to crop
        directories, for example::

            {
                "crops/defect": {"name": "defect", "value": "yes"},
                "crops/material": {"material": "metal"},
            }
        """

        resolved_name = (
            attribute_name
            if attribute_name is not None
            else attribute
            if attribute is not None
            else name
        )
        resolved_value = (
            attribute_value
            if attribute_value is not None
            else to
            if to is not None
            else value
        )
        if not isinstance(crops_dir, Mapping) and (
            resolved_name is None or resolved_value is None
        ):
            raise ValueError("name and value are required")
        resolved_backup_dir = (
            backup_dir
            if out_data is not None
            else self.output_labels_backup if backup_dir is None else backup_dir
        )
        return self._run(
            "ann.att_correct_from_error_crops",
            crops_dir=crops_dir,
            out_data=out_data,
            name=resolved_name,
            value=resolved_value,
            report=report,
            backup_dir=resolved_backup_dir,
            dry_run=dry_run,
            only_val=only_val,
            **kwargs,
        )

    @_logged_direct_operation("ann_att_update_from_map", invalidate_dataset=True)
    def ann_att_update_from_map(
        self,
        attribute_map: Mapping[str, Any],
        *,
        out_data: str | Path | None = None,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | Path | None = None,
        only_val: bool | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
    ) -> int:
        """Update attributes from a Python mapping.

        The source dataset is updated in place by default.  ``out_data``
        writes the edited annotations and schema to a new dataset instead.
        """

        if not isinstance(attribute_map, Mapping):
            raise TypeError("attribute_map must be a mapping")

        from yolo_data_manager.annotation.edit import (
            EditReport,
            set_attributes_from_map,
        )
        from yolo_data_manager.io.loader import load_yolo_dataset
        from yolo_data_manager.io.writer import (
            write_yolo_dataset,
            write_yolo_labels_in_place,
        )

        requested_only_val = self.only_val if only_val is None else bool(only_val)
        split_file = (
            self.split_file if requested_only_val else self._explicit_split_file
        )
        dataset = load_yolo_dataset(
            self.root,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            class_file=self.class_file,
            attribute_file=self.attribute_file,
            task=self.task,
            split_file=split_file,
            only_val=requested_only_val,
            layout=self.layout,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        current, edit_report = set_attributes_from_map(dataset, attribute_map)
        if not isinstance(edit_report, EditReport):
            raise TypeError("attribute update did not produce an edit report")

        backup = None
        if not dry_run:
            if out_data is not None:
                output_root = Path(out_data).expanduser()
                if output_root.resolve() == Path(self.root).resolve():
                    raise ValueError(
                        "out_data must point to a different dataset root; "
                        "omit out_data to edit the source dataset in place"
                    )
                write_yolo_dataset(
                    current,
                    output_root,
                    copy_images=True,
                    keep_empty_labels=True,
                    workers=workers,
                    progress=progress,
                    progress_leave=progress_leave,
                    backup=False,
                    operation="ann.att_update_from_map",
                )
            else:
                resolved_backup_dir = (
                    self.output_labels_backup if backup_dir is None else backup_dir
                )
                backup = write_yolo_labels_in_place(
                    current,
                    workers=workers,
                    progress=progress,
                    progress_leave=progress_leave,
                    backup_dir=resolved_backup_dir,
                    operation="ann_att_update_from_map",
                    backup_result={
                        "action": "update_annotation_attribute_map",
                        "changed": len(edit_report.rows),
                    },
                )

        report_path = (
            Path(report)
            if report is not None
            else self.output_annotation / "att_update_from_map" / "edit_report.csv"
        )
        edit_report.write_csv(report_path)
        print(
            json.dumps(
                {
                    "changed": len(edit_report.rows),
                    "dry_run": dry_run,
                    "out_data": (
                        str(out_data) if out_data is not None and not dry_run else None
                    ),
                    "backup_dir": (
                        str(backup.snapshot_dir)
                        if backup is not None and backup.count
                        else None
                    ),
                    "backup_files": backup.count if backup is not None else 0,
                    "report": str(report_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    def ann_set_attr(
        self,
        name: str,
        value: str,
        *,
        class_: str | list[str] | None = None,
        where_value: str | None = None,
        out: str | None = None,
        out_data: str | Path | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Backward-compatible attribute setter.

        Without ``out`` or ``out_data`` it updates source labels in place.
        Supplying either output argument writes a separate dataset.
        """

        _check_output_aliases(out, out_data)
        if out is not None or out_data is not None:
            return self._run(
                "ann.set_attr",
                name=name,
                value=value,
                class_=class_,
                where_value=where_value,
                out=out,
                out_data=out_data,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
                backup_dir=backup_dir,
                dry_run=dry_run,
                report=report,
                **kwargs,
            )

        workers = int(kwargs.pop("workers", 8))
        progress = bool(kwargs.pop("progress", True))
        progress_leave = bool(kwargs.pop("progress_leave", False))
        only_val = kwargs.pop("only_val", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected ann_set_attr arguments: {unexpected}")
        with operation_scope(
            self.root,
            "ann_set_attr",
            {
                "name": name,
                "value": value,
                "class": class_,
                "where_value": where_value,
                "dry_run": dry_run,
            },
        ):
            return self.ann_att_update_from_map(
                {
                    "update": [
                        {
                            "name": name,
                            "value": value,
                            "class_": class_,
                            "where_value": where_value,
                        }
                    ]
                },
                backup_dir=backup_dir,
                dry_run=dry_run,
                report=report,
                only_val=only_val,
                out_data=out_data,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
            )

    def ann_delete_attr(
        self,
        name: str,
        *,
        value: str | list[str] | None = None,
        nonzero: bool = False,
        out: str | None = None,
        out_data: str | Path | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Delete annotations by attribute (``ydm ann delete-attr``)."""
        _check_output_aliases(out, out_data)
        return self._run(
            "ann.delete_attr",
            name=name,
            value=value,
            nonzero=nonzero,
            out=out,
            out_data=out_data,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    # -- visualization ------------------------------------------------------

    def vis_draw(
        self,
        out: str | None = None,
        *,
        style: str = "cv2",
        limit: int | None = None,
        show_conf: bool = False,
        conf: float | None = None,
        mask_alpha: int = 64,
        fill_mask: bool = True,
        show_attrs: bool = False,
        show_id: bool = False,
        filter_level: Sequence[int | str] | int | str | None = (1,),
        att_seperate: bool = False,
        clean: bool = True,
        only_val: bool | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Draw bounding-boxes / masks on images (``ydm vis draw``)."""
        return self._run(
            "vis.draw",
            out=out,
            style=style,
            limit=limit,
            show_conf=show_conf,
            conf=conf,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attrs=show_attrs,
            show_id=show_id,
            filter_level=filter_level,
            att_seperate=att_seperate,
            clean=clean,
            only_val=only_val,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def vis_crop(
        self,
        out: str | None = None,
        *,
        style: str = "cv2",
        keep_shape: bool = False,
        min_size: int = 1,
        padding: float = 0,
        conf: float | None = None,
        by_attr: bool = False,
        filter_level: Sequence[int | str] | int | str | None = (1,),
        att_seperate: bool = False,
        clean: bool = True,
        only_val: bool | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Crop annotation regions into class folders (``ydm vis crop``)."""
        return self._run(
            "vis.crop",
            out=out,
            style=style,
            keep_shape=keep_shape,
            min_size=min_size,
            padding=padding,
            conf=conf,
            by_attr=by_attr,
            filter_level=filter_level,
            att_seperate=att_seperate,
            clean=clean,
            only_val=only_val,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def vis_manual_box(
        self,
        image: str | Path,
        *,
        label: str | Path | None = None,
        class_id: int | None = None,
        max_width: int = 1400,
        max_height: int = 900,
        min_pixels: int = 2,
        precision: int = 6,
        show_existing: bool = True,
        mask_outside: bool = False,
        out: str | Path | None = None,
        only_val: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Draw one temporary box without changing the source label."""
        return self._run(
            "vis.manual_box",
            image=image,
            label=label,
            class_id=class_id,
            max_width=max_width,
            max_height=max_height,
            min_pixels=min_pixels,
            precision=precision,
            show_existing=show_existing,
            mask_outside=mask_outside,
            out=out,
            only_val=only_val,
            **kwargs,
        )

    # -- export -------------------------------------------------------------

    def export_coco(self, out: str | None = None, **kwargs: Any) -> int:
        """Export to COCO JSON (``ydm export coco``)."""
        return self._run("export.coco", out=out, **kwargs)

    def export_xany(self, out: str | None = None, **kwargs: Any) -> int:
        """Export to x-anylabeling JSON (``ydm export xany``)."""
        return self._run("export.xany", out=out, **kwargs)

    # -- import -------------------------------------------------------------

    def import_labelme(
        self,
        json_dir: str,
        out: str | None = None,
        *,
        classes: str | list[str] | None = None,
        attribute_file: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Import LabelMe JSON directory as YOLO (``ydm import labelme``)."""
        return run_task(
            "import.labelme",
            json_dir=json_dir,
            out=out,
            task=self.task,
            classes=classes,
            attribute_file=attribute_file,
            **kwargs,
        )

    def import_coco(
        self,
        json_path: str,
        images_dir: str,
        out: str | None = None,
        *,
        classes: str | list[str] | None = None,
        copy_images: bool = True,
        **kwargs: Any,
    ) -> int:
        """Import COCO JSON as YOLO (``ydm import coco``)."""
        return run_task(
            "import.coco",
            json_path=json_path,
            images_dir=images_dir,
            out=out,
            task=self.task,
            classes=classes,
            copy_images=copy_images,
            **kwargs,
        )

    def import_voc(
        self,
        annotations_dir: str,
        images_dir: str,
        out: str | None = None,
        *,
        classes: str | list[str] | None = None,
        skip_difficult: bool = True,
        **kwargs: Any,
    ) -> int:
        """Import Pascal VOC XML as YOLO (``ydm import voc``)."""
        return run_task(
            "import.voc",
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            out=out,
            classes=classes,
            skip_difficult=skip_difficult,
            **kwargs,
        )

    def import_mask(
        self,
        images_dir: str,
        masks_dir: str,
        out: str | None = None,
        *,
        class_map: str | Path | Mapping[Any, str] | None = None,
        background: int | str = 0,
        min_area: int = 1,
        copy_images: bool = True,
        **kwargs: Any,
    ) -> int:
        """Import semantic segmentation masks as YOLO segmentation (``ydm import mask``)."""
        if isinstance(class_map, Mapping):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                yaml.safe_dump(dict(class_map), f, allow_unicode=True, sort_keys=False)
                class_map_path = f.name
            try:
                return self.import_mask(
                    images_dir,
                    masks_dir,
                    out,
                    class_map=class_map_path,
                    background=background,
                    min_area=min_area,
                    copy_images=copy_images,
                    **kwargs,
                )
            finally:
                Path(class_map_path).unlink(missing_ok=True)
        return run_task(
            "import.mask",
            images_dir=images_dir,
            masks_dir=masks_dir,
            out=out,
            class_map=class_map,
            background=background,
            min_area=min_area,
            copy_images=copy_images,
            **kwargs,
        )

    # -- convert ------------------------------------------------------------

    def convert_seg2det(
        self,
        out: str | None = None,
        *,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Convert segmentation labels to detection boxes (``ydm convert seg2det``)."""
        return self._run(
            "convert.seg2det",
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            **kwargs,
        )

    def convert_pseudo(
        self,
        *,
        out: str | None = None,
        conf: float = 0.0,
        drop_confidence: bool = True,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Convert predictions to pseudo-labels (``ydm convert pseudo``)."""
        return self._run(
            "convert.pseudo",
            out=out,
            conf=conf,
            drop_confidence=drop_confidence,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            backup_dir=backup_dir,
            dry_run=dry_run,
            **kwargs,
        )

    def resize_images(
        self,
        out: str | Path | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        scale: float | None = None,
        keep_ratio: bool = True,
        interpolation: str = "lanczos",
        fill_color: int | Sequence[int] = (114, 114, 114),
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Resize dataset images and transform labels when letterboxing."""
        return self._run(
            "convert.resize",
            out=out,
            width=width,
            height=height,
            scale=scale,
            keep_ratio=keep_ratio,
            interpolation=interpolation,
            fill_color=fill_color,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def remap_filenames(
        self,
        out: str | Path | None = None,
        *,
        digits: int | None = None,
        start: int = 0,
        mapping_file: str | Path | None = None,
        dry_run: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Copy the dataset with numeric image/label filenames."""
        return self._run(
            "convert.filename_remap",
            out=out,
            digits=digits,
            start=start,
            mapping_file=mapping_file,
            dry_run=dry_run,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    # -- VLM ---------------------------------------------------------------

    def vlm_auto_label(
        self,
        out: str | Path | None = None,
        *,
        config: str | Path | None = None,
        provider: str | None = None,
        prompt: str | None = None,
        limit: int | None = None,
        allow_new_classes: bool = False,
        copy_images: bool = True,
        dry_run: bool = False,
        workers: int | None = None,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Generate YOLO labels with a configured VLM."""
        return self._run(
            "vlm.auto_label",
            out=out,
            config=config,
            provider=provider,
            prompt=prompt,
            limit=limit,
            allow_new_classes=allow_new_classes,
            copy_images=copy_images,
            dry_run=dry_run,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def vlm_verify_errors(
        self,
        out: str | Path | None = None,
        *,
        error_dir: str | Path | None = None,
        pred_root: str | Path | None = None,
        config: str | Path | None = None,
        provider: str | None = None,
        mode: str = "all",
        confidence: float = 0.0,
        limit: int | None = None,
        apply: bool = False,
        yes: bool = False,
        backup_dir: str | Path | None = None,
        dry_run: bool = False,
        workers: int | None = None,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Verify error-analysis crops and optionally apply the correction plan."""
        return self._run(
            "vlm.verify_errors",
            out=out,
            error_dir=error_dir,
            pred_root=pred_root,
            config=config,
            provider=provider,
            mode=mode,
            confidence=confidence,
            limit=limit,
            apply=apply,
            yes=yes,
            backup_dir=backup_dir,
            dry_run=dry_run,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def ydm_assistant(
        self,
        intent: str,
        *,
        config: str | Path | None = None,
        provider: str | None = None,
        execute: bool = False,
        yes: bool = False,
        **kwargs: Any,
    ) -> int:
        """Convert natural-language intent into a safe YoloManager operation."""
        return self._run(
            "vlm.assistant",
            intent=intent,
            config=config,
            provider=provider,
            execute=execute,
            yes=yes,
            **kwargs,
        )

    def vlm_assistant(self, intent: str, **kwargs: Any) -> int:
        """Alias for ydm_assistant."""
        return self.ydm_assistant(intent, **kwargs)

    # -- evaluation ---------------------------------------------------------

    def eval_compare(
        self,
        gt_root: str,
        pred_root: str,
        out: str | None = None,
        *,
        iou: float = 0.5,
        conf: float | None = None,
        **kwargs: Any,
    ) -> int:
        """Compare predictions against GT (``ydm eval compare``)."""
        return run_task(
            "eval.compare",
            gt_root=gt_root,
            pred_root=pred_root,
            out=out,
            iou=iou,
            conf=conf,
            task=self.task,
            layout=self.layout,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            **kwargs,
        )

    def eval_review_pack(
        self,
        gt_root: str,
        pred_root: str,
        out: str | None = None,
        *,
        csv: str | None = None,
        iou: float = 0.5,
        conf: float | None = None,
        status: str | list[str] = "fp,fn",
        **kwargs: Any,
    ) -> int:
        """Generate FP/FN review package (``ydm eval review-pack``)."""
        return run_task(
            "eval.review_pack",
            gt_root=gt_root,
            pred_root=pred_root,
            out=out,
            csv=csv,
            iou=iou,
            conf=conf,
            status=status,
            task=self.task,
            layout=self.layout,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            **kwargs,
        )

    def eval_error_analysis(
        self,
        pred_root: str,
        out: str | None = None,
        *,
        gt_root: str | None = None,
        match_iou: float = 0.5,
        low_iou: float = 0.1,
        conf_thres: float = 0.0,
        nms_iou: float | None = 0.5,
        duplicate_iou: float = 0.9,
        val_source: str | None = None,
        only_val: bool | None = None,
        class_file: str | None = None,
        attribute_file: str | None = None,
        class_rules: str | Path | Mapping[int | str, Mapping[str, Any]] | None = None,
        class_: str | list[str] | None = None,
        exclude_class_: str | list[str] | None = None,
        min_width: float | None = None,
        min_height: float | None = None,
        min_area: float | None = None,
        min_size_logic: str = "or",
        min_pixels: float | None = None,
        review: bool = True,
        crop_padding: int = 12,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        review_workers: int | None = None,
        review_progress: bool = True,
        review_progress_leave: bool = False,
        copy_pred_txt: bool = True,
        conf_curve: bool = False,
        **kwargs: Any,
    ) -> int:
        """Analyze prediction errors; optionally write a 0.1-0.5 confidence curve."""
        resolved_gt_root = gt_root or self.root
        requested_only_val = self.only_val if only_val is None else only_val
        resolved_val_source = val_source
        if resolved_val_source is None and requested_only_val:
            resolved_val_source = self.split_file or _default_existing_path(
                self.root, "val.txt"
            )
        resolved_class_file = (
            class_file
            or self.class_file
            or _default_existing_path(self.root, "class.txt")
        )
        temporary_class_rules_path: str | None = None
        class_rules_path: str | Path | None = class_rules
        if isinstance(class_rules, Mapping):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                yaml.safe_dump(
                    dict(class_rules), f, allow_unicode=True, sort_keys=False
                )
                temporary_class_rules_path = f.name
            class_rules_path = temporary_class_rules_path
        try:
            return run_task(
                "eval.error_analysis",
                gt_root=resolved_gt_root,
                pred_root=pred_root,
                out=out,
                match_iou=match_iou,
                low_iou=low_iou,
                conf_thres=conf_thres,
                nms_iou=nms_iou,
                no_nms=nms_iou is None,
                duplicate_iou=duplicate_iou,
                val_source=resolved_val_source,
                only_val=requested_only_val,
                class_file=resolved_class_file,
                attribute_file=attribute_file or self.attribute_file,
                class_rules=class_rules_path,
                class_=class_,
                exclude_class_=exclude_class_,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                min_size_logic=min_size_logic,
                min_pixels=min_pixels,
                review=review,
                crop_padding=crop_padding,
                review_workers=review_workers,
                review_progress=review_progress,
                review_progress_leave=review_progress_leave,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
                copy_pred_txt=copy_pred_txt,
                conf_curve=conf_curve,
                task=self.task,
                layout=self.layout,
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                **kwargs,
            )
        finally:
            if temporary_class_rules_path is not None:
                Path(temporary_class_rules_path).unlink(missing_ok=True)

    def eval_error_analysis_common(
        self,
        error_dirs: Sequence[str | Path] | Mapping[str, str | Path],
        out: str | None = None,
        *,
        iou: float = 0.5,
        copy_crops: bool = True,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Extract errors shared by multiple ``eval_error_analysis`` runs."""

        if isinstance(error_dirs, Mapping):
            resolved_error_dirs = list(error_dirs.values())
        elif isinstance(error_dirs, (str, Path)):
            resolved_error_dirs = [error_dirs]
        else:
            resolved_error_dirs = list(error_dirs)

        return run_task(
            "eval.error_analysis_common",
            error_dirs=resolved_error_dirs,
            out=out,
            iou=iou,
            copy_crops=copy_crops,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        )

    def eval_metrics(
        self,
        pred_root: str,
        *,
        gt_root: str | None = None,
        out: str | None = None,
        csv: str | None = None,
        print_table: bool = False,
        show_original: bool = False,
        class_: str | list[str] | None = None,
        exclude_class_: str | list[str] | None = None,
        merge_class_map: Mapping[str | int, str | int | Sequence[str | int]]
        | str
        | Path
        | None = None,
        conf_thres: float = 0.0,
        nms_iou: float | None = 0.5,
        min_width: float | None = None,
        min_height: float | None = None,
        min_area: float | None = None,
        min_size_logic: str = "or",
        min_pixels: float | None = None,
        class_rules: str | Path | Mapping[int | str, Mapping[str, Any]] | None = None,
        ignore_empty_classes: bool = True,
        val_source: str | None = None,
        only_val: bool | None = None,
        class_file: str | None = None,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Compute precision/recall/mAP from GT and prediction txt (``ydm eval metrics``)."""
        resolved_gt_root = gt_root or self.root
        requested_only_val = self.only_val if only_val is None else only_val
        resolved_val_source = val_source
        if resolved_val_source is None and requested_only_val:
            resolved_val_source = self.split_file or _default_existing_path(
                self.root, "val.txt"
            )
        resolved_class_file = (
            class_file
            or self.class_file
            or _default_existing_path(self.root, "class.txt")
        )
        temporary_class_rules_path: str | None = None
        class_rules_path: str | Path | None = class_rules
        if isinstance(class_rules, Mapping):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                yaml.safe_dump(
                    dict(class_rules), f, allow_unicode=True, sort_keys=False
                )
                temporary_class_rules_path = f.name
            class_rules_path = temporary_class_rules_path
        try:
            return run_task(
                "eval.metrics",
                gt_root=resolved_gt_root,
                pred_root=pred_root,
                out=out,
                csv=csv,
                print_table=print_table,
                show_original=show_original,
                class_=class_,
                exclude_class_=exclude_class_,
                merge_class_map=merge_class_map,
                conf_thres=conf_thres,
                nms_iou=nms_iou,
                no_nms=nms_iou is None,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                min_size_logic=min_size_logic,
                min_pixels=min_pixels,
                class_rules=class_rules_path,
                ignore_empty_classes=ignore_empty_classes,
                val_source=resolved_val_source,
                only_val=requested_only_val,
                class_file=resolved_class_file,
                workers=workers,
                progress=progress,
                progress_leave=progress_leave,
                task=self.task,
                layout=self.layout,
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                **kwargs,
            )
        finally:
            if temporary_class_rules_path is not None:
                Path(temporary_class_rules_path).unlink(missing_ok=True)
