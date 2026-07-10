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
    "eval.error_analysis": ("eval", "error-analysis"),
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


def _class_values(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _default_existing_path(root: str, name: str) -> str | None:
    path = Path(root) / name
    return str(path) if path.exists() else None


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
        "dataset.yaml",
        "dataset.filter",
        "dataset.duplicates",
        "dataset.bad_images",
        "ann.delete_class",
        "ann.replace_class",
        "ann.merge_class",
        "ann.rename_class",
        "ann.apply_map",
        "ann.set_attr",
        "ann.delete_attr",
        "vis.draw",
        "vis.crop",
        "export.coco",
        "export.xany",
        "convert.seg2det",
        "convert.pseudo",
    }
)


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
        init_layout: bool = True,
        init_check: bool | str | Path = True,
        init_check_fill_missing_txt: bool = False,
    ) -> None:
        self.root = str(root)
        self.layout = layout
        self.task = task
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_file = class_file
        self.attribute_file = attribute_file
        self.split_file = split_file
        self.init_layout = init_layout
        self.init_check = init_check
        self.init_check_fill_missing_txt = init_check_fill_missing_txt

        self._warmup_()

    def _warmup_(self) -> None:
        if self.init_layout:
            self.layout_detect()
        if isinstance(self.init_check, (str, Path)):
            self.check(
                out=str(self.init_check),
                fill_missing_txt=self.init_check_fill_missing_txt,
            )
        elif self.init_check:
            self.check(fill_missing_txt=self.init_check_fill_missing_txt)

    # -- helpers ------------------------------------------------------------

    def _run(self, task: str, **params: Any) -> int:
        """Invoke *task* via ``run_task``, auto-filling common parameters."""
        if task in _ROOT_TASKS:
            params.setdefault("root", self.root)
            params.setdefault("layout", self.layout)
            params.setdefault("images_dir", self.images_dir)
            params.setdefault("labels_dir", self.labels_dir)
            if self.class_file is not None:
                params.setdefault("class_file", self.class_file)
            if self.attribute_file is not None:
                params.setdefault("attribute_file", self.attribute_file)
            if self.split_file is not None:
                params.setdefault("split_file", self.split_file)
        return run_task(task, **params)

    # -- check & stats -----------------------------------------------------

    def check(
        self,
        *,
        out: str | None = None,
        fill_missing_txt: bool = False,
        **kwargs: Any,
    ) -> int:
        """Validate the dataset (``ydm check``)."""
        return self._run(
            "check",
            out=out,
            fill_missing_txt=fill_missing_txt,
            **kwargs,
        )

    def stats(
        self,
        *,
        out: str | None = None,
        class_csv: str | None = None,
        ann_csv: str | None = None,
        attr_csv: str | None = None,
        plots_dir: str | None = None,
        stats_list: str | list[str] | None = None,
        **kwargs: Any,
    ) -> int:
        """Compute dataset statistics (``ydm stats``)."""
        return self._run(
            "stats",
            out=out,
            class_csv=class_csv,
            ann_csv=ann_csv,
            attr_csv=attr_csv,
            plots_dir=plots_dir,
            stats_list=stats_list,
            **kwargs,
        )

    # -- layout -------------------------------------------------------------

    def layout_detect(self) -> int:
        """Detect the YOLO layout under the manager root."""
        return run_task("layout.detect", root=self.root)

    # -- query --------------------------------------------------------------

    def query_class(
        self,
        class_: str | list[str],
        *,
        out: str | None = None,
        copy_images: str | None = None,
        copy_labels: str | None = None,
        filtered_labels: bool = False,
        **kwargs: Any,
    ) -> int:
        """Query annotations by class (``ydm query class``)."""
        return self._run(
            "query.class",
            class_=class_,
            out=out,
            copy_images=copy_images,
            copy_labels=copy_labels,
            filtered_labels=filtered_labels,
            **kwargs,
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
        out: str,
        *,
        copy_images: bool = True,
        **kwargs: Any,
    ) -> int:
        """Select a subset via a txt file (``ydm dataset select``)."""
        return self._run(
            "dataset.select",
            file=file,
            out=out,
            copy_images=copy_images,
            **kwargs,
        )

    def dataset_normalize(
        self,
        out: str,
        *,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Normalize layout into flat images/labels (``ydm dataset normalize``)."""
        return self._run(
            "dataset.normalize",
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
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
        absolute_paths: bool = False,
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
            absolute_paths=absolute_paths,
            **kwargs,
        )

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

    def dataset_filter(
        self,
        out: str,
        *,
        class_: str | list[str] | None = None,
        min_width: float | None = None,
        min_height: float | None = None,
        min_area: float | None = None,
        max_area: float | None = None,
        min_conf: float | None = None,
        copy_images: bool = True,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Filter annotations by geometry/confidence (``ydm dataset filter``)."""
        return self._run(
            "dataset.filter",
            out=out,
            class_=class_,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            max_area=max_area,
            min_conf=min_conf,
            copy_images=copy_images,
            dry_run=dry_run,
            **kwargs,
        )

    def dataset_merge(
        self,
        roots: str | list[str],
        out: str,
        *,
        source_prefix: bool = True,
        rename_duplicates: bool = True,
        copy_images: bool = True,
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
        out: str,
        *,
        compact: bool = False,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Delete annotations of given classes (``ydm ann delete-class``)."""
        return self._run(
            "ann.delete_class",
            class_=class_,
            out=out,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_replace_class(
        self,
        from_: str | list[str],
        to: str,
        out: str,
        *,
        compact: bool = False,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Replace source classes with a target class (``ydm ann replace-class``)."""
        return self._run(
            "ann.replace_class",
            from_=from_,
            to=to,
            out=out,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
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
        compact: bool = True,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Merge source classes into one (``ydm ann merge-class``)."""
        if isinstance(from_, Mapping):
            return self._ann_merge_class_map(
                from_,
                out=out,
                compact=compact,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
                dry_run=dry_run,
                report=report,
            )
        if to is None:
            raise ValueError("to is required when from_ is not a merge mapping")
        return self._run(
            "ann.merge_class",
            from_=from_,
            to=to,
            out=out,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
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
    ) -> int:
        import json

        from yolo_data_manager.annotation.edit import EditReport, merge_classes
        from yolo_data_manager.io.loader import load_yolo_dataset
        from yolo_data_manager.io.writer import write_yolo_dataset

        dataset = load_yolo_dataset(
            self.root,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            class_file=self.class_file,
            attribute_file=self.attribute_file,
            task=self.task,
            split_file=self.split_file,
            layout=self.layout,
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

        if not dry_run:
            if out is None:
                raise ValueError("out is required when dry_run=False")
            write_yolo_dataset(
                current,
                out,
                copy_images=copy_images,
                keep_empty_labels=keep_empty_labels,
            )
        if report:
            combined_report.write_csv(report)
        print(json.dumps({"changed": len(combined_report.rows), "out": None if dry_run else out}, indent=2, ensure_ascii=False))
        return 0

    def ann_rename_class(
        self,
        from_: str,
        to: str,
        out: str,
        *,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Rename a class without changing IDs (``ydm ann rename-class``)."""
        return self._run(
            "ann.rename_class",
            from_=from_,
            to=to,
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_apply_map(
        self,
        map_file: str,
        out: str,
        *,
        compact: bool = True,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Apply a class remap YAML (``ydm ann apply-map``)."""
        return self._run(
            "ann.apply_map",
            map_file=map_file,
            out=out,
            compact=compact,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_set_attr(
        self,
        name: str,
        value: str,
        *,
        class_: str | list[str] | None = None,
        where_value: str | None = None,
        out: str | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Set an attribute on annotations (``ydm ann set-attr``)."""
        return self._run(
            "ann.set_attr",
            name=name,
            value=value,
            class_=class_,
            where_value=where_value,
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    def ann_delete_attr(
        self,
        name: str,
        *,
        value: str | list[str] | None = None,
        nonzero: bool = False,
        out: str | None = None,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        report: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Delete annotations by attribute (``ydm ann delete-attr``)."""
        return self._run(
            "ann.delete_attr",
            name=name,
            value=value,
            nonzero=nonzero,
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
            dry_run=dry_run,
            report=report,
            **kwargs,
        )

    # -- visualization ------------------------------------------------------

    def vis_draw(
        self,
        out: str,
        *,
        limit: int | None = None,
        show_conf: bool = False,
        conf: float | None = None,
        mask_alpha: int = 64,
        fill_mask: bool = True,
        show_attrs: bool = False,
        filter_no_attrs: bool = False,
        workers: int = 1,
        progress: bool = True,
        **kwargs: Any,
    ) -> int:
        """Draw bounding-boxes / masks on images (``ydm vis draw``)."""
        return self._run(
            "vis.draw",
            out=out,
            limit=limit,
            show_conf=show_conf,
            conf=conf,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attrs=show_attrs,
            filter_no_attrs=filter_no_attrs,
            workers=workers,
            progress=progress,
            **kwargs,
        )

    def vis_crop(
        self,
        out: str,
        *,
        keep_shape: bool = False,
        min_size: int = 1,
        conf: float | None = None,
        by_attr: bool = False,
        filter_no_attrs: bool = True,
        workers: int = 1,
        progress: bool = False,
        **kwargs: Any,
    ) -> int:
        """Crop annotation regions into class folders (``ydm vis crop``)."""
        return self._run(
            "vis.crop",
            out=out,
            keep_shape=keep_shape,
            min_size=min_size,
            conf=conf,
            by_attr=by_attr,
            filter_no_attrs=filter_no_attrs,
            workers=workers,
            progress=progress,
            **kwargs,
        )

    # -- export -------------------------------------------------------------

    def export_coco(self, out: str, **kwargs: Any) -> int:
        """Export to COCO JSON (``ydm export coco``)."""
        return self._run("export.coco", out=out, **kwargs)

    def export_xany(self, out: str, **kwargs: Any) -> int:
        """Export to x-anylabeling JSON (``ydm export xany``)."""
        return self._run("export.xany", out=out, **kwargs)

    # -- import -------------------------------------------------------------

    def import_labelme(
        self,
        json_dir: str,
        out: str,
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
        out: str,
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
        out: str,
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

    # -- convert ------------------------------------------------------------

    def convert_seg2det(
        self,
        out: str,
        *,
        copy_images: bool = True,
        keep_empty_labels: bool = True,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        """Convert segmentation labels to detection boxes (``ydm convert seg2det``)."""
        return self._run(
            "convert.seg2det",
            out=out,
            copy_images=copy_images,
            keep_empty_labels=keep_empty_labels,
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
            dry_run=dry_run,
            **kwargs,
        )

    # -- evaluation ---------------------------------------------------------

    def eval_compare(
        self,
        gt_root: str,
        pred_root: str,
        out: str,
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
        out: str,
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
        out: str,
        *,
        gt_root: str | None = None,
        match_iou: float = 0.5,
        low_iou: float = 0.1,
        conf_thres: float = 0.0,
        duplicate_iou: float = 0.9,
        val_source: str | None = None,
        class_file: str | None = None,
        review: bool = False,
        crop_padding: int = 12,
        review_workers: int = 1,
        review_progress: bool = False,
        review_progress_leave: bool = False,
        **kwargs: Any,
    ) -> int:
        """Fine-grained error analysis of predictions vs GT (``ydm eval error-analysis``)."""
        resolved_gt_root = gt_root or self.root
        resolved_val_source = val_source or self.split_file or _default_existing_path(self.root, "val.txt")
        resolved_class_file = class_file or self.class_file or _default_existing_path(self.root, "class.txt")
        return run_task(
            "eval.error_analysis",
            gt_root=resolved_gt_root,
            pred_root=pred_root,
            out=out,
            match_iou=match_iou,
            low_iou=low_iou,
            conf_thres=conf_thres,
            duplicate_iou=duplicate_iou,
            val_source=resolved_val_source,
            class_file=resolved_class_file,
            review=review,
            crop_padding=crop_padding,
            review_workers=review_workers,
            review_progress=review_progress,
            review_progress_leave=review_progress_leave,
            task=self.task,
            layout=self.layout,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            **kwargs,
        )
