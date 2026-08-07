"""Run reusable example functions against datasets supplied at runtime.

The same command can be used for datasets in unrelated directories.  Repeat
``--data-dir`` to process more than one dataset in one invocation::

    python -m example.datasets.run_dataset \
        --data-dir E:/datasets/a \
        --data-dir E:/datasets/b \
        --task stats

The option name ``--data-dir`` deliberately accepts either a dataset root or
a dataset YAML file, matching :class:`yolo_data_manager.YoloManager`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# Allow both ``python -m example.datasets.run_dataset`` and the convenient
# direct form ``python example/datasets/run_dataset.py`` from a checkout.
if __package__ in (None, ""):
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from example.functions.data_filter_small import yolo_filter_small
from example.functions.data_manual_draw_box import yolo_draw
from example.functions.data_merge_class import yolo_merge_class
from example.functions.data_object_label_update import yolo_update as yolo_update_from_crops
from example.functions.data_select_copy import yolo_select_val
from example.functions.data_split import yolo_split
from example.functions.data_sta import yolo_sta
from example.functions.data_vis import yolo_vis
from example.functions.result_error_ana import yolo_error_ana
from example.functions.result_metric import yolo_metric
from example.functions.result_object_label_update import (
    yolo_update as yolo_update_from_error_crops,
)


TASKS = (
    "stats",
    "vis",
    "filter-small",
    "split",
    "select-val",
    "merge-class",
    "manual-box",
    "metric",
    "error-analysis",
    "correct-crops",
    "correct-error-crops",
)


def _load_mapping(value: str | None) -> Mapping[str | int, Any] | None:
    """Load an inline JSON/YAML mapping or a mapping file."""

    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("mapping options must contain a JSON/YAML object")
    return parsed


def _values(values: Sequence[str] | None) -> list[str] | None:
    """Expand repeatable and comma-separated class arguments."""

    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result or None


def _target(value: str | None) -> str | int | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def build_parser(default_task: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call reusable YOLO example functions for external datasets."
    )
    parser.add_argument(
        "--data-dir",
        "--dataset",
        dest="data_dirs",
        action="append",
        required=True,
        help="dataset root or YAML file; repeat for multiple datasets",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default=default_task or "stats",
        help="operation to run (default: stats)",
    )
    parser.add_argument(
        "--only-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="limit the operation to the validation split; default is all input data",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="operation-specific output path")

    # Statistics and visualization.
    parser.add_argument("--stats-list", action="append", default=None)
    parser.add_argument("--class-csv", default=None)
    parser.add_argument("--ann-csv", default=None)
    parser.add_argument("--attr-csv", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-out", default=None)
    parser.add_argument("--crop-out", default=None)
    parser.add_argument("--show-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding", type=float, default=0)

    # Dataset filtering and splitting.
    parser.add_argument("--filter-ratio", type=float, default=0.01)
    parser.add_argument("--logic", choices=["or", "and"], default="or")
    parser.add_argument("--class-rules", default=None, help="JSON/YAML object or file")
    parser.add_argument("--min-width", type=float, default=None)
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=None)
    parser.add_argument("--max-area", type=float, default=None)
    parser.add_argument("--min-conf", type=float, default=None)
    parser.add_argument("--train", type=float, default=0.9)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--selection-file", default=None)
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--merge-class-map", default=None, help="JSON/YAML target-to-source mapping")

    # Prediction evaluation.
    parser.add_argument("--pred-dir", default=None, help="prediction root or labels directory")
    parser.add_argument("--pred-name", default=None, help="run name below --pred-dir")
    parser.add_argument("--abs-path", action="store_true", help="use --pred-dir exactly as supplied")
    parser.add_argument("--class", dest="class_values", action="append", default=None)
    parser.add_argument("--exclude-class", dest="exclude_class_values", action="append", default=None)
    parser.add_argument("--min-pixels", type=float, default=None)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--show-original", action="store_true")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--review", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-pred-txt", action=argparse.BooleanOptionalAction, default=True)

    # Manual box and annotation correction.
    parser.add_argument("--image", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--show-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crops-dir", default=None)
    parser.add_argument("--to", default=None, help="target class; use 'none' to delete")
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--delete-pred-none", action="store_true")
    parser.add_argument("--replace-gt-from-pred", action="store_true")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_dataset_task(data_dir: str | Path, args: argparse.Namespace) -> int:
    """Dispatch one parsed operation for one dataset path."""

    task = args.task
    class_values = _values(args.class_values)
    exclude_values = _values(args.exclude_class_values)
    class_rules = _load_mapping(args.class_rules)
    merge_class_map = _load_mapping(args.merge_class_map)

    if task == "stats":
        return yolo_sta(
            data_dir,
            stats_list=args.stats_list or "all",
            only_val=args.only_val,
            out=args.out,
            class_csv=args.class_csv,
            ann_csv=args.ann_csv,
            attr_csv=args.attr_csv,
            plots_dir=args.plots_dir,
        )
    if task == "vis":
        return yolo_vis(
            data_dir,
            crop=args.crop,
            draw_out=args.draw_out or args.out,
            crop_out=args.crop_out,
            only_val=args.only_val,
            workers=args.workers,
            show_id=args.show_id,
            padding=args.padding,
        )
    if task == "filter-small":
        return yolo_filter_small(
            data_dir,
            filter_ratio=args.filter_ratio,
            logic=args.logic,
            class_rules=class_rules,
            out_dir=args.out,
            class_=class_values,
            min_area=args.min_area,
            max_area=args.max_area,
            min_conf=args.min_conf,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )
    if task == "split":
        return yolo_split(
            data_dir,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
            absolute_paths=args.absolute_paths,
            out=args.out,
        )
    if task == "select-val":
        return yolo_select_val(
            data_dir,
            copy_images=args.copy_images,
            selection_file=args.selection_file,
            out=args.out,
        )
    if task == "merge-class":
        if merge_class_map is None:
            raise ValueError("--merge-class-map is required for task merge-class")
        return yolo_merge_class(
            data_dir,
            args.out,
            merge_class_map,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )
    if task == "manual-box":
        if args.image is None:
            raise ValueError("--image is required for task manual-box")
        return yolo_draw(
            data_dir,
            args.image,
            label=args.label,
            class_id=args.class_id,
            show_existing=args.show_existing,
            out=args.out,
        )
    if task == "metric":
        if args.pred_dir is None:
            raise ValueError("--pred-dir is required for task metric")
        return yolo_metric(
            data_dir,
            args.pred_dir,
            args.pred_name,
            abs_path=args.abs_path,
            workers=args.workers,
            class_=class_values,
            exclude_class_=exclude_values,
            merge_class_map=merge_class_map,
            min_width=args.min_width,
            min_height=args.min_height,
            min_area=args.min_area,
            min_size_logic=args.logic,
            min_pixels=args.min_pixels,
            conf_thres=args.conf_thres,
            only_val=args.only_val,
            show_original=args.show_original,
            out=args.out,
            csv=args.csv,
        )
    if task == "error-analysis":
        if args.pred_dir is None:
            raise ValueError("--pred-dir is required for task error-analysis")
        return yolo_error_ana(
            data_dir,
            args.pred_dir,
            args.pred_name,
            abs_path=args.abs_path,
            only_val=args.only_val,
            workers=args.workers,
            conf_thres=args.conf_thres,
            class_=class_values,
            exclude_class_=exclude_values,
            min_width=args.min_width,
            min_height=args.min_height,
            min_area=args.min_area,
            min_size_logic=args.logic,
            min_pixels=args.min_pixels,
            class_rules=class_rules,
            out=args.out,
            review=args.review,
            copy_pred_txt=args.copy_pred_txt,
        )
    if task == "correct-crops":
        if args.crops_dir is None:
            raise ValueError("--crops-dir is required for task correct-crops")
        return yolo_update_from_crops(
            data_dir,
            args.crops_dir,
            _target(args.to),
            report=args.report,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
            only_val=args.only_val,
        )
    if task == "correct-error-crops":
        if args.crops_dir is None:
            raise ValueError("--crops-dir is required for task correct-error-crops")
        return yolo_update_from_error_crops(
            data_dir,
            args.crops_dir,
            _target(args.to),
            pred_dir=args.pred_dir,
            dedup_iou=args.dedup_iou,
            delete_pred_none=args.delete_pred_none,
            replace_gt_from_pred=args.replace_gt_from_pred,
            report=args.report,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
            only_val=args.only_val,
        )
    raise ValueError(f"unsupported task: {task}")


def main(
    argv: Sequence[str] | None = None,
    *,
    default_task: str | None = None,
) -> int:
    """CLI entry point; return the largest status from all dataset paths."""

    args = build_parser(default_task).parse_args(argv)
    status = 0
    for data_dir in args.data_dirs:
        print(f"\n=== {args.task}: {data_dir} ===")
        status = max(status, int(run_dataset_task(data_dir, args)))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
