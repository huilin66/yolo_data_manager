from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_data_manager.annotation.edit import delete_class, merge_classes, rename_class, replace_class
from yolo_data_manager.annotation.query import query_by_class
from yolo_data_manager.annotation.remap import apply_class_map
from yolo_data_manager.converters.coco import export_coco
from yolo_data_manager.converters.seg_det import segmentation_to_detection
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.validator import validate_dataset
from yolo_data_manager.io.writer import write_yolo_dataset
from yolo_data_manager.stats.compute import compute_stats
from yolo_data_manager.stats.report import write_json_report
from yolo_data_manager.vis.renderer import render_dataset


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ydm", description="YOLO Data Manager")
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="validate a YOLO dataset")
    add_dataset_args(check)
    check.add_argument("--out", default=None, help="optional JSON output path")
    check.set_defaults(handler=handle_check)

    stats = subparsers.add_parser("stats", help="compute dataset statistics")
    add_dataset_args(stats)
    stats.add_argument("--out", default=None, help="optional JSON output path")
    stats.set_defaults(handler=handle_stats)

    query = subparsers.add_parser("query", help="query annotations")
    query_sub = query.add_subparsers(dest="query_command", required=True)
    query_class = query_sub.add_parser("class", help="query labels containing a class")
    add_dataset_args(query_class)
    query_class.add_argument("--class", dest="class_values", required=True, help="class id/name, comma-separated allowed")
    query_class.add_argument("--out", default=None, help="optional CSV output path")
    query_class.set_defaults(handler=handle_query_class)

    ann = subparsers.add_parser("ann", help="edit annotations")
    ann_sub = ann.add_subparsers(dest="ann_command", required=True)

    delete = ann_sub.add_parser("delete-class", help="delete annotations of one or more classes")
    add_dataset_args(delete)
    add_write_args(delete)
    delete.add_argument("--class", dest="class_values", required=True, help="class id/name, comma-separated allowed")
    delete.add_argument("--compact", action="store_true", help="remove classes from class.txt and remap ids")
    delete.set_defaults(handler=handle_delete_class)

    replace = ann_sub.add_parser("replace-class", help="replace one or more classes with another class")
    add_dataset_args(replace)
    add_write_args(replace)
    replace.add_argument("--from", dest="from_values", required=True, help="source class id/name, comma-separated allowed")
    replace.add_argument("--to", dest="to_value", required=True, help="target class id/name")
    replace.add_argument("--compact", action="store_true", help="remove source classes from class.txt and remap ids")
    replace.set_defaults(handler=handle_replace_class)

    merge = ann_sub.add_parser("merge-class", help="merge classes into one class")
    add_dataset_args(merge)
    add_write_args(merge)
    merge.add_argument("--from", dest="from_values", required=True, help="source class id/name, comma-separated allowed")
    merge.add_argument("--to", dest="to_value", required=True, help="target class id/name")
    merge.add_argument("--no-compact", dest="compact", action="store_false", help="keep source class names in class.txt")
    merge.set_defaults(handler=handle_merge_class, compact=True)

    rename = ann_sub.add_parser("rename-class", help="rename a class without changing ids")
    add_dataset_args(rename)
    add_write_args(rename)
    rename.add_argument("--from", dest="from_value", required=True, help="source class id/name")
    rename.add_argument("--to", dest="to_value", required=True, help="new class name")
    rename.set_defaults(handler=handle_rename_class)

    apply_map = ann_sub.add_parser("apply-map", help="apply class rename/merge/drop yaml")
    add_dataset_args(apply_map)
    add_write_args(apply_map)
    apply_map.add_argument("--map", dest="map_file", required=True, help="YAML class map")
    apply_map.add_argument("--no-compact", dest="compact", action="store_false", help="do not compact class ids")
    apply_map.set_defaults(handler=handle_apply_map, compact=True)

    vis = subparsers.add_parser("vis", help="visualize annotations")
    vis_sub = vis.add_subparsers(dest="vis_command", required=True)
    draw = vis_sub.add_parser("draw", help="draw labels on images")
    add_dataset_args(draw)
    draw.add_argument("--out", required=True, help="output image directory")
    draw.add_argument("--limit", type=int, default=None)
    draw.add_argument("--show-conf", action="store_true")
    draw.set_defaults(handler=handle_vis_draw)

    export = subparsers.add_parser("export", help="export to another format")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    coco = export_sub.add_parser("coco", help="export YOLO dataset to COCO JSON")
    add_dataset_args(coco)
    coco.add_argument("--out", required=True, help="output COCO JSON path")
    coco.set_defaults(handler=handle_export_coco)

    convert = subparsers.add_parser("convert", help="convert dataset task/form")
    convert_sub = convert.add_subparsers(dest="convert_command", required=True)
    seg2det = convert_sub.add_parser("seg2det", help="convert YOLO segmentation labels to detection labels")
    add_dataset_args(seg2det)
    add_write_args(seg2det)
    seg2det.set_defaults(handler=handle_seg2det)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="YOLO dataset root")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--attribute-file", default=None)
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    parser.add_argument("--split-file", default=None)


def add_write_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--report", default=None, help="optional edit report CSV path")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false", help="do not copy image files")
    parser.add_argument("--drop-empty-labels", dest="keep_empty_labels", action="store_false", help="do not write empty label files")
    parser.set_defaults(copy_images=True, keep_empty_labels=True)


def load_from_args(args: argparse.Namespace):
    return load_yolo_dataset(
        root=args.root,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=args.class_file,
        attribute_file=args.attribute_file,
        task=args.task,
        split_file=args.split_file,
    )


def handle_check(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    report = validate_dataset(dataset)
    payload = {
        "ok": report.ok,
        "summary": report.summary(),
        "issues": report.to_rows(),
    }
    _emit_json(payload, args.out)
    return 0 if report.ok else 2


def handle_stats(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    payload = compute_stats(dataset)
    _emit_json(payload, args.out)
    return 0


def handle_query_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    result = query_by_class(dataset, _split_values(args.class_values))
    if args.out:
        result.write_csv(args.out)
    print(json.dumps({"matches": len(result), "labels": [str(p) for p in result.label_paths()]}, indent=2, ensure_ascii=False))
    return 0


def handle_delete_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = delete_class(dataset, _split_values(args.class_values), compact=args.compact)
    _write_edit_result(edited, report, args)
    return 0


def handle_replace_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = replace_class(dataset, _split_values(args.from_values), args.to_value, compact=args.compact)
    _write_edit_result(edited, report, args)
    return 0


def handle_merge_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = merge_classes(dataset, _split_values(args.from_values), args.to_value, compact=args.compact)
    _write_edit_result(edited, report, args)
    return 0


def handle_rename_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = rename_class(dataset, args.from_value, args.to_value)
    _write_edit_result(edited, report, args)
    return 0


def handle_apply_map(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, reports = apply_class_map(dataset, args.map_file, compact=args.compact)
    write_yolo_dataset(
        edited,
        args.out,
        copy_images=args.copy_images,
        keep_empty_labels=args.keep_empty_labels,
    )
    if args.report:
        rows = []
        for report in reports:
            rows.extend(report.rows)
        from yolo_data_manager.annotation.edit import EditReport

        EditReport(rows=rows).write_csv(args.report)
    print(json.dumps({"reports": len(reports), "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_vis_draw(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    render_dataset(dataset, args.out, limit=args.limit, show_confidence=args.show_conf)
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_export_coco(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    export_coco(dataset, args.out)
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_seg2det(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited = segmentation_to_detection(dataset)
    write_yolo_dataset(
        edited,
        args.out,
        copy_images=args.copy_images,
        keep_empty_labels=args.keep_empty_labels,
    )
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def _write_edit_result(dataset, report, args: argparse.Namespace) -> None:
    write_yolo_dataset(
        dataset,
        args.out,
        copy_images=args.copy_images,
        keep_empty_labels=args.keep_empty_labels,
    )
    if args.report:
        report.write_csv(args.report)
    print(json.dumps({"changed": len(report.rows), "out": args.out}, indent=2, ensure_ascii=False))


def _split_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _emit_json(payload: dict[str, object], out: str | None) -> None:
    if out:
        write_json_report(payload, out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

