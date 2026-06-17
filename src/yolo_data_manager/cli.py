from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_data_manager.annotation.edit import delete_by_attribute, delete_class, merge_classes, rename_class, replace_class, set_attribute
from yolo_data_manager.annotation.query import copy_query_result, query_by_attribute, query_by_class
from yolo_data_manager.annotation.remap import apply_class_map
from yolo_data_manager.converters.coco import export_coco, import_coco
from yolo_data_manager.converters.labelme import import_labelme_dir
from yolo_data_manager.converters.pseudo import predictions_to_pseudo_labels
from yolo_data_manager.converters.seg_det import segmentation_to_detection
from yolo_data_manager.converters.voc import import_voc_dir
from yolo_data_manager.converters.xanylabeling import export_xanylabeling
from yolo_data_manager.dataset.duplicates import find_duplicate_images, write_duplicate_image_csv
from yolo_data_manager.dataset.filter import filter_by_geometry
from yolo_data_manager.dataset.merge import merge_datasets
from yolo_data_manager.dataset.quality import find_bad_images, write_image_quality_csv
from yolo_data_manager.dataset.select import select_from_file
from yolo_data_manager.dataset.split import split_dataset
from yolo_data_manager.core.schema import write_dataset_yaml
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.validator import validate_dataset
from yolo_data_manager.io.writer import write_split_file, write_yolo_dataset
from yolo_data_manager.stats.compute import compute_stats
from yolo_data_manager.stats.export import write_annotation_csv, write_stats_plots
from yolo_data_manager.stats.report import write_class_counts_csv, write_json_report
from yolo_data_manager.vis.renderer import crop_dataset, render_dataset
from yolo_data_manager.evaluation.compare import compare_datasets, write_compare_csv
from yolo_data_manager.evaluation.review_pack import write_review_pack


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
    stats.add_argument("--class-csv", default=None, help="optional class-count CSV output path")
    stats.add_argument("--ann-csv", default=None, help="optional annotation CSV output path")
    stats.add_argument("--plots-dir", default=None, help="optional directory for PNG plots")
    stats.set_defaults(handler=handle_stats)

    query = subparsers.add_parser("query", help="query annotations")
    query_sub = query.add_subparsers(dest="query_command", required=True)
    query_class = query_sub.add_parser("class", help="query labels containing a class")
    add_dataset_args(query_class)
    query_class.add_argument("--class", dest="class_values", required=True, help="class id/name, comma-separated allowed")
    query_class.add_argument("--out", default=None, help="optional CSV output path")
    query_class.add_argument("--copy-images", default=None, help="copy matching images to this directory")
    query_class.add_argument("--copy-labels", default=None, help="copy matching labels to this directory")
    query_class.add_argument("--filtered-labels", action="store_true", help="when copying labels, keep only matched instances")
    query_class.set_defaults(handler=handle_query_class)
    query_attr = query_sub.add_parser("attr", help="query annotations by attribute")
    add_dataset_args(query_attr)
    query_attr.add_argument("--name", required=True, help="attribute name")
    query_attr.add_argument("--value", default=None, help="attribute value, comma-separated allowed")
    query_attr.add_argument("--nonzero", action="store_true", help="match annotations whose raw attribute value is non-zero")
    query_attr.add_argument("--out", default=None, help="optional CSV output path")
    query_attr.add_argument("--copy-images", default=None, help="copy matching images to this directory")
    query_attr.add_argument("--copy-labels", default=None, help="copy matching labels to this directory")
    query_attr.add_argument("--filtered-labels", action="store_true", help="when copying labels, keep only matched instances")
    query_attr.set_defaults(handler=handle_query_attr)

    dataset_cmd = subparsers.add_parser("dataset", help="dataset management")
    dataset_sub = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_select = dataset_sub.add_parser("select", help="copy a subset from a txt/csv-like file")
    add_dataset_args(dataset_select)
    dataset_select.add_argument("--file", required=True, help="selection file containing image paths/names/stems")
    dataset_select.add_argument("--out", required=True, help="output dataset root")
    dataset_select.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_select.set_defaults(handler=handle_dataset_select, copy_images=True)

    dataset_split = dataset_sub.add_parser("split", help="write train/val/test split txt files")
    add_dataset_args(dataset_split)
    dataset_split.add_argument("--train", type=float, default=0.8)
    dataset_split.add_argument("--val", type=float, default=0.2)
    dataset_split.add_argument("--test", type=float, default=0.0)
    dataset_split.add_argument("--seed", type=int, default=233)
    dataset_split.add_argument("--out", default=None, help="output directory; defaults to dataset root")
    dataset_split.set_defaults(handler=handle_dataset_split)

    dataset_yaml = dataset_sub.add_parser("yaml", help="write dataset.yaml")
    add_dataset_args(dataset_yaml)
    dataset_yaml.add_argument("--out", default=None, help="output yaml path; defaults to root/dataset.yaml")
    dataset_yaml.add_argument("--train", default="images/train")
    dataset_yaml.add_argument("--val", default="images/val")
    dataset_yaml.add_argument("--test", default=None)
    dataset_yaml.set_defaults(handler=handle_dataset_yaml)

    dataset_filter = dataset_sub.add_parser("filter", help="filter annotations by class/geometry/confidence")
    add_dataset_args(dataset_filter)
    dataset_filter.add_argument("--out", required=True, help="output dataset root")
    dataset_filter.add_argument("--class", dest="class_values", default=None, help="class id/name, comma-separated allowed")
    dataset_filter.add_argument("--min-width", type=float, default=None)
    dataset_filter.add_argument("--min-height", type=float, default=None)
    dataset_filter.add_argument("--min-area", type=float, default=None)
    dataset_filter.add_argument("--max-area", type=float, default=None)
    dataset_filter.add_argument("--min-conf", type=float, default=None)
    dataset_filter.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_filter.add_argument("--dry-run", action="store_true")
    dataset_filter.set_defaults(handler=handle_dataset_filter, copy_images=True)

    dataset_merge = dataset_sub.add_parser("merge", help="merge multiple YOLO datasets with class-name alignment")
    dataset_merge.add_argument("--roots", required=True, help="comma-separated dataset roots")
    dataset_merge.add_argument("--out", required=True, help="output dataset root")
    dataset_merge.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    dataset_merge.add_argument("--images-dir", default="images")
    dataset_merge.add_argument("--labels-dir", default="labels")
    dataset_merge.add_argument("--no-source-prefix", dest="source_prefix", action="store_false", help="do not prefix output image names by dataset index")
    dataset_merge.add_argument("--no-rename-duplicates", dest="rename_duplicates", action="store_false", help="fail on duplicate output image names")
    dataset_merge.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_merge.add_argument("--dry-run", action="store_true")
    dataset_merge.set_defaults(handler=handle_dataset_merge, source_prefix=True, rename_duplicates=True, copy_images=True)

    dataset_duplicates = dataset_sub.add_parser("duplicates", help="find duplicate image files by content hash")
    add_dataset_args(dataset_duplicates)
    dataset_duplicates.add_argument("--out", default=None, help="optional duplicate CSV output")
    dataset_duplicates.add_argument("--algorithm", default="sha256")
    dataset_duplicates.set_defaults(handler=handle_dataset_duplicates)

    dataset_bad_images = dataset_sub.add_parser("bad-images", help="find missing or corrupt images")
    add_dataset_args(dataset_bad_images)
    dataset_bad_images.add_argument("--out", default=None, help="optional CSV output")
    dataset_bad_images.set_defaults(handler=handle_dataset_bad_images)

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

    set_attr = ann_sub.add_parser("set-attr", help="set an attribute value on annotations")
    add_dataset_args(set_attr)
    add_write_args(set_attr)
    set_attr.add_argument("--name", required=True, help="attribute name")
    set_attr.add_argument("--value", required=True, help="new attribute value")
    set_attr.add_argument("--class", dest="class_values", default=None, help="optional class id/name filter")
    set_attr.add_argument("--where-value", default=None, help="only update annotations whose current attribute has this value")
    set_attr.set_defaults(handler=handle_set_attr)

    delete_attr = ann_sub.add_parser("delete-attr", help="delete annotations matched by an attribute")
    add_dataset_args(delete_attr)
    add_write_args(delete_attr)
    delete_attr.add_argument("--name", required=True, help="attribute name")
    delete_attr.add_argument("--value", default=None, help="attribute value, comma-separated allowed")
    delete_attr.add_argument("--nonzero", action="store_true")
    delete_attr.set_defaults(handler=handle_delete_attr)

    vis = subparsers.add_parser("vis", help="visualize annotations")
    vis_sub = vis.add_subparsers(dest="vis_command", required=True)
    draw = vis_sub.add_parser("draw", help="draw labels on images")
    add_dataset_args(draw)
    draw.add_argument("--out", required=True, help="output image directory")
    draw.add_argument("--limit", type=int, default=None)
    draw.add_argument("--show-conf", action="store_true")
    draw.add_argument("--conf", type=float, default=None, help="optional confidence threshold")
    draw.add_argument("--mask-alpha", type=int, default=64)
    draw.add_argument("--no-fill-mask", dest="fill_mask", action="store_false")
    draw.set_defaults(fill_mask=True)
    draw.set_defaults(handler=handle_vis_draw)
    crop = vis_sub.add_parser("crop", help="crop annotation regions into class folders")
    add_dataset_args(crop)
    crop.add_argument("--out", required=True, help="output crop directory")
    crop.add_argument("--keep-shape", action="store_true")
    crop.add_argument("--min-size", type=int, default=1)
    crop.add_argument("--conf", type=float, default=None, help="optional confidence threshold")
    crop.set_defaults(handler=handle_vis_crop)

    export = subparsers.add_parser("export", help="export to another format")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    coco = export_sub.add_parser("coco", help="export YOLO dataset to COCO JSON")
    add_dataset_args(coco)
    coco.add_argument("--out", required=True, help="output COCO JSON path")
    coco.set_defaults(handler=handle_export_coco)
    xany = export_sub.add_parser("xany", help="export YOLO dataset to x-anylabeling JSON files")
    add_dataset_args(xany)
    xany.add_argument("--out", required=True, help="output JSON directory")
    xany.set_defaults(handler=handle_export_xany)

    import_cmd = subparsers.add_parser("import", help="import another annotation format")
    import_sub = import_cmd.add_subparsers(dest="import_command", required=True)
    labelme = import_sub.add_parser("labelme", help="import LabelMe JSON directory as YOLO")
    labelme.add_argument("--json-dir", required=True)
    labelme.add_argument("--out", required=True)
    labelme.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    labelme.add_argument("--classes", default=None, help="optional comma-separated class order")
    labelme.set_defaults(handler=handle_import_labelme)
    coco_import = import_sub.add_parser("coco", help="import COCO JSON as YOLO")
    coco_import.add_argument("--json", dest="json_path", required=True)
    coco_import.add_argument("--images-dir", required=True)
    coco_import.add_argument("--out", required=True)
    coco_import.add_argument("--task", choices=["detect", "segment"], default="detect")
    coco_import.add_argument("--classes", default=None, help="optional comma-separated class order")
    coco_import.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    coco_import.set_defaults(handler=handle_import_coco, copy_images=True)
    voc_import = import_sub.add_parser("voc", help="import Pascal VOC XML directory as YOLO")
    voc_import.add_argument("--annotations-dir", required=True)
    voc_import.add_argument("--images-dir", required=True)
    voc_import.add_argument("--out", required=True)
    voc_import.add_argument("--classes", default=None, help="optional comma-separated class order")
    voc_import.add_argument("--keep-difficult", dest="skip_difficult", action="store_false")
    voc_import.set_defaults(handler=handle_import_voc, skip_difficult=True)

    convert = subparsers.add_parser("convert", help="convert dataset task/form")
    convert_sub = convert.add_subparsers(dest="convert_command", required=True)
    seg2det = convert_sub.add_parser("seg2det", help="convert YOLO segmentation labels to detection labels")
    add_dataset_args(seg2det)
    add_write_args(seg2det)
    seg2det.set_defaults(handler=handle_seg2det)
    pseudo = convert_sub.add_parser("pseudo", help="convert prediction labels to pseudo labels")
    add_dataset_args(pseudo)
    add_write_args(pseudo)
    pseudo.add_argument("--conf", type=float, default=0.0, help="confidence threshold")
    pseudo.add_argument("--keep-conf", dest="drop_confidence", action="store_false", help="keep confidence in output labels")
    pseudo.set_defaults(handler=handle_pseudo, drop_confidence=True)

    eval_cmd = subparsers.add_parser("eval", help="evaluate or compare predictions")
    eval_sub = eval_cmd.add_subparsers(dest="eval_command", required=True)
    compare = eval_sub.add_parser("compare", help="compare prediction labels against GT labels")
    compare.add_argument("--gt-root", required=True)
    compare.add_argument("--pred-root", required=True)
    compare.add_argument("--out", required=True, help="CSV output path")
    compare.add_argument("--iou", type=float, default=0.5)
    compare.add_argument("--conf", type=float, default=None)
    compare.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    compare.set_defaults(handler=handle_eval_compare)
    review = eval_sub.add_parser("review-pack", help="write FP/FN review package from GT and predictions")
    review.add_argument("--gt-root", required=True)
    review.add_argument("--pred-root", required=True)
    review.add_argument("--out", required=True, help="review output directory")
    review.add_argument("--csv", default=None, help="optional full compare CSV output")
    review.add_argument("--iou", type=float, default=0.5)
    review.add_argument("--conf", type=float, default=None)
    review.add_argument("--status", default="fp,fn", help="statuses to include, comma-separated")
    review.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    review.set_defaults(handler=handle_eval_review_pack)

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
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing output")
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
    if args.class_csv:
        write_class_counts_csv(payload, args.class_csv)
    if args.ann_csv:
        write_annotation_csv(dataset, args.ann_csv)
    if args.plots_dir:
        write_stats_plots(dataset, args.plots_dir)
    _emit_json(payload, args.out)
    return 0


def handle_query_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    result = query_by_class(dataset, _split_values(args.class_values))
    if args.out:
        result.write_csv(args.out)
    copy_query_result(
        result,
        images_dir=args.copy_images,
        labels_dir=args.copy_labels,
        filtered_labels=args.filtered_labels,
    )
    print(json.dumps({"matches": len(result), "labels": [str(p) for p in result.label_paths()]}, indent=2, ensure_ascii=False))
    return 0


def handle_query_attr(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    values = _split_values(args.value) if args.value else None
    result = query_by_attribute(dataset, args.name, values=values, nonzero=args.nonzero)
    if args.out:
        result.write_csv(args.out)
    copy_query_result(
        result,
        images_dir=args.copy_images,
        labels_dir=args.copy_labels,
        filtered_labels=args.filtered_labels,
    )
    print(json.dumps({"matches": len(result), "labels": [str(p) for p in result.label_paths()]}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_select(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    selected = select_from_file(dataset, args.file)
    write_yolo_dataset(selected, args.out, copy_images=args.copy_images)
    print(json.dumps({"images": len(selected.images), "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_split(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    splits = split_dataset(dataset, train=args.train, val=args.val, test=args.test, seed=args.seed)
    out_dir = Path(args.out) if args.out else Path(args.root)
    for split_name, names in splits.items():
        write_split_file(names, out_dir / f"{split_name}.txt")
    print(json.dumps({name: len(values) for name, values in splits.items()}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_yaml(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out_path = Path(args.out) if args.out else Path(args.root) / "dataset.yaml"
    write_dataset_yaml(dataset.classes, out_path, train=args.train, val=args.val, test=args.test)
    print(json.dumps({"out": str(out_path)}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_filter(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    before = dataset.annotation_count()
    class_ids = {dataset.class_id(value) for value in _split_values(args.class_values)} if args.class_values else None
    filtered = filter_by_geometry(
        dataset,
        class_ids=class_ids,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        max_area=args.max_area,
        min_confidence=args.min_conf,
    )
    after = filtered.annotation_count()
    if not args.dry_run:
        write_yolo_dataset(filtered, args.out, copy_images=args.copy_images)
    print(json.dumps({"before": before, "after": after, "removed": before - after, "out": None if args.dry_run else args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_merge(args: argparse.Namespace) -> int:
    roots = _split_values(args.roots)
    datasets = [
        load_yolo_dataset(root, images_dir=args.images_dir, labels_dir=args.labels_dir, task=args.task)
        for root in roots
    ]
    merged, report = merge_datasets(
        datasets,
        root=args.out,
        rename_duplicates=args.rename_duplicates,
        source_prefix=args.source_prefix,
    )
    if not args.dry_run:
        write_yolo_dataset(merged, args.out, copy_images=args.copy_images)
    print(
        json.dumps(
            {
                "images": report.image_count,
                "annotations": report.annotation_count,
                "classes": report.class_names,
                "renamed_images": report.renamed_images,
                "out": None if args.dry_run else args.out,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def handle_dataset_duplicates(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    groups = find_duplicate_images(dataset, algorithm=args.algorithm)
    if args.out:
        write_duplicate_image_csv(groups, args.out)
    print(json.dumps({"groups": len(groups), "duplicates": [group.__dict__ for group in groups]}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_bad_images(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    issues = find_bad_images(dataset)
    if args.out:
        write_image_quality_csv(issues, args.out)
    print(json.dumps({"issues": len(issues), "bad_images": [issue.__dict__ for issue in issues]}, indent=2, ensure_ascii=False))
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
    if not args.dry_run:
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
    print(json.dumps({"reports": len(reports), "out": None if args.dry_run else args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_set_attr(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    classes = _split_values(args.class_values) if args.class_values else None
    edited, report = set_attribute(
        dataset,
        args.name,
        args.value,
        classes=classes,
        where_value=args.where_value,
    )
    _write_edit_result(edited, report, args)
    return 0


def handle_delete_attr(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    values = _split_values(args.value) if args.value else None
    edited, report = delete_by_attribute(dataset, args.name, values=values, nonzero=args.nonzero)
    _write_edit_result(edited, report, args)
    return 0


def handle_vis_draw(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    render_dataset(
        dataset,
        args.out,
        limit=args.limit,
        show_confidence=args.show_conf,
        confidence_threshold=args.conf,
        mask_alpha=args.mask_alpha,
        fill_mask=args.fill_mask,
    )
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_vis_crop(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    saved = crop_dataset(dataset, args.out, keep_shape=args.keep_shape, min_size=args.min_size, confidence_threshold=args.conf)
    print(json.dumps({"saved": saved, "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_export_coco(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    export_coco(dataset, args.out)
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_export_xany(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    export_xanylabeling(dataset, args.out)
    print(json.dumps({"out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_labelme(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    dataset = import_labelme_dir(args.json_dir, out_root=args.out, task=args.task, class_names=classes)
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_coco(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    dataset = import_coco(
        args.json_path,
        images_dir=args.images_dir,
        out_root=args.out,
        task=args.task,
        class_names=classes,
        copy_images=args.copy_images,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_voc(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    dataset = import_voc_dir(
        args.annotations_dir,
        images_dir=args.images_dir,
        out_root=args.out,
        class_names=classes,
        skip_difficult=args.skip_difficult,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": args.out}, indent=2, ensure_ascii=False))
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


def handle_pseudo(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    pseudo = predictions_to_pseudo_labels(dataset, confidence_threshold=args.conf, drop_confidence=args.drop_confidence)
    if not args.dry_run:
        write_yolo_dataset(
            pseudo,
            args.out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
            include_confidence=not args.drop_confidence,
        )
    print(json.dumps({"annotations": pseudo.annotation_count(), "out": None if args.dry_run else args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_eval_compare(args: argparse.Namespace) -> int:
    gt = load_yolo_dataset(args.gt_root, task=args.task)
    pred = load_yolo_dataset(args.pred_root, task=args.task)
    rows, summary = compare_datasets(gt, pred, iou_threshold=args.iou, confidence_threshold=args.conf)
    write_compare_csv(rows, args.out)
    print(json.dumps({"summary": summary, "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def handle_eval_review_pack(args: argparse.Namespace) -> int:
    gt = load_yolo_dataset(args.gt_root, task=args.task)
    pred = load_yolo_dataset(args.pred_root, task=args.task)
    rows, summary = compare_datasets(gt, pred, iou_threshold=args.iou, confidence_threshold=args.conf)
    if args.csv:
        write_compare_csv(rows, args.csv)
    counts = write_review_pack(rows, gt, args.out, statuses=set(_split_values(args.status)), pred=pred)
    print(json.dumps({"summary": summary, "review": counts, "out": args.out}, indent=2, ensure_ascii=False))
    return 0


def _write_edit_result(dataset, report, args: argparse.Namespace) -> None:
    if not args.dry_run:
        write_yolo_dataset(
            dataset,
            args.out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
        )
    if args.report:
        report.write_csv(args.report)
    print(json.dumps({"changed": len(report.rows), "out": None if args.dry_run else args.out}, indent=2, ensure_ascii=False))


def _split_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _emit_json(payload: dict[str, object], out: str | None) -> None:
    if out:
        write_json_report(payload, out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
