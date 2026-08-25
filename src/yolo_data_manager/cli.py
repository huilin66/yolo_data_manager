from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from yolo_data_manager.annotation.edit import delete_by_attribute, delete_class, merge_classes, rename_class, replace_class, set_attribute
from yolo_data_manager.annotation.crop_correction import (
    correct_gt_labels_from_error_crops,
    correct_labels_from_crops,
)
from yolo_data_manager.annotation.query import copy_query_result, query_by_attribute, query_by_class
from yolo_data_manager.annotation.remap import apply_class_map
from yolo_data_manager.converters.coco import export_coco, import_coco
from yolo_data_manager.converters.labelme import import_labelme_dir
from yolo_data_manager.converters.mask import import_semantic_mask_dir
from yolo_data_manager.converters.pseudo import predictions_to_pseudo_labels
from yolo_data_manager.converters.seg_det import segmentation_to_detection
from yolo_data_manager.converters.voc import import_voc_dir
from yolo_data_manager.converters.xanylabeling import export_xanylabeling
from yolo_data_manager.dataset.duplicates import find_duplicate_images, write_duplicate_image_csv
from yolo_data_manager.dataset.filter import filter_by_geometry
from yolo_data_manager.dataset.merge import merge_datasets
from yolo_data_manager.dataset.quality import find_bad_images, write_image_quality_csv
from yolo_data_manager.dataset.select import select_from_file
from yolo_data_manager.dataset.split import class_counts_for_images, split_dataset
from yolo_data_manager.core.schema import write_dataset_yaml
from yolo_data_manager.io.layout import detect_layout
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.output_paths import (
    default_annotation_output,
    default_conversion_output,
    default_dataset_output,
    default_evaluation_output,
    default_visualization_output,
    ydm_dir,
)
from yolo_data_manager.io.validator import fill_missing_label_files, validate_dataset
from yolo_data_manager.io.writer import (
    move_existing_split_files_to_backup,
    write_split_file,
    write_yolo_dataset,
)
from yolo_data_manager.stats.compute import compute_stats
from yolo_data_manager.stats.export import write_annotation_csv, write_attribute_csv, write_stats_plots
from yolo_data_manager.stats.report import write_class_counts_csv, write_json_report
from yolo_data_manager.tools.image_resize import resize_yolo_dataset, validate_resize_options
from yolo_data_manager.vis.manual_box import draw_manual_box, find_dataset_image
from yolo_data_manager.vis.renderer import crop_dataset, render_dataset
from yolo_data_manager.evaluation.compare import compare_datasets, write_compare_csv
from yolo_data_manager.evaluation.error_analysis import (
    analyze_errors,
    collect_stems_from_source,
    copy_prediction_txt_to_review,
    find_duplicate_gt,
    filter_error_analysis_datasets,
    load_error_analysis_dataset,
    print_error_summary,
    write_duplicate_gt_csv,
    write_error_csvs,
    write_error_review_pack,
)
from yolo_data_manager.evaluation.metrics import (
    compute_detection_metrics,
    format_metrics_table,
    write_metrics_csv,
    write_metrics_json,
    write_size_metrics_csv,
)
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
    check.add_argument(
        "--out",
        default=None,
        help="JSON output path; defaults to <root>/ydm_quality/check.json",
    )
    check.add_argument("--fill-missing-txt", action="store_true", help="create empty txt files for images without matching labels")
    check.add_argument("--print-full", action="store_true", help="also print the full JSON report to terminal")
    check.set_defaults(handler=handle_check)

    stats = subparsers.add_parser("stats", help="compute dataset statistics")
    add_dataset_args(stats)
    stats.add_argument("--out", default=None, help="JSON output path; defaults to <root>/ydm_stats/stats.json")
    stats.add_argument("--class-csv", default=None, help="class-count CSV; defaults to <root>/ydm_stats/class_counts.csv")
    stats.add_argument("--ann-csv", default=None, help="annotation CSV; defaults to <root>/ydm_stats/annotations.csv")
    stats.add_argument("--attr-csv", default=None, help="attribute CSV; defaults to <root>/ydm_stats/attributes.csv")
    stats.add_argument("--plots-dir", default=None, help="PNG plot directory; defaults to <root>/ydm_stats/plots")
    stats.add_argument("--stats-list", default=None, help="comma-separated stats to plot/export; use all for every stats output")
    stats.set_defaults(handler=handle_stats)

    layout_cmd = subparsers.add_parser("layout", help="detect or inspect YOLO dataset layouts")
    layout_sub = layout_cmd.add_subparsers(dest="layout_command", required=True)
    layout_detect = layout_sub.add_parser("detect", help="detect YOLO layout")
    layout_detect.add_argument("--root", required=True)
    add_runtime_args(layout_detect, workers=False)
    layout_detect.set_defaults(handler=handle_layout_detect)

    query = subparsers.add_parser("query", help="query annotations")
    query_sub = query.add_subparsers(dest="query_command", required=True)
    query_class = query_sub.add_parser("class", help="query labels containing a class")
    add_dataset_args(query_class)
    query_class.add_argument("--class", dest="class_values", required=True, help="class id/name, comma-separated allowed")
    query_class.add_argument(
        "--source",
        choices=["gt", "pred"],
        default="gt",
        help="query GT labels from --root or prediction labels from --pred-root",
    )
    query_class.add_argument(
        "--pred-root",
        "--prediction-root",
        dest="pred_root",
        default=None,
        help="prediction dataset root or labels directory; required with --source pred",
    )
    query_class.add_argument("--out", default=None, help="CSV output path; defaults to ydm_quality/query/class_<class>.csv")
    query_class.add_argument("--copy-images", default=None, help="copy matching images to this directory")
    query_class.add_argument("--copy-labels", default=None, help="copy matching labels to this directory")
    query_class.add_argument("--filtered-labels", action="store_true", help="when copying labels, keep only matched instances")
    query_class.set_defaults(handler=handle_query_class)
    query_attr = query_sub.add_parser("attr", help="query annotations by attribute")
    add_dataset_args(query_attr)
    query_attr.add_argument("--name", required=True, help="attribute name")
    query_attr.add_argument("--value", default=None, help="attribute value, comma-separated allowed")
    query_attr.add_argument("--nonzero", action="store_true", help="match annotations whose raw attribute value is non-zero")
    query_attr.add_argument("--out", default=None, help="CSV output path; defaults to ydm_quality/query/attr_<name>.csv")
    query_attr.add_argument("--copy-images", default=None, help="copy matching images to this directory")
    query_attr.add_argument("--copy-labels", default=None, help="copy matching labels to this directory")
    query_attr.add_argument("--filtered-labels", action="store_true", help="when copying labels, keep only matched instances")
    query_attr.set_defaults(handler=handle_query_attr)

    dataset_cmd = subparsers.add_parser("dataset", help="dataset management")
    dataset_sub = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_select = dataset_sub.add_parser("select", help="copy a subset from a txt/csv-like file")
    add_dataset_args(dataset_select)
    dataset_select.add_argument("--file", required=True, help="selection file containing image paths/names/stems")
    dataset_select.add_argument("--out", default=None, help="output dataset root; defaults to <root>/ydm_dataset/select")
    dataset_select.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    dataset_select.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_select.set_defaults(
        handler=handle_dataset_select,
        copy_images=True,
    )

    dataset_normalize = dataset_sub.add_parser("normalize", help="normalize supported YOLO layouts into flat images/labels")
    add_dataset_args(dataset_normalize)
    add_write_args(dataset_normalize)
    dataset_normalize.set_defaults(
        handler=handle_dataset_normalize,
        _output_operation="normalize",
    )

    dataset_split = dataset_sub.add_parser("split", help="write train/val/test split txt files")
    add_dataset_args(dataset_split)
    dataset_split.add_argument("--train", type=float, default=0.8)
    dataset_split.add_argument("--val", type=float, default=0.2)
    dataset_split.add_argument("--test", type=float, default=0.0)
    dataset_split.add_argument("--seed", type=int, default=233)
    dataset_split.add_argument("--out", default=None, help="output directory; defaults to dataset root")
    dataset_split.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory for existing train/val/test txt; defaults to <dataset-root>/labels_backup",
    )
    dataset_split.add_argument("--absolute-paths", action="store_true", help="write absolute image paths instead of image file names")
    dataset_split.add_argument(
        "--train-include-list",
        default=None,
        help="txt file or comma-separated image names/paths forced into train",
    )
    dataset_split.add_argument(
        "--val-include-list",
        default=None,
        help="txt file or comma-separated image names/paths forced into val",
    )
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
    dataset_filter.add_argument("--out", default=None, help="output dataset root; defaults to <root>/ydm_dataset/filter")
    dataset_filter.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    dataset_filter.add_argument("--class", dest="class_values", default=None, help="class id/name, comma-separated allowed")
    dataset_filter.add_argument("--min-width", type=float, default=None)
    dataset_filter.add_argument("--min-height", type=float, default=None)
    dataset_filter.add_argument("--min-size-logic", choices=["or", "and"], default="or", help="combine min-width/min-height removal checks")
    dataset_filter.add_argument("--min-area", type=float, default=None)
    dataset_filter.add_argument("--max-area", type=float, default=None)
    dataset_filter.add_argument("--min-conf", type=float, default=None)
    dataset_filter.add_argument("--class-rules", default=None, help="YAML/JSON per-class filter rules")
    dataset_filter.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_filter.add_argument("--dry-run", action="store_true")
    dataset_filter.set_defaults(
        handler=handle_dataset_filter,
        copy_images=True,
        _output_operation="filter",
    )

    dataset_merge = dataset_sub.add_parser("merge", help="merge multiple YOLO datasets with class-name alignment")
    dataset_merge.add_argument("--roots", required=True, help="comma-separated dataset roots")
    dataset_merge.add_argument("--out", default=None, help="output dataset root; defaults to <first-root>/ydm_dataset/merge")
    dataset_merge.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    dataset_merge.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    dataset_merge.add_argument("--images-dir", default="images")
    dataset_merge.add_argument("--labels-dir", default="labels")
    dataset_merge.add_argument("--no-source-prefix", dest="source_prefix", action="store_false", help="do not prefix output image names by dataset index")
    dataset_merge.add_argument("--no-rename-duplicates", dest="rename_duplicates", action="store_false", help="fail on duplicate output image names")
    dataset_merge.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    dataset_merge.add_argument("--dry-run", action="store_true")
    dataset_merge.set_defaults(
        handler=handle_dataset_merge,
        source_prefix=True,
        rename_duplicates=True,
        copy_images=True,
    )

    dataset_duplicates = dataset_sub.add_parser("duplicates", help="find duplicate image files by content hash")
    add_dataset_args(dataset_duplicates)
    dataset_duplicates.add_argument("--out", default=None, help="duplicate CSV; defaults to <root>/ydm_quality/duplicates.csv")
    dataset_duplicates.add_argument("--algorithm", default="sha256")
    dataset_duplicates.set_defaults(handler=handle_dataset_duplicates)

    dataset_bad_images = dataset_sub.add_parser("bad-images", help="find missing or corrupt images")
    add_dataset_args(dataset_bad_images)
    dataset_bad_images.add_argument("--out", default=None, help="CSV output; defaults to <root>/ydm_quality/bad_images.csv")
    dataset_bad_images.set_defaults(handler=handle_dataset_bad_images)

    ann = subparsers.add_parser("ann", help="edit annotations")
    ann_sub = ann.add_subparsers(dest="ann_command", required=True)

    delete = ann_sub.add_parser("delete-class", help="delete annotations of one or more classes")
    add_dataset_args(delete)
    add_write_args(delete)
    delete.add_argument("--class", dest="class_values", required=True, help="class id/name, comma-separated allowed")
    delete.add_argument("--compact", action="store_true", help="remove classes from class.txt and remap ids")
    delete.set_defaults(handler=handle_delete_class, _output_operation="delete_class")

    replace = ann_sub.add_parser("replace-class", help="replace one or more classes with another class")
    add_dataset_args(replace)
    add_write_args(replace)
    replace.add_argument("--from", dest="from_values", required=True, help="source class id/name, comma-separated allowed")
    replace.add_argument("--to", dest="to_value", required=True, help="target class id/name")
    replace.add_argument("--compact", action="store_true", help="remove source classes from class.txt and remap ids")
    replace.set_defaults(handler=handle_replace_class, _output_operation="replace_class")

    merge = ann_sub.add_parser("merge-class", help="merge classes into one class")
    add_dataset_args(merge)
    add_write_args(merge)
    merge.add_argument("--from", dest="from_values", required=True, help="source class id/name, comma-separated allowed")
    merge.add_argument("--to", dest="to_value", required=True, help="target class id/name")
    merge.add_argument("--no-compact", dest="compact", action="store_false", help="keep source class names in class.txt")
    merge.set_defaults(handler=handle_merge_class, compact=True, _output_operation="merge_class")

    rename = ann_sub.add_parser("rename-class", help="rename a class without changing ids")
    add_dataset_args(rename)
    add_write_args(rename)
    rename.add_argument("--from", dest="from_value", required=True, help="source class id/name")
    rename.add_argument("--to", dest="to_value", required=True, help="new class name")
    rename.set_defaults(handler=handle_rename_class, _output_operation="rename_class")

    apply_map = ann_sub.add_parser("apply-map", help="apply class rename/merge/drop yaml")
    add_dataset_args(apply_map)
    add_write_args(apply_map)
    apply_map.add_argument("--map", dest="map_file", required=True, help="YAML class map")
    apply_map.add_argument("--no-compact", dest="compact", action="store_false", help="do not compact class ids")
    apply_map.set_defaults(handler=handle_apply_map, compact=True, _output_operation="apply_map")

    correct_crops = ann_sub.add_parser(
        "correct-from-crops",
        help="update annotation classes from standard vis-crop filenames",
    )
    add_dataset_args(correct_crops)
    correct_crops.add_argument("--crops-dir", required=True, help="directory containing vis-crop images")
    correct_crops.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    correct_crops.add_argument("--to", dest="to_value", required=True, help="target class id/name; use none/null to delete the annotation")
    correct_crops.add_argument("--report", default=None, help="edit report CSV; defaults to ydm_annotation/correct_from_crops/edit_report.csv")
    correct_crops.add_argument("--dry-run", action="store_true", help="report changes without modifying labels")
    correct_crops.set_defaults(handler=handle_correct_from_crops, _output_operation="correct_from_crops")

    correct_error_crops = ann_sub.add_parser(
        "correct-from-error-crops",
        aliases=["correct-gt-from-error-crops"],
        help="update GT classes from eval_error_analysis pred_gt filenames",
    )
    add_dataset_args(correct_error_crops)
    correct_error_crops.add_argument("--crops-dir", required=True, help="directory containing pred_gt crop images")
    correct_error_crops.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    correct_error_crops.add_argument(
        "--pred-dir",
        "--pred-labels-dir",
        dest="pred_dir",
        default=None,
        help="prediction txt directory used for gtnone append or GT replacement",
    )
    correct_error_crops.add_argument(
        "--dedup-iou",
        type=float,
        default=0.5,
        help="IoU threshold for same-class overlapping predictions; default 0.5",
    )
    correct_error_crops.add_argument(
        "--delete-pred-none",
        action="store_true",
        help="delete GT annotation y for prednone_gty crops",
    )
    correct_error_crops.add_argument(
        "--replace-gt-from-pred",
        action="store_true",
        help="replace GT y with prediction x for predx_gty crops",
    )
    correct_error_crops.add_argument("--to", dest="to_value", required=True, help="target class id/name; use none/null to delete the GT annotation")
    correct_error_crops.add_argument("--report", default=None, help="edit report CSV; defaults to ydm_annotation/correct_from_error_crops/edit_report.csv")
    correct_error_crops.add_argument("--dry-run", action="store_true", help="report changes without modifying labels")
    correct_error_crops.set_defaults(handler=handle_correct_from_error_crops, _output_operation="correct_from_error_crops")

    set_attr = ann_sub.add_parser("set-attr", help="set an attribute value on annotations")
    add_dataset_args(set_attr)
    add_write_args(set_attr)
    set_attr.add_argument("--name", required=True, help="attribute name")
    set_attr.add_argument("--value", required=True, help="new attribute value")
    set_attr.add_argument("--class", dest="class_values", default=None, help="optional class id/name filter")
    set_attr.add_argument("--where-value", default=None, help="only update annotations whose current attribute has this value")
    set_attr.set_defaults(handler=handle_set_attr, _output_operation="set_attr")

    delete_attr = ann_sub.add_parser("delete-attr", help="delete annotations matched by an attribute")
    add_dataset_args(delete_attr)
    add_write_args(delete_attr)
    delete_attr.add_argument("--name", required=True, help="attribute name")
    delete_attr.add_argument("--value", default=None, help="attribute value, comma-separated allowed")
    delete_attr.add_argument("--nonzero", action="store_true")
    delete_attr.set_defaults(handler=handle_delete_attr, _output_operation="delete_attr")

    vis = subparsers.add_parser("vis", help="visualize annotations")
    vis_sub = vis.add_subparsers(dest="vis_command", required=True)
    draw = vis_sub.add_parser("draw", help="draw labels on images")
    add_dataset_args(draw)
    draw.add_argument("--out", default=None, help="output image directory; defaults to <root>/ydm_vis/draw")
    draw.add_argument("--limit", type=int, default=None)
    draw.add_argument("--show-conf", action="store_true")
    draw.add_argument("--conf", type=float, default=None, help="optional confidence threshold")
    draw.add_argument("--mask-alpha", type=int, default=64)
    draw.add_argument("--no-fill-mask", dest="fill_mask", action="store_false")
    draw.add_argument("--show-attrs", action="store_true")
    draw.add_argument("--show-id", action="store_true", help="show annotation order id from YOLO txt before class name")
    draw.add_argument("--filter-no-attrs", action="store_true")
    draw.set_defaults(fill_mask=True)
    draw.set_defaults(handler=handle_vis_draw)
    crop = vis_sub.add_parser("crop", help="crop annotation regions into class folders")
    add_dataset_args(crop)
    crop.add_argument("--out", default=None, help="output crop directory; defaults to <root>/ydm_vis/crop")
    crop.add_argument("--keep-shape", action="store_true")
    crop.add_argument("--min-size", type=int, default=1)
    crop.add_argument(
        "--padding",
        type=_parse_crop_padding,
        default=0,
        help="per-side padding: integer pixels or decimal box-size ratio",
    )
    crop.add_argument("--conf", type=float, default=None, help="optional confidence threshold")
    crop.add_argument("--by-attr", action="store_true", help="also save crops into class/attribute-value folders")
    crop.add_argument("--keep-no-attrs", dest="filter_no_attrs", action="store_false")
    crop.set_defaults(filter_no_attrs=True)
    crop.set_defaults(handler=handle_vis_crop)
    manual_box = vis_sub.add_parser(
        "manual-box",
        help="preview one image/label and draw one temporary box without editing the label",
    )
    add_dataset_args(manual_box)
    manual_box.add_argument(
        "--image",
        required=True,
        help="image filename, path relative to --root, or absolute image path",
    )
    manual_box.add_argument(
        "--label",
        default=None,
        help="optional label path; defaults to the label paired with the image",
    )
    manual_box.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="optional class id used to print a complete YOLO row",
    )
    manual_box.add_argument("--max-width", type=int, default=1400)
    manual_box.add_argument("--max-height", type=int, default=900)
    manual_box.add_argument("--min-pixels", type=int, default=2)
    manual_box.add_argument("--precision", type=int, default=6)
    manual_box.add_argument(
        "--hide-existing",
        dest="show_existing",
        action="store_false",
        help="start with existing txt annotations hidden; press L to toggle them",
    )
    manual_box.add_argument(
        "--mask-outside",
        action="store_true",
        help="after drawing a box, mask the area outside it with black",
    )
    manual_box.add_argument("--out", default=None, help="JSON output path; defaults to <root>/ydm_vis/manual_box/<image>.json")
    manual_box.set_defaults(handler=handle_vis_manual_box, show_existing=True)

    export = subparsers.add_parser("export", help="export to another format")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    coco = export_sub.add_parser("coco", help="export YOLO dataset to COCO JSON")
    add_dataset_args(coco)
    coco.add_argument("--out", default=None, help="output COCO JSON path; defaults to <root>/ydm_conversion/coco/instances.json")
    coco.set_defaults(handler=handle_export_coco)
    xany = export_sub.add_parser("xany", help="export YOLO dataset to x-anylabeling JSON files")
    add_dataset_args(xany)
    xany.add_argument("--out", default=None, help="output JSON directory; defaults to <root>/ydm_conversion/xanylabeling")
    xany.set_defaults(handler=handle_export_xany)

    import_cmd = subparsers.add_parser("import", help="import another annotation format")
    import_sub = import_cmd.add_subparsers(dest="import_command", required=True)
    labelme = import_sub.add_parser("labelme", help="import LabelMe JSON directory as YOLO")
    labelme.add_argument("--json-dir", required=True)
    labelme.add_argument("--out", default=None, help="output dataset root; defaults to <json-dir-parent>/ydm_conversion/import_labelme")
    labelme.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    labelme.add_argument("--classes", default=None, help="optional comma-separated class order")
    labelme.add_argument("--attribute-file", default=None, help="optional attribute.yaml for importing shape attributes")
    add_runtime_args(labelme)
    labelme.set_defaults(handler=handle_import_labelme)
    coco_import = import_sub.add_parser("coco", help="import COCO JSON as YOLO")
    coco_import.add_argument("--json", dest="json_path", required=True)
    coco_import.add_argument("--images-dir", required=True)
    coco_import.add_argument("--out", default=None, help="output dataset root; defaults to <json-parent>/ydm_conversion/import_coco")
    coco_import.add_argument("--task", choices=["detect", "segment"], default="detect")
    coco_import.add_argument("--classes", default=None, help="optional comma-separated class order")
    coco_import.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    add_runtime_args(coco_import)
    coco_import.set_defaults(handler=handle_import_coco, copy_images=True)
    voc_import = import_sub.add_parser("voc", help="import Pascal VOC XML directory as YOLO")
    voc_import.add_argument("--annotations-dir", required=True)
    voc_import.add_argument("--images-dir", required=True)
    voc_import.add_argument("--out", default=None, help="output dataset root; defaults to <annotations-parent>/ydm_conversion/import_voc")
    voc_import.add_argument("--classes", default=None, help="optional comma-separated class order")
    voc_import.add_argument("--keep-difficult", dest="skip_difficult", action="store_false")
    add_runtime_args(voc_import)
    voc_import.set_defaults(handler=handle_import_voc, skip_difficult=True)
    mask_import = import_sub.add_parser("mask", help="import semantic segmentation masks as YOLO segmentation")
    mask_import.add_argument("--images-dir", required=True)
    mask_import.add_argument("--masks-dir", required=True)
    mask_import.add_argument("--out", default=None, help="output dataset root; defaults to <images-parent>/ydm_conversion/import_mask")
    mask_import.add_argument("--class-map", default=None, help="YAML/JSON mapping from mask value/color to class name")
    mask_import.add_argument("--background", default="0", help="background mask value or color")
    mask_import.add_argument("--min-area", type=int, default=1, help="minimum connected-component area in pixels")
    mask_import.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    add_runtime_args(mask_import)
    mask_import.set_defaults(handler=handle_import_mask, copy_images=True)

    convert = subparsers.add_parser("convert", help="convert dataset task/form")
    convert_sub = convert.add_subparsers(dest="convert_command", required=True)
    seg2det = convert_sub.add_parser("seg2det", help="convert YOLO segmentation labels to detection labels")
    add_dataset_args(seg2det)
    add_write_args(seg2det)
    seg2det.set_defaults(
        handler=handle_seg2det,
        _output_operation="seg2det",
    )
    pseudo = convert_sub.add_parser("pseudo", help="convert prediction labels to pseudo labels")
    add_dataset_args(pseudo)
    add_write_args(pseudo)
    pseudo.add_argument("--conf", type=float, default=0.0, help="confidence threshold")
    pseudo.add_argument("--keep-conf", dest="drop_confidence", action="store_false", help="keep confidence in output labels")
    pseudo.set_defaults(
        handler=handle_pseudo,
        drop_confidence=True,
        _output_operation="pseudo",
    )
    resize = convert_sub.add_parser("resize", help="resize dataset images and transform YOLO labels")
    add_dataset_args(resize)
    resize.add_argument("--out", default=None, help="output dataset root; defaults to <root>/ydm_conversion/resize")
    resize.add_argument("--width", type=int, default=None, help="target image width in pixels")
    resize.add_argument("--height", type=int, default=None, help="target image height in pixels")
    resize.add_argument("--scale", type=float, default=None, help="uniform scale factor; cannot be combined with --width/--height")
    resize.add_argument("--keep-ratio", dest="keep_ratio", action="store_true", help="keep aspect ratio and letterbox when both dimensions are provided")
    resize.add_argument("--no-keep-ratio", dest="keep_ratio", action="store_false", help="stretch images to the target dimensions")
    resize.add_argument(
        "--interpolation",
        choices=["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"],
        default="lanczos",
    )
    resize.add_argument("--fill-color", default="114,114,114", help="letterbox fill color as gray or R,G,B")
    resize.add_argument("--drop-empty-labels", dest="keep_empty_labels", action="store_false", help="do not write empty label files")
    resize.add_argument("--dry-run", action="store_true", help="validate and report without writing output")
    resize.set_defaults(
        handler=handle_resize,
        keep_ratio=True,
        keep_empty_labels=True,
    )

    eval_cmd = subparsers.add_parser("eval", help="evaluate or compare predictions")
    eval_sub = eval_cmd.add_subparsers(dest="eval_command", required=True)
    compare = eval_sub.add_parser("compare", help="compare prediction labels against GT labels")
    compare.add_argument("--gt-root", required=True)
    compare.add_argument("--pred-root", required=True)
    compare.add_argument("--out", default=None, help="CSV output path; defaults to <gt-root>/ydm_evaluation/compare.csv")
    compare.add_argument("--iou", type=float, default=0.5)
    compare.add_argument("--conf", type=float, default=None)
    compare.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    compare.add_argument("--layout", choices=["auto", "flat", "split_dirs", "image_list", "mixed"], default="auto")
    compare.add_argument("--images-dir", default="images")
    compare.add_argument("--labels-dir", default="labels")
    add_runtime_args(compare)
    compare.set_defaults(handler=handle_eval_compare)
    review = eval_sub.add_parser("review-pack", help="write FP/FN review package from GT and predictions")
    review.add_argument("--gt-root", required=True)
    review.add_argument("--pred-root", required=True)
    review.add_argument("--out", default=None, help="review output directory; defaults to <gt-root>/ydm_evaluation/review_pack")
    review.add_argument("--csv", default=None, help="full compare CSV; defaults to <out>/compare.csv")
    review.add_argument("--iou", type=float, default=0.5)
    review.add_argument("--conf", type=float, default=None)
    review.add_argument("--status", default="fp,fn", help="statuses to include, comma-separated")
    review.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    review.add_argument("--layout", choices=["auto", "flat", "split_dirs", "image_list", "mixed"], default="auto")
    review.add_argument("--images-dir", default="images")
    review.add_argument("--labels-dir", default="labels")
    add_runtime_args(review)
    review.set_defaults(handler=handle_eval_review_pack)
    error_analysis = eval_sub.add_parser(
        "error-analysis",
        help="fine-grained error analysis: FP/FN sub-types, class errors, duplicate GT",
    )
    error_analysis.add_argument("--gt-root", required=True)
    error_analysis.add_argument("--pred-root", required=True)
    error_analysis.add_argument("--out", default=None, help="output directory; defaults to <gt-root>/ydm_evaluation/error_analysis")
    error_analysis.add_argument("--match-iou", type=float, default=0.5)
    error_analysis.add_argument("--low-iou", type=float, default=0.1)
    error_analysis.add_argument("--conf-thres", type=float, default=0.0, help="confidence threshold for predictions")
    error_analysis.add_argument("--nms-iou", type=float, default=0.5, help="same-class NMS IoU threshold; default 0.5")
    error_analysis.add_argument("--no-nms", dest="nms_iou", action="store_const", const=None, help="disable prediction NMS")
    error_analysis.add_argument("--duplicate-iou", type=float, default=0.9, help="IoU threshold for duplicate GT detection")
    error_analysis.add_argument("--class", dest="class_values", default=None, help="class ids/names to evaluate, comma-separated")
    error_analysis.add_argument("--exclude-class", dest="exclude_class_values", default=None, help="class ids/names to exclude, comma-separated")
    error_analysis.add_argument("--min-width", type=float, default=None, help="ignore boxes narrower than this normalized width")
    error_analysis.add_argument("--min-height", type=float, default=None, help="ignore boxes shorter than this normalized height")
    error_analysis.add_argument("--min-area", type=float, default=None, help="ignore boxes smaller than this normalized area")
    error_analysis.add_argument("--min-size-logic", choices=["or", "and"], default="or", help="combine min-width/min-height checks")
    error_analysis.add_argument("--min-pixels", type=float, default=None, help="ignore boxes whose pixel width or height is smaller than this")
    error_analysis.add_argument("--class-rules", default=None, help="YAML/JSON per-class size filter rules")
    error_analysis.add_argument("--val-source", default=None, help="validation image dir or txt list used to limit evaluated stems")
    error_analysis.add_argument("--only-val", action="store_true", help="use the dataset validation split; default is all data")
    error_analysis.add_argument("--class-file", default=None, help="optional class names file; supports 'id name' or one name per line")
    error_analysis.add_argument("--names", dest="class_file", default=None, help="alias of --class-file")
    error_analysis.add_argument("--review", action="store_true", help="write visual review images and box crops grouped by error type")
    error_analysis.add_argument("--crop-padding", type=int, default=12, help="pixel padding around review crops")
    error_analysis.add_argument("--review-workers", type=int, default=None, help="legacy alias for review visualization workers; defaults to --workers")
    error_analysis.add_argument("--review-progress", action="store_true", help="show progress while writing review visualization")
    error_analysis.add_argument("--review-progress-leave", action="store_true", help="keep review progress bar after completion")
    error_analysis.add_argument("--copy-pred-txt", action="store_true", help="copy prediction txt files into review/pred_txt")
    error_analysis.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    error_analysis.add_argument("--layout", choices=["auto", "flat", "split_dirs", "image_list", "mixed"], default="auto")
    error_analysis.add_argument("--images-dir", default="images")
    error_analysis.add_argument("--labels-dir", default="labels")
    add_runtime_args(error_analysis)
    error_analysis.set_defaults(handler=handle_eval_error_analysis)

    metrics = eval_sub.add_parser("metrics", help="compute Ultralytics-style precision/recall/mAP from GT and prediction txt")
    metrics.add_argument("--gt-root", required=True)
    metrics.add_argument("--pred-root", required=True)
    metrics.add_argument("--out", default=None, help="JSON output path; defaults to <gt-root>/ydm_evaluation/metrics.json")
    metrics.add_argument("--csv", default=None, help="per-class CSV; defaults to <gt-root>/ydm_evaluation/metrics.csv")
    metrics.add_argument("--print-table", action="store_true", help="print an Ultralytics-style metrics table instead of JSON")
    metrics.add_argument(
        "--show-original",
        action="store_true",
        help="when class/exclude/min-pixels/class-rules/merge filters are set, show original metrics before final metrics",
    )
    metrics.add_argument("--class", dest="class_values", default=None, help="class ids/names to evaluate, comma-separated")
    metrics.add_argument("--exclude-class", dest="exclude_class_values", default=None, help="class ids/names to exclude, comma-separated")
    metrics.add_argument(
        "--merge-class-map",
        dest="merge_class_map",
        default=None,
        help="target-to-source class mapping as inline JSON/YAML or a JSON/YAML file",
    )
    metrics.add_argument("--conf-thres", type=float, default=0.0, help="confidence threshold for predictions")
    metrics.add_argument("--nms-iou", type=float, default=0.5, help="same-class NMS IoU threshold; default 0.5")
    metrics.add_argument("--no-nms", dest="nms_iou", action="store_const", const=None, help="disable prediction NMS")
    metrics.add_argument("--min-width", type=float, default=None, help="ignore boxes narrower than this normalized width")
    metrics.add_argument("--min-height", type=float, default=None, help="ignore boxes shorter than this normalized height")
    metrics.add_argument("--min-area", type=float, default=None, help="ignore boxes smaller than this normalized area")
    metrics.add_argument("--min-size-logic", choices=["or", "and"], default="or", help="combine min-width/min-height checks")
    metrics.add_argument("--min-pixels", type=float, default=None, help="ignore boxes whose pixel width or height is smaller than this")
    metrics.add_argument(
        "--class-rules",
        default=None,
        help="YAML/JSON per-class size filter rules; overrides global size filters for matching classes",
    )
    metrics.add_argument("--include-empty-classes", dest="ignore_empty_classes", action="store_false", help="include classes with zero GT instances in metrics output")
    metrics.add_argument("--val-source", default=None, help="validation image dir or txt list used to limit evaluated stems")
    metrics.add_argument("--only-val", action="store_true", help="use the dataset validation split; default is all data")
    metrics.add_argument("--class-file", default=None, help="optional class names file; supports 'id name' or one name per line")
    metrics.add_argument("--names", dest="class_file", default=None, help="alias of --class-file")
    metrics.add_argument("--task", choices=["auto", "detect", "segment"], default="detect")
    metrics.add_argument("--layout", choices=["auto", "flat", "split_dirs", "image_list", "mixed"], default="auto")
    metrics.add_argument("--images-dir", default="images")
    metrics.add_argument("--labels-dir", default="labels")
    add_runtime_args(metrics)
    metrics.set_defaults(handler=handle_eval_metrics, ignore_empty_classes=True)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="YOLO dataset root")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--attribute-file", default=None)
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--only-val", action="store_true", help="use the dataset validation split; default is all data")
    parser.add_argument("--layout", choices=["auto", "flat", "split_dirs", "image_list", "mixed"], default="flat")
    add_runtime_args(parser)


def add_runtime_args(parser: argparse.ArgumentParser, *, workers: bool = True) -> None:
    if workers:
        parser.add_argument("--workers", type=int, default=8, help="worker threads for supported loading/processing steps")
    parser.add_argument("--progress", dest="progress", action="store_true", help="show temporary tqdm progress bars")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="hide tqdm progress bars")
    parser.add_argument("--progress-leave", action="store_true", help="keep progress bars after completion")
    parser.set_defaults(progress=True, progress_leave=False)


def add_write_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=None, help="output dataset root; defaults to the command's <root>/ydm_* directory")
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory; default is <dataset-root>/labels_backup",
    )
    parser.add_argument("--report", default=None, help="edit report CSV; defaults to the command's annotation output directory")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false", help="do not copy image files")
    parser.add_argument("--drop-empty-labels", dest="keep_empty_labels", action="store_false", help="do not write empty label files")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing output")
    parser.set_defaults(copy_images=True, keep_empty_labels=True)


def load_from_args(args: argparse.Namespace, *, progress: bool | None = None, progress_leave: bool | None = None):
    root = args.root
    class_file = args.class_file
    split_file = args.split_file
    root_path = Path(root)
    if root_path.suffix.lower() in {".yaml", ".yml"} and root_path.is_file():
        from yolo_data_manager.scripting import _resolve_manager_root

        resolved_root, yaml_class_file, yaml_split_file = _resolve_manager_root(root_path)
        root = str(resolved_root)
        class_file = class_file or yaml_class_file
        if getattr(args, "only_val", False) and split_file is None:
            split_file = yaml_split_file

    return load_yolo_dataset(
        root=root,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=class_file,
        attribute_file=args.attribute_file,
        task=args.task,
        split_file=split_file,
        only_val=getattr(args, "only_val", False),
        layout=args.layout,
        workers=getattr(args, "workers", 8),
        progress=getattr(args, "progress", True) if progress is None else progress,
        progress_leave=getattr(args, "progress_leave", False) if progress_leave is None else progress_leave,
    )


def _resolved_output_root(root: str | Path) -> Path:
    """Resolve a dataset YAML to its dataset directory for default outputs."""

    root_path = Path(root)
    if root_path.suffix.lower() in {".yaml", ".yml"} and root_path.is_file():
        from yolo_data_manager.scripting import _resolve_manager_root

        return _resolve_manager_root(root_path)[0]
    return root_path


def _default_file_path(args: argparse.Namespace, group: str, filename: str) -> str:
    return str(ydm_dir(_resolved_output_root(args.root), group) / filename)


def _value_or_default(value: str | Path | None, default: str | Path) -> str:
    return str(value) if value is not None else str(default)


def _default_report_path(args: argparse.Namespace, operation: str) -> str:
    return str(default_annotation_output(_resolved_output_root(args.root), operation) / "edit_report.csv")


def _eval_output_root(root: str | Path) -> Path:
    """Use a GT dataset root as the anchor for evaluation reports."""

    return _resolved_output_root(root)


def _safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "query"


def handle_layout_detect(args: argparse.Namespace) -> int:
    _print_status("LAYOUT", f"detecting dataset layout: {args.root}")
    info = detect_layout(args.root, progress=args.progress, progress_leave=args.progress_leave)
    print(json.dumps(info.to_dict(), indent=2, ensure_ascii=False))
    return 0


def handle_check(args: argparse.Namespace) -> int:
    out = _value_or_default(args.out, _default_file_path(args, "quality", "check.json"))
    _print_status("CHECK", f"loading and validating dataset: {args.root}")
    dataset = load_from_args(args, progress=args.progress, progress_leave=args.progress_leave)
    _print_status("CHECK", f"validating {len(dataset.images)} images with {max(1, int(args.workers))} worker(s)")
    report = validate_dataset(
        dataset,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    if args.fill_missing_txt:
        _print_status("CHECK", "creating empty txt files for missing labels")
    created = fill_missing_label_files(dataset) if args.fill_missing_txt else []
    payload = {
        "ok": report.ok,
        "summary": report.summary(),
        "issues": report.to_rows(),
        "fixed": {
            "missing_txt_created": [str(path) for path in created],
            "missing_txt_created_count": len(created),
        },
    }
    write_json_report(payload, out)
    _print_check_summary(payload, out)
    if args.print_full:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if report.ok else 2


def handle_stats(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    payload = compute_stats(dataset)
    stats_dir = ydm_dir(_resolved_output_root(args.root), "stats")
    out = _value_or_default(args.out, stats_dir / "stats.json")
    class_csv = _value_or_default(args.class_csv, stats_dir / "class_counts.csv")
    ann_csv = _value_or_default(args.ann_csv, stats_dir / "annotations.csv")
    attr_csv = _value_or_default(args.attr_csv, stats_dir / "attributes.csv")
    plots_dir = _value_or_default(args.plots_dir, stats_dir / "plots")
    write_class_counts_csv(payload, class_csv)
    write_annotation_csv(dataset, ann_csv)
    write_attribute_csv(dataset, attr_csv)
    write_stats_plots(dataset, plots_dir, stats_list=args.stats_list)
    _emit_json(payload, out)
    return 0


def _load_query_dataset(args: argparse.Namespace):
    source = getattr(args, "source", "gt")
    if source == "gt":
        return load_from_args(args)
    if source != "pred":
        raise ValueError("query source must be 'gt' or 'pred'")

    pred_root_value = getattr(args, "pred_root", None)
    if pred_root_value is None:
        raise ValueError("--pred-root is required when --source pred")
    pred_root = Path(pred_root_value)
    prediction_source = pred_root
    images_dir = getattr(args, "images_dir", "images")
    labels_dir = getattr(args, "labels_dir", "labels")
    if (
        (pred_root / labels_dir).is_dir()
        and not (pred_root / images_dir).exists()
    ):
        prediction_source = pred_root / labels_dir

    class_file = getattr(args, "class_file", None)
    if class_file is None:
        class_file_candidates = [
            prediction_source / "class.txt",
            pred_root / "class.txt",
            pred_root.parent / "class.txt",
            _resolved_output_root(args.root) / "class.txt",
        ]
        class_file = next(
            (str(path) for path in class_file_candidates if path.is_file()),
            None,
        )

    stems = None
    if getattr(args, "only_val", False):
        split_source = getattr(args, "split_file", None)
        if split_source is None:
            root = _resolved_output_root(args.root)
            for candidate in (root / "val.txt", root / "val"):
                if candidate.exists():
                    split_source = str(candidate)
                    break
        stems = collect_stems_from_source(split_source) if split_source else None

    return load_error_analysis_dataset(
        prediction_source,
        task=args.task,
        layout=args.layout,
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_file=class_file,
        stems=stems,
        workers=getattr(args, "workers", 8),
        progress=getattr(args, "progress", True),
        progress_leave=getattr(args, "progress_leave", False),
    )


def handle_query_class(args: argparse.Namespace) -> int:
    source = getattr(args, "source", "gt")
    dataset = _load_query_dataset(args)
    class_values = _split_values(args.class_values)
    result = query_by_class(dataset, class_values)
    query_name = _safe_stem("_".join(class_values))
    out = _value_or_default(
        args.out,
        ydm_dir(_resolved_output_root(args.root), "quality") / "query" / f"class_{query_name}.csv",
    )
    result.write_csv(out)
    copy_query_result(
        result,
        images_dir=args.copy_images,
        labels_dir=args.copy_labels,
        filtered_labels=args.filtered_labels,
    )
    print(
        json.dumps(
            {
                "source": source,
                "classes": class_values,
                "matches": len(result),
                "image_files": result.image_names(),
                "label_files": result.label_names(),
                "labels": [str(p) for p in result.label_paths()],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def handle_query_attr(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    values = _split_values(args.value) if args.value else None
    result = query_by_attribute(dataset, args.name, values=values, nonzero=args.nonzero)
    query_name = _safe_stem(args.name)
    out = _value_or_default(
        args.out,
        ydm_dir(_resolved_output_root(args.root), "quality") / "query" / f"attr_{query_name}.csv",
    )
    result.write_csv(out)
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
    out = _value_or_default(args.out, default_dataset_output(_resolved_output_root(args.root), "select"))
    write_yolo_dataset(
        selected,
        out,
        copy_images=args.copy_images,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
        backup_dir=args.backup_dir,
    )
    print(json.dumps({"images": len(selected.images), "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_normalize(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_dataset_output(_resolved_output_root(args.root), "normalize"),
    )
    if not args.dry_run:
        write_yolo_dataset(
            dataset,
            out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": None if args.dry_run else out}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_split(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    splits = split_dataset(
        dataset,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        absolute_paths=args.absolute_paths,
        train_include_list=args.train_include_list,
        val_include_list=args.val_include_list,
    )
    out_dir = Path(args.out) if args.out else _resolved_output_root(args.root)
    backup_root = (
        Path(args.backup_dir)
        if args.backup_dir is not None
        else _resolved_output_root(args.root) / "labels_backup"
    )
    backup_snapshot = move_existing_split_files_to_backup(out_dir, backup_root)
    for split_name, names in splits.items():
        write_split_file(names, out_dir / f"{split_name}.txt")
    print(
        json.dumps(
            {
                "splits": {name: len(values) for name, values in splits.items()},
                "backup_dir": str(backup_snapshot) if backup_snapshot is not None else None,
                "total_class_counts": class_counts_for_images(dataset),
                "val_class_counts": class_counts_for_images(dataset, splits.get("val", [])),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def handle_dataset_yaml(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out_path = Path(args.out) if args.out else _resolved_output_root(args.root) / "dataset.yaml"
    write_dataset_yaml(dataset.classes, out_path, train=args.train, val=args.val, test=args.test)
    print(json.dumps({"out": str(out_path)}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_filter(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_dataset_output(_resolved_output_root(args.root), "filter"),
    )
    before = dataset.annotation_count()
    class_ids = {dataset.class_id(value) for value in _split_values(args.class_values)} if args.class_values else None
    filtered = filter_by_geometry(
        dataset,
        class_ids=class_ids,
        min_width=args.min_width,
        min_height=args.min_height,
        min_size_logic=args.min_size_logic,
        min_area=args.min_area,
        max_area=args.max_area,
        min_confidence=args.min_conf,
        class_rules=_read_class_rules(args.class_rules),
    )
    after = filtered.annotation_count()
    if not args.dry_run:
        write_yolo_dataset(
            filtered,
            out,
            copy_images=args.copy_images,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    print(json.dumps({"before": before, "after": after, "removed": before - after, "out": None if args.dry_run else out}, indent=2, ensure_ascii=False))
    return 0


def _read_class_rules(path: str | None) -> dict | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def handle_dataset_merge(args: argparse.Namespace) -> int:
    roots = _split_values(args.roots)
    if not roots:
        raise ValueError("--roots must contain at least one dataset root")
    out = _value_or_default(args.out, default_dataset_output(_resolved_output_root(roots[0]), "merge"))
    datasets = [
        load_yolo_dataset(
            root,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            task=args.task,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
        )
        for root in roots
    ]
    merged, report = merge_datasets(
        datasets,
        root=out,
        rename_duplicates=args.rename_duplicates,
        source_prefix=args.source_prefix,
    )
    if not args.dry_run:
        write_yolo_dataset(
            merged,
            out,
            copy_images=args.copy_images,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    print(
        json.dumps(
            {
                "images": report.image_count,
                "annotations": report.annotation_count,
                "classes": report.class_names,
                "renamed_images": report.renamed_images,
                "out": None if args.dry_run else out,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def handle_dataset_duplicates(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    groups = find_duplicate_images(
        dataset,
        algorithm=args.algorithm,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    out = _value_or_default(args.out, _default_file_path(args, "quality", "duplicates.csv"))
    write_duplicate_image_csv(groups, out)
    print(json.dumps({"groups": len(groups), "duplicates": [group.__dict__ for group in groups], "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_dataset_bad_images(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    issues = find_bad_images(
        dataset,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    out = _value_or_default(args.out, _default_file_path(args, "quality", "bad_images.csv"))
    write_image_quality_csv(issues, out)
    print(json.dumps({"issues": len(issues), "bad_images": [issue.__dict__ for issue in issues], "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_delete_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = delete_class(dataset, _split_values(args.class_values), compact=args.compact)
    _write_edit_result(edited, report, args, operation="delete_class")
    return 0


def handle_replace_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = replace_class(dataset, _split_values(args.from_values), args.to_value, compact=args.compact)
    _write_edit_result(edited, report, args, operation="replace_class")
    return 0


def handle_merge_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = merge_classes(dataset, _split_values(args.from_values), args.to_value, compact=args.compact)
    _write_edit_result(edited, report, args, operation="merge_class")
    return 0


def handle_rename_class(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, report = rename_class(dataset, args.from_value, args.to_value)
    _write_edit_result(edited, report, args, operation="rename_class")
    return 0


def handle_apply_map(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited, reports = apply_class_map(dataset, args.map_file, compact=args.compact)
    out = _value_or_default(
        args.out,
        default_annotation_output(_resolved_output_root(args.root), "apply_map"),
    )
    report_path = args.report or _default_report_path(args, "apply_map")
    if not args.dry_run:
        write_yolo_dataset(
            edited,
            out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    if report_path:
        rows = []
        for report in reports:
            rows.extend(report.rows)
        from yolo_data_manager.annotation.edit import EditReport

        EditReport(rows=rows).write_csv(report_path)
    print(json.dumps({"reports": len(reports), "out": None if args.dry_run else out, "report": report_path}, indent=2, ensure_ascii=False))
    return 0


def handle_correct_from_crops(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    result, edit_report = correct_labels_from_crops(
        dataset,
        args.crops_dir,
        _parse_optional_class_value(args.to_value),
        backup_dir=getattr(args, "backup_dir", None),
        dry_run=args.dry_run,
    )
    report_path = args.report or _default_report_path(args, "correct_from_crops")
    edit_report.write_csv(report_path)
    payload = result.to_dict()
    payload["dry_run"] = args.dry_run
    payload["report"] = report_path
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def handle_correct_from_error_crops(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    result, edit_report = correct_gt_labels_from_error_crops(
        dataset,
        args.crops_dir,
        _parse_optional_class_value(args.to_value),
        pred_labels_dir=getattr(args, "pred_dir", None),
        dedup_iou=getattr(args, "dedup_iou", 0.5),
        delete_pred_none=getattr(args, "delete_pred_none", False),
        replace_gt_from_pred=getattr(args, "replace_gt_from_pred", False),
        backup_dir=getattr(args, "backup_dir", None),
        dry_run=args.dry_run,
    )
    report_path = args.report or _default_report_path(args, "correct_from_error_crops")
    edit_report.write_csv(report_path)
    payload = result.to_dict()
    payload["dry_run"] = args.dry_run
    payload["report"] = report_path
    print(json.dumps(payload, indent=2, ensure_ascii=False))
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
    _write_edit_result(edited, report, args, operation="set_attr")
    return 0


def handle_delete_attr(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    values = _split_values(args.value) if args.value else None
    edited, report = delete_by_attribute(dataset, args.name, values=values, nonzero=args.nonzero)
    _write_edit_result(edited, report, args, operation="delete_attr")
    return 0


def handle_vis_draw(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_visualization_output(_resolved_output_root(args.root), "draw"),
    )
    render_dataset(
        dataset,
        out,
        limit=args.limit,
        show_confidence=args.show_conf,
        confidence_threshold=args.conf,
        mask_alpha=args.mask_alpha,
        fill_mask=args.fill_mask,
        show_attributes=args.show_attrs,
        show_txt_id=args.show_id,
        filter_no_attributes=args.filter_no_attrs,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_vis_crop(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_visualization_output(_resolved_output_root(args.root), "crop"),
    )
    saved = crop_dataset(
        dataset,
        out,
        keep_shape=args.keep_shape,
        min_size=args.min_size,
        padding=args.padding,
        confidence_threshold=args.conf,
        by_attribute=args.by_attr,
        filter_no_attributes=args.filter_no_attrs,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"saved": saved, "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_vis_manual_box(args: argparse.Namespace) -> int:
    try:
        dataset = load_from_args(args, progress=False, progress_leave=False)
        image = find_dataset_image(dataset, args.image)
        label_path = Path(args.label) if args.label is not None else image.label_path
        if label_path is not None and not label_path.is_absolute():
            root_label_path = dataset.root / label_path
            if root_label_path.exists() or not label_path.exists():
                label_path = root_label_path
        output_arg = args.out or str(
            default_visualization_output(_resolved_output_root(args.root), "manual_box")
            / f"{image.stem}.json"
        )
        if output_arg is not None:
            output_path = Path(output_arg).resolve()
            protected_paths = {image.path.resolve()}
            if label_path is not None:
                protected_paths.add(label_path.resolve())
            if output_path in protected_paths:
                raise ValueError("--out must be a separate JSON path, not the source image or label")

        result = draw_manual_box(
            image.path,
            label_path=label_path,
            class_id=args.class_id,
            class_names=dataset.classes.names,
            max_width=args.max_width,
            max_height=args.max_height,
            min_pixels=args.min_pixels,
            precision=args.precision,
            show_existing=args.show_existing,
            mask_outside=args.mask_outside,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ydm vis manual-box failed: {exc}", file=sys.stderr)
        return 2

    if result is None:
        payload = {"cancelled": True, "image": str(image.path)}
    else:
        payload = result.to_dict()

    output_path = Path(output_arg)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def handle_export_coco(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_conversion_output(_resolved_output_root(args.root), "coco") / "instances.json",
    )
    export_coco(dataset, out)
    print(json.dumps({"out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_export_xany(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_conversion_output(_resolved_output_root(args.root), "xanylabeling"),
    )
    export_xanylabeling(
        dataset,
        out,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_labelme(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    out = _value_or_default(
        args.out,
        default_conversion_output(Path(args.json_dir).parent, "import_labelme"),
    )
    dataset = import_labelme_dir(
        args.json_dir,
        out_root=out,
        task=args.task,
        class_names=classes,
        attribute_file=args.attribute_file,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_coco(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    out = _value_or_default(
        args.out,
        default_conversion_output(Path(args.json_path).parent, "import_coco"),
    )
    dataset = import_coco(
        args.json_path,
        images_dir=args.images_dir,
        out_root=out,
        task=args.task,
        class_names=classes,
        copy_images=args.copy_images,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_voc(args: argparse.Namespace) -> int:
    classes = _split_values(args.classes) if args.classes else None
    out = _value_or_default(
        args.out,
        default_conversion_output(Path(args.annotations_dir).parent, "import_voc"),
    )
    dataset = import_voc_dir(
        args.annotations_dir,
        images_dir=args.images_dir,
        out_root=out,
        class_names=classes,
        skip_difficult=args.skip_difficult,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_import_mask(args: argparse.Namespace) -> int:
    out = _value_or_default(
        args.out,
        default_conversion_output(Path(args.images_dir).parent, "import_mask"),
    )
    dataset = import_semantic_mask_dir(
        args.images_dir,
        args.masks_dir,
        out_root=out,
        class_map=_read_class_rules(args.class_map),
        background=args.background,
        min_area=args.min_area,
        copy_images=args.copy_images,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"images": len(dataset.images), "annotations": dataset.annotation_count(), "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_seg2det(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    edited = segmentation_to_detection(dataset)
    out = _value_or_default(
        args.out,
        default_conversion_output(_resolved_output_root(args.root), "seg2det"),
    )
    write_yolo_dataset(
        edited,
        out,
        copy_images=args.copy_images,
        keep_empty_labels=args.keep_empty_labels,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
        backup_dir=args.backup_dir,
    )
    print(json.dumps({"out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_pseudo(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    pseudo = predictions_to_pseudo_labels(dataset, confidence_threshold=args.conf, drop_confidence=args.drop_confidence)
    out = _value_or_default(
        args.out,
        default_conversion_output(_resolved_output_root(args.root), "pseudo"),
    )
    if not args.dry_run:
        write_yolo_dataset(
            pseudo,
            out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
            include_confidence=not args.drop_confidence,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    print(json.dumps({"annotations": pseudo.annotation_count(), "out": None if args.dry_run else out}, indent=2, ensure_ascii=False))
    return 0


def handle_resize(args: argparse.Namespace) -> int:
    dataset = load_from_args(args)
    out = _value_or_default(
        args.out,
        default_conversion_output(_resolved_output_root(args.root), "resize"),
    )
    fill_color = _parse_fill_color(args.fill_color)
    validate_resize_options(
        width=args.width,
        height=args.height,
        scale=args.scale,
        interpolation=args.interpolation,
    )
    if args.dry_run:
        payload = {
            "images": len(dataset.images),
            "annotations": dataset.annotation_count(),
            "out": None,
            "dry_run": True,
        }
    else:
        result = resize_yolo_dataset(
            dataset,
            out,
            width=args.width,
            height=args.height,
            scale=args.scale,
            keep_ratio=args.keep_ratio,
            interpolation=args.interpolation,
            fill_color=fill_color,
            keep_empty_labels=args.keep_empty_labels,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
        )
        payload = result.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _eval_load_kwargs(args: argparse.Namespace) -> dict:
    """Shared kwargs for loading GT/pred datasets in eval handlers."""
    return {
        "task": args.task,
        "layout": args.layout,
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "workers": args.workers,
        "progress": args.progress,
        "progress_leave": args.progress_leave,
    }


def handle_eval_compare(args: argparse.Namespace) -> int:
    kw = _eval_load_kwargs(args)
    gt = load_yolo_dataset(args.gt_root, **kw)
    pred = load_yolo_dataset(args.pred_root, **kw)
    rows, summary = compare_datasets(gt, pred, iou_threshold=args.iou, confidence_threshold=args.conf)
    out = _value_or_default(args.out, ydm_dir(_eval_output_root(args.gt_root), "evaluation") / "compare.csv")
    write_compare_csv(rows, out)
    print(json.dumps({"summary": summary, "out": out}, indent=2, ensure_ascii=False))
    return 0


def handle_eval_review_pack(args: argparse.Namespace) -> int:
    kw = _eval_load_kwargs(args)
    gt = load_yolo_dataset(args.gt_root, **kw)
    pred = load_yolo_dataset(args.pred_root, **kw)
    rows, summary = compare_datasets(gt, pred, iou_threshold=args.iou, confidence_threshold=args.conf)
    out = _value_or_default(args.out, default_evaluation_output(_eval_output_root(args.gt_root), "review_pack"))
    csv_out = args.csv or str(Path(out) / "compare.csv")
    write_compare_csv(rows, csv_out)
    counts = write_review_pack(
        rows,
        gt,
        out,
        statuses=set(_split_values(args.status)),
        pred=pred,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    print(json.dumps({"summary": summary, "review": counts, "out": out, "csv": csv_out}, indent=2, ensure_ascii=False))
    return 0


def handle_eval_error_analysis(args: argparse.Namespace) -> int:
    out = _value_or_default(
        args.out,
        default_evaluation_output(_eval_output_root(args.gt_root), "error_analysis"),
    )
    val_source = _resolve_eval_val_source(args.gt_root, args.val_source, getattr(args, "only_val", False))
    stems = collect_stems_from_source(val_source)
    gt = load_error_analysis_dataset(
        args.gt_root,
        task=args.task,
        layout=args.layout,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=args.class_file,
        stems=stems,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    pred = load_error_analysis_dataset(
        args.pred_root,
        task=args.task,
        layout=args.layout,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=args.class_file,
        stems=stems,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    class_values = _split_values(args.class_values) if getattr(args, "class_values", None) else None
    exclude_class_values = (
        _split_values(args.exclude_class_values)
        if getattr(args, "exclude_class_values", None)
        else None
    )
    min_width = getattr(args, "min_width", None)
    min_height = getattr(args, "min_height", None)
    min_area = getattr(args, "min_area", None)
    min_size_logic = getattr(args, "min_size_logic", "or")
    min_pixels = getattr(args, "min_pixels", None)
    class_rules = _read_class_rules(getattr(args, "class_rules", None))
    if any(
        value is not None
        for value in (
            class_values,
            exclude_class_values,
            min_width,
            min_height,
            min_area,
            min_pixels,
            class_rules,
        )
    ):
        gt, pred = filter_error_analysis_datasets(
            gt,
            pred,
            class_ids=class_values,
            exclude_class_ids=exclude_class_values,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            min_size_logic=min_size_logic,
            min_pixels=min_pixels,
            class_rules=class_rules,
        )
    error_rows, summary = analyze_errors(
        gt,
        pred,
        match_iou=args.match_iou,
        low_iou=args.low_iou,
        conf_thres=args.conf_thres,
        nms_iou=args.nms_iou,
    )
    dup_rows = find_duplicate_gt(gt, duplicate_iou=args.duplicate_iou)
    write_error_csvs(error_rows, out)
    write_duplicate_gt_csv(dup_rows, out)
    review_counts = (
        write_error_review_pack(
            error_rows,
            gt,
            pred,
            out,
            crop_padding=args.crop_padding,
            workers=args.review_workers if args.review_workers is not None else args.workers,
            progress=args.review_progress or args.progress,
            progress_leave=args.review_progress_leave or args.progress_leave,
        )
        if args.review
        else {}
    )
    copied_pred_txt = copy_prediction_txt_to_review(pred, out, stems=stems) if args.copy_pred_txt else []
    print_error_summary(error_rows, dup_rows)
    print(
        json.dumps(
            {
                "summary": summary,
                "nms_iou": args.nms_iou,
                "duplicate_gt_pairs": len(dup_rows),
                "review": review_counts,
                "pred_txt_copied": len(copied_pred_txt),
                "out": out,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def handle_eval_metrics(args: argparse.Namespace) -> int:
    output_dir = ydm_dir(_eval_output_root(args.gt_root), "evaluation")
    out = _value_or_default(args.out, output_dir / "metrics.json")
    csv_out = _value_or_default(args.csv, output_dir / "metrics.csv")
    csv_path = Path(csv_out)
    size_csv_out = csv_path.with_name(
        f"{csv_path.stem}_size{csv_path.suffix or '.csv'}"
    )
    val_source = _resolve_eval_val_source(args.gt_root, args.val_source, getattr(args, "only_val", False))
    stems = collect_stems_from_source(val_source)
    gt = load_error_analysis_dataset(
        args.gt_root,
        task=args.task,
        layout=args.layout,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=args.class_file,
        stems=stems,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    pred = load_error_analysis_dataset(
        args.pred_root,
        task=args.task,
        layout=args.layout,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_file=args.class_file,
        stems=stems,
        workers=args.workers,
        progress=args.progress,
        progress_leave=args.progress_leave,
    )
    merge_class_map = _load_merge_class_map(args.merge_class_map)
    class_rules = _read_class_rules(args.class_rules)
    class_values = _split_values(args.class_values) if args.class_values else None
    exclude_class_values = _split_values(args.exclude_class_values) if args.exclude_class_values else None
    show_original = args.show_original and (
        bool(class_values)
        or bool(exclude_class_values)
        or args.min_pixels is not None
        or bool(merge_class_map)
        or bool(class_rules)
    )
    original_metrics = None
    if show_original:
        original_metrics = compute_detection_metrics(
            gt,
            pred,
            conf_thres=args.conf_thres,
            nms_iou=args.nms_iou,
            min_width=args.min_width,
            min_height=args.min_height,
            min_area=args.min_area,
            min_size_logic=args.min_size_logic,
            ignore_empty_classes=args.ignore_empty_classes,
        )
    metrics = compute_detection_metrics(
        gt,
        pred,
        class_ids=class_values,
        exclude_class_ids=exclude_class_values,
        merge_class_map=merge_class_map,
        conf_thres=args.conf_thres,
        nms_iou=args.nms_iou,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        min_size_logic=args.min_size_logic,
        min_pixels=args.min_pixels,
        class_rules=class_rules,
        ignore_empty_classes=args.ignore_empty_classes,
    )
    write_metrics_json(metrics, out)
    write_metrics_csv(metrics, csv_out)
    write_size_metrics_csv(metrics, size_csv_out)
    if original_metrics is not None and args.print_table:
        print("Original metrics:")
        print(format_metrics_table(original_metrics))
        print()
        print("Final metrics:")
        print(format_metrics_table(metrics))
    elif original_metrics is not None:
        print(
            json.dumps(
                {
                    "report_type": "detection_metrics_comparison",
                    "original": original_metrics.to_dict(),
                    "final": metrics.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.print_table:
        print(format_metrics_table(metrics))
    else:
        print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _write_edit_result(
    dataset,
    report,
    args: argparse.Namespace,
    *,
    operation: str | None = None,
) -> None:
    operation = operation or getattr(args, "_output_operation", "edit")
    out = _value_or_default(
        args.out,
        default_annotation_output(_resolved_output_root(args.root), operation),
    )
    report_path = args.report or _default_report_path(args, operation)
    if not args.dry_run:
        write_yolo_dataset(
            dataset,
            out,
            copy_images=args.copy_images,
            keep_empty_labels=args.keep_empty_labels,
            workers=args.workers,
            progress=args.progress,
            progress_leave=args.progress_leave,
            backup_dir=args.backup_dir,
        )
    report.write_csv(report_path)
    print(
        json.dumps(
            {
                "changed": len(report.rows),
                "out": None if args.dry_run else out,
                "report": report_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _split_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_fill_color(value: str) -> int | tuple[int, ...]:
    values = _split_values(value)
    if len(values) not in {1, 3, 4}:
        raise ValueError("fill color must be a gray value or R,G,B/R,G,B,A")
    try:
        numbers = tuple(int(item) for item in values)
    except ValueError as exc:
        raise ValueError("fill color values must be integers") from exc
    if any(number < 0 or number > 255 for number in numbers):
        raise ValueError("fill color values must be between 0 and 255")
    return numbers[0] if len(numbers) == 1 else numbers


def _parse_optional_class_value(value: str | None) -> str | None:
    if value is None or value.strip().lower() in {"none", "null"}:
        return None
    return value


def _parse_crop_padding(value: str) -> int | float:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("padding must be an integer or decimal value")
    try:
        if any(marker in text.lower() for marker in (".", "e")):
            return float(text)
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "padding must be an integer pixel value or a floating-point ratio"
        ) from exc


def _resolve_eval_val_source(root: str | Path, val_source: str | None, only_val: bool) -> str | None:
    if val_source is not None:
        return val_source
    if not only_val:
        return None

    root_path = Path(root)
    for candidate in (root_path / "val.txt", root_path / "images" / "val", root_path / "val"):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"only_val=True requires a validation source (val.txt or a val directory) under {root_path}"
    )


def _load_merge_class_map(value: str | None) -> dict[object, object] | None:
    if value is None:
        return None

    try:
        candidate = Path(value)
        if candidate.is_file():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        else:
            data = yaml.safe_load(value)
    except OSError:
        data = yaml.safe_load(value)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("merge-class-map must be a mapping of target class to source classes")
    return data


def _emit_json(payload: dict[str, object], out: str | None) -> None:
    if out:
        write_json_report(payload, out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_status(tag: str, message: str) -> None:
    print(f"\033[36m[{tag}] {message}\033[0m", file=sys.stderr)


def _print_check_summary(payload: dict[str, object], out: str) -> None:
    summary = payload.get("summary")
    fixed = payload.get("fixed")
    issue_counts = summary if isinstance(summary, dict) else {}
    fixed_counts = fixed if isinstance(fixed, dict) else {}
    warning_count = sum(int(count) for key, count in issue_counts.items() if str(key).startswith("warning:"))
    error_count = sum(int(count) for key, count in issue_counts.items() if str(key).startswith("error:"))
    created_count = int(fixed_counts.get("missing_txt_created_count", 0) or 0)

    if error_count or warning_count:
        color = "\033[31m"
        reset = "\033[0m"
        print(
            f"{color}[CHECK WARNING] errors={error_count}, warnings={warning_count}, "
            f"missing_txt_created={created_count}. Full report: {out}{reset}",
            file=sys.stderr,
        )
        for key, count in sorted(issue_counts.items(), key=lambda item: (not str(item[0]).startswith("error:"), str(item[0]))):
            print(f"{color}  {key}: {count}{reset}", file=sys.stderr)
        return

    print(f"\033[32m[CHECK OK] no issues. Full report: {out}\033[0m", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
