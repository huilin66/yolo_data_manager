# CLI Usage

This document lists common `ydm` commands and parameters. The README provides a short overview; this file is the detailed CLI reference.

## Installation

```bash
python -m pip install .
ydm --help
```

Development mode:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Run without installing:

```powershell
$env:PYTHONPATH = "src"
python -m yolo_data_manager.cli check --root path/to/yolo
```

## Common Loading Arguments

Most commands that read a YOLO dataset support:

| Flag | Description |
|---|---|
| `--root` | YOLO dataset root |
| `--layout` | `auto`, `flat`, `split_dirs`, `image_list`, `mixed` |
| `--task` | `auto`, `detect`, `segment` |
| `--images-dir` | Image directory name, default `images` |
| `--labels-dir` | Label directory name, default `labels` |
| `--class-file` | Class file path |
| `--attribute-file` | Attribute schema path |
| `--split-file` | Image list file |
| `--only-val` | Process only the validation split; default is all data |

## Common Runtime Arguments

Most commands that load, write, validate, visualize, or evaluate datasets support the same runtime flags:

| Flag | Default | Description |
|---|---|---|
| `--workers` | `8` | Worker threads for supported loading, validation, writing, visualization, and review steps |
| `--progress` | enabled | Show temporary tqdm progress bars |
| `--no-progress` | disables progress | Hide tqdm progress bars |
| `--progress-leave` | `False` | Keep progress bars after completion |

The default style is `workers=8`, tqdm enabled, and `leave=False`. A few pure conversion commands do not use threads internally yet, but keep the same CLI style where applicable.

## Layout and Validation

```bash
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --task auto
ydm check --root path/to/yolo --layout auto
ydm check --root path/to/yolo --layout flat --fill-missing-txt --out validation.json
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
```

`layout detect` emits `report_type: layout_detect`. It is a layout detection result, not a dataset validation/check result. The output also includes `class_source`, `class_count`, and `classes` so you can confirm whether classes were read from `class.txt`, `classes.txt`, `dataset.yaml`, or `data.yaml`.

`check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. If `--out` is omitted, the default file is `<root>/check_result.json`. Use `--print-full` only when you also want the full JSON printed to the terminal.

`--fill-missing-txt` creates empty label txt files for images without labels and reports the created files in JSON.

## Query

```bash
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm query class --root path/to/yolo --class person --copy-images out/images --copy-labels out/labels
ydm query class --root path/to/yolo --class person --copy-labels out/labels --filtered-labels
ydm query attr --root path/to/yolo --name defect --value yes --out defect.csv
ydm query attr --root path/to/yolo --name defect --nonzero --copy-labels out/labels
```

## Annotation Edits

```bash
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --compact
ydm ann delete-class --root path/to/yolo --class ignore --out yolo_clean --compact
ydm ann replace-class --root path/to/yolo --from old_name --to new_name --out yolo_replaced
ydm ann rename-class --root path/to/yolo --from cls_a --to cls_b --out yolo_renamed
ydm ann apply-map --root path/to/yolo --map class_map.yaml --out yolo_mapped
ydm ann set-attr --root path/to/yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root path/to/yolo --name defect --value yes --out yolo_attr_clean
ydm ann correct-from-crops --root path/to/yolo --crops-dir image_vis/crop/car --to defect --report crop_correction.csv
ydm ann correct-from-error-crops --root path/to/yolo --crops-dir result_ana/val-52/review/pred_gt/pred_car_gt_background/crops --pred-dir result_ana/val-52/review/pred_txt --dedup-iou 0.5 --to defect --delete-pred-none --only-val --report gt_correction.csv
```

Write operations target `--out` and do not overwrite the source dataset in place.
`correct-from-crops` is the exception: it updates the corresponding source label files directly. Use `--dry-run` first when reviewing changes. Standard `vis crop` names use `<image_stem>_<1-based annotation index>.<extension>`. Use `--to none` or `--to null` to delete the corresponding annotation.
`correct-from-error-crops` uses the `y` in `xxx_predx_gty` to locate the GT annotation. When `--pred-dir` is provided, a crop with `gt none` appends prediction txt record `x` to the corresponding GT label, omitting prediction confidence. Without `--pred-dir`, `gt none` crops are skipped.
Added predictions and `--replace-gt-from-pred` replacement boxes are deduplicated by same-class IoU on the same image; overlapping candidates keep the higher-confidence prediction, and a suppressed replacement deletes its duplicate GT row. Use `--dedup-iou` to change the default `0.5` threshold.
With `--delete-pred-none`, `prednone_gty` deletes GT annotation `y` even when `--to` names an update class. For deletion-only review crops, use `--to none --delete-pred-none`; `predx_gty` continues to follow `--to`. With `--replace-gt-from-pred` and `--pred-dir`, `predx_gty` replaces GT row `y` completely with prediction row `x` (class and geometry), `prednone_gty` deletes, and `predx_gtnone` appends.

## Dataset Operations

```bash
ydm dataset select --root path/to/yolo --file val.txt --out yolo_val
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 --seed 233
ydm dataset split --root path/to/yolo --train 0.8 --val 0.1 --test 0.1 --absolute-paths
ydm dataset yaml --root path/to/yolo --out dataset.yaml
ydm dataset merge --roots data1,data2 --out merged_yolo
ydm dataset duplicates --root path/to/yolo --out duplicate_images.csv
ydm dataset bad-images --root path/to/yolo --out bad_images.csv
```

`dataset split` prints total box counts by class and validation box counts by class.

## Filtering

Global filtering:

```bash
ydm dataset filter --root path/to/yolo --min-area 0.001 --out yolo_filtered
ydm dataset filter --root path/to/yolo --min-width 0.01 --min-height 0.01 --min-size-logic and --out yolo_filtered
```

`--min-size-logic or` is the default: remove boxes when width or height is below the threshold.  
`--min-size-logic and` removes boxes only when both width and height are below the threshold.

Per-class rules:

```bash
ydm dataset filter --root path/to/yolo --class-rules filter_rules.yaml --out yolo_filtered
```

`filter_rules.yaml`:

```yaml
person:
  min_width: 0.01
  min_height: 0.01
  min_size_logic: and

car:
  min_area: 0.0005

defect:
  min_width: 0.005
  min_height: 0.005
```

## Statistics

```bash
ydm stats --root path/to/yolo --out stats.json
ydm stats --root path/to/yolo --ann-csv annotations.csv --attr-csv attributes.csv --plots-dir stats_plots
ydm stats --root path/to/yolo --plots-dir labels_sta --stats-list all
ydm stats --root path/to/yolo --plots-dir labels_sta --stats-list image_shape,box_shape_pix,box_pos_center
```

`--stats-list` supports:

```text
all, class_counts, box_number, box_width, box_height, box_area,
image_shape, box_shape, box_shape_pix, box_shape_rate,
box_pos_start, box_pos_center, box_pos_end, attribute, legacy_csv
```

Selecting `box_shape`, `box_shape_pix`, `box_shape_rate`, `box_width`, or `box_height` also creates the per-class folders `box_shape_ratios/`, `box_shape_pixels/`, `aspect_ratio/`, `width_image_ratio/`, and `height_image_ratio/`. The `box_width` and `box_height` selections additionally create the class-comparison boxplots `box_width_boxplot.png` and `box_height_boxplot.png`.

## Visualization and Cropping

```bash
ydm vis draw --root path/to/yolo --out images_vis
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis draw --root path/to/yolo --out images_vis --show-id
ydm vis draw --root path/to/yolo --out images_vis --workers 16
ydm vis draw --root path/to/yolo --out images_vis --no-progress
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm vis crop --root path/to/yolo --out crops --workers 16
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5 --out manual_box.json
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --hide-existing
```

`--show-id` displays the 1-based annotation order from the label txt file. Crop filenames also use 1-based object ids.
`vis manual-box` only reads and displays the selected image and matching txt file. Drag one temporary box and press Enter to print pixel and normalized YOLO coordinates; the label is never modified. Use the mouse wheel or `+/-` to zoom and `0` to reset the view. Existing annotations are shown by default; press `L` to toggle them or use `--hide-existing` to start hidden. With `--class-id`, it also prints a complete YOLO row for manual insertion. `--out` writes a separate JSON file only.

## Import and Export

```bash
ydm export coco --root path/to/yolo --out instances.json
ydm export xany --root path/to/yolo --out xany_json

ydm import labelme --json-dir labelme_json --out yolo --task segment
ydm import coco --json instances.json --images-dir images --out yolo --task segment
ydm import voc --annotations-dir Annotations --images-dir JPEGImages --out yolo
```

Semantic mask import:

```bash
ydm import mask --images-dir images --masks-dir masks --out yolo_seg --class-map class_map.yaml --background 0 --min-area 20
```

`class_map.yaml`:

```yaml
0: background
1: crack
2: spalling
```

RGB mask example:

```yaml
"#ff0000": crack
"0,255,0": spalling
```

## Conversion

```bash
ydm convert seg2det --root yolo_seg --out yolo_det
ydm convert pseudo --root pred_yolo --conf 0.5 --out pseudo_yolo
```

## Evaluation and Error Analysis

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --out compare.csv --iou 0.5
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --out review_pack --iou 0.5
ydm eval metrics --gt-root gt_yolo --pred-root pred_yolo --out metrics.json --csv metrics.csv
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --out vehicle_metrics.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --out vehicle_no_small.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --print-table
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --exclude-class ignore,background
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --merge-class-map '{"vehicle":["car","truck"]}'
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car --min-pixels 15 --show-original
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --match-iou 0.5 --low-iou 0.1 --duplicate-iou 0.9
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --val-source val.txt --class-file class.txt --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class car,bus --exclude-class ignore --min-width 0.01 --min-height 0.01 --min-size-logic and --min-pixels 8 --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class-rules error_rules.yaml --out error_report
ydm eval error-analysis --gt-root gt_labels --pred-root pred_labels --names class.txt --out error_report
```

`eval metrics` computes Precision, Recall, mAP@0.5, and mAP@0.5:0.95. `--class` evaluates only selected classes, while `--exclude-class` independently excludes classes; both can be used together. `--merge-class-map` accepts a target-to-source class mapping as inline JSON/YAML or as a JSON/YAML file, for example `{"vehicle":["car","truck"]}`. The mapping is applied to both GT and predictions before class selection, matching, and aggregation. With `--show-original`, when class, merge, or `--min-pixels` filters are supplied, the original metrics are printed before the final metrics; the original run omits those filters/remapping but preserves other filters. JSON output uses `report_type=detection_metrics_comparison` with `original` and `final` entries, while `--out` still writes the final metrics. Classes with `Instances=0` are omitted from output and mean metrics by default; add `--include-empty-classes` to keep them for false-positive checks. Small-object filtering supports `--min-width`, `--min-height`, `--min-area`, `--min-size-logic`, or pixel filtering with `--min-pixels`. Add `--print-table` to print an aligned Ultralytics-style table for manual comparison.

`eval error-analysis` supports `--class` to keep selected classes and `--exclude-class` to exclude classes independently. `--min-width`, `--min-height`, `--min-area`, `--min-size-logic`, and `--min-pixels` filter both GT and predictions. Width/height/area use normalized YOLO coordinates; `--min-pixels` checks pixel width or height. It still accepts legacy `--review-workers`, `--review-progress`, and `--review-progress-leave`; new scripts should prefer the common runtime flags.
`--class-rules` accepts a YAML/JSON file and overrides the global size rule per class using `width`, `height`, and `logic`; classes without a rule use the global parameters.

Review output:

```text
review/
  pred_gt/
    confusion_matrix.png
    pred_classA_gt_classB/
      images/
      crops/
  pred_txt/
```

Crop filename format:

```text
image_pred<pred_txt_order>_gt<gt_txt_order>.jpg
```

Missing sides use `none`.
