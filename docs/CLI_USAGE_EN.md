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

## Default Output Paths

Explicit `--out`, `--csv`, and `--plots-dir` values always take precedence. When omitted, outputs are organized as follows:

```text
<dataset_root>/
  labels_backup/                         # timestamped label backups
  ydm_quality/                           # check, query, duplicates, bad-images
  ydm_stats/                             # stats.json, CSV files, plots/
  ydm_vis/                               # draw/, crop/, manual_box/
  ydm_evaluation/                        # compare, review_pack, error_analysis, metrics
  ydm_dataset/                           # select, normalize, filter, merge
  ydm_annotation/                        # annotation edits and reports
  ydm_conversion/                        # import/export and task conversions
  train.txt / val.txt / test.txt         # split stays at the dataset root
  dataset.yaml                            # stays at the dataset root
```

Multimodal data uses the same functional groups. It may add modality subdirectories
under `ydm_stats`, `ydm_vis`, or `ydm_conversion` when required, but it does not create
a separate `ydm_multimodal` feature directory.

## Layout and Validation

```bash
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --task auto
ydm check --root path/to/yolo --layout auto
ydm check --root path/to/yolo --layout flat --fill-missing-txt --out validation.json
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
```

`layout detect` emits `report_type: layout_detect`. It is a layout detection result, not a dataset validation/check result. The output also includes `class_source`, `class_count`, and `classes` so you can confirm whether classes were read from `class.txt`, `classes.txt`, `dataset.yaml`, or `data.yaml`.

`check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. If `--out` is omitted, the default file is `<root>/ydm_quality/check.json`. Use `--print-full` only when you also want the full JSON printed to the terminal.

`--fill-missing-txt` creates empty label txt files for images without labels and reports the created files in JSON.

## Query

```bash
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm query class --root path/to/yolo --class person --copy-images out/images --copy-labels out/labels
ydm query class --root path/to/yolo --class person --copy-labels out/labels --filtered-labels
ydm query class --root gt_yolo --source pred --pred-root pred_yolo --class car --class-file gt_yolo/class.txt --out pred_car.csv
ydm query attr --root path/to/yolo --name defect --value yes --out defect.csv
ydm query attr --root path/to/yolo --name defect --nonzero --copy-labels out/labels
```

`query class` searches GT under `--root` by default. To query predictions, use `--source pred --pred-root ...`; `--pred-root` may be a full YOLO prediction root or its `labels` directory. If the prediction directory has no class file, pass the GT/shared names file with `--class-file`. The terminal JSON includes matching `image_files` and `label_files`, while the CSV contains one row per matching annotation.

## Annotation Edits

```bash
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --compact
ydm ann delete-class --root path/to/yolo --class ignore --out yolo_clean --compact
ydm ann replace-class --root path/to/yolo --from old_name --to new_name --out yolo_replaced
ydm ann rename-class --root path/to/yolo --from cls_a --to cls_b --out yolo_renamed
ydm ann apply-map --root path/to/yolo --map class_map.yaml --out yolo_mapped
ydm ann set-attr --root path/to/yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root path/to/yolo --name defect --value yes --out yolo_attr_clean
ydm dataset filter --root path/to/yolo --min-area 0.001 --out yolo_filtered --backup-dir label_backups
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --backup-dir label_backups
ydm ann correct-from-crops --root path/to/yolo --crops-dir ydm_vis/crop/car --to defect --backup-dir label_backups --report crop_correction.csv
ydm ann correct-from-error-crops --root path/to/yolo --crops-dir result_ana/val-52/review/pred_gt/pred_car_gt_background/crops --pred-dir result_ana/val-52/review/pred_txt --dedup-iou 0.5 --to defect --delete-pred-none --backup-dir label_backups --only-val --report gt_correction.csv
```

When `--out` is omitted, write operations use their corresponding `ydm_dataset` or `ydm_annotation` subdirectory and do not overwrite the source dataset in place.
`correct-from-crops` is the exception: it updates the corresponding source label files directly. Use `--dry-run` first when reviewing changes. Standard `vis crop` names use `<image_stem>_<1-based annotation index>.<extension>`. Use `--to none` or `--to null` to delete the corresponding annotation.
`correct-from-error-crops` uses the `y` in `xxx_predx_gty` to locate the GT annotation. When `--pred-dir` is provided, a crop with `gt none` appends prediction txt record `x` to the corresponding GT label, omitting prediction confidence. Without `--pred-dir`, `gt none` crops are skipped.
Added predictions and `--replace-gt-from-pred` replacement boxes are deduplicated by same-class IoU on the same image; overlapping candidates keep the higher-confidence prediction, and a suppressed replacement deletes its duplicate GT row. Use `--dedup-iou` to change the default `0.5` threshold.
With `--delete-pred-none`, `prednone_gty` deletes GT annotation `y` even when `--to` names an update class. For deletion-only review crops, use `--to none --delete-pred-none`; `predx_gty` continues to follow `--to`. With `--replace-gt-from-pred` and `--pred-dir`, `predx_gty` replaces GT row `y` completely with prediction row `x` (class and geometry), `prednone_gty` deletes, and `predx_gtnone` appends.
Commands that write GT label txt files, including `dataset filter`, `dataset merge`, and `ann` edits, support `--backup-dir` to snapshot current input labels before writing. When omitted, the default is `<dataset-root>/labels_backup`; passing it overrides the default. Crop correction backs up only labels it actually changes. Each run creates a `YYYYMMDD_HHMMSS_microseconds` snapshot directory and preserves paths relative to the dataset root; a label file is copied at most once per run. `--dry-run` does not create a backup.

## Dataset Operations

```bash
ydm dataset select --root path/to/yolo --file val.txt --out yolo_val
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 --seed 233
ydm dataset split --root path/to/yolo --train 0.8 --val 0.1 --test 0.1 --absolute-paths
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 \
  --train-include-list train_include.txt --val-include-list val_include.txt
ydm dataset yaml --root path/to/yolo --out dataset.yaml
ydm dataset merge --roots data1,data2 --out merged_yolo
ydm dataset duplicates --root path/to/yolo --out duplicate_images.csv
ydm dataset bad-images --root path/to/yolo --out bad_images.csv
```

`dataset split` prints total box counts by class and validation box counts by class.
`--train-include-list` and `--val-include-list` accept txt files or comma-separated image names/paths. Listed images are removed from the random pool before splitting and then forced into the corresponding split; the two parameters may not overlap.
If `train.txt`, `val.txt`, or `test.txt` already exists in the output directory, split moves it before writing into `<dataset-root>/labels_backup/<timestamp>/`; use `--backup-dir` to override the backup directory.

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
ydm stats --root path/to/yolo
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
ydm vis draw --root path/to/yolo
ydm vis draw --root path/to/yolo --style cv2
ydm vis crop --root path/to/yolo --style pil --workers 16
ydm vis crop --root path/to/yolo --padding 20
ydm vis crop --root path/to/yolo --out crops --padding 0.2
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis draw --root path/to/yolo --out images_vis --show-attrs --filter-no-attrs --att-seperate
ydm vis draw --root path/to/yolo --out images_vis --show-id
ydm vis draw --root path/to/yolo --out images_vis --workers 16
ydm vis draw --root path/to/yolo --out images_vis --no-progress
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm vis crop --root path/to/yolo --out crops --att-seperate
ydm vis crop --root path/to/yolo --out crops --workers 16
ydm vis draw --root path/to/yolo --out images_vis --no-clean
ydm vis crop --root path/to/yolo --out crops --no-clean
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5 --out manual_box.json
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --hide-existing
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --mask-outside
```

`vis draw` and `vis crop` clear the output directory before running by default, so stale files from previous runs are removed. Pass `--no-clean` to keep existing outputs.
`--show-id` displays the 1-based annotation order from the label txt file. Crop filenames also use 1-based object ids.
`vis manual-box` only reads and displays the selected image and matching txt file. Drag one temporary box and press Enter to print pixel and normalized YOLO coordinates; the label is never modified. Use the mouse wheel or `+/-` to zoom and `0` to reset the view. Existing annotations are shown by default; press `L` to toggle them or use `--hide-existing` to start hidden. With `--class-id`, it also prints a complete YOLO row for manual insertion. `--out` writes a separate JSON file only.
With `--mask-outside`, a valid selected box remains visible while the area outside it is masked black; press `R` to redraw the selection.

## Import and Export

```bash
ydm export coco --root path/to/yolo
ydm export xany --root path/to/yolo

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
ydm convert resize --root yolo_data --width 640 --height 640 --out yolo_640
ydm convert resize --root yolo_data --scale 0.5 --out yolo_half
```

`convert resize` preserves the aspect ratio by default. When both `--width` and `--height` are specified, it uses gray letterboxing and transforms detection boxes and segmentation polygons accordingly. Use `--no-keep-ratio` to stretch directly to the target dimensions. The default output is `<root>/ydm_conversion/resize`; the source dataset is not overwritten.

## Evaluation and Error Analysis

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --iou 0.5
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --iou 0.5
ydm eval metrics --gt-root gt_yolo --pred-root pred_yolo
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --out vehicle_metrics.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --out vehicle_no_small.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --print-table
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --exclude-class ignore,background
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --merge-class-map '{"vehicle":["car","truck"]}'
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --min-width 0.01 --class-rules metrics_rules.yaml
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car --min-pixels 15 --show-original
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --match-iou 0.5 --low-iou 0.1 --duplicate-iou 0.9
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --val-source val.txt --class-file class.txt --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class car,bus --exclude-class ignore --min-width 0.01 --min-height 0.01 --min-size-logic and --min-pixels 8 --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class-rules error_rules.yaml --out error_report
ydm eval error-analysis --gt-root gt_labels --pred-root pred_labels --names class.txt --out error_report
```

`eval metrics` computes Precision, Recall, mAP@0.5, and mAP@0.5:0.95. `--class` evaluates only selected classes, while `--exclude-class` independently excludes classes; both can be used together. `--merge-class-map` accepts a target-to-source class mapping as inline JSON/YAML or as a JSON/YAML file, for example `{"vehicle":["car","truck"]}`. The mapping is applied to both GT and predictions before class selection, matching, and aggregation. With `--show-original`, when class, merge, `--class-rules`, or `--min-pixels` filters are supplied, the original metrics are printed before the final metrics; the original run omits those filters/remapping but preserves other filters. JSON output uses `report_type=detection_metrics_comparison` with `original` and `final` entries, while `--out` still writes the final metrics. Classes with `Instances=0` are omitted from output and mean metrics by default; add `--include-empty-classes` to keep them for false-positive checks. Small-object filtering supports `--min-width`, `--min-height`, `--min-area`, `--min-size-logic`, or pixel filtering with `--min-pixels`. Add `--print-table` to print an aligned Ultralytics-style table for manual comparison.
`eval metrics` also accepts `--class-rules` as a YAML/JSON file to override the global size rule per class. Supported fields are `width`/`min_width`, `height`/`min_height`, `min_area`, `min_pixels`, and `logic`/`min_size_logic`; classes without a rule use the global parameters. With `--merge-class-map`, rules match the merged target class names.

```yaml
Hollow:
  width: 0.03
  height: 0.03
  logic: or
Leakage:
  min_pixels: 20
```

Metrics also report COCO-style small, medium, and large target metrics by pixel area: area `< 32²` is small, `32² <= area < 96²` is medium, and area `>= 96²` is large. They are stored under `size_metrics` in JSON and written separately to `metrics_size.csv`; valid image dimensions are required for size classification.

`eval error-analysis` supports `--class` to keep selected classes and `--exclude-class` to exclude classes independently. `--min-width`, `--min-height`, `--min-area`, `--min-size-logic`, and `--min-pixels` filter both GT and predictions. Width/height/area use normalized YOLO coordinates; `--min-pixels` checks pixel width or height. It still accepts legacy `--review-workers`, `--review-progress`, and `--review-progress-leave`; new scripts should prefer the common runtime flags.
`--class-rules` accepts a YAML/JSON file and overrides the global size rule per class using `width`, `height`, and `logic`; classes without a rule use the global parameters.
Both evaluation commands apply confidence-prioritized, class-aware NMS by default with `--nms-iou 0.5`; use `--no-nms` to disable it.

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
