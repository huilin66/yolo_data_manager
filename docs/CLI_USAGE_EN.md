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

## Layout and Validation

```bash
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --task auto
ydm check --root path/to/yolo --layout auto
ydm check --root path/to/yolo --layout flat --fill-missing-txt --out validation.json
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
```

`layout detect` emits `report_type: layout_detect`. It is a layout detection result, not a dataset validation/check result. The output also includes `class_source`, `class_count`, and `classes` so you can confirm whether classes were read from `class.txt`, `classes.txt`, `dataset.yaml`, or `data.yaml`.

`check` uses multi-threaded validation and a tqdm progress bar by default with `leave=False`. It writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. If `--out` is omitted, the default file is `<root>/check_result.json`. Use `--workers 16` to tune threads, `--no-progress` to disable the bar, `--progress-leave` to keep it, and `--print-full` only when you also want the full JSON printed to the terminal.

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
```

Write operations target `--out` and do not overwrite the source dataset in place.

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

## Visualization and Cropping

```bash
ydm vis draw --root path/to/yolo --out images_vis
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis draw --root path/to/yolo --out images_vis --show-id
ydm vis draw --root path/to/yolo --out images_vis --workers 8 --progress
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm vis crop --root path/to/yolo --out crops --workers 8 --progress
```

`--show-id` displays the 1-based annotation order from the label txt file. Crop filenames also use 1-based object ids.

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
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --match-iou 0.5 --low-iou 0.1 --duplicate-iou 0.9
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --review-workers 8 --review-progress --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --val-source val.txt --class-file class.txt --out error_report
ydm eval error-analysis --gt-root gt_labels --pred-root pred_labels --names class.txt --out error_report
```

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
