# YOLO Data Manager Handoff

This document summarizes the project design, current scope, module boundaries, and follow-up work. User-facing docs are in [README](../README.md), [PYTHON_USAGE_EN.md](PYTHON_USAGE_EN.md), and [CLI_USAGE_EN.md](CLI_USAGE_EN.md).

## Version Maintenance Requirement

After every code update, the project version must be updated before handoff. The single source of truth is `[project].version` in the root `pyproject.toml`: increment the patch version for backward-compatible fixes or features, and use a minor/major increment when a new public API or behavior change is incompatible. Before delivering code, tests, or documentation changes, verify that the version matches the change.

## Goals and Principles

YOLO Data Manager exists to keep dataset loading, conversion, statistics, visualization, path handling, and temporary project logic out of one-off scripts.

Core principles:

1. Convert all supported formats into one internal dataset model.
2. Query, edit, statistics, and visualization depend only on that model.
3. Import/export modules own format boundaries and should not duplicate business logic.
4. Write operations default to a new output directory and should not modify the source dataset in place; default analysis outputs use `ydm_`-prefixed functional groups.

## Default Output Paths and Modality Boundary

The default output groups below a dataset root are:

```text
labels_backup/       timestamped backups before label writes
ydm_quality/         check, query, duplicates, bad-images
ydm_stats/           stats JSON, CSV files, plots/
ydm_vis/             draw/, crop/, manual_box/
ydm_evaluation/      compare, review_pack, error_analysis, metrics
ydm_dataset/         select, normalize, filter, merge
ydm_annotation/      annotation edits and reports
ydm_conversion/      format import/export and task conversions
train.txt/val.txt/test.txt, dataset.yaml  remain at the dataset root
```

Explicit `out`, `csv`, and `plots_dir` values always take precedence. `labels_backup`
does not use the `ydm_` prefix because it is part of the write-safety policy. Multimodality
is a dataset property, not a separate feature group: single-modal and multimodal operations
share these functional directories. Only operations that need to separate image sources add
subdirectories such as `ydm_stats/plots/rgb/`, `ydm_vis/draw/depth/`, or
`ydm_conversion/uint8/depth/`; there is no `ydm_multimodal/` directory.

`MultiModalYoloManager` is a modality-aware loader/cache for associating multiple image
folders with one shared label set. It is not a second business workflow. Its check,
statistics, visualization, crop, and uint8 conversion outputs use the same groups as
`YoloManager`; operations without safe all-modality write semantics must not silently modify
only one image source.

## Example Code and External Datasets

```text
example/                 one caller script per dataset, with paths and parameters
example/functions/       secondarily organized reusable functions
tools/                   standalone helpers such as TT100K conversion
```

`example/dataset_template.py` is the dataset-caller template. Copy it, rename
it for a dataset, import the needed functions from `example.functions`, and
set the dataset path and parameters directly in that file. There is no generic
dataset runner and no need for `run_ydm.py`; separate example files keep
different dataset configurations isolated.

## Current Feature Groups

### Loading and Validation

- Detect `images/`, `labels/`, `class.txt`, `classes.txt`, `dataset.yaml`, and `attribute.yaml`.
- Match images and labels by file stem, not directory order.
- Support YOLO detection, YOLO segmentation, prediction confidence, and multi-attribute labels.
- Support layouts: `flat`, `split_dirs`, `image_list`, `mixed`, and `auto`.
- Normalize different layouts into standard `images/` and `labels/`.
- Support global and class-scoped attributes.
- Validate missing images, missing labels, orphan labels, invalid class ids, invalid coordinates, invalid box sizes, and invalid polygons.

### Import and Export

Implemented:

- YOLO -> COCO
- YOLO -> x-anylabeling
- YOLO segmentation -> YOLO detection
- LabelMe -> YOLO
- COCO -> YOLO
- VOC -> YOLO
- semantic segmentation mask -> YOLO segmentation

Semantic mask import conventions:

- Single-channel masks use pixel values, for example `0=background`, `1=crack`.
- RGB masks use colors, for example `#ff0000=crack`.
- Each connected component becomes one YOLO segmentation polygon.
- Background is not written to labels.
- `min_area` filters tiny connected components.
- If OpenCV is installed, contours are used. Without OpenCV, the importer falls back to bounding-rectangle polygons.

### Dataset Operations

- select/copy subset
- split train/val/test
- merge datasets with class-name alignment
- remap class ids
- generate `dataset.yaml`
- duplicate image hash detection
- bad image detection
- filter by class, area, width, height, confidence
- `min_size_logic=or/and`
- per-class filtering rules

### Annotation Query

Query returns both:

- label-level matches: which txt files contain a class or attribute
- instance-level rows: image, label, line number, class id/name, box/polygon, attributes, confidence

### Annotation Edits

Supported edit operations:

- delete class
- replace class
- merge classes
- merge multiple class groups by dict
- rename class
- apply YAML class map
- set attribute
- delete by attribute

Edits write to a new output directory and can emit reports.

### Statistics

Implemented statistics include:

- image count, label count, annotation count
- class distribution
- objects per image
- empty images
- box width/height/area/aspect ratio
- image size statistics
- polygon point count
- attribute distribution
- class-attribute cross distribution
- annotation CSV
- attribute long-form CSV
- optional PNG plots

### Visualization

Supported:

- detection boxes
- segmentation polygons
- class name, confidence, attributes
- 1-based txt annotation order id
- crop output
- attribute crop grouping
- confidence threshold
- multi-threaded rendering
- progress bars

### Evaluation and Error Analysis

Supported:

- GT vs prediction comparison by class and IoU
- TP/FP/FN CSV
- FP/FN review pack
- fine-grained error analysis:
  - background FP
  - localisation FP
  - duplicate prediction
  - class error
  - FN class error
  - FN low IoU
  - FN no prediction
- duplicate GT detection
- Ultralytics-style confusion matrix with `background`
- `review/pred_gt/pred_<pred_class>_gt_<gt_class>` folders
- review crop names: `image_pred<pred_txt_order>_gt<gt_txt_order>`
- optional prediction txt copy to `review/pred_txt`
- review visualization multi-threading and progress bars
- `eval metrics` supports `--class` inclusion, `--exclude-class` exclusion, and a shared `--merge-class-map` for GT and predictions
- `eval metrics --show-original` prints the original metrics before class/merge/`min_pixels` filtering for comparison
- Dataset loading processes all data by default; use `--only-val` or Python `only_val=True` to limit processing to validation data. YAML `val` no longer implicitly limits ordinary statistics or visualization.
- Added `ann correct-from-crops` / `ann_correct_from_crops`: locate source annotation rows from `vis_crop` filenames (`<image_stem>_<1-based index>`) and correct their classes in place; `to=None` (CLI `--to none`) deletes the annotation.
- Added `ann correct-from-error-crops` / `ann_correct_from_error_crops`: use the y index in `xxx_predx_gty` filenames to locate and correct or delete GT rows; `predx` is review context only.
- All GT-label writing entry points support `--backup-dir` / `backup_dir`: current input labels are copied into a timestamped snapshot before writing; the default directory is `labels_backup` under the dataset root; crop correction backs up only labels it changes, and `dry-run` creates no backup.
- `vis crop` supports `padding`: integers expand each side by pixels, decimals expand each side by the box width/height ratio, and crops are clamped to image boundaries.
- Added `--delete-pred-none` / `delete_pred_none=True`: force deletion of GT row y for `prednone_gty` crops, even when `--to` / `to` is an update class.
- Added `--replace-gt-from-pred` / `replace_gt_from_pred=True`: with prediction txt, replace GT row y completely with prediction x for `predx_gty`; same-image same-class replacements use `dedup_iou` and delete suppressed duplicate GT rows; delete `prednone_gty` and append `predx_gtnone`.

## Package Structure

```text
yolo_data_manager/
  core/
    models.py
    geometry.py
    schema.py
    errors.py
    multimodal.py
  io/
    loader.py
    writer.py
    validator.py
    output_paths.py
    multimodal.py
  annotation/
    query.py
    edit.py
    remap.py
  dataset/
    split.py
    select.py
    filter.py
    merge.py
    duplicates.py
    quality.py
  converters/
    coco.py
    labelme.py
    mask.py
    pseudo.py
    seg_det.py
    voc.py
    xanylabeling.py
  evaluation/
    compare.py
    error_analysis.py
    review_pack.py
  stats/
    compute.py
    export.py
    report.py
    multimodal.py
  vis/
    renderer.py
    multimodal.py
  multimodal_manager.py
  cli.py
  scripting.py
```

## Internal Model

```text
YoloDataset
  root
  classes
  attributes
  images: list[YoloImage]
  orphan_labels

YoloImage
  path
  label_path
  width
  height
  annotations: list[YoloAnnotation]

YoloAnnotation
  class_id
  box
  polygon
  attributes
  confidence
  line_no
```

Detection, segmentation, attributes, and predictions should continue to flow through this model.

## Safety Rules

- When output arguments are omitted, use the unified `ydm_*` default groups; explicit `--out` wins.
- Avoid in-place mutation unless an explicit future feature adds it carefully.
- Preserve user data and prefer copy/write-new-directory workflows.
- Use `dry_run` and report files for potentially destructive annotation edits.
- Compact/remap operations must update `class.txt` and label class ids together.

## Follow-Up Work

Potential next migration targets:

- richer visualization from existing `data_vis/yolo_vis.py`
- remaining statistics from `data_vis/yolo_sta.py`
- specialized importers from `dataformat_swift`
- richer x-anylabeling attribute round-trip
- additional mask polygon simplification controls
- CVAT / Roboflow / Datumaro import paths

## Test Command

```bash
python -m pytest -q
```
