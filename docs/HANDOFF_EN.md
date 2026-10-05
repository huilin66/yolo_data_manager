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

5. Every CLI and YoloManager operation records its start, completion, duration, and exception status. By default, local-date logs are written to ydm_log/YYYY-MM-DD.log under the dataset root, and console operation messages include a local timestamp.

## Default Output Paths and Modality Boundary

The default output groups below a dataset root are:

```text
labels_backup/       timestamped backups; first run stores labels/ and backup_metadata.json at the root
ydm_quality/         check, query, duplicates, bad-images
ydm_stats/           stats JSON, CSV files, basic_info.csv, plots/
ydm_vis/             draw/, crop/, draw_att/, crop_att/, manual_box/
ydm_evaluation/      compare, review_pack, error_analysis, metrics
ydm_dataset/         select, normalize, filter, merge
ydm_annotation/      annotation edits and reports
ydm_conversion/      format import/export and task conversions
ydm_vlm/             automatic labels, error-review plans, and YDM assistant
ydm_log/              daily operation logs (YYYY-MM-DD.log)
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

The first real label backup stores an immutable baseline directly in the backup root as
`labels/` and `backup_metadata.json`. It contains the source labels and available schema
files before the first write and is never overwritten. The `source_labels` value remains
as a compatibility alias for `ydm ann restore-backup`; selecting an operation timestamp
rolls back later snapshots in reverse order. The current files are backed up before a
restore by default.

## Example Code and External Datasets

```text
example/                 one caller script per dataset, with paths and parameters
tools/                   standalone helpers such as TT100K conversion
```

`example/dataset_template.py` is the dataset-caller template. Copy it, rename
it for a dataset, create a `YoloManager`, and set the dataset path and
parameters directly in that file. There is no generic dataset runner and no
need for `run_ydm.py`; separate example files keep different dataset
configurations isolated.

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
- split train/val/test ratios must be non-negative and sum to approximately 1.0; when `test=0`, train is allocated first from the train ratio and all remainder goes to val. Split enables `ensure_class_presence` by default: it first spreads annotated classes across each non-empty split, then approximately matches per-class box counts to the train/val/test ratios. When samples are insufficient it prioritizes train/test/val, or train/val when `test=0`. Because images are indivisible, include lists, image granularity, and capacity limits mean this is best effort; `seed` makes the result reproducible, and the strategy can be disabled with `--no-ensure-class-presence` or `ensure_class_presence=False`. `require_labels=True` / `--require-labels` restricts splitting to images with a corresponding `.txt` label.
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
- annotation CSV, including an inferred `split` column when available
- split inference for stats and annotation CSV uses root `train.txt`/`val.txt`/`test.txt` exclusively when any exists, and falls back to split directories only when all three are absent
- basic_info.csv with image, class, and standalone attribute/value counts grouped by total, train, val, and test
- attribute_num.png and attribute_num.csv for aggregate attribute counts
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
- `style=pil/cv2` rendering backends, with `cv2` as the default
- `show_attrs=True, att_seperate=True` copies rendered images into `draw_att/attribute/level`; `vis_crop(..., att_seperate=True)` copies actual crop files into `crop_att/attribute/level` rather than draw files; `filter_level` excludes selected levels by 1-based index or level name, defaults to `[1]`, and accepts an empty list to disable filtering
- `vis draw` / `vis crop` clear the output directory before running by default (`clean=True`), so stale files from previous runs (e.g. outdated crops) do not accumulate; pass `clean=False` in Python or `--no-clean` in the CLI to keep existing outputs. The cleanup is guarded: it refuses to clear a directory that is the dataset root (or an ancestor) or that contains source images/labels (raises `ValueError`), so source data cannot be deleted accidentally.

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
- `eval_error_analysis_common` / `ydm eval error-analysis-common` reads multiple error-analysis outputs, extracts shared `fn_no_pred` GT and shared same-class background/class-error predictions by IoU, and writes common crops plus original/common/remaining count tables per input
- `eval_error_analysis_common` also confidence-fuses each common prediction group into `common_pred_txt/` and remaps crop `predX` indices; after manual crop selection, pass it to `ann_correct_from_error_crops(..., pred_dir=common_pred_txt, replace_gt_from_pred=True)` for GT updates
- `eval_error_analysis` optionally accepts `conf_curve=True` / `--conf-curve` to count review error-group files at confidence thresholds 0.1 through 0.5 and write `conf_curve.csv`, `conf_curve_summary.csv`, and `conf_curve.png`
- attribute error analysis on one-to-one matched same-class boxes, with `attribute_error.csv`, an optional attribute-error review pack, and `review/attribute_error/attribute_<name>/confusion_matrix.png` for each attribute; matrix rows are predicted values, columns are true values, and correct/incorrect matches are included; external prediction labels can share the GT `attribute.yaml`
- attribute-error crops use `predX_gtY` to locate prediction/GT label rows; use `ann_att_correct_from_error_crops` / `correct_gt_attributes_from_error_crops` for single directories or directory-to-attribute-rule mappings, preserving class and geometry with `dry_run`/`backup_dir` support
- added `ann_att_correct_from_crops` / `correct_gt_attributes_from_crops` for standard `vis_crop` filenames (`<image_stem>_<1-based index>`), with directory-to-attribute-rule mappings and the default `labels_backup` snapshot
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
- `ann_correct_from_error_crops` also accepts a `{crop_dir: target_class}` mapping, processes multiple error-crop directories in one task with one label-backup snapshot, and deletes the selected GT row for a `None` target; the single-directory `(crops_dir, to)` form remains compatible.
- Added `ann_update_from_map(class_map)`: accepts a Python dictionary and applies `rename`, `merge`, and `drop` in order in place; one timestamped snapshot backs up all labels and the class source by default, with no temporary YAML map.
- `convert filename-remap` writes the identical filename mapping JSON to both the source and output dataset `ydm_conversion/filename_mapping.json`; an explicitly supplied mapping path receives an additional copy.
- All GT-label writing entry points support `--backup-dir` / `backup_dir`: current input labels are copied into a timestamped snapshot before writing; the default directory is `labels_backup` under the dataset root; crop correction backs up only labels it changes, and `dry-run` creates no backup.
- `vis crop` supports `padding`: integers expand each side by pixels, decimals expand each side by the box width/height ratio, and crops are clamped to image boundaries.
- Added `--delete-pred-none` / `delete_pred_none=True`: force deletion of GT row y for `prednone_gty` crops, even when `--to` / `to` is an update class.
- Added `--replace-gt-from-pred` / `replace_gt_from_pred=True`: with prediction txt, replace GT row y completely with prediction x for `predx_gty`; same-image same-class replacements use `dedup_iou` and delete suppressed duplicate GT rows; delete `prednone_gty` and append `predx_gtnone`.

## VLM Extension

- VLM settings are loaded from .env; qwen is built in for DashScope and local OpenAI-compatible services.
- vlm auto-label calls the configured VLM and writes a new YOLO dataset from validated JSON detections.
- vlm verify-errors reads eval_error_analysis class and attribute crops and writes correction_plan.json plus crop maps for the existing annotation correction APIs.
- ydm assistant converts natural-language intent into an allowlisted YoloManager method; write operations require explicit confirmation.
- Providers implement VLMProvider and can be extended with register_vlm_provider. Model output is schema-validated and is never executed as shell or Python code.

See README, CLI_USAGE, and PYTHON_USAGE for configuration and examples. The default VLM output group is ydm_vlm under the dataset root.

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
