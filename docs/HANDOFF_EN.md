# YOLO Data Manager Handoff

This document summarizes the project design, current scope, module boundaries, and follow-up work. User-facing docs are in [README](../README.md), [PYTHON_USAGE_EN.md](PYTHON_USAGE_EN.md), and [CLI_USAGE_EN.md](CLI_USAGE_EN.md).

## Goals and Principles

YOLO Data Manager exists to keep dataset loading, conversion, statistics, visualization, path handling, and temporary project logic out of one-off scripts.

Core principles:

1. Convert all supported formats into one internal dataset model.
2. Query, edit, statistics, and visualization depend only on that model.
3. Import/export modules own format boundaries and should not duplicate business logic.
4. Write operations default to a new output directory and should not modify the source dataset in place.

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

## Package Structure

```text
yolo_data_manager/
  core/
    models.py
    geometry.py
    schema.py
    errors.py
  io/
    loader.py
    writer.py
    validator.py
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
  vis/
    renderer.py
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

- Default writes go to `--out`.
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
