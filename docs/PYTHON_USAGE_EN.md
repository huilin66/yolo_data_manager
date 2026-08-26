# Python Usage

This document describes Python-first usage. For CLI usage, see [CLI_USAGE_EN.md](CLI_USAGE_EN.md). For handoff notes, see [HANDOFF_EN.md](HANDOFF_EN.md).

## Installation

Runtime install:

```bash
python -m pip install .
```

Development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## YoloManager

`YoloManager` stores the dataset root and common loading options. Methods that operate on the current dataset automatically reuse those values.

The code block below shows common initialization patterns and frequent calls, not the complete API. See the method quick-reference section for the full list.

```python
from yolo_data_manager import YoloManager

mgr = YoloManager("datasets/my_yolo", layout="auto", init_check=False)
mgr = YoloManager(r"E:\repository\yolo8\ultralytics\cfg\datasets\data_fire.yaml", layout="auto")
mgr = YoloManager(
    "datasets/my_yolo",
    layout="flat",
    init_layout_progress=True,
    init_layout_progress_leave=False,
    init_check_workers=16,
    init_check_progress=True,
    init_check_progress_leave=False,
)

mgr.check(out="validation.json", fill_missing_txt=True)
mgr.layout_detect()
```

`YoloManager(..., layout="auto")` initializes by detecting layout, loading images/labels, and then running check.

`root` may also be an Ultralytics-style `data.yaml/dataset.yaml`. In that case `path` becomes the dataset root and `names` becomes the class source. Dataset operations process all data by default; set `only_val=True` explicitly to use the YAML `val` entry (or `val.txt`/a `val` directory under the dataset root).

## Common Runtime Arguments

Most methods that load, write, validate, visualize, or evaluate datasets support the same runtime keyword arguments:

| Argument | Default | Description |
|---|---|---|
| `workers` | `8` | Worker threads for supported loading, validation, writing, visualization, and review steps |
| `progress` | `True` | Show temporary tqdm progress bars |
| `progress_leave` | `False` | Keep progress bars after completion |
| `only_val` | `False` | Process only the validation split; default is all data |

```python
mgr.check(workers=16)
mgr.stats(only_val=True)
mgr.vis_draw(out="images_vis", progress=False)
mgr.vis_crop(clean=False)  # keep outputs from previous runs instead of clearing
mgr.eval_error_analysis(pred_root="pred", out="error_report", review=True, workers=16)
```

Lower-level functions such as `load_yolo_dataset()` and `validate_dataset()` default to quiet progress-free execution so they are pleasant as library calls. `YoloManager` and the CLI show progress by default.

`check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. If `out` is omitted, the default file is `<root>/ydm_quality/check.json`.

### Default output paths

The Python API and CLI use the same defaults. Explicit `out`, `csv`, or `plots_dir` values always take precedence:

```text
<root>/labels_backup/       timestamped backups before label writes
<root>/ydm_quality/         check, query, duplicates, bad-images
<root>/ydm_stats/           stats.json, CSV files, plots/
<root>/ydm_vis/             draw/, crop/, manual_box/
<root>/ydm_evaluation/      compare, review_pack, error_analysis, metrics
<root>/ydm_dataset/         select, normalize, filter, merge
<root>/ydm_annotation/      annotation-edit outputs and edit_report.csv
<root>/ydm_conversion/      format import/export and task conversions
<root>/train.txt, val.txt, test.txt, dataset.yaml
```

`dataset_split()` keeps split files in the dataset root, and `dataset_yaml()` keeps
`dataset.yaml` there as well. `only_val` changes the data scope, not the output group.
`train_include_list` and `val_include_list` accept an image-name/path list or a
txt file with one image name/path per line. These images are removed from the
random pool before splitting, then forced into train or val. The two lists may
not overlap. Relative image paths are matched from the dataset root; bare file
names and stems are also supported.
If `train.txt`, `val.txt`, or `test.txt` already exists in the output directory,
split moves it before writing into `<dataset-root>/labels_backup/<timestamp>/`;
pass `backup_dir` to override the backup directory.
Multimodal data uses these same functional groups; modality subdirectories are added only
where needed, and there is no separate `ydm_multimodal` feature module.

The default locations are also available directly from `YoloManager`:

```python
mgr.output_stats
mgr.output_vis
mgr.output_evaluation
mgr.output_labels_backup
mgr.output_dataset_yaml
```

These properties return `Path` objects and do not create directories; the corresponding
manager operation creates outputs as needed.

`layout_detect()` prints a layout detection result, not a validation/check result. The output has `report_type: layout_detect` and includes `class_source`, `class_count`, and `classes`.

## Statistics

```python
mgr.stats()
mgr.stats(out="stats.json", class_csv="class_counts.csv", attr_csv="attributes.csv")
mgr.stats(plots_dir="labels_sta", stats_list=["all"])
mgr.stats(plots_dir="labels_sta", stats_list=["image_shape", "box_shape_pix", "box_pos_center"])
```

`stats_list` supports:

```text
all, class_counts, box_number, box_width, box_height, box_area,
image_shape, box_shape, box_shape_pix, box_shape_rate,
box_pos_start, box_pos_center, box_pos_end, attribute, legacy_csv
```

Selecting `box_shape`, `box_shape_pix`, `box_shape_rate`, `box_width`, or `box_height` also writes the per-class folders `box_shape_ratios/`, `box_shape_pixels/`, `aspect_ratio/`, `width_image_ratio/`, and `height_image_ratio/` respectively. Each folder contains one plot per class. `box_width` and `box_height` also write `box_width_boxplot.png` and `box_height_boxplot.png`, with classes on the x-axis and normalized box width or height on the y-axis.

## Query

`class_` is used because `class` is a Python keyword.

```python
mgr.query_class(class_=["car", "truck"], out="vehicles.csv")
mgr.query_class(class_=["person"], copy_images="persons/images", copy_labels="persons/labels", filtered_labels=True)
mgr.query_class(
    class_="car",
    source="pred",
    pred_root="datasets/pred_labels",
    class_file="datasets/class.txt",
    out="pred_car.csv",
)
mgr.query_attr(name="occluded", value=["yes"], out="occluded.csv")
mgr.query_attr(name="quality", nonzero=True)
```

The query CSV contains one row per matching annotation. For direct Python access, `query_by_class(dataset, ["car"])` returns a `QueryResult`; use `result.image_names()` or `result.label_names()` to get unique matching filenames.

## Dataset Operations

```python
mgr.dataset_normalize(out="normalized_yolo")
mgr.dataset_split(train=0.8, val=0.1, test=0.1, seed=233)
mgr.dataset_split(train=0.8, val=0.1, test=0.1, absolute_paths=True)
mgr.dataset_split(
    train=0.8,
    val=0.2,
    train_include_list=["images/keep_train_001.jpg", "keep_train_002.jpg"],
    val_include_list="val_include.txt",
)
mgr.dataset_select(file="val.txt", out="val_subset")
mgr.dataset_yaml(out="dataset.yaml", train="images/train", val="images/val")
mgr.dataset_duplicates(out="duplicates.csv")
mgr.dataset_bad_images(out="bad_images.csv")
```

Filtering:

```python
mgr.dataset_filter(out="filtered", min_area=0.001, class_=["car", "truck"], backup_dir="label_backups")
mgr.dataset_filter(out="filtered_small", min_width=0.01, min_height=0.01, min_size_logic="and")
mgr.dataset_filter(
    out="filtered_by_class",
    class_rules={
        "person": {"min_width": 0.01, "min_height": 0.01, "min_size_logic": "and"},
        "car": {"min_area": 0.0005},
        "defect": {"min_width": 0.005, "min_height": 0.005},
    },
)
```

`min_size_logic="or"` removes boxes when width or height is too small. `min_size_logic="and"` removes boxes only when both width and height are too small.
Per-class rules also accept the concise form `{"class_name": {"width": 0.03, "height": 0.03, "logic": "or"}}`; `width` and `height` are normalized YOLO dimensions, and `logic` is `or` or `and`.

Merging:

```python
mgr.dataset_merge(roots=["datasets/part1", "datasets/part2"], out="merged_yolo", source_prefix=True)
```

## Annotation Edits

```python
mgr.ann_merge_class(from_=["crack", "break"], to="defect", out="yolo_merged", compact=True,
                    backup_dir="label_backups")
mgr.ann_merge_class({"vehicle": ["car", "truck"], "human": ["person"]}, out="yolo_merged_multi")
mgr.ann_delete_class(class_=["ignore"], out="yolo_clean", compact=True)
mgr.ann_replace_class(from_=["old_name"], to="new_name", out="yolo_replaced")
mgr.ann_rename_class(from_="cls_a", to="cls_b", out="yolo_renamed")
mgr.ann_apply_map(map_file="class_map.yaml", out="yolo_mapped")
mgr.ann_set_attr(name="defect", value="yes", class_=["sign"], out="yolo_attr")
mgr.ann_delete_attr(name="quality", value=["bad"], out="yolo_clean")
mgr.ann_correct_from_crops(
    crops_dir="ydm_vis/crop/car",
    to="defect",
    only_val=True,
    report="crop_correction.csv",
    backup_dir="label_backups",
    dry_run=True,
)
```

Write operations output to a new directory. Commands that write GT label txt files accept `backup_dir` to snapshot current input labels before writing; crop correction is the in-place exception and backs up only labels it changes. Use `dry_run=True` when you want to inspect the effect first.
Pass `to=None` to delete the corresponding annotation instead of assigning a class.
Pass `backup_dir="label_backups"` to override the default backup directory. If omitted, backups go to `<dataset-root>/labels_backup`. Each source txt is backed up at most once per run; `dry_run=True` creates no backup.
Use `mgr.ann_correct_from_error_crops(...)` for `eval_error_analysis` crops; in `xxx_predx_gty`, the 1-based `y` locates the GT annotation. Provide `pred_dir` to append prediction txt record `x` for `gtnone` crops, without prediction confidence. Added predictions use same-class IoU deduplication (default `dedup_iou=0.5`) and keep the higher-confidence candidate.
The same `backup_dir` option applies to `ann_correct_from_error_crops`.
Set `delete_pred_none=True` to delete the GT row `y` for `prednone_gty` crops even when `to` is an update class. For deletion-only crops, pass `to=None` and `delete_pred_none=True`.
Set `replace_gt_from_pred=True` with `pred_dir` to replace GT row `y` completely with prediction row `x` for `predx_gty` crops; same-class overlapping replacements use `dedup_iou` and keep the higher-confidence prediction, while the suppressed duplicate GT row is deleted. `prednone_gty` is deleted and `predx_gtnone` is appended.

## Visualization

```python
mgr.vis_draw(out="images_vis", show_conf=True, show_attrs=True, style="cv2")  # default; use style="pil" for the PIL backend
mgr.vis_draw(out="images_vis", conf=0.25, fill_mask=True, mask_alpha=64)
mgr.vis_draw(out="images_vis", show_id=True)
mgr.vis_draw(out="images_vis", workers=16)
mgr.vis_draw(out="images_vis", progress=False)
mgr.vis_crop(out="crops", by_attr=True, min_size=32)
mgr.vis_crop(out="crops", workers=16)
mgr.vis_crop(out="crops", padding=20)    # add 20 pixels on each side
mgr.vis_crop(out="crops", padding=0.2)   # add 20% of box width/height on each side
mgr.vis_manual_box(
    image="images/0001.jpg",
    class_id=5,
    show_existing=False,
    mask_outside=True,
    out="manual_box.json",
)
```

`vis_manual_box` displays the selected image and its matching YOLO txt, then
lets the user draw one temporary box. It prints pixel and normalized YOLO
coordinates (and a complete row when `class_id` is provided) without changing
the source label file. Existing annotations are shown by default; press `L` to
toggle them, or pass `show_existing=False` to start hidden.
Use the mouse wheel or `+/-` to zoom and `0` to reset the view.
With `mask_outside=True`, a valid selected box remains visible while the area
outside it is masked black; press `R` to redraw.

`show_id=True` displays the 1-based annotation order from the label txt file.

## Import and Export

```python
mgr.export_coco(out="instances.json")
mgr.export_xany(out="xany_json")

mgr.import_labelme(json_dir="labelme_json", out="yolo_out", task="segment")
mgr.import_coco(json_path="instances.json", images_dir="images", out="yolo_out")
mgr.import_voc(annotations_dir="Annotations", images_dir="JPEGImages", out="yolo_out")
mgr.import_mask(
    images_dir="images",
    masks_dir="masks",
    out="yolo_seg",
    class_map={0: "background", 1: "crack", 2: "spalling"},
    background=0,
    min_area=20,
)
```

`import_mask` converts semantic segmentation masks to YOLO segmentation. Single-channel masks use pixel values; RGB masks can use keys such as `"#ff0000"` or `"255,0,0"`.

## Conversion

```python
mgr.convert_seg2det(out="yolo_det")
mgr.convert_pseudo(out="pseudo_labels", conf=0.5, drop_confidence=True)
mgr.resize_images(out="yolo_640", width=640, height=640, keep_ratio=True)
mgr.resize_images(out="yolo_half", scale=0.5)
```

`resize_images` keeps the aspect ratio by default. For a letterboxed resize, labels are transformed automatically; `keep_ratio=False` performs a direct stretch, so normalized YOLO coordinates retain their values. The default output is `ydm_conversion/resize` under the manager root.

## Evaluation and Error Analysis

```python
mgr.eval_compare(gt_root="datasets/gt", pred_root="datasets/pred", out="compare.csv", iou=0.5)
mgr.eval_review_pack(gt_root="datasets/gt", pred_root="datasets/pred", out="review_pack", status=["fp", "fn"])
mgr.eval_metrics(pred_root="datasets/pred_labels", class_=["car", "bus"], min_pixels=8, out="metrics.json", csv="metrics.csv")
mgr.eval_metrics(pred_root="datasets/pred_labels", class_=["car", "bus"], print_table=True)
mgr.eval_metrics(
    pred_root="datasets/pred_labels",
    exclude_class_=["ignore", "background"],
    merge_class_map={"vehicle": ["car", "truck"]},
    class_rules={
        "Hollow": {"width": 0.03, "height": 0.03, "logic": "or"},
        "Leakage": {"min_pixels": 20},
    },
    show_original=True,
)
mgr.eval_metrics(pred_root="datasets/pred_labels", ignore_empty_classes=False)

mgr.eval_error_analysis(pred_root="datasets/pred_labels", out="error_report")
mgr.eval_error_analysis(
    pred_root="datasets/pred_labels",
    out="error_report",
    match_iou=0.5,
    low_iou=0.1,
    conf_thres=0.25,
    nms_iou=0.5,
    duplicate_iou=0.9,
    review=True,
    workers=8,
    progress=True,
    progress_leave=False,
    copy_pred_txt=True,
)
mgr.eval_error_analysis(
    pred_root="datasets/pred_labels",
    out="error_report",
    class_=["car", "bus"],
    exclude_class_=["ignore"],
    min_width=0.01,
    min_height=0.01,
    min_area=0.0005,
    min_size_logic="and",
    min_pixels=8,
    class_rules={
        "Efflorescene Low Risk": {"width": 0.03, "height": 0.03, "logic": "or"},
        "Broken High Risk": {"width": 0.01, "height": 0.02, "logic": "and"},
    },
)
```

`eval_metrics` uses `class_` to select classes and the independent `exclude_class_` parameter to exclude classes; both can be supplied together. `merge_class_map` accepts a target-to-source mapping such as `{"vehicle": ["car", "truck"]}` and applies it to GT and predictions before class selection, matching, and aggregation. Class selection and exclusion use the merged target class names. With `show_original=True`, when class, merge, `class_rules`, or `min_pixels` filters are supplied, the original metrics are output before the final metrics; JSON output contains `original` and `final`, while the `out` file still stores the final metrics.
`eval_metrics` also accepts `class_rules` to override the global size filter per class. Rules may use class names or ids and support `width`/`min_width`, `height`/`min_height`, `min_area`, `min_pixels`, and `logic`/`min_size_logic`. A configured class uses its complete rule, unmatched classes keep the global parameters, and rules match merged target class names when `merge_class_map` is used.
`eval_metrics` also reports COCO-style small, medium, and large target metrics by pixel area: area `< 32²` is small, `32² <= area < 96²` is medium, and area `>= 96²` is large. They are stored under `size_metrics` in JSON and written separately to `metrics_size.csv`; valid image dimensions are required for size classification.

Statistics, visualization, and evaluation process all data by default; set `only_val=True` or provide `val_source` explicitly to limit processing to validation data.

When `gt_root` or `class_file` is omitted, `YoloManager` falls back to the manager root and `class.txt` when available. Evaluation uses all data by default; set `only_val=True` or provide `val_source` to limit it to validation data.

`eval_error_analysis` supports the same class and size filters: `class_` selects classes, `exclude_class_` excludes classes, and `min_width`, `min_height`, `min_area`, `min_size_logic`, and `min_pixels` filter both GT and predictions. Width and height/area use normalized YOLO coordinates; `min_pixels` checks pixel width or height.
`class_rules` overrides the global size rule per class using `width`, `height`, and `logic`; classes without a rule continue to use the global parameters.
`eval_error_analysis` and `eval_metrics` apply confidence-prioritized, class-aware NMS first by default (`nms_iou=0.5`), then use the same one-to-one IoU matching rule. Pass `nms_iou=None` to disable NMS; disabled-NMS duplicates are marked as `duplicate_prediction` in error analysis and counted as FPs in metrics.

## Multimodal YOLO Datasets

Multimodality is a dataset property, not a separate functional module. `MultiModalYoloManager` is a modality-aware loading and association adapter for datasets with one shared YOLO label set and multiple aligned image folders. It follows the same output groups as `YoloManager` (`ydm_quality`, `ydm_stats`, `ydm_vis`, and `ydm_conversion`). Methods whose all-modality write semantics are not yet defined, such as edit, split, and merge, are intentionally not exposed. It is a Python API and currently has no CLI command.

Each image or label filename is normalized to a scene stem by removing its extension and its configured source `suffix`. For example, `visible/0001_V.jpg`, `infrared/0001_T.png`, and `labels/0001_gt.txt` all associate with scene `0001`.

```python
from yolo_data_manager import (
    MultiModalYoloManager,
)

root = r"E:\datasets\mdet_train"

# Empty configuration matches unchanged stems. Extensions may differ.
mgr = MultiModalYoloManager(
    root,
    image_dirs=["visible", "infrared", "depth"],
    labels_dir="labels",
    class_file="class.txt",
    task="detect",
)

stats = mgr.stats(stats_list=["all"])
mgr.vis_draw(show_id=True, workers=8)
mgr.vis_crop(workers=8)

mgr.check()  # prints a compact summary and writes ydm_quality/multimodal_check.json
```

`check()` also returns `image_type_summary`: the source-image count and groups by `format / Pillow mode / dtype / channel count / resolution` for every modality. This makes mixed inputs such as `JPEG/RGB/uint8` and `PNG/I;16/uint16` in one depth folder visible immediately.

Write non-`uint8` images to a new modality output directory as 8-bit PNG without changing source images, labels, or the cached dataset. Selected `uint8` images are copied unchanged. Supply a fixed range for depth images when brightness must remain comparable between files:

```python
converted = mgr.convert_to_uint8(
    # omit out to use ydm_conversion/uint8/
    modalities=["depth"],
    stretch=True,
    value_range=(0, 20000),  # maps this raw depth range to display values 0–255
    preserve_zero=True,      # keeps invalid depth 0 black
    workers=8,
)
# Output: ydm_conversion/uint8/depth/<original-relative-path>; non-uint8 files become .png
```

Without `value_range`, `stretch=True` applies a per-image min-max stretch over nonzero valid values. It improves detail but makes brightness incomparable across images. `stretch=False` only clips source values to `0–255`, which is usually unsuitable for `uint16` depth data. `overwrite=False` by default prevents replacing an existing output image.

Use `image_params` and `label_params` when filenames carry suffixes. The dictionary key is the logical image type and binds to an image folder with the same name by default. Use `dir` when the type and folder name differ.

```python
mgr = MultiModalYoloManager(
    root,
    image_dirs=["visible", "thermal", "depth_map"],
    image_params={
        "rgb": {"dir": "visible", "suffix": "_V"},
        "infrared": {"dir": "thermal", "suffix": "_T"},
        "depth": {"dir": "depth_map", "suffix": "_D", "required": False},
    },
    labels_dir="labels",
    label_params={"suffix": "_gt"},
    class_file="class.txt",
    task="detect",
)
```

The default is for every image type to be required. `required=False` allows a scene to remain usable when that modality is absent. `annotation_stats` counts each shared label once, whereas `modalities.<type>.stats` contains per-modality image and pixel-level box statistics. Rendered outputs are separated by type, for example `ydm_vis/draw/rgb/` and `ydm_vis/draw/infrared/`; these are modality subdirectories of the same visualization group.

## Functional API

You can call tasks directly with `run_task`.

```python
from pathlib import Path
from yolo_data_manager import run_task

code = run_task(
    "stats",
    root=Path("datasets/my_yolo"),
    layout="auto",
    out="stats.json",
)
```

Task names use module-style identifiers, such as `query.class`, `ann.set_attr`, `vis.draw`, and `eval.error_analysis`.

## Example Functions and Dataset Callers

`example/functions/` contains the secondarily organized reusable functions.
Files directly under `example/` are dataset-specific callers. Copy
`example/dataset_template.py`, rename it for a dataset, set its path, and
select the functions and parameters to run:

```python
from example.functions import yolo_sta, yolo_vis

DATA_DIR = r"/path/to/my_dataset.yaml"

yolo_sta(DATA_DIR, stats_list=["all"], only_val=False)
yolo_vis(DATA_DIR, crop=True, only_val=False)
```

There is no generic `example/datasets/` runner and no need for `run_ydm.py`.
TT100K conversion is an independent repository tool at
`tools/convert_tt100k.py`.

## Parameter Notes

| Python Parameter | CLI Flag | Note |
|---|---|---|
| `class_` | `--class` | Avoids Python keyword conflict |
| `from_` | `--from` | Avoids Python keyword conflict |
| `map_file` | `--map` | Avoids built-in name conflict |
| `json_path` | `--json` | Avoids module name conflict |
| `class_map` | `--class-map` | Used by mask import |

Lists become comma-separated values. `None` values are omitted. Boolean options follow CLI semantics, for example `copy_images=False` becomes `--no-copy-images`.
