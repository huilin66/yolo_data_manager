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

```python
from yolo_data_manager import YoloManager

mgr = YoloManager("datasets/my_yolo", layout="auto", init_check=False)

mgr.check(out="validation.json", fill_missing_txt=True)
mgr.layout_detect()
```

`layout_detect()` prints a layout detection result, not a validation/check result. The output has `report_type: layout_detect` and includes `class_source`, `class_count`, and `classes`.

## Statistics

```python
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

## Query

`class_` is used because `class` is a Python keyword.

```python
mgr.query_class(class_=["car", "truck"], out="vehicles.csv")
mgr.query_class(class_=["person"], copy_images="persons/images", copy_labels="persons/labels", filtered_labels=True)
mgr.query_attr(name="occluded", value=["yes"], out="occluded.csv")
mgr.query_attr(name="quality", nonzero=True)
```

## Dataset Operations

```python
mgr.dataset_normalize(out="normalized_yolo")
mgr.dataset_split(train=0.8, val=0.1, test=0.1, seed=233)
mgr.dataset_split(train=0.8, val=0.1, test=0.1, absolute_paths=True)
mgr.dataset_select(file="val.txt", out="val_subset")
mgr.dataset_yaml(out="dataset.yaml", train="images/train", val="images/val")
mgr.dataset_duplicates(out="duplicates.csv")
mgr.dataset_bad_images(out="bad_images.csv")
```

Filtering:

```python
mgr.dataset_filter(out="filtered", min_area=0.001, class_=["car", "truck"])
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

Merging:

```python
mgr.dataset_merge(roots=["datasets/part1", "datasets/part2"], out="merged_yolo", source_prefix=True)
```

## Annotation Edits

```python
mgr.ann_merge_class(from_=["crack", "break"], to="defect", out="yolo_merged", compact=True)
mgr.ann_merge_class({"vehicle": ["car", "truck"], "human": ["person"]}, out="yolo_merged_multi")
mgr.ann_delete_class(class_=["ignore"], out="yolo_clean", compact=True)
mgr.ann_replace_class(from_=["old_name"], to="new_name", out="yolo_replaced")
mgr.ann_rename_class(from_="cls_a", to="cls_b", out="yolo_renamed")
mgr.ann_apply_map(map_file="class_map.yaml", out="yolo_mapped")
mgr.ann_set_attr(name="defect", value="yes", class_=["sign"], out="yolo_attr")
mgr.ann_delete_attr(name="quality", value=["bad"], out="yolo_clean")
```

Write operations output to a new directory. Use `dry_run=True` when you want to inspect the effect first.

## Visualization

```python
mgr.vis_draw(out="images_vis", show_conf=True, show_attrs=True)
mgr.vis_draw(out="images_vis", conf=0.25, fill_mask=True, mask_alpha=64)
mgr.vis_draw(out="images_vis", show_id=True)
mgr.vis_draw(out="images_vis", workers=8, progress=True)
mgr.vis_crop(out="crops", by_attr=True, min_size=32)
mgr.vis_crop(out="crops", workers=8, progress=True)
```

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
```

## Evaluation and Error Analysis

```python
mgr.eval_compare(gt_root="datasets/gt", pred_root="datasets/pred", out="compare.csv", iou=0.5)
mgr.eval_review_pack(gt_root="datasets/gt", pred_root="datasets/pred", out="review_pack", status=["fp", "fn"])

mgr.eval_error_analysis(pred_root="datasets/pred_labels", out="error_report")
mgr.eval_error_analysis(
    pred_root="datasets/pred_labels",
    out="error_report",
    match_iou=0.5,
    low_iou=0.1,
    conf_thres=0.25,
    duplicate_iou=0.9,
    review=True,
    review_workers=8,
    review_progress=True,
    review_progress_leave=False,
    copy_pred_txt=True,
)
```

When `gt_root`, `val_source`, or `class_file` are omitted, `YoloManager` falls back to the manager root, `val.txt`, and `class.txt` when available.

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

## Parameter Notes

| Python Parameter | CLI Flag | Note |
|---|---|---|
| `class_` | `--class` | Avoids Python keyword conflict |
| `from_` | `--from` | Avoids Python keyword conflict |
| `map_file` | `--map` | Avoids built-in name conflict |
| `json_path` | `--json` | Avoids module name conflict |
| `class_map` | `--class-map` | Used by mask import |

Lists become comma-separated values. `None` values are omitted. Boolean options follow CLI semantics, for example `copy_images=False` becomes `--no-copy-images`.
