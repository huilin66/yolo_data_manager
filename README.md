# YOLO Data Manager

[中文文档](README_CN.md)

YOLO Data Manager is a Python package and CLI for managing YOLO datasets. It normalizes different dataset sources into one internal model, then provides loading, validation, import/export, dataset operations, annotation query/edit, statistics, visualization, and prediction error analysis on top of that model.

## Documentation

- [Python Usage](docs/PYTHON_USAGE_EN.md)
- [CLI Usage](docs/CLI_USAGE_EN.md)
- [Project Handoff](docs/HANDOFF_EN.md)
- [中文 Python 使用](docs/PYTHON_USAGE.md)
- [中文 CLI 使用](docs/CLI_USAGE.md)
- [中文交接说明](docs/HANDOFF.md)

## Installation

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Feature Map

| Area | What It Does | Common Parameters |
|---|---|---|
| Load and validate | Check missing images/labels, orphan labels, invalid classes, invalid geometry | `layout`, `task`, `fill_missing_txt` |
| Layout management | Detect and normalize YOLO dataset layouts | `images_dir`, `labels_dir`, `split_file` |
| Query | Find images, labels, and instances by class or attribute | `class_`, `name`, `value`, `copy_images` |
| Annotation edits | Delete, replace, merge, rename classes; set/delete attributes | `compact`, `dry_run`, `report` |
| Dataset operations | select, split, merge, filter, yaml, duplicate/bad-image checks | `train`, `val`, `absolute_paths`, `class_rules` |
| Statistics | Class distribution, object counts, box shapes, image shapes, attributes, plots | `stats_list`, `plots_dir`, `ann_csv` |
| Visualization | Draw boxes/masks, show confidence/attributes/txt order id, crop objects | `show_id`, `show_conf`, `workers` |
| Import/export | Convert between YOLO and LabelMe/COCO/VOC/masks/x-anylabeling | `class_map`, `background`, `min_area` |
| Evaluation | Compare GT vs predictions, build FP/FN review packs, error analysis, confusion matrix | `match_iou`, `low_iou`, `review_workers` |

`layout detect` output is a layout detection result, not a validation/check result. It includes `report_type`, `class_source`, `class_count`, and `classes`.

## Python Quick Demo

```python
from yolo_data_manager import YoloManager

mgr = YoloManager("datasets/my_yolo", layout="auto", init_check=False)

mgr.check(out="validation.json", fill_missing_txt=True)
mgr.stats(plots_dir="stats", stats_list=["all"])
mgr.vis_draw(out="vis", show_id=True, show_conf=True)

mgr.dataset_filter(
    out="filtered",
    min_width=0.01,
    min_height=0.01,
    min_size_logic="and",
    class_rules={
        "person": {"min_width": 0.01, "min_height": 0.01},
        "car": {"min_area": 0.0005},
    },
)

mgr.eval_error_analysis(
    pred_root="datasets/pred_labels",
    out="error_report",
    review=True,
    workers=8,
    copy_pred_txt=True,
)
```

## CLI Quick Demo

```bash
ydm check --root path/to/yolo --layout auto --fill-missing-txt --out validation.json
ydm stats --root path/to/yolo --plots-dir stats --stats-list all
ydm vis draw --root path/to/yolo --out vis --show-id --show-conf
ydm dataset filter --root path/to/yolo --out filtered --min-width 0.01 --min-height 0.01 --min-size-logic and
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --out metrics.json --csv metrics.csv
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --out error_report --review --workers 8 --copy-pred-txt
```

## Output Conventions

- Write operations default to a new output directory and do not overwrite the source dataset in place.
- The CLI and `YoloManager` use common runtime defaults: `workers=8`, temporary tqdm progress bars, and `leave=False`. Tune them with `--workers/--no-progress/--progress-leave` or Python `workers/progress/progress_leave`.
- `check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. Without an output path, the default report is `<root>/check_result.json`.
- Standard YOLO output includes `images/`, `labels/`, `class.txt`, and `dataset.yaml`.
- Error-analysis review output includes `review/pred_gt`, `confusion_matrix.png`, grouped `pred_<pred_class>_gt_<gt_class>` folders, and optional `review/pred_txt`.
- Review crop names use `image_pred<pred_txt_order>_gt<gt_txt_order>`, with `none` for missing sides.

## Git Ignore Policy

The project `.gitignore` excludes local datasets, generated visualization/statistics outputs, training runs, caches, and common model-weight formats such as `.pt`, `.pth`, `.onnx`, `.engine`, `.safetensors`, and `.weights`.
