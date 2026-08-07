# YOLO Data Manager

[中文文档](README_CN.md)

YOLO Data Manager is a Python package and CLI for managing YOLO datasets. A dataset may be single-modal or may contain multiple aligned image modalities sharing one label set; modality is a dataset property, not a separate feature module. All inputs are normalized into one internal model, then handled through the same loading, validation, import/export, dataset-operation, annotation, statistics, visualization, and prediction-analysis workflows.

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
| Visualization | Draw boxes/masks, show confidence/attributes/txt order id, crop objects, temporary manual boxes | `show_id`, `show_conf`, `workers` |
| Import/export | Convert between YOLO and LabelMe/COCO/VOC/masks/x-anylabeling | `class_map`, `background`, `min_area` |
| Evaluation | Compare GT vs predictions, build FP/FN review packs, error analysis, confusion matrix | `match_iou`, `low_iou`, `review_workers` |

`MultiModalYoloManager` provides modality-aware loading, scene alignment, and caching while reusing the same functional output groups as the single-modal manager; it does not add a separate multimodal output module.

`layout detect` output is a layout detection result, not a validation/check result. It includes `report_type`, `class_source`, `class_count`, and `classes`.

## Multimodal Loading and Validation

`MultiModalYoloManager` associates a shared YOLO label folder with multiple image folders. It derives a common scene stem from each filename, optionally removing a per-type suffix: for example, `visible/0001_V.jpg`, `infrared/0001_T.png`, and `labels/0001_gt.txt` associate with scene `0001`.

With empty image and label configuration, matching uses identical filename stems and standard `labels/<stem>.txt` labels. `image_params` and `label_params` configure modality/label suffixes when names differ. `check()` reports missing modalities, orphan images or labels, suffix mismatches, and duplicate scene images. The manager caches the associated dataset, so `stats()`, `vis_draw()`, and `vis_crop()` reuse parsed labels rather than loading once per image folder.

Multimodal support is currently a Python API; use `MultiModalYoloManager` for modality-aware loading. Its first supported operations are `check`, `stats`, `vis_draw`, `vis_crop`, and uint8 conversion. It uses the same output groups as single-modal workflows. Full parameters and examples are in [Python Usage](docs/PYTHON_USAGE_EN.md#multimodal-yolo-datasets).

## Python Quick Demo

```python
from yolo_data_manager import YoloManager

mgr = YoloManager("datasets/my_yolo", layout="auto", init_check=False)
mgr_yaml = YoloManager(r"E:\repository\yolo8\ultralytics\cfg\datasets\data_fire.yaml", layout="auto", init_check=False)

mgr.check(fill_missing_txt=True)
mgr.stats(stats_list=["all"])
mgr.vis_draw(show_id=True, show_conf=True)

mgr.dataset_filter(
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
    review=True,
    workers=8,
    copy_pred_txt=True,
)
```

## CLI Quick Demo

```bash
ydm check --root path/to/yolo --layout auto --fill-missing-txt --out validation.json
ydm stats --root path/to/yolo --stats-list all
ydm vis draw --root path/to/yolo --show-id --show-conf
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5
ydm dataset filter --root path/to/yolo --min-width 0.01 --min-height 0.01 --min-size-logic and
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --show-original --print-table
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --exclude-class ignore --min-width 0.01 --min-height 0.01 --min-size-logic and --min-pixels 8 --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --names class.txt --class-rules error_rules.yaml --out error_report
```

## Output Conventions

- Write operations default to a new output directory and do not overwrite the source dataset in place.
- The CLI and `YoloManager` use common runtime defaults: `workers=8`, temporary tqdm progress bars, and `leave=False`. Tune them with `--workers/--no-progress/--progress-leave` or Python `workers/progress/progress_leave`.
- `check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. Without an output path, the default report is `<root>/ydm_quality/check.json`.
- Default analysis outputs use `ydm_quality/`, `ydm_stats/`, `ydm_vis/`, `ydm_evaluation/`, `ydm_dataset/`, `ydm_annotation/`, and `ydm_conversion/`; `labels_backup/` remains unprefixed.
- `train.txt`, `val.txt`, `test.txt`, and `dataset.yaml` remain at the dataset root. Multimodal workflows add `rgb/`, `depth/`, and similar subdirectories only inside the relevant functional group; there is no `ydm_multimodal/` directory.
- Standard YOLO output includes `images/`, `labels/`, `class.txt`, and `dataset.yaml`.
- Error-analysis review output includes `review/pred_gt`, `confusion_matrix.png`, grouped `pred_<pred_class>_gt_<gt_class>` folders, and optional `review/pred_txt`.
- Review crop names use `image_pred<pred_txt_order>_gt<gt_txt_order>`, with `none` for missing sides.

## Git Ignore Policy

The project `.gitignore` excludes local datasets, generated visualization/statistics outputs, training runs, caches, and common model-weight formats such as `.pt`, `.pth`, `.onnx`, `.engine`, `.safetensors`, and `.weights`.
