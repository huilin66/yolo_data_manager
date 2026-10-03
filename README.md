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

## Standalone Web UI

YDM includes an independent local web workspace whose visual language follows
MultiAnno without coupling the two applications at runtime. It loads a YOLO
dataset root or `data.yaml`, then shows image thumbnails, annotated previews,
class counts, and train/val/test split summaries.

Install the optional web dependencies and start it:

The web frontend requires Node.js and npm; npm is normally installed with Node.js.
Use Node.js 20.19+ (or 22.12+). `ydm web` starts both the frontend development
server and the YDM API. If you only use `--api-only`, Node.js/npm is not required.
The UI supports English and Chinese; use the language button in the top-right
corner to switch, and the choice is remembered by the current browser.
If the frontend reports that `react-i18next` or `i18next` cannot be resolved,
run `npm install` again in the frontend directory and restart `ydm web`.
The top-right VLM assistant icon is always available in the workspace. Clicking
it opens a collapsible right-side panel; dataset details remain at the bottom
of the central workspace. After a dataset is loaded, it can turn natural-language
requests into statistics, validation, query, visualization, and annotation-
operation plans. Operations that can modify dataset files are shown for manual
confirmation before execution. VLM settings prefer the project `.env` file;
the web endpoints are `GET /api/vlm/status` and `POST /api/vlm/assistant`.

```bash
python -m pip install -e ".[web]"
cd src/yolo_data_manager/web/frontend
npm install
cd ../../../..
ydm web --open
```

The default frontend URL is `http://127.0.0.1:5174`; the API listens on
`http://127.0.0.1:8091`. To start only the API, use `ydm web --api-only`.
MultiAnno can later add a launcher entry without sharing YDM's data model or
runtime process.

## Feature Map

| Area | What It Does | Common Parameters |
|---|---|---|
| Load and validate | Check missing images/labels, orphan labels, invalid classes, invalid geometry | `layout`, `task`, `fill_missing_txt` |
| Layout management | Detect and normalize YOLO dataset layouts | `images_dir`, `labels_dir`, `split_file` |
| Query | Find images, labels, and instances by class or attribute | `class_`, `name`, `value`, `copy_images` |
| Annotation edits | Delete, replace, merge, rename classes; set/delete attributes | `compact`, `dry_run`, `report` |
| Dataset operations | select, split, merge, filter, resize, filename-remap, yaml, duplicate/bad-image checks | `train`, `val`, `ensure_class_presence`, `absolute_paths`, `class_rules` |
| Statistics | Class distribution, object counts, box shapes, image shapes, attributes, compact split tables, plots | `stats_list`, `plots_dir`, `ann_csv`, `basic_info_csv` |
| Visualization | Draw boxes/masks, show confidence/attributes/txt order id, crop objects, attribute-separated image groups, temporary manual boxes | `show_id`, `show_conf`, `att_seperate`, `workers` |
| Import/export | Convert between YOLO and LabelMe/COCO/VOC/masks/x-anylabeling | `class_map`, `background`, `min_area` |
| Evaluation | Compare GT vs predictions, build FP/FN review packs, class/attribute error analysis, confusion matrix | `match_iou`, `low_iou`, `attribute_file`, `review_workers` |

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
mgr.vis_draw(show_id=True, show_conf=True, style="cv2")
mgr.vis_draw(show_attrs=True, filter_level=[1], att_seperate=True)
mgr.vis_crop(out="crops", filter_level=[1], att_seperate=True)  # crops are copied into crop_att/attribute/value

mgr.anno_update_by_size(
    min_width=0.01,
    min_height=0.01,
    min_size_logic="and",
    class_rules={
        "person": {"min_width": 0.01, "min_height": 0.01},
        "car": {"min_area": 0.0005},
    },
)
# Omitting out updates source labels in place and backs them up under labels_backup/.

mgr.eval_error_analysis(
    pred_root="datasets/pred_labels",
    review=True,
    workers=8,
    copy_pred_txt=True,
)
# Attribute mismatches are written to error_report/attribute_error.csv;
# review=True also creates review/attribute_error/attribute_<name>/...
```

## Example Organization

Files directly under `example/` are dataset-specific callers: copy
`example/dataset_template.py`, rename it for the dataset, set its path, and
select the manager methods and parameters to run:

```python
from yolo_data_manager import YoloManager

DATA_DIR = r"/path/to/my_dataset.yaml"

manager = YoloManager(DATA_DIR, layout="auto", init_check=False)
manager.stats(stats_list=["all"], only_val=False)
manager.vis_draw(only_val=False, style="cv2")
manager.vis_crop(only_val=False, style="cv2")
```

There is no generic `example/datasets/` runner and no need for `run_ydm.py`.
TT100K conversion is an independent repository tool at
`tools/convert_tt100k.py`.

## CLI Quick Demo

```bash
ydm check --root path/to/yolo --layout auto --fill-missing-txt --out validation.json
ydm stats --root path/to/yolo --stats-list all
ydm vis draw --root path/to/yolo --show-id --show-conf
ydm vis draw --root path/to/yolo --style cv2
ydm vis draw --root path/to/yolo --show-attrs --filter-level 1 --att-seperate
ydm vis crop --root path/to/yolo --style cv2 --workers 16
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5
ydm dataset filter --root path/to/yolo --min-width 0.01 --min-height 0.01 --min-size-logic and
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --show-original --print-table
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --exclude-class ignore --min-width 0.01 --min-height 0.01 --min-size-logic and --min-pixels 8 --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --names class.txt --class-rules error_rules.yaml --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --review --conf-curve
ydm eval error-analysis-common --error-dir error_report_model_a --error-dir error_report_model_b --out common_error_report --iou 0.5 --workers 8
```

`eval error-analysis-common` 直接读取多个 `eval_error_analysis` 输出目录，不重新运行模型。默认提取每次都出现的 `fn_no_pred` GT，以及每次都出现、类别相同且预测框 IoU 达到阈值的 `background_fp` / `class_error_pred` 预测。报告读取、按图像匹配、crop 定位和复制支持 `--workers` 并行，最终报告保持稳定顺序。结果写入 `common_error_summary.csv`（各运行原始、公共、剩余数量）、公共明细 CSV 和 `common_error_summary.json`；如果输入分析结果包含 review crop，还会生成可直接交给 `ann_correct_from_error_crops` 的 `common_crops/`。

`eval error-analysis --conf-curve` 会按 `0.1、0.2、0.3、0.4、0.5` 重新统计 review 分组文件数量，生成 `conf_curve.csv`、`conf_curve_summary.csv` 和 `conf_curve.png`。每条曲线对应一个 `pred_<预测类别>_gt_<真实类别>` 或属性错误目录。

## Output Conventions

- Write operations default to a new output directory and do not overwrite the source dataset in place.
- The CLI and `YoloManager` use common runtime defaults: `workers=8`, temporary tqdm progress bars, and `leave=False`. Tune them with `--workers/--no-progress/--progress-leave` or Python `workers/progress/progress_leave`.
- `check` writes the full validation report to JSON, while the terminal prints only a red warning/error summary or a green OK summary. Without an output path, the default report is `<root>/ydm_quality/check.json`.
- Default analysis outputs use `ydm_quality/`, `ydm_stats/`, `ydm_vis/`, `ydm_evaluation/`, `ydm_dataset/`, `ydm_annotation/`, and `ydm_conversion/`; `labels_backup/` remains unprefixed.
- CLI and YoloManager operations write timestamped daily logs to `ydm_log/YYYY-MM-DD.log` under the dataset root; console operation messages include local timestamps.
- `vis draw` and `vis crop` clear their output directory before running by default (pass `clean=False` in Python or `--no-clean` in the CLI to keep existing outputs).
- `train.txt`, `val.txt`, `test.txt`, and `dataset.yaml` remain at the dataset root. Multimodal workflows add `rgb/`, `depth/`, and similar subdirectories only inside the relevant functional group; there is no `ydm_multimodal/` directory.
- Standard YOLO output includes `images/`, `labels/`, `class.txt`, and `dataset.yaml`.
- Error-analysis review output includes `review/pred_gt`, `confusion_matrix.png`, grouped `pred_<pred_class>_gt_<gt_class>` folders, and optional `review/pred_txt`.
- When an attribute schema is available, error analysis also writes `attribute_error.csv`; `review=True` adds `review/attribute_error/attribute_<name>/gt_<gt_value>_pred_<pred_value>/` with the matched image and crop.
- Review crop names use `image_pred<pred_txt_order>_gt<gt_txt_order>`, with `none` for missing sides.

## VLM

VLM settings are loaded from .env in the project root or current working directory. The built-in qwen provider supports DashScope and local OpenAI-compatible endpoints:

    VLM_PROVIDER=qwen
    VLM_BASE_URL=http://127.0.0.1:18001/v1
    VLM_MODEL=qwen3-vl-30b
    VLM_API_KEY=your-key
    VLM_TIMEOUT=120
    VLM_WORKERS=4

Generate YOLO annotations:

    ydm vlm auto-label --root yolo_data --out yolo_vlm

Verify eval_error_analysis crops and write correction_plan.json:

    ydm vlm verify-errors --root gt_yolo --error-dir gt_yolo/ydm_evaluation/error_analysis

The natural-language assistant only creates a plan by default. Add --execute --yes to permit a write operation:

    ydm vlm assistant --root yolo_data --intent "show class statistics"

VLM outputs default to ydm_vlm/. The generated class_crop_map and attribute_crop_map can be passed directly to the existing crop-correction APIs; applied edits still use the labels_backup policy.

## Git Ignore Policy

The project `.gitignore` excludes local datasets, generated visualization/statistics outputs, training runs, caches, and common model-weight formats such as `.pt`, `.pth`, `.onnx`, `.engine`, `.safetensors`, and `.weights`.
