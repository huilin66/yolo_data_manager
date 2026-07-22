# YOLO Data Manager

[English README](README.md)

YOLO Data Manager 是一个用于管理 YOLO 数据集的 Python 工具包和命令行工具。它把不同来源的数据先读成统一内部模型，然后在同一套接口上完成加载校验、导入导出、数据集管理、标注查询修改、统计、可视化和预测错误分析。

## 文档入口

- [Python 详细使用](docs/PYTHON_USAGE.md)
- [CLI 详细使用](docs/CLI_USAGE.md)
- [项目交接说明](docs/HANDOFF.md)
- [Python Usage in English](docs/PYTHON_USAGE_EN.md)
- [CLI Usage in English](docs/CLI_USAGE_EN.md)
- [Handoff in English](docs/HANDOFF_EN.md)

## 安装

```bash
python -m pip install .
```

开发和测试：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## 功能地图

| 功能 | 说明 | 常用参数 |
|---|---|---|
| 加载与校验 | 检查缺图、缺 label、孤儿 label、类别越界、坐标异常 | `layout`、`task`、`fill_missing_txt` |
| 布局管理 | 自动识别并标准化不同 YOLO 目录结构 | `images_dir`、`labels_dir`、`split_file` |
| 查询 | 按类别或属性查找图片、label、实例行 | `class_`、`name`、`value`、`copy_images` |
| 标注修改 | 删除、替换、合并、重命名类别，设置或删除属性 | `compact`、`dry_run`、`report` |
| 数据集管理 | select、split、merge、filter、yaml、重复图、坏图检测 | `train`、`val`、`absolute_paths`、`class_rules` |
| 统计 | 类别分布、目标数、框宽高面积、图片尺寸、属性统计、图表 | `stats_list`、`plots_dir`、`ann_csv` |
| 可视化 | 画框、画 mask、显示 confidence/属性/txt 顺序号、裁剪目标 | `show_id`、`show_conf`、`workers` |
| 导入导出 | 在 YOLO 与 LabelMe/COCO/VOC/mask/x-anylabeling 之间转换 | `class_map`、`background`、`min_area` |
| 评估分析 | GT vs pred 对比、FP/FN review、细粒度错误分析、混淆矩阵 | `match_iou`、`low_iou`、`review_workers` |

`layout detect` 输出是布局检测结果，不是 `check` 校验结果；结果中会包含 `report_type`、`class_source`、`class_count`、`classes`。

## Python 快速 Demo

```python
from yolo_data_manager import YoloManager

mgr = YoloManager(r"E:\datasets\my_yolo", layout="auto", init_check=False)
mgr_yaml = YoloManager(r"E:\repository\yolo8\ultralytics\cfg\datasets\data_fire.yaml", layout="auto", init_check=False)

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
    pred_root=r"E:\datasets\pred_labels",
    out="error_report",
    review=True,
    workers=8,
    copy_pred_txt=True,
)
```

## CLI 快速 Demo

```bash
ydm check --root path/to/yolo --layout auto --fill-missing-txt --out validation.json
ydm stats --root path/to/yolo --plots-dir stats --stats-list all
ydm vis draw --root path/to/yolo --out vis --show-id --show-conf
ydm dataset filter --root path/to/yolo --out filtered --min-width 0.01 --min-height 0.01 --min-size-logic and
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --out metrics.json --csv metrics.csv --print-table
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --out error_report --review --workers 8 --copy-pred-txt
```

## 输出约定

- 写操作默认输出到新目录，不覆盖原数据。
- CLI 和 `YoloManager` 默认使用统一运行参数：`workers=8`、显示临时 tqdm、`leave=False`。可用 `--workers/--no-progress/--progress-leave` 或 Python 的 `workers/progress/progress_leave` 调整。
- `check` 完整校验结果写入 JSON，终端只输出红色 warning/error 摘要或绿色 OK 摘要。不指定输出路径时默认写到 `<root>/check_result.json`。
- 标准 YOLO 输出包含 `images/`、`labels/`、`class.txt`、`dataset.yaml`。
- error analysis 的 review 输出包含 `pred_gt/`、`confusion_matrix.png`、按 `pred_<预测类别>_gt_<真实类别>` 组织的图片和 crop。
- review crop 文件名使用 `原图名_pred预测txt顺序id_gtGTtxt顺序id`，没有的一侧为 `none`。

## Git Ignore 策略

项目 `.gitignore` 默认忽略本地数据、生成的可视化/统计输出、训练 run、缓存，以及常见模型权重格式：`.pt`、`.pth`、`.onnx`、`.engine`、`.safetensors`、`.weights` 等。
