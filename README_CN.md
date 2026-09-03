# YOLO Data Manager

[English README](README.md)

YOLO Data Manager 是一个用于管理 YOLO 数据集的 Python 工具包和命令行工具。数据集可以是单模态，也可以是共享 label、多个对齐图像模态；模态是数据集属性，不是独立的功能模块。所有数据先读成统一内部模型，再用同一套接口完成加载校验、导入导出、数据集管理、标注查询修改、统计、可视化和预测错误分析。

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
| 数据集管理 | select、split、merge、filter、resize、yaml、重复图、坏图检测 | `train`、`val`、`absolute_paths`、`class_rules` |
| 统计 | 类别分布、目标数、框宽高面积、图片尺寸、属性统计、图表 | `stats_list`、`plots_dir`、`ann_csv` |
| 可视化 | 画框、画 mask、显示 confidence/属性/txt 顺序号、裁剪目标、临时手动画框 | `show_id`、`show_conf`、`workers` |
| 导入导出 | 在 YOLO 与 LabelMe/COCO/VOC/mask/x-anylabeling 之间转换 | `class_map`、`background`、`min_area` |
| 评估分析 | GT vs pred 对比、FP/FN review、类别/属性错误分析、混淆矩阵 | `match_iou`、`low_iou`、`attribute_file`、`review_workers` |

多模态通过 `MultiModalYoloManager` 提供模态感知的加载、scene 对齐和缓存；它复用统计、校验、可视化和转换的同一套功能目录，不增加独立的多模态输出模块。

`layout detect` 输出是布局检测结果，不是 `check` 校验结果；结果中会包含 `report_type`、`class_source`、`class_count`、`classes`。

## 多模态加载与校验

`MultiModalYoloManager` 用于将一份共享 YOLO label 目录关联到多个图像目录。它从每个文件名得到共同的场景 stem，并可去除每个 type 的 suffix：例如 `visible/0001_V.jpg`、`infrared/0001_T.png` 与 `labels/0001_gt.txt` 会关联为场景 `0001`。

当图像和 label 配置为空时，按相同文件 stem 关联，标签默认使用 `labels/<stem>.txt`。图像名或标签名带后缀时，可用 `image_params`、`label_params` 配置。`check()` 会报告缺失模态、孤儿图像或 label、suffix 不匹配和重复 scene 图像。manager 会缓存关联结果，因此连续调用 `stats()`、`vis_draw()`、`vis_crop()` 不会针对每个图像目录重复读取和解析 label。

多模态当前为 Python API，入口是 `MultiModalYoloManager`；首期支持 `check`、`stats`、`vis_draw`、`vis_crop` 和 uint8 转换。它使用与单模态相同的输出分组；完整参数与示例见 [Python 详细使用](docs/PYTHON_USAGE.md#多模态-yolo-数据集)。

## Python 快速 Demo

```python
from yolo_data_manager import YoloManager

mgr = YoloManager(r"E:\datasets\my_yolo", layout="auto", init_check=False)
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
    pred_root=r"E:\datasets\pred_labels",
    review=True,
    workers=8,
    copy_pred_txt=True,
)
# 有属性 schema 时会额外写出 error_report/attribute_error.csv；
# review=True 还会生成 review/attribute_error/attribute_<name>/...
```

## 示例代码组织

`example/functions/` 保存二次整理后的可复用函数；`example/` 根目录下的文件则对应具体数据集，直接填写路径、选择参数并调用这些函数。复制 `example/dataset_template.py`，按数据集改名即可：

```python
from example.functions import yolo_sta, yolo_vis

DATA_DIR = r"/path/to/my_dataset.yaml"

yolo_sta(DATA_DIR, stats_list=["all"], only_val=False)
yolo_vis(DATA_DIR, crop=True, only_val=False)
```

不再保留通用 `example/datasets/` 调用器，也不再需要 `run_ydm.py`。TT100K 转换属于独立工具，入口为 `tools/convert_tt100k.py`。

## CLI 快速 Demo

```bash
ydm check --root path/to/yolo --layout auto --fill-missing-txt --out validation.json
ydm stats --root path/to/yolo --stats-list all
ydm vis draw --root path/to/yolo --show-id --show-conf
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5
ydm dataset filter --root path/to/yolo --min-width 0.01 --min-height 0.01 --min-size-logic and
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --show-original --print-table
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_labels --review --workers 8 --copy-pred-txt
```

## 输出约定

- 写操作默认输出到新目录，不覆盖原数据。
- CLI 和 `YoloManager` 默认使用统一运行参数：`workers=8`、显示临时 tqdm、`leave=False`。可用 `--workers/--no-progress/--progress-leave` 或 Python 的 `workers/progress/progress_leave` 调整。
- `check` 完整校验结果写入 JSON，终端只输出红色 warning/error 摘要或绿色 OK 摘要。不指定输出路径时默认写到 `<root>/ydm_quality/check.json`。
- 默认分析输出使用统一的 `ydm_quality/`、`ydm_stats/`、`ydm_vis/`、`ydm_evaluation/`、`ydm_dataset/`、`ydm_annotation/`、`ydm_conversion/` 目录；`labels_backup/` 保持不带前缀。
- `train.txt`、`val.txt`、`test.txt` 和 `dataset.yaml` 默认保留在数据集根目录。多模态只在同一功能目录下按需增加 `rgb/`、`depth/` 等子目录，不创建 `ydm_multimodal/`。
- 标准 YOLO 输出包含 `images/`、`labels/`、`class.txt`、`dataset.yaml`。
- error analysis 的 review 输出包含 `pred_gt/`、`confusion_matrix.png`、按 `pred_<预测类别>_gt_<真实类别>` 组织的图片和 crop。
- 有属性 schema 时，error analysis 还会写出 `attribute_error.csv`；`review=True` 会在 `review/attribute_error/attribute_<属性名>/gt_<GT值>_pred_<预测值>/` 下保存匹配图片和 crop。
- review crop 文件名使用 `原图名_pred预测txt顺序id_gtGTtxt顺序id`，没有的一侧为 `none`。

## Git Ignore 策略

项目 `.gitignore` 默认忽略本地数据、生成的可视化/统计输出、训练 run、缓存，以及常见模型权重格式：`.pt`、`.pth`、`.onnx`、`.engine`、`.safetensors`、`.weights` 等。
