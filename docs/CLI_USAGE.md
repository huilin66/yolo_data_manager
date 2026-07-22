# CLI 使用指南

本文档记录 `ydm` 命令行的常用调用方式。README 只保留快速入口，完整命令示例放在这里。

## 安装与运行

```bash
python -m pip install .
ydm --help
```

开发模式：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

不安装时：

```powershell
$env:PYTHONPATH = "src"
python -m yolo_data_manager.cli check --root path/to/yolo
```

## 全局加载参数

大多数读取 YOLO 数据集的命令都支持：

| 参数 | 说明 |
|---|---|
| `--root` | YOLO 数据集根目录 |
| `--layout` | `auto`、`flat`、`split_dirs`、`image_list`、`mixed` |
| `--task` | `auto`、`detect`、`segment` |
| `--images-dir` | 图片目录名，默认 `images` |
| `--labels-dir` | label 目录名，默认 `labels` |
| `--class-file` | 类别文件路径 |
| `--attribute-file` | 属性配置路径 |
| `--split-file` | 图片列表文件路径 |

## 统一运行参数

大多数读取、写入、校验、可视化和评估命令都支持同一组运行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--workers` | `8` | 支持并行的加载、校验、写入、可视化、复核等步骤使用的线程数 |
| `--progress` | 开启 | 显示临时 tqdm 进度条 |
| `--no-progress` | 关闭进度条 | 不显示 tqdm 进度条 |
| `--progress-leave` | `False` | 任务结束后保留进度条 |

默认风格是 `workers=8`、显示 tqdm、`leave=False`。少数纯格式转换命令暂不使用线程，但仍会保持相同的 CLI 风格。

## 加载、布局与校验

```bash
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --task auto
ydm check --root path/to/yolo --layout auto
ydm check --root path/to/yolo --layout flat --fill-missing-txt --out validation.json
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
```

`layout detect` 输出的 `report_type` 是 `layout_detect`，这是布局检测结果，不是 `check` 校验结果。输出中还会包含 `class_source`、`class_count`、`classes`，用于确认类别是从 `class.txt`、`classes.txt`、`dataset.yaml` 还是 `data.yaml` 读取到的。

`check` 完整校验结果会写入 JSON 文件，终端只输出红色 warning/error 摘要或绿色 OK 摘要。`--out` 不指定时默认写到 `<root>/check_result.json`。如确实需要在终端打印完整 JSON，可加 `--print-full`。

`--fill-missing-txt` 会为没有 label 的图片创建空 txt，并在 JSON 中列出创建结果。

## 查询

```bash
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm query class --root path/to/yolo --class person --copy-images out/images --copy-labels out/labels
ydm query class --root path/to/yolo --class person --copy-labels out/labels --filtered-labels
ydm query attr --root path/to/yolo --name defect --value yes --out defect.csv
ydm query attr --root path/to/yolo --name defect --nonzero --copy-labels out/labels
```

## 标注修改

```bash
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --compact
ydm ann delete-class --root path/to/yolo --class ignore --out yolo_clean --compact
ydm ann replace-class --root path/to/yolo --from old_name --to new_name --out yolo_replaced
ydm ann rename-class --root path/to/yolo --from cls_a --to cls_b --out yolo_renamed
ydm ann apply-map --root path/to/yolo --map class_map.yaml --out yolo_mapped
ydm ann set-attr --root path/to/yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root path/to/yolo --name defect --value yes --out yolo_attr_clean
```

写操作默认输出到 `--out`，不原地覆盖原数据。

## 数据集管理

```bash
ydm dataset select --root path/to/yolo --file val.txt --out yolo_val
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 --seed 233
ydm dataset split --root path/to/yolo --train 0.8 --val 0.1 --test 0.1 --absolute-paths
ydm dataset yaml --root path/to/yolo --out dataset.yaml
ydm dataset merge --roots data1,data2 --out merged_yolo
ydm dataset duplicates --root path/to/yolo --out duplicate_images.csv
ydm dataset bad-images --root path/to/yolo --out bad_images.csv
```

split 会打印总类别 box 数量和 val 类别 box 数量，方便检查验证集分布。

## 过滤

全局过滤：

```bash
ydm dataset filter --root path/to/yolo --min-area 0.001 --out yolo_filtered
ydm dataset filter --root path/to/yolo --min-width 0.01 --min-height 0.01 --min-size-logic and --out yolo_filtered
```

`--min-size-logic or` 是默认逻辑：宽或高小于阈值就删除。  
`--min-size-logic and` 表示宽和高都小于阈值才删除。

按类别过滤：

```bash
ydm dataset filter --root path/to/yolo --class-rules filter_rules.yaml --out yolo_filtered
```

`filter_rules.yaml`：

```yaml
person:
  min_width: 0.01
  min_height: 0.01
  min_size_logic: and

car:
  min_area: 0.0005

defect:
  min_width: 0.005
  min_height: 0.005
```

## 统计

```bash
ydm stats --root path/to/yolo --out stats.json
ydm stats --root path/to/yolo --ann-csv annotations.csv --attr-csv attributes.csv --plots-dir stats_plots
ydm stats --root path/to/yolo --plots-dir labels_sta --stats-list all
ydm stats --root path/to/yolo --plots-dir labels_sta --stats-list image_shape,box_shape_pix,box_pos_center
```

`--stats-list` 支持：

```text
all, class_counts, box_number, box_width, box_height, box_area,
image_shape, box_shape, box_shape_pix, box_shape_rate,
box_pos_start, box_pos_center, box_pos_end, attribute, legacy_csv
```

## 可视化与裁剪

```bash
ydm vis draw --root path/to/yolo --out images_vis
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis draw --root path/to/yolo --out images_vis --show-id
ydm vis draw --root path/to/yolo --out images_vis --workers 16
ydm vis draw --root path/to/yolo --out images_vis --no-progress
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm vis crop --root path/to/yolo --out crops --workers 16
```

`--show-id` 显示 txt 中从 1 开始的标注顺序号。crop 文件名也从 1 开始。

## 导入导出

```bash
ydm export coco --root path/to/yolo --out instances.json
ydm export xany --root path/to/yolo --out xany_json

ydm import labelme --json-dir labelme_json --out yolo --task segment
ydm import coco --json instances.json --images-dir images --out yolo --task segment
ydm import voc --annotations-dir Annotations --images-dir JPEGImages --out yolo
```

语义分割 mask 导入：

```bash
ydm import mask --images-dir images --masks-dir masks --out yolo_seg --class-map class_map.yaml --background 0 --min-area 20
```

`class_map.yaml`：

```yaml
0: background
1: crack
2: spalling
```

RGB mask：

```yaml
"#ff0000": crack
"0,255,0": spalling
```

## 转换

```bash
ydm convert seg2det --root yolo_seg --out yolo_det
ydm convert pseudo --root pred_yolo --conf 0.5 --out pseudo_yolo
```

## 评估与错误分析

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --out compare.csv --iou 0.5
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --out review_pack --iou 0.5
ydm eval metrics --gt-root gt_yolo --pred-root pred_yolo --out metrics.json --csv metrics.csv
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --out vehicle_metrics.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --out vehicle_no_small.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --print-table
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --match-iou 0.5 --low-iou 0.1 --duplicate-iou 0.9
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --val-source val.txt --class-file class.txt --out error_report
ydm eval error-analysis --gt-root gt_labels --pred-root pred_labels --names class.txt --out error_report
```

`eval metrics` 计算 Precision、Recall、mAP@0.5、mAP@0.5:0.95，并支持 `--class` 只评估指定类别；未选类别的 GT 和预测都会被忽略。默认不输出、不计入 `Instances=0` 的类别；如需保留这些空 GT 类用于排查误检，可加 `--include-empty-classes`。小目标过滤可使用 `--min-width`、`--min-height`、`--min-area`、`--min-size-logic`，或按像素使用 `--min-pixels`。加 `--print-table` 可输出接近 Ultralytics 的对齐表格，方便人工对比。

`eval error-analysis` 仍兼容旧参数 `--review-workers`、`--review-progress`、`--review-progress-leave`；新脚本建议直接使用统一运行参数。

review 输出：

```text
review/
  pred_gt/
    confusion_matrix.png
    pred_classA_gt_classB/
      images/
      crops/
  pred_txt/
```

crop 文件名格式：

```text
原图名_pred预测txt顺序id_gtGTtxt顺序id.jpg
```

没有对应对象时使用 `none`。
