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
| `--only-val` | 仅处理验证集；默认处理全部数据 |

## 统一运行参数

大多数读取、写入、校验、可视化和评估命令都支持同一组运行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--workers` | `8` | 支持并行的加载、校验、写入、可视化、复核等步骤使用的线程数 |
| `--progress` | 开启 | 显示临时 tqdm 进度条 |
| `--no-progress` | 关闭进度条 | 不显示 tqdm 进度条 |
| `--progress-leave` | `False` | 任务结束后保留进度条 |

默认风格是 `workers=8`、显示 tqdm、`leave=False`。少数纯格式转换命令暂不使用线程，但仍会保持相同的 CLI 风格。

## 默认输出路径

除特别说明外，显式传入的 `--out`、`--csv`、`--plots-dir` 优先；省略时使用以下约定：

```text
<dataset_root>/
  labels_backup/                         # label 写入前的时间戳备份
  ydm_quality/                           # check、query、duplicates、bad-images
  ydm_stats/                             # stats.json、CSV、plots/
  ydm_vis/                               # draw/、crop/、manual_box/
  ydm_evaluation/                        # compare、review_pack、error_analysis、metrics
  ydm_dataset/                           # select、normalize、filter、merge
  ydm_annotation/                        # 各类标注编辑输出和 report
  ydm_conversion/                        # coco、xanylabeling、import、seg2det、pseudo
  train.txt / val.txt / test.txt         # split 仍写在数据集根目录
  dataset.yaml                            # 默认仍写在数据集根目录
```

`check` 默认写入 `ydm_quality/check.json`；`stats` 默认同时写入
`ydm_stats/stats.json`、类别/标注/属性 CSV 和 `ydm_stats/plots/`。多模态数据不创建独立的
`ydm_multimodal` 功能目录，而是在相同的 `ydm_quality`、`ydm_stats`、`ydm_vis`、
`ydm_conversion` 目录中按模态建立必要的子目录。

## 加载、布局与校验

```bash
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --task auto
ydm check --root path/to/yolo --layout auto
ydm check --root path/to/yolo --layout flat --fill-missing-txt --out validation.json
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
```

`layout detect` 输出的 `report_type` 是 `layout_detect`，这是布局检测结果，不是 `check` 校验结果。输出中还会包含 `class_source`、`class_count`、`classes`，用于确认类别是从 `class.txt`、`classes.txt`、`dataset.yaml` 还是 `data.yaml` 读取到的。

`check` 完整校验结果会写入 JSON 文件，终端只输出红色 warning/error 摘要或绿色 OK 摘要。`--out` 不指定时默认写到 `<root>/ydm_quality/check.json`。如确实需要在终端打印完整 JSON，可加 `--print-full`。

`--fill-missing-txt` 会为没有 label 的图片创建空 txt，并在 JSON 中列出创建结果。

## 查询

```bash
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm query class --root path/to/yolo --class person --copy-images out/images --copy-labels out/labels
ydm query class --root path/to/yolo --class person --copy-labels out/labels --filtered-labels
ydm query class --root gt_yolo --source pred --pred-root pred_yolo --class car --class-file gt_yolo/class.txt --out pred_car.csv
ydm query attr --root path/to/yolo --name defect --value yes --out defect.csv
ydm query attr --root path/to/yolo --name defect --nonzero --copy-labels out/labels
```

`query class` 默认查询 `--root` 中的 GT。查询预测结果时使用 `--source pred --pred-root ...`；`--pred-root` 可以是完整 YOLO 预测目录，也可以直接是 `labels` 目录。预测目录没有类别文件时，可使用 GT 的 `--class-file`。终端 JSON 会同时输出匹配的 `image_files` 和 `label_files`，CSV 中每条记录对应一个匹配标注。

## 标注修改

```bash
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --compact
ydm ann delete-class --root path/to/yolo --class ignore --out yolo_clean --compact
ydm ann replace-class --root path/to/yolo --from old_name --to new_name --out yolo_replaced
ydm ann rename-class --root path/to/yolo --from cls_a --to cls_b --out yolo_renamed
ydm ann apply-map --root path/to/yolo --map class_map.yaml --out yolo_mapped
ydm ann set-attr --root path/to/yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root path/to/yolo --name defect --value yes --out yolo_attr_clean
ydm dataset filter --root path/to/yolo --min-area 0.001 --out yolo_filtered --backup-dir label_backups
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out yolo_merged --backup-dir label_backups
ydm ann correct-from-crops --root path/to/yolo --crops-dir ydm_vis/crop/car --to defect --backup-dir label_backups --report crop_correction.csv
ydm ann correct-from-error-crops --root path/to/yolo --crops-dir result_ana/val-52/review/pred_gt/pred_car_gt_background/crops --pred-dir result_ana/val-52/review/pred_txt --dedup-iou 0.5 --to defect --delete-pred-none --backup-dir label_backups --only-val --report gt_correction.csv
```

写操作省略 `--out` 时默认输出到对应的 `ydm_dataset` 或 `ydm_annotation` 子目录，不原地覆盖原数据。
`correct-from-crops` 是按 crop 文件名直接修改源数据对应 label 的例外；建议先使用 `--dry-run`，或保留 `--report` 作为修改记录。`vis crop` 文件名 `<image_stem>_<序号>.<扩展名>` 中的序号从 1 开始。`--to none` 或 `--to null` 会删除对应标注。
`correct-from-error-crops` 使用 `xxx_predx_gty` 文件名中的 `y` 定位 GT 标注序号。提供 `--pred-dir` 后，`gt none` 的 crop 会使用预测 txt 中第 `x` 条记录追加到对应 GT label；追加时会去掉 prediction confidence。未提供 `--pred-dir` 时，`gt none` crop 会跳过。
追加预测以及 `--replace-gt-from-pred` 产生的替换框，默认按同一类别、同一图片的 IoU `0.5` 去重，重叠候选保留置信度更高的预测；替换框被去重时，对应的重复 GT 也会删除。可用 `--dedup-iou` 调整阈值。
指定 `--delete-pred-none` 后，`prednone_gty` 会删除对应的第 `y` 条 GT 标注，即使 `--to` 设置了目标类别。只处理删除时可使用 `--to none --delete-pred-none`；`predx_gty` 仍按 `--to` 执行类别更新或删除。指定 `--replace-gt-from-pred` 后，需要同时提供 `--pred-dir`，`predx_gty` 会用预测第 `x` 条记录完整替换 GT 第 `y` 条（类别和 geometry），`prednone_gty` 删除，`predx_gtnone` 追加。
会写出 GT label txt 的命令（包括 `dataset filter`、`dataset merge` 和 `ann` 编辑命令）都支持 `--backup-dir`：写出前先备份当前输入 label。未指定时默认使用 `<数据集根目录>/labels_backup`，指定后可覆盖默认路径。crop 校正只备份实际修改的 txt。每次运行会在备份目录下创建 `YYYYMMDD_HHMMSS_microseconds` 时间戳子目录，并保留相对于数据集根目录的路径；同一 txt 在一次运行中只备份一次。`--dry-run` 不会创建备份。

## 数据集管理

```bash
ydm dataset select --root path/to/yolo --file val.txt --out yolo_val
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 --seed 233
ydm dataset split --root path/to/yolo --train 0.8 --val 0.1 --test 0.1 --absolute-paths
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 \
  --train-include-list train_include.txt --val-include-list val_include.txt
ydm dataset yaml --root path/to/yolo --out dataset.yaml
ydm dataset merge --roots data1,data2 --out merged_yolo
ydm dataset duplicates --root path/to/yolo --out duplicate_images.csv
ydm dataset bad-images --root path/to/yolo --out bad_images.csv
```

split 会打印总类别 box 数量和 val 类别 box 数量，方便检查验证集分布。
`--train-include-list` 和 `--val-include-list` 可以传 txt 文件，也可以传逗号分隔的图片名/路径。指定的图片会先从随机池中排除，再强制加入对应 split；两个参数不能包含同一张图片。
如果输出目录中已存在 `train.txt`、`val.txt` 或 `test.txt`，写入前会将其移动到 `<数据集根目录>/labels_backup/<时间戳>/`；可通过 `--backup-dir` 指定其他备份目录。

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
ydm stats --root path/to/yolo
ydm stats --root path/to/yolo --out stats.json --ann-csv annotations.csv --attr-csv attributes.csv --plots-dir stats_plots
ydm stats --root path/to/yolo --stats-list all
ydm stats --root path/to/yolo --plots-dir labels_sta --stats-list image_shape,box_shape_pix,box_pos_center
```

`--stats-list` 支持：

```text
all, class_counts, box_number, box_width, box_height, box_area,
image_shape, box_shape, box_shape_pix, box_shape_rate,
box_pos_start, box_pos_center, box_pos_end, attribute, legacy_csv
```

选择 `box_shape`、`box_shape_pix`、`box_shape_rate`、`box_width`、`box_height` 时，还会按类别生成 `box_shape_ratios/`、`box_shape_pixels/`、`aspect_ratio/`、`width_image_ratio/`、`height_image_ratio/` 五个目录；`box_width` 和 `box_height` 还会生成按类别比较的 `box_width_boxplot.png` 和 `box_height_boxplot.png`。

## 可视化与裁剪

```bash
ydm vis draw --root path/to/yolo
ydm vis crop --root path/to/yolo --padding 20
ydm vis crop --root path/to/yolo --out crops --padding 0.2
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis draw --root path/to/yolo --out images_vis --show-id
ydm vis draw --root path/to/yolo --out images_vis --workers 16
ydm vis draw --root path/to/yolo --out images_vis --no-progress
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm vis crop --root path/to/yolo --out crops --workers 16
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --class-id 5 --out manual_box.json
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --hide-existing
ydm vis manual-box --root path/to/yolo --image images/0001.jpg --mask-outside
```

`--show-id` 显示 txt 中从 1 开始的标注顺序号。crop 文件名也从 1 开始。
`vis manual-box` 只读取并显示指定 image 与同名 txt，鼠标拖拽一个临时框后按 Enter 输出像素坐标和 YOLO 归一化坐标；它不会修改 label。滚轮或 `+/-` 可缩放，按 `0` 恢复整图。已有标注默认显示，按 `L` 可切换显示/隐藏，也可用 `--hide-existing` 启动时隐藏。指定 `--class-id` 时还会输出可手动粘贴的完整 YOLO 行，`--out` 只写独立 JSON。
使用 `--mask-outside` 时，拖出有效框后框外区域会显示为黑色，按 `R` 可重新选择区域。

## 导入导出

```bash
ydm export coco --root path/to/yolo
ydm export xany --root path/to/yolo

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
ydm convert resize --root yolo_data --width 640 --height 640 --out yolo_640
ydm convert resize --root yolo_data --scale 0.5 --out yolo_half
```

`convert resize` 默认保持宽高比；同时指定 `--width` 和 `--height` 时会使用灰色 letterbox，并同步变换检测框和分割多边形。使用 `--no-keep-ratio` 可直接拉伸到目标尺寸。输出默认位于 `<root>/ydm_conversion/resize`，原始数据不会被覆盖。

## 评估与错误分析

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --iou 0.5
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --iou 0.5
ydm eval metrics --gt-root gt_yolo --pred-root pred_yolo
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --out vehicle_metrics.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --min-pixels 8 --out vehicle_no_small.json
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car,bus --print-table
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --exclude-class ignore,background
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --merge-class-map '{"vehicle":["car","truck"]}'
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --min-width 0.01 --class-rules metrics_rules.yaml
ydm eval metrics --gt-root gt_yolo --pred-root pred_labels --names class.txt --class car --min-pixels 15 --show-original
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --match-iou 0.5 --low-iou 0.1 --duplicate-iou 0.9
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --workers 8 --copy-pred-txt
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --val-source val.txt --class-file class.txt --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class car,bus --exclude-class ignore --min-width 0.01 --min-height 0.01 --min-size-logic and --min-pixels 8 --out error_report
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --names class.txt --class-rules error_rules.yaml --out error_report
ydm eval error-analysis --gt-root gt_labels --pred-root pred_labels --names class.txt --out error_report
```

`eval metrics` 计算 Precision、Recall、mAP@0.5、mAP@0.5:0.95。`--class` 只评估指定类别，`--exclude-class` 单独排除指定类别；两者可同时使用，未选/被排除类别的 GT 和预测都会被忽略。`--merge-class-map` 接受目标类别到原始类别列表的 JSON/YAML 映射，也可以传入映射文件，例如 `{"vehicle":["car","truck"]}`；映射会同时作用于 GT 和预测，并在类别选择、匹配和统计前生效。设置 `--show-original` 时，如果使用了类别、合并、`--class-rules` 或 `--min-pixels` 参数，会在最终结果前输出原始结果；原始结果不应用这些筛选/合并参数，但保留其他过滤参数。JSON 输出为 `detection_metrics_comparison`，包含 `original` 和 `final`；`--out` 文件仍写入最终结果。默认不输出、不计入 `Instances=0` 的类别；如需保留这些空 GT 类用于排查误检，可加 `--include-empty-classes`。小目标过滤可使用 `--min-width`、`--min-height`、`--min-area`、`--min-size-logic`，或按像素使用 `--min-pixels`。加 `--print-table` 可输出接近 Ultralytics 的对齐表格，方便人工对比。
`eval metrics` 的 `--class-rules` 接收 YAML/JSON 文件，按类别覆盖全局尺寸规则。支持 `width`/`min_width`、`height`/`min_height`、`min_area`、`min_pixels` 和 `logic`/`min_size_logic`；未配置的类别使用全局参数。若同时使用 `--merge-class-map`，规则按合并后的目标类别名匹配。

```yaml
Hollow:
  width: 0.03
  height: 0.03
  logic: or
Leakage:
  min_pixels: 20
```

metrics 还会按 COCO 风格的像素面积输出 small、medium、large 目标指标：面积 `< 32²` 为 small、`32² <= 面积 < 96²` 为 medium、面积 `>= 96²` 为 large。JSON 中位于 `size_metrics`，并额外写出 `metrics_size.csv`；图片需要有有效宽高才能进行尺寸分类。

`eval error-analysis` 支持 `--class` 只保留指定类别，`--exclude-class` 独立排除类别；`--min-width`、`--min-height`、`--min-area`、`--min-size-logic` 和 `--min-pixels` 会同时过滤 GT 与预测。宽高/面积使用归一化 YOLO 尺寸，`--min-pixels` 按像素宽度或高度判断。仍兼容旧参数 `--review-workers`、`--review-progress`、`--review-progress-leave`；新脚本建议直接使用统一运行参数。
`--class-rules` 接收 YAML/JSON 文件，按类别覆盖全局尺寸规则；规则字段可使用 `width`、`height`、`logic`，未配置的类别使用全局参数。
两个评估命令默认按类别执行置信度优先的 NMS，阈值为 `--nms-iou 0.5`；使用 `--no-nms` 可关闭。

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
