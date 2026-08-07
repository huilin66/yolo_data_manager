# YOLO Data Manager Handoff

本文档用于项目交接，说明当前设计原则、功能边界、包结构、关键约定和后续迁移方向。面向使用者的教程在 [README](../README.md)，Python 细节在 [PYTHON_USAGE.md](PYTHON_USAGE.md)，CLI 细节在 [CLI_USAGE.md](CLI_USAGE.md)。

## 目标与原则

`YOLO Data Manager` 用来统一管理 YOLO 数据集，避免现有脚本中“读取、转换、统计、可视化、路径配置、临时逻辑”混在一起的问题。

核心原则：

1. 所有格式先读成统一内部模型。
2. 查询、编辑、统计、可视化都只依赖内部模型。
3. 导入导出只负责格式边界，不重复实现业务逻辑。
4. 默认不原地破坏数据，写操作优先输出到新目录，并支持 `dry-run/report/backup`。
5. 当 `progress=True` 时，每个可能耗时的阶段都必须先输出可保留的阶段提示；即使动态进度条使用 `leave=False` 清除，也不得让用户在处理期间看到无状态的空白。`progress=False` 是用户主动关闭提示的例外。
6. 并行任务收到 `Ctrl+C` 时必须取消尚未开始的工作，清理进度显示并立即向调用方传播中断；不得因线程池退出等待整个待处理队列。已生成的输出可保留，后续任务不再继续。
7. 多模态图像可混有不同格式、模式和位深；`check()` 必须报告每个模态的源图像类型计数。需要显示型转换时，只能写入新的输出目录；`uint16` 等非 `uint8` 图像须显式拉伸或指定固定值域后再转换，不能直接当作 RGB 显示。

## 当前功能分组

### 1. 加载与校验

- 识别 `images/labels/class.txt/classes.txt/dataset.yaml/attribute.yaml`
- 按文件 stem 匹配 image 和 label，不依赖目录排序
- 支持 YOLO detection、YOLO segmentation、带属性多任务标签、预测 confidence
- 支持不同 YOLO 布局：`flat`、`split_dirs`、`image_list`、`mixed`、`auto`
- 支持把不同布局 normalize 成标准 `images/labels` 组织
- 支持全局 attribute 和按类别组织的 class-scoped attribute
- 校验缺图、缺标签、孤儿 label、类别越界、坐标越界、负宽高、多边形点数异常

典型命令：

```bash
ydm layout detect --root yolo_data
ydm check --root yolo_data --layout auto
ydm dataset normalize --root yolo_data --layout auto --out yolo_normalized
```

### 2. 导入导出

第一阶段：

- YOLO -> COCO
- YOLO segmentation -> YOLO detection
- YOLO -> x-anylabeling
- LabelMe -> YOLO 简化导入
- COCO -> YOLO
- VOC -> YOLO
- semantic segmentation mask -> YOLO segmentation

迁移阶段：

- LabelMe -> YOLO 完整迁移，包括多任务属性
- YOLO -> x-anylabeling 属性细节对齐
- Bosch/GTSDB/TZ 专用数据源

语义 mask 导入约定：

- 单通道 mask 使用像素值作为类别值，例如 `0=background, 1=crack`
- RGB mask 使用颜色作为类别值，例如 `#ff0000=crack`
- 每个连通区域转为一个 YOLO segmentation polygon
- `background` 不输出到 label
- `min_area` 过滤小连通区域
- 有 OpenCV 时使用 contour polygon；无 OpenCV 时退回外接矩形 polygon，避免强制引入重依赖

### 3. 数据集管理

- split train/val/test
- select/copy 子集
- merge 数据集
- class id remap
- 删除空标注
- 保留/删除空 label 文件
- 输出操作 report
- 按类别、面积、宽高、confidence 过滤标注
- 支持 `min_size_logic=or/and` 控制宽高小框过滤逻辑
- 支持每个类别单独设置过滤规则
- 多数据集合并，按类别名对齐并自动 remap class id
- 生成 `dataset.yaml`

典型命令：

```bash
ydm dataset select --root yolo --file val.txt --out yolo_val
ydm dataset split --root yolo --train 0.8 --val 0.2 --test 0.0 --seed 233
ydm dataset split --root yolo --train 0.8 --val 0.1 --test 0.1 --absolute-paths
ydm dataset filter --root yolo --min-area 0.001 --out yolo_filtered
ydm dataset filter --root yolo --min-width 0.01 --min-height 0.01 --min-size-logic and --out yolo_filtered
ydm dataset filter --root yolo --class-rules filter_rules.yaml --out yolo_filtered
ydm dataset merge --roots yolo_a,yolo_b --out yolo_merged
ydm dataset duplicates --root yolo --out duplicate_images.csv
ydm dataset bad-images --root yolo --out bad_images.csv
ydm dataset yaml --root yolo --out dataset.yaml
```

### 4. 标注查询

查询结果分两层：

- label-level：哪些 `.txt` 包含目标类别
- instance-level：具体到每一行标注，包括 image、label、line_no、class_id、class_name、bbox/polygon、attributes、confidence

典型命令：

```bash
ydm query class --root yolo --class surface --out surface.csv
ydm query class --root yolo --class 3 --out class3.csv
ydm query class --root yolo --class surface --copy-images query/images --copy-labels query/labels
ydm query class --root yolo --class surface --copy-labels query/labels --filtered-labels
ydm query attr --root yolo --name defect --value yes --out defect.csv
ydm query attr --root yolo --name defect --nonzero --copy-labels query/labels
```

支持两类 attribute yaml：

```yaml
attributes:
  defect: [no, yes]
  color: [red, green]
```

```yaml
attributes:
  sign:
    defect: [no, yes]
  road:
    material: [asphalt, concrete]
```

### 5. 标注修改

需要区分两种行为：

- 只改标注行：保留 `class.txt` 编号体系
- 改类别体系：删除/合并类别后同步更新 `class.txt`，并重排 label class id

典型命令：

```bash
ydm ann delete-class --root yolo --class ignore --out yolo_clean
ydm ann drop-class --root yolo --class ignore --out yolo_clean --compact
ydm ann replace-class --root yolo --from old --to new --out yolo_fixed
ydm ann merge-class --root yolo --from crack,break,peeling --to defect --out yolo_merged --compact
ydm ann rename-class --root yolo --from old_name --to new_name --out yolo_renamed
ydm ann set-attr --root yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root yolo --name defect --value yes --out yolo_attr_clean
```

### 6. 统计

统计模块只负责产出结构化结果，绘图模块单独处理：

- 图片数、label 数、标注数
- 类别分布
- 每图目标数
- 空图/空 label
- bbox 宽高、面积、长宽比
- segmentation polygon 点数、外接框
- 属性分布、类别-属性交叉分布
- annotation CSV 明细
- attribute long-form CSV 明细
- 可选 PNG 图表输出

### 7. 可视化

- detection box
- segmentation polygon
- class name/confidence
- txt 中从 1 开始的标注顺序号
- crop
- gallery
- prediction threshold
- 多线程渲染和进度条
- 后续迁移现有 `data_vis/yolo_vis.py` 中更完整的 OpenCV 风格

典型命令：

```bash
ydm vis draw --root yolo --out images_vis
ydm vis draw --root yolo --out images_vis --show-conf --show-attrs --filter-no-attrs --mask-alpha 80
ydm vis draw --root yolo --out images_vis --show-id
ydm vis crop --root yolo --out crops --by-attr
```

### 8. 预测结果对比

- GT vs prediction 按 class + IoU 贪心匹配
- 输出 TP/FP/FN 明细 CSV
- 支持 confidence threshold
- 细粒度错误分析：background FP、localisation FP、duplicate prediction、class error、FN 子类型
- 生成 Ultralytics 风格完整混淆矩阵，包含 `background`
- review 目录按 `pred_<预测类别>_gt_<真实类别>` 组织，包含 background 情况
- review crop 文件名使用 `原图名_pred预测txt顺序id_gtGTtxt顺序id`
- 可复制预测 txt 到 `review/pred_txt`
- review 图和 crop 支持多线程与进度条
- `eval metrics` 支持 `--class` 选择类别、`--exclude-class` 排除类别，以及对 GT/预测同时生效的 `--merge-class-map`
- `eval metrics --show-original` 可在类别/合并/`min_pixels` 过滤结果前输出原始结果用于对比
- 数据集加载默认处理全部数据；使用 `--only-val` 或 Python 的 `only_val=True` 才限制到验证集，YAML 的 `val` 不再隐式限制普通统计和可视化
- 新增 `ann correct-from-crops` / `ann_correct_from_crops`：按 `vis_crop` 的 `<image_stem>_<1-based index>` 文件名定位原始标注行并直接校正类别；`to=None`（CLI 使用 `--to none`）时删除该标注
- 新增 `ann correct-from-error-crops` / `ann_correct_from_error_crops`：按 `eval_error_analysis` 的 `xxx_predx_gty` 文件名中的 y 定位 GT 行并校正或删除，`predx` 只用于复核上下文
- `vis crop` 支持 `padding`：整数按每边像素扩展，小数按 box 宽度/高度比例扩展，并自动限制在图像边界内
- `ann correct-from-error-crops` / `ann_correct_from_error_crops` 支持 `--delete-pred-none` / `delete_pred_none=True`：对 `prednone_gty` 强制删除第 y 条 GT，即使 `--to` / `to` 是更新类别
- 支持 `--replace-gt-from-pred` / `replace_gt_from_pred=True`：结合预测 txt，用 `predx_gty` 的预测 x 完整替换 GT y（类别和 geometry），同图同类替换框按 `dedup_iou` 去重并删除被抑制的重复 GT；`prednone_gty` 删除，`predx_gtnone` 追加

典型命令：

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --out compare.csv --iou 0.5 --conf 0.3
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --out review_pack --iou 0.5
ydm eval error-analysis --gt-root gt_yolo --pred-root pred_yolo --out error_report --review --workers 8 --copy-pred-txt
ydm convert pseudo --root pred_yolo --conf 0.5 --out pseudo_yolo
```

## 包结构

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
  converters/
    coco.py
    labelme.py
    mask.py
    seg_det.py
    voc.py
    xanylabeling.py
  stats/
    compute.py
    report.py
  vis/
    renderer.py
  cli.py
```

## 内部模型

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

这样 detection、segmentation、multi-attribute、prediction 都能进入同一套查询、编辑、统计、可视化流程。

## 迁移计划

### Phase 1: 核心可用

- core model
- YOLO loader/writer
- validator
- query class
- delete/replace/merge/rename class
- stats JSON
- basic visualization
- COCO export

### Phase 2: 迁移旧能力

- 迁移 `data_vis/yolo_vis.py` 的多属性可视化、crop、confidence、mask overlay
- 迁移 `data_vis/yolo_sta.py` 的图表输出
- 迁移 `dataformat_swift/yolo2xanylabeling.py`
- 迁移 `dataformat_swift/labelme2yolo.py`

### Phase 3: 数据源适配

- Bosch
- GTSDB
- TZ XML
- 项目内其他定制格式

## 写操作安全策略

- 默认输出到 `--out`
- 原地修改必须显式 `--in-place`
- 支持 `--dry-run`
- 支持 `--report edit_report.csv`
- 支持 `--keep-empty-labels`
- 支持 `--dry-run`
- class compact/remap 操作必须输出 remap 表

## Git Ignore 策略

项目根目录 `.gitignore` 默认忽略：

- Python 缓存、构建产物、虚拟环境
- 数据目录：`data/ datasets/ dataset/ raw/ processed/ images/ labels/ annotations/`
- 训练与分析输出：`runs/ outputs/ work_dirs/ images_vis/ labels_sta/ cache/`
- 模型权重：`.pt .pth .ckpt .onnx .engine .trt .safetensors .weights .h5`
- 大型压缩包与视频
