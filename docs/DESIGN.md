# YOLO Data Manager 项目设计

## 目标

`YOLO Data Manager` 用来统一管理 YOLO 数据集，避免现有脚本中“读取、转换、统计、可视化、路径配置、临时逻辑”混在一起的问题。

核心原则：

1. 所有格式先读成统一内部模型。
2. 查询、编辑、统计、可视化都只依赖内部模型。
3. 导入导出只负责格式边界，不重复实现业务逻辑。
4. 默认不原地破坏数据，写操作优先输出到新目录，并支持 `dry-run/report/backup`。

## 功能分组

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

迁移阶段：

- COCO/VOC -> YOLO
- LabelMe -> YOLO 完整迁移，包括多任务属性
- YOLO -> x-anylabeling 属性细节对齐
- Bosch/GTSDB/TZ 专用数据源

### 3. 数据集管理

- split train/val/test
- select/copy 子集
- merge 数据集
- class id remap
- 删除空标注
- 保留/删除空 label 文件
- 输出操作 report
- 按类别、面积、宽高、confidence 过滤标注
- 多数据集合并，按类别名对齐并自动 remap class id
- 生成 `dataset.yaml`

典型命令：

```bash
ydm dataset select --root yolo --file val.txt --out yolo_val
ydm dataset split --root yolo --train 0.8 --val 0.2 --test 0.0 --seed 233
ydm dataset filter --root yolo --min-area 0.001 --out yolo_filtered
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
- crop
- gallery
- prediction threshold
- 后续迁移现有 `data_vis/yolo_vis.py` 中更完整的 OpenCV 风格

典型命令：

```bash
ydm vis draw --root yolo --out images_vis
ydm vis draw --root yolo --out images_vis --show-conf --show-attrs --filter-no-attrs --mask-alpha 80
ydm vis crop --root yolo --out crops --by-attr
```

### 8. 预测结果对比

- GT vs prediction 按 class + IoU 贪心匹配
- 输出 TP/FP/FN 明细 CSV
- 支持 confidence threshold

典型命令：

```bash
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --out compare.csv --iou 0.5 --conf 0.3
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --out review_pack --iou 0.5
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
    seg_det.py
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
