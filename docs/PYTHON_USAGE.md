# Python 使用指南

YOLO Data Manager 可以完全通过 Python 脚本运行。README 是主入口教程；本文档记录更完整的 Python 调用方式。命令行用法见 [CLI_USAGE.md](CLI_USAGE.md)，项目交接说明见 [HANDOFF.md](HANDOFF.md)。

普通使用，包含所有运行功能：

```powershell
cd E:\repository\yolo_data_manager
python -m pip install .
```

开发和测试，包含所有运行功能与测试依赖：

```powershell
cd E:\repository\yolo_data_manager
python -m pip install -e ".[dev]"
python -m pytest -q
```

## YoloManager（推荐）

通过 `YoloManager` 类管理数据集：初始化时绑定 `root`，后续调用无需重复传入路径。

下面代码块只展示常用初始化方式和高频方法，不是完整 API；完整方法列表见后面的“方法速查”。

```python
from yolo_data_manager import YoloManager

mgr = YoloManager(r"E:\datasets\my_yolo", layout="auto")
mgr = YoloManager(r"E:\repository\yolo8\ultralytics\cfg\datasets\data_fire.yaml", layout="auto")
mgr = YoloManager(r"E:\datasets\my_yolo", layout="flat", init_check=False)
mgr = YoloManager(r"E:\datasets\my_yolo", layout="flat", init_check=r"E:\datasets\my_yolo\stats\validation.json")
mgr = YoloManager(r"E:\datasets\my_yolo", layout="flat",
                  init_layout_progress=True, init_layout_progress_leave=False,
                  init_check_workers=16, init_check_progress=True,
                  init_check_progress_leave=False)

# 校验
mgr.check()
mgr.check(out="validation.json")
mgr.check(out="validation.json", fill_missing_txt=True)

# 统计
mgr.stats(out="stats.json", class_csv="class_counts.csv", attr_csv="attributes.csv")
mgr.stats(plots_dir="labels_sta", stats_list=["all"])
mgr.stats(plots_dir="labels_sta", stats_list=["image_shape", "box_shape_pix", "box_pos_center"])

# 布局检测
mgr.layout_detect()

# 按类别查询 —— class_ 是 Python 关键字别名
mgr.query_class(class_=["car", "truck"], out="vehicles.csv")
mgr.query_class(class_=["person"], copy_images="persons/images", copy_labels="persons/labels",
                filtered_labels=True)

# 按属性查询
mgr.query_attr(name="occluded", value=["yes"], out="occluded.csv")
mgr.query_attr(name="quality", nonzero=True)

# 数据集管理
mgr.dataset_normalize(out=r"E:\datasets\normalized_yolo")
mgr.dataset_split(train=0.8, val=0.1, test=0.1, seed=233)
mgr.dataset_split(train=0.8, val=0.1, test=0.1, seed=233, absolute_paths=True)
mgr.dataset_filter(out="filtered", min_area=0.001, class_=["car", "truck"], backup_dir="label_backups")
mgr.dataset_filter(out="filtered_small", min_width=0.01, min_height=0.01,
                   min_size_logic="and")
mgr.dataset_filter(
    out="filtered_by_class",
    class_rules={
        "person": {"min_width": 0.01, "min_height": 0.01, "min_size_logic": "and"},
        "car": {"min_area": 0.0005},
        "defect": {"min_width": 0.005, "min_height": 0.005},
    },
)
mgr.dataset_select(file="val.txt", out="val_subset")
mgr.dataset_yaml(out="dataset.yaml", train="images/train", val="images/val")
mgr.dataset_duplicates(out="duplicates.csv")
mgr.dataset_bad_images(out="bad_images.csv")

# 多数据集合并 —— roots 不在 mgr 上，独立传入
mgr.dataset_merge(roots=[r"E:\datasets\part1", r"E:\datasets\part2"],
                  out="merged_yolo", source_prefix=True, backup_dir="label_backups")

# 标注修改 —— 修改操作写入新目录，不覆盖原数据；建议先 dry_run=True
mgr.ann_merge_class(from_=["crack", "break"], to="defect", out="yolo_merged", compact=True,
                    backup_dir="label_backups")
mgr.ann_merge_class({"vehicle": ["car", "truck"], "human": ["person"]}, out="yolo_merged_multi")
mgr.ann_delete_class(class_=["ignore"], out="yolo_clean", compact=True)
mgr.ann_replace_class(from_=["old_name"], to="new_name", out="yolo_replaced")
mgr.ann_rename_class(from_="cls_a", to="cls_b", out="yolo_renamed")
mgr.ann_apply_map(map_file="class_map.yaml", out="yolo_mapped")
mgr.ann_set_attr(name="defect", value="yes", class_=["sign"], out="yolo_attr")
mgr.ann_delete_attr(name="quality", value=["bad"], out="yolo_clean")
mgr.ann_correct_from_crops(
    crops_dir="ydm_vis/crop/car",
    to="defect",
    only_val=True,
    report="crop_correction.csv",
    backup_dir="label_backups",
    dry_run=True,
)

# `ann_correct_from_crops` 会按 `<image_stem>_<1-based annotation index>.<扩展名>` 解析 `vis_crop` 结果，递归处理属性子目录，并直接更新对应源 label；确认无误后去掉 `dry_run=True`。
# 将 `to=None` 传入时，会删除对应的整条标注。
mgr.ann_correct_from_error_crops(
    crops_dir="result_ana/val-52/review/pred_gt/pred_car_gt_background/crops",
    pred_dir="result_ana/val-52/review/pred_txt",
    dedup_iou=0.5,
    delete_pred_none=True,
    to="defect",
    only_val=True,
    report="gt_correction.csv",
    backup_dir="label_backups",
    dry_run=True,
)
# error-analysis crop 使用 `xxx_predx_gty`；提供 pred_dir 后，gtnone 会按 predx 从预测 txt 追加到 GT。
# delete_pred_none=True 时，prednone_gty 会删除第 y 条 GT，即使 to 设置为更新类别。
# replace_gt_from_pred=True 时，predx_gty 会用预测第 x 条完整替换 GT 第 y 条（类别和 geometry），并按 dedup_iou 对同图同类替换框去重；被抑制的重复 GT 会删除。
# backup_dir 指定写出 GT 前的备份目录；省略时默认是 `<数据集根目录>/labels_backup`。每次实际写入会创建带时间戳的快照子目录，dry_run=True 不会创建备份。

# 可视化
mgr.vis_draw(out="images_vis", show_conf=True, show_attrs=True)
mgr.vis_draw(out="images_vis", conf=0.25, fill_mask=True, mask_alpha=64)
mgr.vis_draw(out="images_vis", show_id=True)  # 显示 txt 中从 1 开始的标注顺序号
mgr.vis_draw(out="images_vis", workers=16)
mgr.vis_draw(out="images_vis", progress=False)
mgr.vis_crop(out="crops", by_attr=True, min_size=32)
mgr.vis_crop(out="crops", workers=16)
mgr.vis_crop(out="crops", padding=20)    # 每边增加 20 像素
mgr.vis_crop(out="crops", padding=0.2)   # 每边增加 box 宽/高的 20%
mgr.vis_crop(out="crops", padding=20)    # 每边增加 20 像素
mgr.vis_crop(out="crops", padding=0.2)   # 每边增加 box 宽/高的 20%

# 临时手动画一个 box：只读取并显示 image/txt，不修改原 label；按 Enter 后输出坐标
mgr.vis_manual_box(
    image="images/0001.jpg",
    class_id=5,
    show_existing=False,
    out="manual_box.json",
)

# 交互窗口中按 L 可切换已有标注显示/隐藏；滚轮或 +/- 缩放，0 恢复整图；临时框始终独立显示

# 已有标注默认显示；show_existing=False 可在启动时隐藏已有标注。

# 导出
mgr.export_coco(out="instances.json")
mgr.export_xany(out="xany_json")

# 转换
mgr.convert_seg2det(out="yolo_det")
mgr.convert_pseudo(out="pseudo_labels", conf=0.5, drop_confidence=True)

# 评估 —— gt_root / pred_root 独立传入
mgr.eval_compare(gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
                 out="compare.csv", iou=0.5)
mgr.eval_review_pack(gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
                     out="review_pack", status=["fp", "fn"])
mgr.eval_metrics(
    pred_root=r"E:\datasets\pred_labels",
    exclude_class_=["ignore", "background"],
    merge_class_map={"vehicle": ["car", "truck"]},
    show_original=True,
    out="metrics.json",
)

# 细粒度错误分析 —— 7 种错误子类型 + 重复 GT 检测
mgr.eval_error_analysis(gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
                        out="error_report")
mgr.eval_error_analysis(gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
                        out="error_report", match_iou=0.5, low_iou=0.1,
                        conf_thres=0.25, nms_iou=0.5, duplicate_iou=0.9)
mgr.eval_error_analysis(gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
                        out="error_report", val_source=r"E:\datasets\val.txt",
                        class_file=r"E:\datasets\class.txt")
mgr.eval_error_analysis(
    gt_root=r"E:\datasets\gt", pred_root=r"E:\datasets\pred",
    out="error_report", class_=["car", "bus"],
    exclude_class_=["ignore"], min_width=0.01, min_height=0.01,
    min_area=0.0005, min_size_logic="and", min_pixels=8,
    class_rules={
        "Efflorescene Low Risk": {"width": 0.03, "height": 0.03, "logic": "or"},
        "Broken High Risk": {"width": 0.01, "height": 0.02, "logic": "and"},
    },
)
mgr.eval_error_analysis(pred_root=r"E:\datasets\pred", out="error_report",
                        review=True, crop_padding=12)
mgr.eval_error_analysis(pred_root=r"E:\datasets\pred", out="error_report",
                        review=True, workers=16, copy_pred_txt=True)

# 导入 —— 独立参数，不使用 mgr 的 root
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

`YoloManager(..., layout="auto")` 初始化时会先做 layout 扫描，再加载图片和 label，最后执行 check。

`YoloManager` 的 `root` 也可以直接传 Ultralytics 风格的 `data.yaml/dataset.yaml`。此时会读取 YAML 的 `path` 作为数据集根目录，读取 `names` 作为类别来源。默认所有数据操作都处理完整数据集；需要只处理验证集时显式设置 `only_val=True`，此时使用 YAML 的 `val`（或数据集目录下的 `val.txt`/`val` 目录）。

## 统一运行参数

大多数加载、写入、校验、可视化和评估方法都支持同一组运行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `workers` | `8` | 支持并行的加载、校验、写入、可视化、复核等步骤使用的线程数 |
| `progress` | `True` | 显示临时 tqdm 进度条 |
| `progress_leave` | `False` | 任务结束后保留进度条 |
| `only_val` | `False` | 仅处理验证集；默认处理全部数据 |

```python
mgr.check(workers=16)
mgr.stats(only_val=True)
mgr.vis_draw(out="images_vis", progress=False)
mgr.eval_error_analysis(pred_root="pred", out="error_report", review=True, workers=16)
```

底层函数如 `load_yolo_dataset()`、`validate_dataset()` 默认不显示进度条，方便作为库函数安静调用；`YoloManager` 和 CLI 默认显示进度条。

`check` 完整校验结果会写入 JSON 文件，终端只输出红色 warning/error 摘要或绿色 OK 摘要。`out` 不指定时默认写到 `<root>/ydm_quality/check.json`。

### 默认输出路径

Python API 与 CLI 使用相同的默认输出规则；显式传入 `out`、`csv` 或 `plots_dir` 时仍以显式路径为准：

```text
<root>/labels_backup/       label 写入前的时间戳备份
<root>/ydm_quality/         check、query、duplicates、bad-images
<root>/ydm_stats/           stats.json、CSV、plots/
<root>/ydm_vis/             draw/、crop/、manual_box/
<root>/ydm_evaluation/      compare、review_pack、error_analysis、metrics
<root>/ydm_dataset/         select、normalize、filter、merge
<root>/ydm_annotation/      标注编辑输出和 edit_report.csv
<root>/ydm_conversion/      格式导入导出和任务转换
<root>/train.txt、val.txt、test.txt、dataset.yaml
```

`dataset_split()` 仍把 split 文件写在数据集根目录；`dataset_yaml()` 默认也写在根目录。
`only_val` 只改变处理的数据范围，不改变默认输出分组。多模态数据沿用这些相同的功能目录，
必要时在 `ydm_stats/plots/`、`ydm_vis/draw/` 等目录下按模态建立子目录，不存在独立的
`ydm_multimodal` 功能模块。

这些默认位置也可以直接从 `YoloManager` 获取：

```python
mgr.output_stats
mgr.output_vis
mgr.output_evaluation
mgr.output_labels_backup
mgr.output_dataset_yaml
```

上述属性返回 `Path`，不会创建目录；实际操作仍会在调用对应 manager 方法时按需创建输出。

这些默认位置也可以直接从 `YoloManager` 获取：

```python
mgr.output_stats
mgr.output_vis
mgr.output_evaluation
mgr.output_labels_backup
mgr.output_dataset_yaml
```

上述属性返回 `Path`，不会创建目录；实际操作仍会在调用对应 manager 方法时按需创建输出。

`layout_detect()` 打印的是布局检测结果，不是 `check` 校验结果。输出中 `report_type` 为 `layout_detect`，并包含 `class_source`、`class_count`、`classes`，可用于确认类别文件来源。

`stats_list` 支持：`all`、`class_counts`、`box_number`、`box_width`、`box_height`、`box_area`、`image_shape`、`box_shape`、`box_shape_pix`、`box_shape_rate`、`box_pos_start`、`box_pos_center`、`box_pos_end`、`attribute`、`legacy_csv`。

选择 `box_shape`、`box_shape_pix`、`box_shape_rate`、`box_width`、`box_height` 时，会额外按类别输出 `box_shape_ratios/`、`box_shape_pixels/`、`aspect_ratio/`、`width_image_ratio/`、`height_image_ratio/` 五个目录，每个目录内为每个类别生成一张图。`box_width` 和 `box_height` 还会分别生成 `box_width_boxplot.png`、`box_height_boxplot.png`，箱线图横轴为类别，纵轴为归一化 box 宽度或高度。

选择 `box_shape`、`box_shape_pix`、`box_shape_rate`、`box_width`、`box_height` 时，会额外按类别输出 `box_shape_ratios/`、`box_shape_pixels/`、`aspect_ratio/`、`width_image_ratio/`、`height_image_ratio/` 五个目录，每个目录内为每个类别生成一张图。`box_width` 和 `box_height` 还会分别生成 `box_width_boxplot.png`、`box_height_boxplot.png`，箱线图横轴为类别，纵轴为归一化 box 宽度或高度。

`dataset_split` 会写出 `train.txt`、`val.txt`、`test.txt`，并在输出中显示 `total_class_counts` 和 `val_class_counts`，方便检查验证集类别分布。

`dataset_filter` 中 `min_width` 和 `min_height` 默认按 `or` 逻辑删除小框：`w < min_width` 或 `h < min_height` 即删除。设置 `min_size_logic="and"` 时，只有 `w < min_width` 且 `h < min_height` 才删除。`class_rules` 可以给不同类别设置不同过滤规则；类别没有命中规则时，继续使用全局过滤参数。
类别规则也支持简写字段：`{"类别": {"width": 0.03, "height": 0.03, "logic": "or"}}`，其中 `width`/`height` 是归一化 YOLO 尺寸，`logic` 为 `or` 或 `and`。
类别规则也支持简写字段：`{"类别": {"width": 0.03, "height": 0.03, "logic": "or"}}`，其中 `width`/`height` 是归一化 YOLO 尺寸，`logic` 为 `or` 或 `and`。

`eval_error_analysis(review=True)` 会在 `review/pred_gt` 下生成按 `pred_<预测类别>_gt_<真实类别>` 组织的复核图片和 crop，并写出 Ultralytics 风格 `confusion_matrix.png`。`copy_pred_txt=True` 会把参与分析的预测 txt 复制到 `review/pred_txt`。

`eval_error_analysis` 的 `class_` 只保留指定类别，`exclude_class_` 独立排除类别；两者可以同时使用。`min_width`、`min_height`、`min_area` 和 `min_pixels` 会同时过滤 GT 与预测，宽高/面积使用归一化 YOLO 尺寸，`min_pixels` 按像素宽度或高度判断；`min_size_logic` 支持 `"or"` 或 `"and"`，语义与 `dataset_filter` 一致。
`class_rules` 可以按类别覆盖全局尺寸规则，格式为 `{类别: {"width": ..., "height": ..., "logic": "or" 或 "and"}}`；命中类别使用自己的规则，未命中类别继续使用全局参数。
`eval_error_analysis` 与 `eval_metrics` 默认先按类别执行置信度优先的 NMS（`nms_iou=0.5`），再使用相同的一对一 IoU 匹配规则；传入 `nms_iou=None` 可关闭 NMS。关闭 NMS 时，重复预测会在错误分析中标记为 `duplicate_prediction`，并在 metrics 中作为 FP 统计。

`eval_metrics` 使用 `class_` 指定只评估的类别，使用独立的 `exclude_class_` 排除类别；两者可以同时传入。`merge_class_map` 接受“目标类别: 原始类别列表”的字典，例如 `{"vehicle": ["car", "truck"]}`，并在 GT 和预测的类别选择、匹配、统计前同时应用。类别选择和排除使用合并后的目标类别名。设置 `show_original=True` 后，如果使用了类别、合并或 `min_pixels` 参数，会在最终结果前输出原始结果；JSON 输出包含 `original` 和 `final`，而 `out` 文件仍保存最终结果。
统计、可视化和评估默认处理全部数据；设置 `only_val=True` 或显式提供 `val_source` 才限制为验证集。

`import_mask` 用于把语义分割 mask 转成 YOLO segmentation。单通道 mask 使用像素值作为类别 id；RGB mask 可在 `class_map` 中使用 `"#ff0000"` 或 `"255,0,0"` 作为 key。若环境中有 OpenCV，会用轮廓提取；否则退回为外接矩形 polygon。

### 初始化参数

`YoloManager` 构造时存储的共用参数，在后续调用有 `--root` 的任务时自动填充：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `root` | (必填) | 数据集根目录 |
| `layout` | `"auto"` | 布局模式：auto / flat / split_dirs / image_list / mixed |
| `task` | `"auto"` | 任务类型：auto / detect / segment |
| `images_dir` | `"images"` | 图片子目录名 |
| `labels_dir` | `"labels"` | 标注子目录名 |
| `class_file` | `None` | class.txt 路径（默认 root/class.txt） |
| `attribute_file` | `None` | attribute.yaml 路径（默认 root/attribute.yaml） |
| `split_file` | `None` | 显式指定图片列表时仅处理该列表；YAML 中的 `val` 不会默认生效 |
| `only_val` | `False` | 是否让后续数据操作默认仅处理验证集 |
| `init_layout` | `True` | 初始化时是否执行一次 layout detect |
| `init_layout_progress` | `True` | 初始化 layout detect 是否显示 tqdm 进度条 |
| `init_layout_progress_leave` | `False` | 初始化 layout detect 是否保留进度条 |
| `init_check` | `True` | 初始化时是否自动 check；也可传入 JSON 路径 |
| `init_check_fill_missing_txt` | `False` | 初始化自动 check 时是否补全缺失的空 label txt |
| `init_check_workers` | `8` | 初始化自动 check 的线程数 |
| `init_check_progress` | `True` | 初始化自动 check 是否显示 tqdm 进度条 |
| `init_check_progress_leave` | `False` | 初始化自动 check 是否保留进度条 |

### 方法速查

| 方法 | 对应 CLI |
|---|---|
| `check()` | `ydm check` |
| `stats()` | `ydm stats` |
| `layout_detect()` | `ydm layout detect` |
| `query_class(class_=..., ...)` | `ydm query class` |
| `query_attr(name=..., ...)` | `ydm query attr` |
| `dataset_select(file=..., out=...)` | `ydm dataset select` |
| `dataset_normalize(out=...)` | `ydm dataset normalize` |
| `dataset_split(train=..., val=..., ...)` | `ydm dataset split` |
| `dataset_yaml(out=..., ...)` | `ydm dataset yaml` |
| `dataset_filter(out=..., ...)` | `ydm dataset filter` |
| `dataset_merge(roots=..., out=...)` | `ydm dataset merge` |
| `dataset_duplicates(out=...)` | `ydm dataset duplicates` |
| `dataset_bad_images(out=...)` | `ydm dataset bad-images` |
| `ann_delete_class(class_=..., out=...)` | `ydm ann delete-class` |
| `ann_replace_class(from_=..., to=..., out=...)` | `ydm ann replace-class` |
| `ann_merge_class(from_=..., to=..., ...)` | `ydm ann merge-class` |
| `ann_rename_class(from_=..., to=..., out=...)` | `ydm ann rename-class` |
| `ann_apply_map(map_file=..., out=...)` | `ydm ann apply-map` |
| `ann_correct_from_crops(crops_dir=..., to=...)` | `ydm ann correct-from-crops` |
| `ann_correct_from_error_crops(crops_dir=..., to=...)` | `ydm ann correct-from-error-crops` |
| `ann_set_attr(name=..., value=..., ...)` | `ydm ann set-attr` |
| `ann_delete_attr(name=..., ...)` | `ydm ann delete-attr` |
| `vis_draw(out=..., ...)` | `ydm vis draw` |
| `vis_crop(out=..., ...)` | `ydm vis crop` |
| `export_coco(out=...)` | `ydm export coco` |
| `export_xany(out=...)` | `ydm export xany` |
| `import_labelme(json_dir=..., out=...)` | `ydm import labelme` |
| `import_coco(json_path=..., images_dir=..., out=...)` | `ydm import coco` |
| `import_voc(annotations_dir=..., images_dir=..., out=...)` | `ydm import voc` |
| `import_mask(images_dir=..., masks_dir=..., out=...)` | `ydm import mask` |
| `convert_seg2det(out=...)` | `ydm convert seg2det` |
| `convert_pseudo(out=..., ...)` | `ydm convert pseudo` |
| `eval_compare(gt_root=..., pred_root=..., out=...)` | `ydm eval compare` |
| `eval_review_pack(gt_root=..., pred_root=..., out=...)` | `ydm eval review-pack` |
| `eval_error_analysis(gt_root=..., pred_root=..., out=...)` | `ydm eval error-analysis` |
| `eval_metrics(pred_root=..., class_=["car", "bus"], min_pixels=8, out=...)` | `ydm eval metrics` |
| `eval_metrics(pred_root=..., class_=["car", "bus"], print_table=True)` | `ydm eval metrics --print-table` |
| `eval_metrics(pred_root=..., exclude_class_=["ignore"], merge_class_map={"vehicle": ["car", "truck"]})` | `ydm eval metrics --exclude-class ignore --merge-class-map ...` |
| `eval_metrics(pred_root=..., class_=["car"], min_pixels=15, show_original=True)` | `ydm eval metrics --class car --min-pixels 15 --show-original` |
| `eval_metrics(pred_root=..., ignore_empty_classes=False)` | `ydm eval metrics --include-empty-classes` |

所有方法返回 `int` 退出码（0 = 成功），底层调用 `run_task()`。

## 函数式调用

如果不习惯面向对象风格，可以直接使用 `run_task()` 函数。每次调用需要显式传入 `root`。

```python
from pathlib import Path
from yolo_data_manager import run_task

code = run_task(
    "check",
    root=Path(r"E:\datasets\my_yolo"),
    layout="auto",
    out="validation.json",
)
if code != 0:
    print("数据集存在校验问题")
```

任务名使用 `模块.操作` 形式，例如 `query.class`、`ann.set_attr`、`vis.draw`、`eval.error_analysis`。
由于 `class` 和 `from` 是 Python 关键字，对应参数写成 `class_` 和 `from_`。

## 示例函数与数据集调用脚本

`example/functions/` 是二次整理后的可复用函数；`example/` 根目录下的文件是具体数据集的调用脚本。复制 `example/dataset_template.py`，按数据集改名并修改路径和参数：

```python
from example.functions import yolo_sta, yolo_vis

DATA_DIR = r"/path/to/my_dataset.yaml"

yolo_sta(DATA_DIR, stats_list=["all"], only_val=False)
yolo_vis(DATA_DIR, crop=True, only_val=False)
```

不再使用通用 `example/datasets/` 调用器，也不再需要 `run_ydm.py`。TT100K 转换作为独立工具放在 `tools/convert_tt100k.py`。

## 加载并获得 Python 对象

需要继续处理返回数据时，可以绕过任务调度器，直接调用底层 API：

```python
from yolo_data_manager import load_yolo_dataset
from yolo_data_manager.io.validator import validate_dataset

dataset = load_yolo_dataset(
    r"E:\datasets\my_yolo",
    layout="auto",
    task="auto",
)
print("图片数:", len(dataset.images))
print("标注数:", dataset.annotation_count())
print("类别:", dataset.classes.names)

report = validate_dataset(dataset)
print("是否通过:", report.ok)
for issue in report.issues:
    print(issue)
```

## 多模态 YOLO 数据集

多模态是数据集的模态属性，不是独立的功能模块。`MultiModalYoloManager` 只是针对“一份共享 YOLO label、多个对齐图像目录”的加载与关联适配器；统计、校验、可视化和图像转换仍遵循 `YoloManager` 的相同输出分组和默认路径。当前适配器提供关联检查、统计、绘制、crop 和 uint8 转换；尚未定义安全的全模态写入语义的查询、编辑、split、merge 等方法不会静默退化为只处理某一路图像。它目前不提供 CLI 命令。

图像关联使用场景 stem：每个图像或 label 文件先去扩展名，再去掉其 source 配置的 `suffix`，得到同一个 `scene_stem`。例如 `visible/0001_V.jpg`、`infrared/0001_T.png` 和 `labels/0001_gt.txt` 可关联为场景 `0001`。

```python
from yolo_data_manager import (
    MultiModalYoloManager,
)

root = r"E:\datasets\mdet_train"

# 空配置：visible/0001.jpg、infrared/0001.png、depth/0001.tif、labels/0001.txt
# 会按相同 stem 自动关联。图像扩展名可以不同。
mgr = MultiModalYoloManager(
    root,
    image_dirs=["visible", "infrared", "depth"],
    labels_dir="labels",
    class_file="class.txt",
    task="detect",
)

# Manager 首次使用时加载并缓存；以下操作均复用同一份关联结果。
# 省略输出参数时使用 ydm_stats/、ydm_vis/ 等统一功能目录。
stats = mgr.stats(stats_list=["all"])
mgr.vis_draw(show_id=True, workers=8)
mgr.vis_crop(workers=8)

# 检查未关联文件、缺失模态、suffix 不匹配或重复场景图。
mgr.check()  # 终端输出简洁摘要；完整报告默认写入 ydm_quality/multimodal_check.json
```

`check()` 的 `image_type_summary` 同时按每个模态输出源图像总数及 `format / Pillow mode / dtype / 通道数 / 分辨率` 的分组数量。例如可直接发现同一 depth 目录中混有 `JPEG/RGB/uint8` 和 `PNG/I;16/uint16`。

对于非 `uint8` 的原始图像，可写入新的模态图像目录并转换为 8 位 PNG；原图、label 和当前缓存的数据集均不会被修改。已是 `uint8` 的选中图像会原样复制。深度图建议提供固定值域，以便不同图片具有可比较的亮度：

```python
converted = mgr.convert_to_uint8(
    # 省略 out 时默认写入 ydm_conversion/uint8/
    modalities=["depth"],
    stretch=True,
    value_range=(0, 20000),  # 将该原始深度范围映射到显示值 0–255
    preserve_zero=True,      # 保留无效深度 0 为黑色
    workers=8,
)
# 输出：ydm_conversion/uint8/depth/<原相对路径>；非 uint8 文件写为 .png
```

不传 `value_range` 时，`stretch=True` 按每张非 `uint8` 图像的非零有效值 min-max 拉伸；适合观察细节，但不同图片的亮度不具可比性。`stretch=False` 则仅把原始数值裁剪到 `0–255`，通常不适用于 `uint16` 深度图。默认 `overwrite=False`，若目标文件已存在会中止以避免覆盖。

若文件名有模态后缀，使用 `image_params` 和 `label_params` 配置。字典 key 是逻辑图像 type；默认它绑定到同名的图像目录。若 type 和目录名不同，可用 `dir` 显式绑定。

```python
mgr = MultiModalYoloManager(
    root,
    image_dirs=["visible", "thermal", "depth_map"],
    image_params={
        "rgb": {"dir": "visible", "suffix": "_V"},
        "infrared": {"dir": "thermal", "suffix": "_T"},
        "depth": {"dir": "depth_map", "suffix": "_D", "required": False},
    },
    labels_dir="labels",
    label_params={"suffix": "_gt"},
    class_file="class.txt",
    task="detect",
)
```

上述配置将 `0001_V.jpg`、`0001_T.png`、`0001_D.tif` 和 `0001_gt.txt` 都归一为 scene stem `0001`。`required=False` 的模态缺失不会排除该场景；默认所有图像 type 都是必需的。统计结果的 `annotation_stats` 只按 scene 统计一次标注，`modalities.<type>.stats` 则分别给出每种图像的尺寸和像素级框统计。可视化输出按 type 分目录，例如 `ydm_vis/draw/rgb/`、`ydm_vis/draw/infrared/`，避免文件覆盖；这只是同一 `vis` 功能下的模态子目录。

## 查询结果对象

```python
from yolo_data_manager import load_yolo_dataset
from yolo_data_manager.annotation.query import query_by_attribute, query_by_class

dataset = load_yolo_dataset(r"E:\datasets\my_yolo", layout="auto")
cars = query_by_class(dataset, ["car"])
occluded = query_by_attribute(dataset, "occluded", values=["yes"])
for match in occluded.matches:
    print(match.image.path, match.annotation.to_yolo_line())
```

## 查看支持的任务

```python
from yolo_data_manager.scripting import TASK_COMMANDS

for task in TASK_COMMANDS:
    print(task)
```

## 参数说明

参数名与 `ydm` 命令一致，只需把连字符改成下划线，例如 `show-attrs` → `show_attrs`。

| Python 参数 | CLI 标志 | 说明 |
|---|---|---|
| `class_` | `--class` | Python 关键字，需加下划线 |
| `from_` | `--from` | Python 关键字，需加下划线 |
| `map_file` | `--map` | 避免与内置函数冲突 |
| `json_path` | `--json` | 避免与模块名冲突 |

布尔值行为：`copy_images=False` → `--no-copy-images`，`compact=False` → `--no-compact`。
列表/元组/集合自动转为逗号分隔字符串：`["a", "b"]` → `a,b`。
`None` 值会被忽略，不传给 CLI。
