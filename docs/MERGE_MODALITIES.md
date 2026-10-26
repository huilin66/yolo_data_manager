# 单模态合并为多模态

包级函数 `merge_modalities` 将多个单模态 YOLO 数据集合并为多个图像目录和一份共享 `labels`。图像默认按不含后缀的文件名（stem，忽略大小写）匹配，同一来源不能出现重复 stem。该方法只组织数据，不进行图像配准；各模态的归一化标注坐标必须已对齐。

```python
from yolo_data_manager import merge_modalities, MultiModalYoloManager

result = merge_modalities(
    sources=[
        {"root": "/data/rgb", "name": "rgb"},
        {"root": "/data/thermal", "name": "thermal"},
        {"root": "/data/depth"},  # 默认 image3
    ],
    out_data="/data/multimodal",
    unmatched="keep",
    merge_labels=True,
    inherit_splits=False,  # 默认不继承原数据划分
    deduplicate=False,
    dedup_iou=0.9,
    workers=8,
    progress=True,
)

multimodal = MultiModalYoloManager(
    result["out"], image_dirs=result["image_dirs"],
)
dataset = multimodal.load()
```

`sources` 定义全部来源，至少两个；无需创建 manager，也没有隐含的主数据集。每项可指定 `images_dir`、`labels_dir`、`layout`、`class_file`、`attribute_file` 和 `task`。默认读取标准 `images/labels` 的 flat 布局；原数据为 split 文件夹时可以指定 `layout="split_dirs"`。`root` 也接受 Ultralytics data YAML。省略 name 时目录按来源位置命名为 image1、image2 等，名称不能重名或占用 labels 等保留路径。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| unmatched | keep | 保留所有样本，允许某些模态缺图 |
| unmatched="drop" | | 只保留所有来源共有的样本 |
| unmatched="zero_fill" | | 保留所有样本，用全零图像补齐缺失模态 |
| merge_labels | True | 合并所有来源同一 stem 的标注，按类别名称统一 ID |
| merge_labels=False | | 只使用 sources[0] 的标签；缺少时生成空 txt，并记录 primary_labels_missing |
| deduplicate | False | 默认不去重，完整保留合并标注 |
| deduplicate=True | | 同类别且属性一致的框按 dedup_iou 去重，保留来源顺序靠前的框；多边形仅删除完全相同的点集 |
| dry_run | False | True 时检查和统计，只创建输出目录中的 ydm_log，不生成数据文件 |

补零默认检查该模态全部源图像的尺寸、通道数和数据类型。若规格不统一或模态没有可参考图像，必须显式指定该来源的补零规格：

```python
{"root": "/data/depth", "name": "depth", "zero_fill": {
    "width": 640, "height": 512, "channels": 1, "dtype": "uint16",
}}
```

支持 uint8/uint16 的 1、3、4 通道补零（PNG），以及 float32/int32 的单通道补零（TIFF）；原图直接复制，不转换原图的数据类型。参数 `zero_fill` 只影响新生成的缺失图像。

输出包含各图像目录、`labels/`、`class.txt`、可选 `attribute.yaml`、`dataset.yaml`、`multimodal.yaml` 和 `merge_report.json`。YAML 记录 image_dirs/labels_dir，供用户配置现有多模态 manager；不会修改第三方训练框架的多模态读取方式。默认不继承原数据划分，不生成 train.txt、val.txt、test.txt，也不在 YAML 中写入 split。显式设置 inherit_splits=True 时，源数据有划分才生成这三个 txt；按 scene 保留原归属，每行指向新目录中该 scene 实际存在的一张图像，其他模态通过 stem 对齐。无原始 split 时不自动划分。仅开启 inherit_splits 时校验 split 冲突，冲突会在写入前报错。

属性定义必须兼容：全局属性要求名称和值定义一致；class-scoped 属性可以合并互不冲突的类别定义，同类定义冲突时报错。字段顺序不同时会重排标注属性以保持语义。

`keep` 模式允许输出不完整 scene；现有多模态加载器的 `complete_scenes` 只包含模态齐全且有标签的样本，不完整样本仍保存在 `scenes` 和输出文件中。需要所有 scene 完整时选择 drop 或 zero_fill。

所有来源只读，不生成源标签备份。日志按天保存到 out_data/ydm_log，并记录开始、阶段、结束和异常。out_data 必须是所有源目录之外的新目录、空目录或仅含 ydm_log 的目录，避免混入旧数据结果。原 mgr.dataset_merge_modalities 仍可兼容调用，但 manager root 不参与来源选择和日志保存。报告包含样本数量、缺失模态、补零规格与数量、空标签、标签合并与去重数量、split 统计和来源路径。耗时阶段提供提示，复制和补零、标签写入支持多线程。
