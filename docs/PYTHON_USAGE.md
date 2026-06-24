# Python 脚本使用指南

YOLO Data Manager 可以完全通过 Python 脚本运行。推荐先以可编辑模式安装：

```powershell
cd E:\repository\yolo_data_manager
python -m pip install -e .
```

## 可编辑脚本入口

项目提供 `scripts/run_ydm.py`。修改其中的 `TASK` 和 `PARAMS`，然后运行：

```powershell
python scripts/run_ydm.py
```

默认配置用于数据统计：

```python
TASK = "stats"
PARAMS = {
    "root": Path(r"E:\datasets\my_yolo"),
    "layout": "auto",
    "out": Path(r"E:\datasets\reports\stats.json"),
    "class_csv": Path(r"E:\datasets\reports\class_counts.csv"),
    "attr_csv": Path(r"E:\datasets\reports\attributes.csv"),
}
```

路径可使用 `Path` 或字符串。值为 `None` 的参数会被忽略，多值参数可以直接使用列表。

## 在自己的脚本中调用

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

任务名使用 `模块.操作` 形式，例如 `query.class`、`ann.set_attr`、`vis.draw`。
由于 `class` 和 `from` 是 Python 关键字，对应参数写成 `class_` 和 `from_`。

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

## 查询类别和属性

```python
from yolo_data_manager import run_task

run_task(
    "query.class",
    root=r"E:\datasets\my_yolo",
    layout="auto",
    class_=["car", "truck"],
    out="vehicle_labels.csv",
    copy_images="vehicle_query/images",
    copy_labels="vehicle_query/labels",
    filtered_labels=True,
)

run_task(
    "query.attr",
    root=r"E:\datasets\my_yolo",
    attribute_file=r"E:\datasets\my_yolo\attribute.yaml",
    name="occluded",
    value=["yes"],
    out="occluded.csv",
)
```

查询结果也可以作为对象使用：

```python
from yolo_data_manager import load_yolo_dataset
from yolo_data_manager.annotation.query import query_by_attribute, query_by_class

dataset = load_yolo_dataset(r"E:\datasets\my_yolo", layout="auto")
cars = query_by_class(dataset, ["car"])
occluded = query_by_attribute(dataset, "occluded", values=["yes"])
for match in occluded.matches:
    print(match.image.path, match.annotation.to_yolo_line())
```

## 修改类别和属性

修改操作写入新的输出目录，不会覆盖原数据。重要操作建议先设置 `dry_run=True`。

```python
from yolo_data_manager import run_task

run_task(
    "ann.merge_class",
    root=r"E:\datasets\my_yolo",
    from_=["crack", "break"],
    to="defect",
    compact=True,
    out=r"E:\datasets\my_yolo_merged",
    report="merge_report.csv",
)

run_task(
    "ann.delete_class",
    root=r"E:\datasets\my_yolo",
    class_=["ignore"],
    compact=True,
    out=r"E:\datasets\my_yolo_clean",
)

run_task(
    "ann.set_attr",
    root=r"E:\datasets\my_yolo",
    attribute_file=r"E:\datasets\my_yolo\attribute.yaml",
    name="occluded",
    value="yes",
    class_=["car"],
    out=r"E:\datasets\my_yolo_attr",
)

run_task(
    "ann.delete_attr",
    root=r"E:\datasets\my_yolo",
    name="quality",
    value=["bad"],
    out=r"E:\datasets\my_yolo_quality_clean",
)
```

## 数据集管理

```python
from yolo_data_manager import run_task

run_task(
    "dataset.normalize",
    root=r"E:\datasets\source_yolo",
    layout="auto",
    out=r"E:\datasets\normalized_yolo",
)

run_task(
    "dataset.split",
    root=r"E:\datasets\normalized_yolo",
    train=0.8,
    val=0.1,
    test=0.1,
    seed=233,
    out=r"E:\datasets\normalized_yolo",
)

run_task(
    "dataset.filter",
    root=r"E:\datasets\my_yolo",
    min_area=0.001,
    class_=["car", "truck"],
    out=r"E:\datasets\my_yolo_filtered",
)

run_task(
    "dataset.merge",
    roots=[r"E:\datasets\part1", r"E:\datasets\part2"],
    out=r"E:\datasets\merged_yolo",
    source_prefix=True,
)

run_task("dataset.duplicates", root=r"E:\datasets\my_yolo", out="duplicates.csv")
run_task("dataset.bad_images", root=r"E:\datasets\my_yolo", out="bad_images.csv")
```

## 统计与可视化

```python
from yolo_data_manager import run_task

run_task(
    "stats",
    root=r"E:\datasets\my_yolo",
    out="stats.json",
    class_csv="class_counts.csv",
    ann_csv="annotations.csv",
    attr_csv="attributes.csv",
    plots_dir="stats_plots",
)

run_task(
    "vis.draw",
    root=r"E:\datasets\my_yolo",
    out="images_vis",
    show_conf=True,
    show_attrs=True,
    filter_no_attrs=False,
    conf=0.25,
)

run_task(
    "vis.crop",
    root=r"E:\datasets\my_yolo",
    out="crops",
    by_attr=True,
    filter_no_attrs=False,
)
```

## 导入、导出和评估

```python
from yolo_data_manager import run_task

run_task("export.coco", root=r"E:\datasets\my_yolo", out="instances.json")
run_task("export.xany", root=r"E:\datasets\my_yolo", out="xany_json")

run_task(
    "import.labelme",
    json_dir=r"E:\datasets\labelme_json",
    out=r"E:\datasets\labelme_yolo",
    task="segment",
    attribute_file=r"E:\datasets\attribute.yaml",
)

run_task(
    "eval.compare",
    gt_root=r"E:\datasets\gt_yolo",
    pred_root=r"E:\datasets\pred_yolo",
    out="compare.csv",
    iou=0.5,
    conf=0.25,
)

run_task(
    "eval.review_pack",
    gt_root=r"E:\datasets\gt_yolo",
    pred_root=r"E:\datasets\pred_yolo",
    out="review_pack",
    status=["fp", "fn"],
)
```

## 查看支持的任务

```python
from yolo_data_manager.scripting import TASK_COMMANDS

for task in TASK_COMMANDS:
    print(task)
```

当前任务覆盖检查、统计、布局检测、查询、数据集管理、标注修改、可视化、导入导出、格式转换和评估。参数名与 `ydm` 命令一致，只需把连字符改成下划线，例如 `show-attrs` 写成 `show_attrs`。
