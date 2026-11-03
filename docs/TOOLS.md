# 工具脚本

## 划分 `vis_crop` 结果

`tools/split_vis_crops.py` 为 `vis_crop` 生成 `train.txt`、`val.txt`、`test.txt`。
它只处理 crop 图像列表，不生成 YOLO 标注；因此输出适合人工 review 或样本列表管理。

### 复用原数据集划分

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --split-dir dataset \
  --out dataset/ydm_vis/crop_split \
  --mode source_split
```

脚本根据 crop 文件名中的原图 stem（例如 `image_001_3.jpg`）查找原数据集
`train.txt`、`val.txt`、`test.txt`，同一原图的所有 crop 会进入同一个 split。

### 按原图随机划分

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --out dataset/ydm_vis/crop_split \
  --mode stem_random \
  --train 0.8 --val 0.2 --test 0 --seed 233
```

### 对 crop 文件完全随机划分

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --out dataset/ydm_vis/crop_split \
  --mode crop_random \
  --train 0.8 --val 0.2 --test 0 --seed 233
```

三种模式分别为：

- `source_split`：复用原数据集的 split txt；
- `stem_random`：按原图 stem 随机，同一原图不会跨 split；
- `crop_random`：每个 crop 独立随机，允许同一原图的 crop 分到不同 split。

默认 txt 中使用相对于 `--crops-dir` 的路径。使用 `--absolute-paths` 写绝对路径，
使用 `--copy-files` 同时复制到 `<out>/train`、`<out>/val`、`<out>/test`。每次运行还会生成
`split_report.json`，记录数量、模式、比例和未分配文件。
