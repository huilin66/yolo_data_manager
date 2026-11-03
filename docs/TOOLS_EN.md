# YDM Common Tools

## Split `vis_crop` results

`yolo_data_manager.tools.split_vis_crops` creates `train.txt`, `val.txt`, and `test.txt` for
YDM `vis_crop` images. It creates crop file lists, not YOLO label files, so the
output is intended for review or sample-list management.

Reuse an existing dataset split:

```bash
python -m yolo_data_manager.tools.split_vis_crops \
  --crops-dir dataset/ydm_vis/crop \
  --split-dir dataset \
  --mode source_split
```

Randomize by source image stem while keeping all crops from one source image in
the same split:

```bash
python -m yolo_data_manager.tools.split_vis_crops \
  --crops-dir dataset/ydm_vis/crop \
  --mode stem_random --train 0.8 --val 0.2 --test 0 --seed 233
```

Randomize every crop file independently:

```bash
python -m yolo_data_manager.tools.split_vis_crops \
  --crops-dir dataset/ydm_vis/crop \
  --mode crop_random --train 0.8 --val 0.2 --test 0 --seed 233
```

The modes are `source_split`, `stem_random`, and `crop_random`. The CLI uses
paths relative to `--crops-dir` by default; the Python function defaults to
absolute paths and accepts `absolute_paths=False` for relative paths. Use
`--absolute-paths` in the CLI for absolute entries, and `--copy-files` to copy files into `<out>/train`, `<out>/val`, and
`<out>/test`. If `--out` is omitted, output defaults directly to `<crops-dir>`.
Thus `source_split` can generate only the three TXT files without copying crop
images. A `split_report.json` is also written with counts, settings, and
unassigned-file information.

## Multimodal helpers

```python
from yolo_data_manager.tools.mdet_tools import (
    convert_depth_to_uint8,
    load_mdet_manager,
    yolo_check,
    yolo_sta,
    yolo_vis,
)
```

These helpers reuse `MultiModalYoloManager` for multimodal loading, checking,
statistics, depth-image display conversion, and visualization.
