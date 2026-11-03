# Standalone Tools

## Split `vis_crop` results

`tools/split_vis_crops.py` creates `train.txt`, `val.txt`, and `test.txt` for
YDM `vis_crop` images. It creates crop file lists, not YOLO label files, so the
output is intended for review or sample-list management.

Reuse an existing dataset split:

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --split-dir dataset \
  --out dataset/ydm_vis/crop_split \
  --mode source_split
```

Randomize by source image stem while keeping all crops from one source image in
the same split:

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --out dataset/ydm_vis/crop_split \
  --mode stem_random --train 0.8 --val 0.2 --test 0 --seed 233
```

Randomize every crop file independently:

```bash
python tools/split_vis_crops.py \
  --crops-dir dataset/ydm_vis/crop \
  --out dataset/ydm_vis/crop_split \
  --mode crop_random --train 0.8 --val 0.2 --test 0 --seed 233
```

The modes are `source_split`, `stem_random`, and `crop_random`. By default,
txt entries are relative to `--crops-dir`; use `--absolute-paths` for absolute
entries and `--copy-files` to copy files into `<out>/train`, `<out>/val`, and
`<out>/test`. A `split_report.json` is written with counts, settings, and
unassigned-file information.
