# YOLO Data Manager

YOLO Data Manager is a Python toolkit and CLI for managing YOLO datasets.

It is designed around one internal dataset model, then layers loading, import/export,
dataset management, annotation query/edit, statistics, and visualization on top.

## First commands

```bash
ydm check --root path/to/yolo --task auto
ydm layout detect --root path/to/yolo
ydm check --root path/to/yolo --layout auto
ydm dataset normalize --root path/to/yolo --layout auto --out normalized_yolo
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm query class --root path/to/yolo --class person --copy-images out/images --copy-labels out/labels
ydm query attr --root path/to/yolo --name defect --value yes --out defect.csv
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out path/to/yolo_merged --compact
ydm ann set-attr --root path/to/yolo --name defect --value yes --class sign --out yolo_attr_fixed
ydm ann delete-attr --root path/to/yolo --name defect --value yes --out yolo_attr_clean
ydm dataset select --root path/to/yolo --file val.txt --out path/to/yolo_val
ydm dataset split --root path/to/yolo --train 0.8 --val 0.2 --seed 233
ydm dataset filter --root path/to/yolo --min-area 0.001 --out path/to/yolo_filtered
ydm dataset merge --roots data1,data2 --out merged_yolo
ydm dataset duplicates --root path/to/yolo --out duplicate_images.csv
ydm dataset bad-images --root path/to/yolo --out bad_images.csv
ydm dataset yaml --root path/to/yolo --out dataset.yaml
ydm stats --root path/to/yolo --out stats.json
ydm stats --root path/to/yolo --ann-csv annotations.csv --attr-csv attributes.csv --plots-dir stats_plots
ydm vis draw --root path/to/yolo --out images_vis
ydm vis draw --root path/to/yolo --out images_vis --show-conf --show-attrs --filter-no-attrs
ydm vis crop --root path/to/yolo --out crops --by-attr
ydm export coco --root path/to/yolo --out instances.json
ydm export xany --root path/to/yolo --out xany_json
ydm import labelme --json-dir labelme_json --out yolo --task segment
ydm import coco --json instances.json --images-dir images --out yolo
ydm import voc --annotations-dir Annotations --images-dir JPEGImages --out yolo
ydm convert pseudo --root pred_yolo --conf 0.5 --out pseudo_yolo
ydm eval compare --gt-root gt_yolo --pred-root pred_yolo --out compare.csv --iou 0.5
ydm eval review-pack --gt-root gt_yolo --pred-root pred_yolo --out review_pack --iou 0.5
```

Install in editable mode from the project root:

```bash
python -m pip install -e .
ydm check --root path/to/yolo
```

Or run without installing by adding `src` to `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
python -m yolo_data_manager.cli check --root path/to/yolo
```

## Project Status

This is the first implementation pass. The foundation is in place:

- YOLO detection/segmentation/multi-attribute label loading
- layout detection for flat folders, split folders, image-list txt files, and mixed folders
- dataset validation
- class-level annotation query
- attribute-level annotation query
- class delete/replace/merge/rename operations
- attribute set/delete operations
- global and class-scoped `attribute.yaml` support
- attribute statistics, CSV export, visualization, and crop grouping
- dataset writing
- basic statistics
- PIL-based visualization
- YOLO segmentation to detection conversion
- basic COCO export
- x-anylabeling export
- simplified LabelMe import
- COCO and VOC import
- dataset select/split
- dataset filtering and dataset.yaml generation
- multi-dataset merge with class-name alignment
- duplicate image-name and duplicate-annotation validation
- duplicate image hash detection
- bad image detection
- GT vs prediction comparison
- FP/FN review package generation
- query-result image/label copying
- annotation CSV export and optional statistics plots

## Repository Ignore Policy

The project `.gitignore` excludes local datasets, generated visualization/statistics
outputs, training runs, caches, and common model-weight formats such as `.pt`,
`.pth`, `.onnx`, `.engine`, `.safetensors`, and `.weights`.

Next migration targets are the richer visualization/statistics from the existing
`data_vis` scripts and the specialized importers in `dataformat_swift`.
