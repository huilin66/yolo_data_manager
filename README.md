# YOLO Data Manager

YOLO Data Manager is a Python toolkit and CLI for managing YOLO datasets.

It is designed around one internal dataset model, then layers loading, import/export,
dataset management, annotation query/edit, statistics, and visualization on top.

## First commands

```bash
ydm check --root path/to/yolo --task auto
ydm query class --root path/to/yolo --class person --out person_labels.csv
ydm ann merge-class --root path/to/yolo --from crack,break --to defect --out path/to/yolo_merged --compact
ydm stats --root path/to/yolo --out stats.json
ydm vis draw --root path/to/yolo --out images_vis
ydm export coco --root path/to/yolo --out instances.json
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
- dataset validation
- class-level annotation query
- class delete/replace/merge/rename operations
- dataset writing
- basic statistics
- PIL-based visualization
- YOLO segmentation to detection conversion
- basic COCO export

Next migration targets are the richer visualization/statistics from the existing
`data_vis` scripts and the specialized importers in `dataformat_swift`.
