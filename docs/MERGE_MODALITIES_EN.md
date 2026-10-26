# Merge single-modality YOLO datasets into multimodal YOLO

```python
from yolo_data_manager import merge_modalities, MultiModalYoloManager

result = merge_modalities(
    sources=[
        {"root": "/data/rgb", "name": "rgb"},
        {"root": "/data/thermal", "name": "thermal"},
        {"root": "/data/depth"},  # Defaults to image3.
    ],
    out_data="/data/multimodal",
    unmatched="keep",
    merge_labels=True,
    inherit_splits=False,  # Source splits are not inherited by default.
    deduplicate=False,
    dedup_iou=0.9,
)
dataset = MultiModalYoloManager(result["out"], image_dirs=result["image_dirs"]).load()
```

All sources must be supplied explicitly (at least two). No manager instance or primary dataset is required. Sources accept root, name, images_dir, labels_dir, layout, class_file, attribute_file and task. The default layout is flat images/labels; specify split_dirs where appropriate. A source root may also be a data YAML. Default output folder names are image1, image2, etc.

Images align by case-insensitive filename stem, ignoring extension. Duplicate stems within a source are rejected; conflicting scene split assignments are checked only with inherit_splits=True. Images and annotations must already be spatially aligned: this operation does not register images.

- unmatched="keep" (default): retain the union of scenes, permitting missing modality images.
- unmatched="drop": retain only scenes present in every modality.
- unmatched="zero_fill": retain the union and generate zero-valued images for missing modalities.
- merge_labels=True (default): combine annotations with class IDs mapped by class name.
- merge_labels=False: use only sources[0] labels; absent labels become empty txt files and are reported in primary_labels_missing.
- deduplicate=False (default): retain all annotations. When enabled, suppress same-class, same-attribute boxes at dedup_iou, keeping the earliest source. Polygons are deduplicated only when their points are identical.
- dry_run=True: validate and report, creating only output operation logs and no dataset files.

Zero filling infers width, height, channels and dtype from all existing images of the missing modality. For heterogeneous or empty modalities, specify the source's zero_fill={"width":640,"height":512,"channels":1,"dtype":"uint16"}. Supported zero image formats are PNG for uint8/uint16 with 1/3/4 channels, and TIFF for single-channel float32/int32. Existing images are copied without conversion.

The output contains the image directories, shared labels/, class.txt, optional attribute.yaml, dataset.yaml, multimodal.yaml, and merge_report.json. YAML records image_dirs and labels_dir for configuring the existing multimodal manager; this does not add multimodal loading to external training frameworks. By default no split txt files or YAML split entries are generated. Set inherit_splits=True to preserve existing scene membership in train.txt/val.txt/test.txt, each entry referencing an existing exported image; other modalities align by stem. Sources without splits are not automatically split.

Attribute schemas must be compatible. Global definitions must agree; class-scoped schemas can combine non-conflicting class definitions. Attribute values are reordered when field order differs. Reports include missing images, zero-filled counts/specifications, labels, deduplication, splits and source paths.

The existing multimodal loader exposes incomplete scenes in scenes, but complete_scenes includes only samples with all required modalities and labels. Use drop or zero_fill when complete samples are required.

out_data must be new, empty, or contain only ydm_log, and be outside all source dataset directories. Sources are read-only. Daily operation logs are written to out_data/ydm_log, including for dry runs. The old manager method remains a compatibility delegate; its root is not used. No source label backup is needed. Image copying, zero filling and label output support worker threads and stage progress.
