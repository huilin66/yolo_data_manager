from pathlib import Path
import json

from PIL import Image

from yolo_data_manager.annotation.edit import merge_classes
from yolo_data_manager.annotation.query import copy_query_result, query_by_class
from yolo_data_manager.converters.labelme import import_labelme_dir
from yolo_data_manager.converters.seg_det import segmentation_to_detection
from yolo_data_manager.converters.xanylabeling import export_xanylabeling
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.validator import validate_dataset
from yolo_data_manager.vis.renderer import crop_dataset


def make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "a.jpg")
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "b.jpg")
    (root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "b.txt").write_text("1 0.1 0.1 0.2 0.1\n", encoding="utf-8")
    return root


def test_load_query_and_validate(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    assert len(dataset.images) == 2
    assert dataset.annotation_count() == 3
    assert validate_dataset(dataset).ok

    result = query_by_class(dataset, ["car"])
    assert len(result) == 2
    assert [path.name for path in result.label_paths()] == ["a.txt", "b.txt"]


def test_merge_classes_with_compact(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    edited, report = merge_classes(dataset, ["person"], "car", compact=True)

    assert len(report.rows) == 1
    assert edited.classes.names == ["car"]
    assert {ann.class_id for image in edited.images for ann in image.annotations} == {0}


def test_segmentation_to_detection(tmp_path):
    root = tmp_path / "seg"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(root / "images" / "poly.jpg")
    (root / "class.txt").write_text("shape\n", encoding="utf-8")
    (root / "labels" / "poly.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n", encoding="utf-8")

    dataset = load_yolo_dataset(root, task="segment")
    det = segmentation_to_detection(dataset)
    ann = det.images[0].annotations[0]

    assert ann.polygon is None
    assert ann.box is not None
    assert tuple(round(value, 6) for value in ann.box.as_tuple()) == (0.2, 0.25, 0.2, 0.3)


def test_query_copy_and_crop(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)
    result = query_by_class(dataset, ["car"])

    copy_query_result(result, images_dir=tmp_path / "images_out", labels_dir=tmp_path / "labels_out")
    assert sorted(path.name for path in (tmp_path / "images_out").iterdir()) == ["a.jpg", "b.jpg"]
    assert sorted(path.name for path in (tmp_path / "labels_out").iterdir()) == ["a.txt", "b.txt"]

    saved = crop_dataset(dataset, tmp_path / "crops")
    assert saved == 3
    assert (tmp_path / "crops" / "person" / "a_0.jpg").exists()


def test_export_xanylabeling(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    export_xanylabeling(dataset, tmp_path / "xany")
    data = json.loads((tmp_path / "xany" / "a.json").read_text(encoding="utf-8"))

    assert data["imagePath"] == "a.jpg"
    assert len(data["shapes"]) == 2
    assert data["shapes"][0]["label"] == "person"


def test_import_labelme(tmp_path):
    labelme_dir = tmp_path / "labelme"
    labelme_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(labelme_dir / "img.jpg")
    data = {
        "imagePath": "img.jpg",
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [
            {
                "label": "surface",
                "shape_type": "rectangle",
                "points": [[10, 20], [50, 60]],
            }
        ],
    }
    (labelme_dir / "img.json").write_text(json.dumps(data), encoding="utf-8")

    dataset = import_labelme_dir(labelme_dir, out_root=tmp_path / "yolo_out", task="detect")

    assert dataset.classes.names == ["surface"]
    assert dataset.annotation_count() == 1
    assert (tmp_path / "yolo_out" / "labels" / "img.txt").exists()
