from pathlib import Path
import json

from PIL import Image

from yolo_data_manager.annotation.edit import delete_by_attribute, merge_classes, set_attribute
from yolo_data_manager.annotation.query import copy_query_result, query_by_attribute, query_by_class
from yolo_data_manager.converters.coco import import_coco
from yolo_data_manager.converters.labelme import import_labelme_dir
from yolo_data_manager.converters.pseudo import predictions_to_pseudo_labels
from yolo_data_manager.converters.seg_det import segmentation_to_detection
from yolo_data_manager.converters.voc import import_voc_dir
from yolo_data_manager.converters.xanylabeling import export_xanylabeling
from yolo_data_manager.core.schema import write_dataset_yaml
from yolo_data_manager.dataset.filter import filter_by_geometry
from yolo_data_manager.dataset.duplicates import find_duplicate_images
from yolo_data_manager.dataset.merge import merge_datasets
from yolo_data_manager.dataset.quality import find_bad_images
from yolo_data_manager.evaluation.compare import compare_datasets
from yolo_data_manager.evaluation.review_pack import write_review_pack
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.layout import detect_layout
from yolo_data_manager.io.validator import validate_dataset
from yolo_data_manager.io.writer import write_yolo_dataset
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


def test_layout_detect_split_dirs_and_normalize(tmp_path):
    root = tmp_path / "split_yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "class.txt").write_text("obj\n", encoding="utf-8")

    info = detect_layout(root)
    dataset = load_yolo_dataset(root, layout="auto")
    write_yolo_dataset(dataset, tmp_path / "normalized")

    assert info.layout == "split_dirs"
    assert dataset.annotation_count() == 1
    assert (tmp_path / "normalized" / "images" / "a.jpg").exists()


def test_image_list_layout(tmp_path):
    root = tmp_path / "list_yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "class.txt").write_text("obj\n", encoding="utf-8")
    (root / "train.txt").write_text("images/a.jpg\n", encoding="utf-8")

    info = detect_layout(root)
    dataset = load_yolo_dataset(root, layout="auto")

    assert info.layout == "image_list"
    assert len(dataset.images) == 1
    assert dataset.annotation_count() == 1


def test_merge_classes_with_compact(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    edited, report = merge_classes(dataset, ["person"], "car", compact=True)

    assert len(report.rows) == 1
    assert edited.classes.names == ["car"]
    assert {ann.class_id for image in edited.images for ann in image.annotations} == {0}


def test_filter_by_geometry(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    filtered = filter_by_geometry(dataset, min_area=0.05)

    assert filtered.annotation_count() == 1
    assert filtered.images[0].annotations[0].class_id == 0


def test_merge_datasets_with_output_name_prefix(tmp_path):
    root1 = make_dataset(tmp_path / "yolo1")
    root2 = make_dataset(tmp_path / "yolo2")
    dataset1 = load_yolo_dataset(root1)
    dataset2 = load_yolo_dataset(root2)

    merged, report = merge_datasets([dataset1, dataset2], root=tmp_path / "merged")

    assert report.image_count == 4
    assert merged.annotation_count() == 6
    assert merged.classes.names == ["person", "car"]
    assert sorted(image.file_name for image in merged.images) == ["d0_a.jpg", "d0_b.jpg", "d1_a.jpg", "d1_b.jpg"]


def test_query_by_attribute_and_duplicate_validation(tmp_path):
    root = tmp_path / "attr_yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("sign\n", encoding="utf-8")
    (root / "attribute.yaml").write_text("attributes:\n  defect: [no, yes]\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text(
        "0 1 1 0.5 0.5 0.2 0.2\n0 1 1 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    dataset = load_yolo_dataset(root)

    result = query_by_attribute(dataset, "defect", values=["yes"])
    report = validate_dataset(dataset)

    assert len(result) == 2
    assert report.summary()["warning:duplicate_annotation"] == 1


def test_self_intersecting_polygon_validation(tmp_path):
    root = tmp_path / "poly_bad"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("shape\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text("0 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1\n", encoding="utf-8")
    dataset = load_yolo_dataset(root, task="segment")

    report = validate_dataset(dataset)

    assert report.summary()["warning:polygon_self_intersection"] == 1


def test_set_and_delete_attribute(tmp_path):
    root = tmp_path / "attr_edit_yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("sign\n", encoding="utf-8")
    (root / "attribute.yaml").write_text("attributes:\n  defect: [no, yes]\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text("0 1 0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset = load_yolo_dataset(root)

    edited, report = set_attribute(dataset, "defect", "yes")
    deleted, delete_report = delete_by_attribute(edited, "defect", values=["yes"])

    assert len(report.rows) == 1
    assert edited.images[0].annotations[0].attributes == [1.0]
    assert len(delete_report.rows) == 1
    assert deleted.annotation_count() == 0


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


def test_duplicate_image_hash(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    groups = find_duplicate_images(dataset)

    assert len(groups) == 1
    assert sorted(groups[0].images) == ["a.jpg", "b.jpg"]


def test_find_bad_images(tmp_path):
    root = tmp_path / "bad_images"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "images" / "bad.jpg").write_text("not an image", encoding="utf-8")
    (root / "class.txt").write_text("x\n", encoding="utf-8")
    (root / "labels" / "bad.txt").write_text("", encoding="utf-8")
    dataset = load_yolo_dataset(root)

    issues = find_bad_images(dataset)

    assert len(issues) == 1
    assert issues[0].code == "bad_image"


def test_compare_datasets(tmp_path):
    gt_root = make_dataset(tmp_path / "gt")
    pred_root = tmp_path / "pred"
    (pred_root / "images").mkdir(parents=True)
    (pred_root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "a.jpg")
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "b.jpg")
    (pred_root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    (pred_root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.3 0.9\n1 0.9 0.9 0.1 0.1 0.8\n", encoding="utf-8")
    (pred_root / "labels" / "b.txt").write_text("", encoding="utf-8")
    gt = load_yolo_dataset(gt_root)
    pred = load_yolo_dataset(pred_root)

    rows, summary = compare_datasets(gt, pred, iou_threshold=0.5)

    assert summary == {"tp": 1, "fp": 1, "fn": 2}
    assert len(rows) == 4


def test_pseudo_labels_and_review_pack(tmp_path):
    gt_root = make_dataset(tmp_path / "gt_pack")
    pred_root = tmp_path / "pred_pack"
    (pred_root / "images").mkdir(parents=True)
    (pred_root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "a.jpg")
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "b.jpg")
    (pred_root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    (pred_root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.3 0.9\n1 0.9 0.9 0.1 0.1 0.2\n", encoding="utf-8")
    (pred_root / "labels" / "b.txt").write_text("", encoding="utf-8")
    gt = load_yolo_dataset(gt_root)
    pred = load_yolo_dataset(pred_root)

    pseudo = predictions_to_pseudo_labels(pred, confidence_threshold=0.5)
    rows, _ = compare_datasets(gt, pred, iou_threshold=0.5)
    counts = write_review_pack(rows, gt, tmp_path / "review", statuses={"fn", "fp"})

    assert pseudo.annotation_count() == 1
    assert pseudo.images[0].annotations[0].confidence is None
    assert counts == {"fn": 2, "fp": 1}
    assert (tmp_path / "review" / "fn" / "review.csv").exists()


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


def test_import_coco(tmp_path):
    images_dir = tmp_path / "coco_images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(images_dir / "a.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 7, "name": "sign"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 40, 30]}],
    }
    json_path = tmp_path / "coco.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    dataset = import_coco(json_path, images_dir, out_root=tmp_path / "yolo_coco")

    assert dataset.classes.names == ["sign"]
    assert dataset.annotation_count() == 1
    assert (tmp_path / "yolo_coco" / "labels" / "a.txt").read_text(encoding="utf-8").strip() == "0 0.3 0.35 0.4 0.3"


def test_import_voc_and_dataset_yaml(tmp_path):
    images_dir = tmp_path / "JPEGImages"
    ann_dir = tmp_path / "Annotations"
    images_dir.mkdir()
    ann_dir.mkdir()
    Image.new("RGB", (100, 80), color="white").save(images_dir / "a.jpg")
    (ann_dir / "a.xml").write_text(
        """
<annotation>
  <filename>a.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>car</name>
    <difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    dataset = import_voc_dir(ann_dir, images_dir, out_root=tmp_path / "yolo_voc")
    write_dataset_yaml(dataset.classes, tmp_path / "dataset.yaml", train="images", val="images")

    assert dataset.classes.names == ["car"]
    assert dataset.annotation_count() == 1
    yaml_text = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
    assert "names:" in yaml_text
    assert "car" in yaml_text
