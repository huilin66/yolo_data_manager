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
from yolo_data_manager.dataset.split import class_counts_for_images, split_dataset
from yolo_data_manager.evaluation.compare import compare_datasets
from yolo_data_manager.evaluation.error_analysis import (
    analyze_errors,
    collect_stems_from_source,
    find_duplicate_gt,
    load_error_analysis_dataset,
    write_duplicate_gt_csv,
    write_error_csvs,
    write_error_review_pack,
)
from yolo_data_manager.evaluation.review_pack import write_review_pack
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.layout import detect_layout
from yolo_data_manager.io.validator import validate_dataset
from yolo_data_manager.io.writer import write_yolo_dataset
from yolo_data_manager.stats.compute import compute_stats
from yolo_data_manager.stats.export import write_attribute_csv, write_stats_plots
from yolo_data_manager.scripting import YoloManager, build_task_argv
from yolo_data_manager.vis.renderer import crop_dataset
from yolo_data_manager.vis.renderer import render_dataset


def test_build_python_task_argv():
    argv = build_task_argv(
        "ann.merge_class",
        root=Path("dataset"),
        from_=["crack", "break"],
        to="defect",
        compact=False,
        copy_images=False,
        dry_run=True,
    )

    assert argv == [
        "ann",
        "merge-class",
        "--root",
        "dataset",
        "--from",
        "crack,break",
        "--to",
        "defect",
        "--no-compact",
        "--no-copy-images",
        "--dry-run",
    ]

    stats_argv = build_task_argv(
        "stats",
        root=Path("dataset"),
        plots_dir=Path("stats_plots"),
        stats_list=["image_shape", "box_pos_center"],
    )
    assert stats_argv[-2:] == ["--stats-list", "image_shape,box_pos_center"]

    vis_argv = build_task_argv("vis.draw", root=Path("dataset"), out="vis", workers=4, progress=True)
    assert vis_argv[-3:] == ["--workers", "4", "--progress"]

    error_argv = build_task_argv(
        "eval.error_analysis",
        gt_root=Path("dataset"),
        pred_root=Path("pred"),
        out=Path("error_out"),
        review=True,
        review_workers=4,
        review_progress=True,
        review_progress_leave=False,
    )
    assert "--review" in error_argv
    assert "--review-workers" in error_argv
    assert "--review-progress" in error_argv
    assert "--review-progress-leave" not in error_argv


def test_yolo_manager_methods(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    mgr = YoloManager(root, layout="flat", task="detect")

    assert mgr.root == str(root)
    assert mgr.layout == "flat"

    # check delegates to run_task with auto-filled root
    code = mgr.check()
    assert code == 0

    # stats with output
    code = mgr.stats(out=str(tmp_path / "stats.json"))
    assert code == 0
    assert (tmp_path / "stats.json").exists()

    # query_class with class_ alias
    code = mgr.query_class(class_=["car"], out=str(tmp_path / "cars.csv"))
    assert code == 0

    # vis_crop uses filter_no_attrs → --keep-no-attrs logic
    argv = build_task_argv("vis.crop", root=str(root), out="crops",
                           filter_no_attrs=False)
    assert "--keep-no-attrs" in argv


def test_yolo_manager_check_can_fill_missing_txt(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    (root / "labels" / "b.txt").unlink()
    out = tmp_path / "validation.json"
    mgr = YoloManager(root, layout="flat", task="detect", init_check=False)

    code = mgr.check(out=str(out), fill_missing_txt=True)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert code == 0
    assert (root / "labels" / "b.txt").exists()
    assert payload["fixed"]["missing_txt_created_count"] == 1
    assert any(issue["code"] == "missing_label" and issue["image"] == "b.jpg" for issue in payload["issues"])


def test_yolo_manager_init_check_can_write_to_path(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    out = tmp_path / "init_validation.json"

    YoloManager(root, layout="flat", task="detect", init_layout=False, init_check=out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["summary"] == {}


def test_yolo_manager_init_check_can_fill_missing_txt(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    (root / "labels" / "b.txt").unlink()
    out = tmp_path / "init_validation.json"

    YoloManager(
        root,
        layout="flat",
        task="detect",
        init_layout=False,
        init_check=out,
        init_check_fill_missing_txt=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert (root / "labels" / "b.txt").exists()
    assert payload["fixed"]["missing_txt_created_count"] == 1
    assert any(issue["code"] == "missing_label" and issue["image"] == "b.jpg" for issue in payload["issues"])


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


def test_split_dataset_can_write_absolute_paths(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    relative = split_dataset(dataset, train=1.0, val=0.0, test=0.0, seed=1)
    absolute = split_dataset(dataset, train=1.0, val=0.0, test=0.0, seed=1, absolute_paths=True)

    assert sorted(relative["train"]) == ["a.jpg", "b.jpg"]
    assert sorted(absolute["train"]) == sorted(
        str((root / "images" / name).resolve()) for name in ["a.jpg", "b.jpg"]
    )


def test_split_class_counts_for_images(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)
    absolute_b = [str((root / "images" / "b.jpg").resolve())]

    assert class_counts_for_images(dataset) == {"person": 1, "car": 2}
    assert class_counts_for_images(dataset, absolute_b) == {"person": 0, "car": 1}


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


def test_yolo_manager_merge_class_mapping(tmp_path):
    root = tmp_path / "merge_map_yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("person\ncar\ntruck\nbike\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text(
        "1 0.2 0.2 0.1 0.1\n2 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n",
        encoding="utf-8",
    )
    out = tmp_path / "merged"
    report = tmp_path / "merge_report.csv"
    mgr = YoloManager(root, layout="flat", task="detect", init_check=False)

    code = mgr.ann_merge_class(
        {
            "vehicle": ["car", "truck"],
            "human": ["person"],
        },
        out=str(out),
        compact=False,
        report=str(report),
    )

    assert code == 0
    assert (out / "class.txt").read_text(encoding="utf-8").splitlines() == [
        "person",
        "car",
        "truck",
        "bike",
        "vehicle",
        "human",
    ]
    assert (out / "labels" / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "4 0.2 0.2 0.1 0.1",
        "4 0.4 0.4 0.1 0.1",
        "5 0.6 0.6 0.1 0.1",
    ]
    assert "merge_class" in report.read_text(encoding="utf-8")


def test_filter_by_geometry(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)

    filtered = filter_by_geometry(dataset, min_area=0.05)

    assert filtered.annotation_count() == 1
    assert filtered.images[0].annotations[0].class_id == 0


def test_stats_list_outputs_legacy_plots_and_csv(tmp_path):
    root = make_dataset(tmp_path / "yolo")
    dataset = load_yolo_dataset(root)
    out_dir = tmp_path / "plots"

    write_stats_plots(dataset, out_dir, stats_list=["image_shape", "box_shape_pix", "box_pos_center", "legacy_csv"])
    stats = compute_stats(dataset)

    assert (out_dir / "image_shape.png").exists()
    assert (out_dir / "image_shape.csv").exists()
    assert (out_dir / "box_shape_pix.png").exists()
    assert (out_dir / "box_pos_center.png").exists()
    assert (out_dir / "sta_box.csv").exists()
    assert stats["box_width_pix"]["count"] == 3
    assert stats["box_pos_center_x"]["count"] == 3


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


def test_class_scoped_attribute_full_flow(tmp_path):
    root = tmp_path / "class_attr_yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("sign\nroad\n", encoding="utf-8")
    (root / "attribute.yaml").write_text(
        "attributes:\n"
        "  sign:\n"
        "    defect: [no, yes]\n"
        "  road:\n"
        "    material: [asphalt, concrete]\n",
        encoding="utf-8",
    )
    (root / "labels" / "a.txt").write_text(
        "0 1 1 0.5 0.5 0.2 0.2\n1 1 1 0.3 0.3 0.2 0.2\n",
        encoding="utf-8",
    )
    dataset = load_yolo_dataset(root)

    query = query_by_attribute(dataset, "defect", values=["yes"])
    edited, edit_report = set_attribute(dataset, "material", "asphalt", classes=["road"])
    stats = compute_stats(edited)
    write_attribute_csv(edited, tmp_path / "attributes.csv")
    render_dataset(edited, tmp_path / "vis", show_attributes=True, filter_no_attributes=True)
    saved = crop_dataset(edited, tmp_path / "crops", by_attribute=True)

    assert len(query) == 1
    assert edited.images[0].annotations[1].attributes == [0.0]
    assert len(edit_report.rows) == 1
    assert stats["attribute_counts"]["defect"]["yes"] == 1
    assert stats["class_attribute_counts"]["road"]["material"]["asphalt"] == 1
    assert "material" in (tmp_path / "attributes.csv").read_text(encoding="utf-8")
    assert (tmp_path / "vis" / "a.jpg").exists()
    assert saved == 4
    assert (tmp_path / "crops" / "sign" / "defect-yes" / "a_0.jpg").exists()


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

    render_dataset(dataset, tmp_path / "vis_mt", workers=2, progress=True)
    saved = crop_dataset(dataset, tmp_path / "crops", workers=2, progress=True)
    assert saved == 3
    assert (tmp_path / "vis_mt" / "a.jpg").exists()
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


def test_import_labelme_with_attributes(tmp_path):
    labelme_dir = tmp_path / "labelme_attr"
    labelme_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(labelme_dir / "img.jpg")
    attr_file = tmp_path / "attribute.yaml"
    attr_file.write_text(
        "attributes:\n"
        "  surface:\n"
        "    defect: [no, yes]\n",
        encoding="utf-8",
    )
    data = {
        "imagePath": "img.jpg",
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [
            {
                "label": "surface",
                "shape_type": "rectangle",
                "points": [[10, 20], [50, 60]],
                "attributes": {"defect": "yes"},
            }
        ],
    }
    (labelme_dir / "img.json").write_text(json.dumps(data), encoding="utf-8")

    dataset = import_labelme_dir(labelme_dir, out_root=tmp_path / "yolo_attr_out", task="detect", attribute_file=attr_file)

    assert dataset.annotation_attributes(dataset.images[0].annotations[0]) == {"defect": "yes"}
    assert (tmp_path / "yolo_attr_out" / "labels" / "img.txt").read_text(encoding="utf-8").startswith("0 1 1 ")


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


def test_error_analysis(tmp_path):
    """Fine-grained error analysis: TP, FP sub-types, FN sub-types, duplicate GT."""
    # -- GT dataset: 2 images, one with a duplicate GT pair --
    gt_root = tmp_path / "gt_ea"
    (gt_root / "images").mkdir(parents=True)
    (gt_root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(gt_root / "images" / "a.jpg")
    Image.new("RGB", (100, 80), color="white").save(gt_root / "images" / "b.jpg")
    (gt_root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    # a.jpg: person + car (well-separated)
    (gt_root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"
        "1 0.3 0.3 0.3 0.3\n",
        encoding="utf-8",
    )
    # b.jpg: two nearly-identical cars (duplicate GT)
    (gt_root / "labels" / "b.txt").write_text(
        "1 0.2 0.2 0.3 0.2\n"
        "1 0.205 0.205 0.29 0.19\n",
        encoding="utf-8",
    )

    # -- Pred dataset --
    pred_root = tmp_path / "pred_ea"
    (pred_root / "images").mkdir(parents=True)
    (pred_root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "a.jpg")
    Image.new("RGB", (100, 80), color="white").save(pred_root / "images" / "b.jpg")
    (pred_root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    # a.jpg: TP + class_error_pred + background_fp
    (pred_root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.3 0.4 0.90\n"
        "1 0.5 0.5 0.3 0.4 0.60\n"
        "1 0.85 0.15 0.15 0.15 0.50\n",
        encoding="utf-8",
    )
    # b.jpg: TP matching first GT car; 2nd GT car unmatched -> FN
    (pred_root / "labels" / "b.txt").write_text(
        "1 0.2 0.2 0.3 0.2 0.85\n",
        encoding="utf-8",
    )

    gt = load_yolo_dataset(gt_root, task="detect")
    pred = load_yolo_dataset(pred_root, task="detect")

    # --- analyze_errors ---
    rows, summary = analyze_errors(gt, pred, match_iou=0.5, low_iou=0.1)

    assert summary.get("tp") == 2, f"expected 2 TP, got {summary}"
    assert summary.get("class_error_pred") == 1, f"expected 1 class_error_pred, got {summary}"
    assert summary.get("background_fp") == 1, f"expected 1 background_fp, got {summary}"
    assert summary.get("fn_low_iou") == 1, f"expected 1 fn_low_iou, got {summary}"
    assert summary.get("fn_no_pred") == 1, f"expected 1 fn_no_pred, got {summary}"

    # Status-level counts
    tp_rows = [r for r in rows if r.status == "tp"]
    fp_rows = [r for r in rows if r.status == "fp"]
    fn_rows = [r for r in rows if r.status == "fn"]
    assert len(tp_rows) == 2
    assert len(fp_rows) == 2  # class_error_pred + background_fp
    assert len(fn_rows) == 2  # fn_no_pred (a.jpg car) + fn_low_iou (b.jpg duplicate)

    # Verify confidence is captured
    assert tp_rows[0].pred_conf is not None

    # --- find_duplicate_gt ---
    dup_rows = find_duplicate_gt(gt, duplicate_iou=0.9)
    assert len(dup_rows) == 1
    assert dup_rows[0].type == "same_class_duplicate"
    assert dup_rows[0].iou >= 0.9

    # --- CSV output (smoke test) ---
    write_error_csvs(rows, tmp_path / "error_out")
    write_duplicate_gt_csv(dup_rows, tmp_path / "error_out")
    review_counts = write_error_review_pack(
        rows,
        gt,
        pred,
        tmp_path / "error_out",
        workers=2,
        progress=True,
        progress_leave=False,
    )
    assert (tmp_path / "error_out" / "fp_report.csv").exists()
    assert (tmp_path / "error_out" / "fn_report.csv").exists()
    assert (tmp_path / "error_out" / "class_error.csv").exists()
    assert (tmp_path / "error_out" / "tp_report.csv").exists()
    assert (tmp_path / "error_out" / "duplicate_gt.csv").exists()
    assert (tmp_path / "error_out" / "false_positive_background.csv").exists()
    assert (tmp_path / "error_out" / "false_negative_missed_gt.csv").exists()
    confusion_dir = tmp_path / "error_out" / "review" / "pred_gt" / "pred_car_gt_person"
    assert review_counts["pred_gt/pred_car_gt_person"] == 1
    assert any((confusion_dir / "images").iterdir())
    assert any((confusion_dir / "crops").iterdir())
    assert (tmp_path / "error_out" / "review" / "pred_gt" / "confusion_matrix.png").exists()


def test_error_analysis_label_dirs_val_source_and_id_names(tmp_path):
    gt_labels = tmp_path / "gt_labels"
    pred_labels = tmp_path / "pred_labels"
    gt_labels.mkdir()
    pred_labels.mkdir()
    names = tmp_path / "names.txt"
    names.write_text("0 person\n1 car\n", encoding="utf-8")
    val_txt = tmp_path / "val.txt"
    val_txt.write_text("/images/a.jpg\n", encoding="utf-8")

    (gt_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (gt_labels / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    (pred_labels / "b.txt").write_text("1 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    stems = collect_stems_from_source(val_txt)
    gt = load_error_analysis_dataset(gt_labels, task="detect", class_file=names, stems=stems)
    pred = load_error_analysis_dataset(pred_labels, task="detect", class_file=names, stems=stems)
    rows, summary = analyze_errors(gt, pred, match_iou=0.5, low_iou=0.1)

    assert stems == {"a"}
    assert [image.stem for image in gt.images] == ["a"]
    assert [image.stem for image in pred.images] == ["a"]
    assert gt.class_name(0) == "person"
    assert summary == {"tp": 1}
    assert rows[0].class_name == "person"


def test_yolo_manager_error_analysis_defaults_to_manager_root(tmp_path, monkeypatch):
    root = make_dataset(tmp_path / "yolo")
    (root / "val.txt").write_text("images/a.jpg\n", encoding="utf-8")
    captured = {}

    def fake_run_task(task_name, **params):
        captured["command"] = task_name
        captured.update(params)
        return 0

    import yolo_data_manager.scripting as scripting

    monkeypatch.setattr(scripting, "run_task", fake_run_task)
    mgr = YoloManager(root, layout="flat", task="detect", init_check=False, init_layout=False)

    code = mgr.eval_error_analysis(
        pred_root="pred_labels",
        out="error_report",
        review=True,
        review_workers=4,
        review_progress=True,
    )

    assert code == 0
    assert captured["command"] == "eval.error_analysis"
    assert captured["gt_root"] == str(root)
    assert captured["pred_root"] == "pred_labels"
    assert captured["val_source"] == str(root / "val.txt")
    assert captured["class_file"] == str(root / "class.txt")
    assert captured["review"] is True
    assert captured["review_workers"] == 4
    assert captured["review_progress"] is True
    assert captured["review_progress_leave"] is False
