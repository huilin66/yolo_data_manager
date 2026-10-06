from pathlib import Path

from PIL import Image

from yolo_data_manager import YoloManager, load_yolo_dataset
from yolo_data_manager.annotation.crop_correction import correct_labels_from_crops


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("old\nkeep\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    return root


def test_class_map_out_data_does_not_modify_source(tmp_path):
    root = _make_dataset(tmp_path / "source")
    output = tmp_path / "class_output"
    original_label = (root / "labels" / "a.txt").read_text(encoding="utf-8")

    manager = YoloManager(root, layout="flat", task="detect", init_layout=False, init_check=False)
    assert manager.ann_update_from_map(
        {"rename": {"old": "new"}},
        out_data=output,
        progress=False,
    ) == 0

    assert (root / "labels" / "a.txt").read_text(encoding="utf-8") == original_label
    assert not (root / "labels_backup").exists()
    assert (output / "images" / "a.jpg").is_file()
    assert (output / "class.txt").read_text(encoding="utf-8").splitlines() == ["new", "keep"]
    assert (output / "labels" / "a.txt").read_text(encoding="utf-8") == original_label


def test_crop_correction_out_data_copies_dataset_and_keeps_source(tmp_path):
    root = _make_dataset(tmp_path / "source")
    output = tmp_path / "crop_output"
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.new("RGB", (8, 8), "white").save(crops / "a_1.jpg")
    original_label = (root / "labels" / "a.txt").read_text(encoding="utf-8")

    result, _ = correct_labels_from_crops(
        load_yolo_dataset(root, task="detect"),
        crops,
        "keep",
        out_data=output,
    )

    assert result.changed == 1
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8") == original_label
    assert (output / "images" / "a.jpg").is_file()
    assert (output / "labels" / "a.txt").read_text(encoding="utf-8") == (
        "1 0.5 0.5 0.4 0.4\n"
    )


def test_manager_crop_correction_accepts_out_data(tmp_path):
    root = _make_dataset(tmp_path / "source")
    output = tmp_path / "manager_crop_output"
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.new("RGB", (8, 8), "white").save(crops / "a_1.jpg")

    manager = YoloManager(root, layout="flat", task="detect", init_layout=False, init_check=False)
    assert manager.ann_correct_from_crops(
        crops,
        "keep",
        out_data=output,
        progress=False,
    ) == 0

    assert (root / "labels" / "a.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.4 0.4\n"
    )
    assert (output / "labels" / "a.txt").read_text(encoding="utf-8") == (
        "1 0.5 0.5 0.4 0.4\n"
    )


def test_attribute_map_out_data_does_not_modify_source(tmp_path):
    root = tmp_path / "attribute_source"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    (root / "attribute.yaml").write_text(
        "attributes:\n  defect: [no, yes]\n",
        encoding="utf-8",
    )
    original_label = "0 1 0 0.5 0.5 0.4 0.4\n"
    (root / "labels" / "a.txt").write_text(original_label, encoding="utf-8")
    output = tmp_path / "attribute_output"

    manager = YoloManager(root, layout="flat", task="detect", init_layout=False, init_check=False)
    assert manager.ann_att_update_from_map(
        {"update": {"defect": {"no": "yes"}}},
        out_data=output,
        progress=False,
    ) == 0

    assert (root / "labels" / "a.txt").read_text(encoding="utf-8") == original_label
    assert (output / "labels" / "a.txt").read_text(encoding="utf-8") == (
        "0 1 1 0.5 0.5 0.4 0.4\n"
    )
