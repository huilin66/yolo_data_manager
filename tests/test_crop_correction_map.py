import json
from pathlib import Path

from PIL import Image

from yolo_data_manager.scripting import YoloManager, build_task_argv


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "class.txt").write_text("person\ncar\n", encoding="utf-8")
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    Image.new("RGB", (20, 20), "white").save(root / "images" / "b.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    (root / "labels" / "b.txt").write_text(
        "1 0.3 0.3 0.2 0.2\n",
        encoding="utf-8",
    )
    return root


def test_crop_class_mapping_uses_one_backup_snapshot(tmp_path):
    root = _make_dataset(tmp_path / "dataset")
    to_car = tmp_path / "crops_car"
    to_person = tmp_path / "crops_person"
    to_delete = tmp_path / "crops_delete"
    for crop_dir in (to_car, to_person, to_delete):
        crop_dir.mkdir()
    Image.new("RGB", (5, 5), "white").save(to_car / "a_1.jpg")
    Image.new("RGB", (5, 5), "white").save(to_person / "a_2.jpg")
    Image.new("RGB", (5, 5), "white").save(to_delete / "b_1.jpg")

    backup_root = tmp_path / "backups"
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    code = manager.ann_correct_from_crops(
        {
            to_car: "car",
            to_person: "person",
            to_delete: None,
        },
        backup_dir=backup_root,
        progress=False,
    )

    assert code == 0
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.5 0.5 0.2 0.2",
        "0 0.4 0.4 0.2 0.2",
    ]
    assert (root / "labels" / "b.txt").read_text(encoding="utf-8") == ""
    snapshots = list(backup_root.iterdir())
    assert len(snapshots) == 1
    assert sorted(path.name for path in snapshots[0].rglob("*.txt")) == ["a.txt", "b.txt"]


def test_crop_class_mapping_is_encoded_as_one_task_argument(tmp_path):
    crops = tmp_path / "crops"
    argv = build_task_argv(
        "ann.correct_from_crops",
        root=tmp_path / "dataset",
        crops_dir={crops: "Leakage", tmp_path / "delete": None},
        progress=False,
    )

    encoded = argv[argv.index("--crops-dir") + 1]
    assert json.loads(encoded) == {
        str(crops): "Leakage",
        str(tmp_path / "delete"): None,
    }
    assert "--to" not in argv


def test_error_crop_class_mapping_uses_one_backup_snapshot(tmp_path):
    root = _make_dataset(tmp_path / "error_dataset")
    to_car = tmp_path / "error_crops_car"
    to_person = tmp_path / "error_crops_person"
    to_delete = tmp_path / "error_crops_delete"
    for crop_dir in (to_car, to_person, to_delete):
        crop_dir.mkdir()
    Image.new("RGB", (5, 5), "white").save(to_car / "a_pred1_gt1.jpg")
    Image.new("RGB", (5, 5), "white").save(to_person / "a_pred2_gt2.jpg")
    Image.new("RGB", (5, 5), "white").save(to_delete / "b_pred1_gt1.jpg")

    backup_root = tmp_path / "error_backups"
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    code = manager.ann_correct_from_error_crops(
        {
            to_car: "car",
            to_person: "person",
            to_delete: None,
        },
        backup_dir=backup_root,
        progress=False,
    )

    assert code == 0
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.5 0.5 0.2 0.2",
        "0 0.4 0.4 0.2 0.2",
    ]
    assert (root / "labels" / "b.txt").read_text(encoding="utf-8") == ""
    snapshots = list(backup_root.iterdir())
    assert len(snapshots) == 1
    assert sorted(path.name for path in snapshots[0].rglob("*.txt")) == ["a.txt", "b.txt"]


def test_error_crop_class_mapping_is_encoded_without_to(tmp_path):
    crops = tmp_path / "error_crops"
    argv = build_task_argv(
        "ann.correct_from_error_crops",
        root=tmp_path / "dataset",
        crops_dir={crops: "Leakage", tmp_path / "delete": None},
        progress=False,
    )

    encoded = argv[argv.index("--crops-dir") + 1]
    assert json.loads(encoded) == {
        str(crops): "Leakage",
        str(tmp_path / "delete"): None,
    }
    assert "--to" not in argv
