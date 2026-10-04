import json
from pathlib import Path

from PIL import Image

from yolo_data_manager import YoloManager


def operation_snapshots(backup_root: Path) -> list[Path]:
    return sorted(
        path for path in backup_root.iterdir() if path.name != "source_labels"
    )


def _make_class_map_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "class.txt").write_text(
        "Hollow High Risk\n"
        "Hollow Low Risk\n"
        "Leakage High Risk\n"
        "background\n"
        "Hollow High Risk Line\n"
        "Temperature Medium Risk\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    Image.new("RGB", (20, 20), "white").save(root / "images" / "b.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.1 0.1\n"
        "1 0.2 0.2 0.1 0.1\n"
        "2 0.3 0.3 0.1 0.1\n"
        "3 0.4 0.4 0.1 0.1\n"
        "4 0.5 0.5 0.1 0.1\n"
        "5 0.6 0.6 0.1 0.1\n",
        encoding="utf-8",
    )
    (root / "labels" / "b.txt").write_text(
        "3 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    return root


def test_ann_update_from_map_updates_labels_classes_and_backups(tmp_path):
    root = _make_class_map_dataset(tmp_path / "dataset")
    manager = YoloManager(root, layout="flat", task="detect", init_check=False, init_layout=False)

    code = manager.ann_update_from_map(
        {
            "merge": {
                "Hollow Confirmed": ["Hollow High Risk"],
                "Hollow Suspected": ["Hollow Low Risk"],
                "Leakage": ["Leakage High Risk"],
            },
            "drop": [
                "background",
                "Hollow High Risk Line",
                "Temperature Medium Risk",
            ],
        },
        progress=False,
    )

    assert code == 0
    assert (root / "class.txt").read_text(encoding="utf-8").splitlines() == [
        "Hollow Confirmed",
        "Hollow Suspected",
        "Leakage",
    ]
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.1 0.1 0.1 0.1",
        "1 0.2 0.2 0.1 0.1",
        "2 0.3 0.3 0.1 0.1",
    ]
    assert (root / "labels" / "b.txt").read_text(encoding="utf-8") == ""
    assert (root / "images" / "b.jpg").is_file()

    snapshots = operation_snapshots(root / "labels_backup")
    assert len(snapshots) == 1
    assert sorted(
        path.relative_to(snapshots[0]).as_posix()
        for path in snapshots[0].rglob("*")
        if path.is_file()
    ) == [
        "backup_metadata.json",
        "class.txt",
        "labels/a.txt",
        "labels/b.txt",
    ]
    metadata = json.loads(
        (snapshots[0] / "backup_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["method"] == "ann_update_from_map"
    assert metadata["status"] == "completed"
    assert metadata["backup_files"] == 3
    assert metadata["result"]["changed"] == 7
    assert (snapshots[0] / "class.txt").read_text(encoding="utf-8").splitlines()[0] == "Hollow High Risk"


def test_ann_update_from_map_updates_yaml_class_source_and_backups_it(tmp_path):
    root = tmp_path / "yaml_dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images\nval: images\nnames: [old, drop]\nnc: 2\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    manager = YoloManager(yaml_path, layout="flat", task="detect", init_check=False, init_layout=False)
    manager.ann_update_from_map(
        {"rename": {"old": "new"}, "drop": ["drop"]},
        progress=False,
    )

    text = yaml_path.read_text(encoding="utf-8")
    assert "names:\n- new\n" in text
    assert "nc: 1" in text
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.5 0.5 0.2 0.2",
    ]
    snapshots = operation_snapshots(root / "labels_backup")
    assert len(snapshots) == 1
    assert (snapshots[0] / "dataset.yaml").is_file()
