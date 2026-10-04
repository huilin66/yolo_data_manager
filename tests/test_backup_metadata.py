import json

from yolo_data_manager.io.backup import LabelBackup


def test_label_backup_writes_operation_metadata(tmp_path):
    dataset_root = tmp_path / "dataset"
    label_path = dataset_root / "labels" / "image.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    backup = LabelBackup(
        dataset_root,
        method="test.operation",
    )
    backup.backup(label_path)
    metadata_path = backup.write_metadata(
        result={"changed": 1, "action": "test"},
    )

    assert metadata_path == backup.snapshot_dir / "backup_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "test.operation"
    assert metadata["status"] == "completed"
    assert metadata["backup_files"] == 1
    assert metadata["files"] == ["labels/image.txt"]
    assert metadata["result"] == {"changed": 1, "action": "test"}
    assert metadata["created_at"]
    assert metadata["updated_at"]
