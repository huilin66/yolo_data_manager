import json
from pathlib import Path

from yolo_data_manager.io.backup import LabelBackup, restore_label_backup


def test_restore_label_backup_saves_current_state_first(tmp_path):
    root = tmp_path / "dataset"
    label_path = root / "labels" / "image.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    source_backup = LabelBackup(root, method="test.source")
    source_backup.backup(label_path)
    source_backup.write_metadata(result={"changed": 1})
    original = label_path.read_text(encoding="utf-8")

    label_path.write_text("1 0.9 0.9 0.2 0.2\n", encoding="utf-8")
    result = restore_label_backup(
        root,
        source_backup.timestamp,
        workers=1,
    )

    assert label_path.read_text(encoding="utf-8") == original
    assert result["restored_files"] == 1
    assert result["current_backup_files"] == 1
    current_metadata = json.loads(
        (root / "labels_backup" / Path(result["current_backup_dir"]).name / "backup_metadata.json")
        .read_text(encoding="utf-8")
    )
    assert current_metadata["method"] == "ann.restore_backup"
    assert current_metadata["result"]["restored_timestamp"] == source_backup.timestamp
