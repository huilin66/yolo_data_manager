import json
from pathlib import Path

from yolo_data_manager.io.backup import (
    LabelBackup,
    ensure_source_labels_backup,
    restore_label_backup,
)


def test_first_backup_creates_immutable_source_labels_baseline(tmp_path):
    root = tmp_path / "dataset"
    labels = root / "labels"
    labels.mkdir(parents=True)
    first = labels / "first.txt"
    second = labels / "second.txt"
    first.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    second.write_text("1 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    backup = LabelBackup(root, method="test.source")
    backup.backup(first)
    source_dir = root / "labels_backup" / "source_labels"

    assert source_dir.is_dir()
    assert (source_dir / "labels" / "first.txt").read_text(encoding="utf-8") == (
        "0 0.1 0.1 0.1 0.1\n"
    )
    assert (source_dir / "labels" / "second.txt").read_text(encoding="utf-8") == (
        "1 0.2 0.2 0.1 0.1\n"
    )
    metadata = json.loads(
        (source_dir / "backup_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["method"] == "source_labels"

    first.write_text("9 0.9 0.9 0.9 0.9\n", encoding="utf-8")
    assert ensure_source_labels_backup(root) == source_dir
    assert (source_dir / "labels" / "first.txt").read_text(encoding="utf-8") == (
        "0 0.1 0.1 0.1 0.1\n"
    )


def test_restore_backup_reconstructs_selected_point_in_time(tmp_path):
    root = tmp_path / "dataset"
    labels = root / "labels"
    labels.mkdir(parents=True)
    first = labels / "first.txt"
    second = labels / "second.txt"
    first.write_text("0 initial\n", encoding="utf-8")
    second.write_text("0 initial\n", encoding="utf-8")

    first_backup = LabelBackup(root, method="test.first")
    first_backup.backup(first)
    first_backup.backup(second)
    first.write_text("0 after-first\n", encoding="utf-8")
    second.write_text("0 after-first\n", encoding="utf-8")
    first_backup.write_metadata(result={"step": 1})

    second_backup = LabelBackup(root, method="test.second")
    second_backup.backup(first)
    first.write_text("0 after-second\n", encoding="utf-8")
    second_backup.write_metadata(result={"step": 2})

    result = restore_label_backup(
        root,
        second_backup.timestamp,
        backup_current=False,
        workers=1,
    )

    assert first.read_text(encoding="utf-8") == "0 after-first\n"
    assert second.read_text(encoding="utf-8") == "0 after-first\n"
    assert result["snapshots_applied"] == [second_backup.timestamp]

    restore_source = restore_label_backup(
        root,
        "source_labels",
        backup_current=False,
        workers=1,
    )
    assert first.read_text(encoding="utf-8") == "0 initial\n"
    assert second.read_text(encoding="utf-8") == "0 initial\n"
    assert restore_source["snapshots_applied"] == ["source_labels"]


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
