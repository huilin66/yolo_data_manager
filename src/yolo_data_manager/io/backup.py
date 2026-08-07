"""Timestamped backups for YOLO label files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from threading import Lock


class LabelBackup:
    """Copy label files into one timestamped snapshot directory."""

    def __init__(self, dataset_root: str | Path, backup_dir: str | Path | None = None) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.base_dir = (
            Path(backup_dir)
            if backup_dir is not None
            else self.dataset_root / "labels_backup"
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.snapshot_dir = self.base_dir / self.timestamp
        self._copied: set[Path] = set()
        self._lock = Lock()

    @property
    def count(self) -> int:
        return len(self._copied)

    def backup(self, label_path: str | Path) -> None:
        source = Path(label_path).resolve()
        with self._lock:
            if source in self._copied or not source.is_file():
                return
            try:
                relative = source.relative_to(self.dataset_root)
            except ValueError:
                relative = Path("external") / source.name

            destination = self.snapshot_dir / relative
            if destination.exists():
                stem = destination.stem
                suffix = destination.suffix
                counter = 1
                while destination.exists():
                    destination = destination.with_name(f"{stem}_{counter}{suffix}")
                    counter += 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self._copied.add(source)
