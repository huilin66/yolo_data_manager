"""Timestamped backups for YOLO label files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from threading import Lock
from typing import Any

from yolo_data_manager.logging_utils import current_operation
from yolo_data_manager.runtime import iter_progress, normalize_workers


BACKUP_METADATA_NAME = "backup_metadata.json"
_METADATA_LOCK = Lock()


def _jsonable(value: Any) -> Any:
    """Convert common project values into JSON-safe metadata values."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_snapshot_metadata(
    snapshot_dir: str | Path,
    *,
    dataset_root: str | Path,
    method: str | None,
    created_at: str,
    files: Iterable[str],
    result: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> Path:
    """Write standard metadata for one timestamped backup snapshot."""

    snapshot_path = Path(snapshot_dir)
    relative_files = sorted({str(path).replace("\\", "/") for path in files})
    payload = {
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "status": status,
        "method": method or "label_backup",
        "dataset_root": str(Path(dataset_root).resolve()),
        "snapshot_dir": str(snapshot_path.resolve()),
        "backup_files": len(relative_files),
        "files": relative_files,
        "result": _jsonable(dict(result or {})),
    }
    metadata_path = snapshot_path / BACKUP_METADATA_NAME
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    with _METADATA_LOCK:
        snapshot_path.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(metadata_path)
    return metadata_path


def restore_label_backup(
    dataset_root: str | Path,
    timestamp: str | Path,
    *,
    backup_dir: str | Path | None = None,
    backup_current: bool = True,
    dry_run: bool = False,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, Any]:
    """Restore files from one timestamped label-backup snapshot.

    ``timestamp`` may be a snapshot directory name or a direct snapshot path.
    The current versions of the files being restored are backed up first by
    default, making the restore operation reversible.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    backup_root = (
        Path(backup_dir).expanduser()
        if backup_dir is not None
        else root / "labels_backup"
    )
    requested = Path(timestamp).expanduser()
    snapshot = (
        requested
        if requested.is_dir()
        else backup_root / requested
    )
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        available = (
            sorted(path.name for path in backup_root.iterdir() if path.is_dir())
            if backup_root.is_dir()
            else []
        )
        suffix = f" Available snapshots: {', '.join(available)}" if available else ""
        raise FileNotFoundError(
            f"backup snapshot not found: {snapshot}.{suffix}"
        )

    metadata = _read_backup_metadata(snapshot)
    metadata_root = metadata.get("dataset_root")
    if metadata_root:
        recorded_root = Path(str(metadata_root)).expanduser().resolve()
        if recorded_root != root:
            raise ValueError(
                "backup dataset root does not match the requested dataset root: "
                f"{recorded_root} != {root}"
            )

    entries, missing = _backup_restore_entries(snapshot, root)
    current_backup: LabelBackup | None = None
    if not dry_run and backup_current:
        current_backup = LabelBackup(
            root,
            backup_root,
            method="ann.restore_backup",
        )
        for _relative, destination, _source in entries:
            current_backup.backup(destination)

    if not dry_run:
        _copy_restore_entries(
            entries,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )

    result: dict[str, Any] = {
        "action": "restore_backup",
        "dataset_root": str(root),
        "snapshot": str(snapshot),
        "timestamp": snapshot.name,
        "dry_run": dry_run,
        "backup_current": backup_current,
        "files": len(entries),
        "restored_files": 0 if dry_run else len(entries),
        "missing_files": missing,
        "missing_count": len(missing),
        "current_backup_dir": (
            str(current_backup.snapshot_dir)
            if current_backup is not None and current_backup.count
            else None
        ),
        "current_backup_files": (
            current_backup.count if current_backup is not None else 0
        ),
    }
    if current_backup is not None and current_backup.count:
        current_backup.write_metadata(
            method="ann.restore_backup",
            result={
                "action": "restore_backup",
                "restored_from": str(snapshot),
                "restored_timestamp": snapshot.name,
                "restored_files": len(entries),
                "missing_files": len(missing),
            },
        )
        result["current_backup_metadata"] = str(current_backup.metadata_path)
    else:
        result["current_backup_metadata"] = None
    return result


def _read_backup_metadata(snapshot: Path) -> dict[str, Any]:
    metadata_path = snapshot / BACKUP_METADATA_NAME
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _backup_restore_entries(
    snapshot: Path,
    dataset_root: Path,
) -> tuple[list[tuple[Path, Path, Path]], list[str]]:
    entries: list[tuple[Path, Path, Path]] = []
    missing: list[str] = []
    for source in sorted(snapshot.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot)
        if relative.name == BACKUP_METADATA_NAME or relative.name.startswith("."):
            continue
        destination = _safe_restore_destination(dataset_root, relative)
        entries.append((relative, destination, source))
    return entries, missing


def _safe_restore_destination(dataset_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe backup path: {relative}")
    destination = (dataset_root / relative).resolve()
    if destination != dataset_root and dataset_root not in destination.parents:
        raise ValueError(f"backup path escapes dataset root: {relative}")
    return destination


def _copy_restore_entries(
    entries: list[tuple[Path, Path, Path]],
    *,
    workers: int,
    progress: bool,
    progress_leave: bool,
) -> None:
    def copy_one(entry: tuple[Path, Path, Path]) -> None:
        _relative, destination, source = entry
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for entry in iter_progress(
            entries,
            enabled=progress,
            total=len(entries),
            desc="restore backup",
            leave=progress_leave,
        ):
            copy_one(entry)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(copy_one, entry) for entry in entries]
        for future in iter_progress(
            as_completed(futures),
            enabled=progress,
            total=len(futures),
            desc="restore backup",
            leave=progress_leave,
        ):
            future.result()


class LabelBackup:
    """Copy label files into one timestamped snapshot directory."""

    def __init__(
        self,
        dataset_root: str | Path,
        backup_dir: str | Path | None = None,
        *,
        method: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.base_dir = (
            Path(backup_dir)
            if backup_dir is not None
            else self.dataset_root / "labels_backup"
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.created_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        self.method = method or current_operation() or "label_backup"
        self.snapshot_dir = self.base_dir / self.timestamp
        self._copied: set[Path] = set()
        self._destinations: dict[Path, Path] = {}
        self._metadata_started = False
        self._lock = Lock()

    @property
    def count(self) -> int:
        return len(self._copied)

    @property
    def metadata_path(self) -> Path:
        """Path of the metadata JSON for this snapshot."""

        return self.snapshot_dir / BACKUP_METADATA_NAME

    @property
    def files(self) -> list[str]:
        """Return backed-up paths relative to the snapshot directory."""

        return sorted(
            destination.relative_to(self.snapshot_dir).as_posix()
            for destination in self._destinations.values()
        )

    def write_metadata(
        self,
        *,
        method: str | None = None,
        result: Mapping[str, Any] | None = None,
        status: str = "completed",
    ) -> Path | None:
        """Write or update metadata after the backup operation completes."""

        if self.count == 0:
            return None
        if method:
            self.method = method
        return write_snapshot_metadata(
            self.snapshot_dir,
            dataset_root=self.dataset_root,
            method=self.method,
            created_at=self.created_at,
            files=self.files,
            result=result,
            status=status,
        )

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
            self._destinations[source] = destination
            if not self._metadata_started:
                self.write_metadata(status="in_progress")
                self._metadata_started = True


__all__ = [
    "BACKUP_METADATA_NAME",
    "LabelBackup",
    "restore_label_backup",
    "write_snapshot_metadata",
]
