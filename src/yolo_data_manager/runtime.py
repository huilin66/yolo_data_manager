from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import TypeVar

from yolo_data_manager.logging_utils import console_event

DEFAULT_WORKERS = 8
DEFAULT_PROGRESS = True
DEFAULT_PROGRESS_LEAVE = False

T = TypeVar("T")


@dataclass(frozen=True)
class ProgressUpdate:
    """A transport-friendly snapshot of one progress stage.

    The callback is intentionally independent from ``tqdm``.  Existing
    terminal progress bars keep their current behavior, while callers such as
    the web server can receive the same timing information as structured data.
    """

    stage: str
    current: int
    total: int | None
    percent: float | None
    elapsed_seconds: float
    rate: float | None
    eta_seconds: float | None
    unit: str = "item"
    done: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "elapsed_seconds": self.elapsed_seconds,
            "rate": self.rate,
            "eta_seconds": self.eta_seconds,
            "unit": self.unit,
            "done": self.done,
        }


ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressTracker:
    """Report progress without changing the underlying processing loop."""

    def __init__(
        self,
        *,
        stage: str,
        total: int | None,
        callback: ProgressCallback | None,
        unit: str = "item",
        min_interval: float = 0.25,
    ) -> None:
        self.stage = stage
        self.current = 0
        self.total = total
        self.unit = unit
        self._callback = callback
        self._min_interval = max(0.0, min_interval)
        self._started = time.perf_counter()
        self._last_emit = 0.0
        self._closed = False
        self._emit(force=True)

    def update(self, value: int = 1) -> None:
        if self._closed:
            return
        self.current += value
        reached_total = self.total is not None and self.current >= self.total
        self._emit(force=reached_total)

    def set_total(self, total: int | None) -> None:
        if self._closed:
            return
        self.total = total
        self._emit()

    def close(self) -> None:
        if self._closed:
            return
        if self.total is None:
            self.total = self.current
        self._emit(force=True, done=True)
        self._closed = True

    def _emit(self, *, force: bool = False, done: bool = False) -> None:
        if self._callback is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_emit < self._min_interval:
            return

        elapsed = max(0.0, now - self._started)
        rate = self.current / elapsed if elapsed > 0 and self.current > 0 else None
        if self.total is None:
            percent = None
            eta = None
        elif self.total <= 0:
            percent = 100.0
            eta = 0.0
        else:
            percent = min(100.0, self.current * 100.0 / self.total)
            eta = (
                max(0.0, self.total - self.current) / rate
                if rate and self.current < self.total
                else 0.0 if self.current >= self.total else None
            )

        update = ProgressUpdate(
            stage=self.stage,
            current=self.current,
            total=self.total,
            percent=percent,
            elapsed_seconds=elapsed,
            rate=rate,
            eta_seconds=eta,
            unit=self.unit,
            done=done,
        )
        self._last_emit = now
        try:
            self._callback(update)
        except Exception:  # noqa: BLE001 - progress reporting must never break processing
            self._callback = None


class _ProgressProxy:
    """Keep the existing progress-bar API while also reporting snapshots."""

    def __init__(self, inner, tracker: ProgressTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    def update(self, value: int = 1) -> None:
        self._inner.update(value)
        self._tracker.update(value)

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._tracker.close()


def normalize_workers(workers: int | None) -> int:
    return max(1, int(workers if workers is not None else DEFAULT_WORKERS))


def progress_stage(desc: str, *, enabled: bool) -> None:
    """Emit a persistent phase label before a potentially long operation.

    Live progress bars commonly use ``leave=False`` to keep terminal output
    compact.  The phase label deliberately remains visible so users are never
    left with an unexplained blank interval between operations.
    """

    if enabled:
        console_event("INFO", f"{desc}...")


def iter_progress(
    items: Iterable[T],
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    leave: bool = DEFAULT_PROGRESS_LEAVE,
    progress_callback: ProgressCallback | None = None,
) -> Iterable[T]:
    tracked_items: Iterable[T] = items
    if enabled:
        progress_stage(desc, enabled=True)
        try:
            from tqdm import tqdm
        except ImportError:
            tracked_items = _simple_progress(items, total=total, desc=desc)
        else:
            tracked_items = tqdm(items, total=total, desc=desc, leave=leave)

    if progress_callback is None:
        return tracked_items

    tracker = ProgressTracker(
        stage=desc,
        total=total,
        callback=progress_callback,
    )

    def _tracked() -> Iterator[T]:
        try:
            for item in tracked_items:
                yield item
                tracker.update()
        finally:
            tracker.close()

    return _tracked()


def create_progress_bar(
    *,
    total: int | None,
    desc: str,
    enabled: bool,
    leave: bool = DEFAULT_PROGRESS_LEAVE,
    progress_callback: ProgressCallback | None = None,
):
    """Create a manually updated progress bar for parallel work.

    The bar is created before worker tasks are submitted so callers can show
    ``0/N`` immediately, even when the first task is slow or blocked on I/O.
    """

    if not enabled:
        progress_bar = _NoopProgress()
    else:
        progress_stage(desc, enabled=True)
        try:
            from tqdm import tqdm
        except ImportError:
            progress_bar = _SimpleProgress(desc=desc, total=total)
        else:
            progress_bar = tqdm(total=total, desc=desc, leave=leave)

    if progress_callback is None:
        return progress_bar
    return _ProgressProxy(
        progress_bar,
        ProgressTracker(stage=desc, total=total, callback=progress_callback),
    )


def scan_matching_files(
    root: Path,
    matcher: Callable[[Path], bool],
    *,
    progress: bool = False,
    progress_leave: bool = DEFAULT_PROGRESS_LEAVE,
    desc: str = "scan files",
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    progress_stage(desc, enabled=progress)
    tracker = ProgressTracker(
        stage=desc,
        total=None,
        callback=progress_callback,
        unit="file",
    ) if progress_callback is not None else None
    if not root.exists():
        if tracker is not None:
            tracker.close()
        return []

    paths: list[Path] = []
    progress_bar = dynamic_file_progress(desc=desc, leave=progress_leave) if progress else None
    scanned_files = 0
    try:
        for dirpath, _, filenames in os.walk(root):
            scanned_files += len(filenames)
            if tracker is not None:
                tracker.set_total(scanned_files)
            if progress_bar is not None:
                progress_bar.total = (progress_bar.total or 0) + len(filenames)
                progress_bar.refresh()
            for filename in filenames:
                path = Path(dirpath) / filename
                if matcher(path):
                    paths.append(path)
                if progress_bar is not None:
                    progress_bar.update(1)
                if tracker is not None:
                    tracker.update()
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if tracker is not None:
            tracker.set_total(scanned_files)
            tracker.close()
    return sorted(paths)


def count_matching_files(
    root: Path,
    matcher: Callable[[Path], bool],
    *,
    progress: bool = False,
    progress_leave: bool = DEFAULT_PROGRESS_LEAVE,
    desc: str = "scan files",
    progress_callback: ProgressCallback | None = None,
) -> int:
    return len(
        scan_matching_files(
            root,
            matcher,
            progress=progress,
            progress_leave=progress_leave,
            desc=desc,
            progress_callback=progress_callback,
        )
    )


def dynamic_file_progress(*, desc: str, leave: bool):
    try:
        from tqdm import tqdm
    except ImportError:
        return _SimpleDynamicProgress(desc=desc)
    return tqdm(total=0, desc=desc, leave=leave, unit="file")


def _simple_progress(items: Iterable[T], *, total: int | None, desc: str) -> Iterator[T]:
    step = max(1, (total or 20) // 20)
    for idx, item in enumerate(items, start=1):
        if total is None:
            if idx == 1 or idx % 100 == 0:
                console_event("INFO", f"{desc}: {idx}", stream=sys.stdout)
        elif idx == 1 or idx == total or idx % step == 0:
            console_event("INFO", f"{desc}: {idx}/{total}", stream=sys.stdout)
        yield item


class _SimpleDynamicProgress:
    def __init__(self, *, desc: str) -> None:
        self.desc = desc
        self.total = 0
        self.n = 0

    def update(self, value: int) -> None:
        self.n += value
        if self.n == 1 or self.n == self.total or self.n % 100 == 0:
            console_event("INFO", f"{self.desc}: {self.n}/{self.total}", stream=sys.stdout)

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


class _SimpleProgress:
    def __init__(self, *, desc: str, total: int | None) -> None:
        self.desc = desc
        self.total = total
        self.n = 0
        self.step = max(1, (total or 20) // 20)
        self._print()

    def update(self, value: int = 1) -> None:
        self.n += value
        if self.n == 1 or self.n == self.total or self.n % self.step == 0:
            self._print()

    def close(self) -> None:
        return None

    def _print(self) -> None:
        total = "?" if self.total is None else str(self.total)
        console_event("INFO", f"{self.desc}: {self.n}/{total}")


class _NoopProgress:
    def update(self, value: int = 1) -> None:
        return None

    def close(self) -> None:
        return None
