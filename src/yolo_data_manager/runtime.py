from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import os
from pathlib import Path
import sys
from typing import TypeVar

DEFAULT_WORKERS = 8
DEFAULT_PROGRESS = True
DEFAULT_PROGRESS_LEAVE = False

T = TypeVar("T")


def normalize_workers(workers: int | None) -> int:
    return max(1, int(workers if workers is not None else DEFAULT_WORKERS))


def progress_stage(desc: str, *, enabled: bool) -> None:
    """Emit a persistent phase label before a potentially long operation.

    Live progress bars commonly use ``leave=False`` to keep terminal output
    compact.  The phase label deliberately remains visible so users are never
    left with an unexplained blank interval between operations.
    """

    if enabled:
        print(f"{desc}...", file=sys.stderr, flush=True)


def iter_progress(
    items: Iterable[T],
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    leave: bool = DEFAULT_PROGRESS_LEAVE,
) -> Iterable[T]:
    if not enabled:
        return items
    progress_stage(desc, enabled=True)
    try:
        from tqdm import tqdm
    except ImportError:
        return _simple_progress(items, total=total, desc=desc)
    return tqdm(items, total=total, desc=desc, leave=leave)


def create_progress_bar(
    *,
    total: int | None,
    desc: str,
    enabled: bool,
    leave: bool = DEFAULT_PROGRESS_LEAVE,
):
    """Create a manually updated progress bar for parallel work.

    The bar is created before worker tasks are submitted so callers can show
    ``0/N`` immediately, even when the first task is slow or blocked on I/O.
    """

    if not enabled:
        return _NoopProgress()
    progress_stage(desc, enabled=True)
    try:
        from tqdm import tqdm
    except ImportError:
        return _SimpleProgress(desc=desc, total=total)
    return tqdm(total=total, desc=desc, leave=leave)


def scan_matching_files(
    root: Path,
    matcher: Callable[[Path], bool],
    *,
    progress: bool = False,
    progress_leave: bool = DEFAULT_PROGRESS_LEAVE,
    desc: str = "scan files",
) -> list[Path]:
    progress_stage(desc, enabled=progress)
    if not root.exists():
        return []

    paths: list[Path] = []
    progress_bar = dynamic_file_progress(desc=desc, leave=progress_leave) if progress else None
    try:
        for dirpath, _, filenames in os.walk(root):
            if progress_bar is not None:
                progress_bar.total = (progress_bar.total or 0) + len(filenames)
                progress_bar.refresh()
            for filename in filenames:
                path = Path(dirpath) / filename
                if matcher(path):
                    paths.append(path)
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return sorted(paths)


def count_matching_files(
    root: Path,
    matcher: Callable[[Path], bool],
    *,
    progress: bool = False,
    progress_leave: bool = DEFAULT_PROGRESS_LEAVE,
    desc: str = "scan files",
) -> int:
    return len(
        scan_matching_files(
            root,
            matcher,
            progress=progress,
            progress_leave=progress_leave,
            desc=desc,
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
                print(f"{desc}: {idx}")
        elif idx == 1 or idx == total or idx % step == 0:
            print(f"{desc}: {idx}/{total}")
        yield item


class _SimpleDynamicProgress:
    def __init__(self, *, desc: str) -> None:
        self.desc = desc
        self.total = 0
        self.n = 0

    def update(self, value: int) -> None:
        self.n += value
        if self.n == 1 or self.n == self.total or self.n % 100 == 0:
            print(f"{self.desc}: {self.n}/{self.total}")

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
        print(f"{self.desc}: {self.n}/{total}", file=sys.stderr, flush=True)


class _NoopProgress:
    def update(self, value: int = 1) -> None:
        return None

    def close(self) -> None:
        return None
