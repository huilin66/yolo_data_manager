from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import os
from pathlib import Path
from typing import TypeVar

DEFAULT_WORKERS = 8
DEFAULT_PROGRESS = True
DEFAULT_PROGRESS_LEAVE = False

T = TypeVar("T")


def normalize_workers(workers: int | None) -> int:
    return max(1, int(workers if workers is not None else DEFAULT_WORKERS))


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
    try:
        from tqdm import tqdm
    except ImportError:
        return _simple_progress(items, total=total, desc=desc)
    return tqdm(items, total=total, desc=desc, leave=leave)


def scan_matching_files(
    root: Path,
    matcher: Callable[[Path], bool],
    *,
    progress: bool = False,
    progress_leave: bool = DEFAULT_PROGRESS_LEAVE,
    desc: str = "scan files",
) -> list[Path]:
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
