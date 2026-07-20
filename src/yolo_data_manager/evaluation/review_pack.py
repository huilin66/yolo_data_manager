from __future__ import annotations

import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from yolo_data_manager.core.models import YoloDataset
from yolo_data_manager.evaluation.compare import CompareRow, write_compare_csv
from yolo_data_manager.runtime import iter_progress, normalize_workers


def write_review_pack(
    rows: list[CompareRow],
    gt: YoloDataset,
    out_dir: str | Path,
    statuses: set[str] | None = None,
    pred: YoloDataset | None = None,
    workers: int = 8,
    progress: bool = False,
    progress_leave: bool = False,
) -> dict[str, int]:
    selected_statuses = statuses or {"fp", "fn"}
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_by_name = {image.file_name: image.path for image in gt.images}
    if pred is not None:
        for image in pred.images:
            image_by_name.setdefault(image.file_name, image.path)
    rows_by_status: dict[str, list[CompareRow]] = defaultdict(list)
    rows_by_status_image: dict[tuple[str, str], list[CompareRow]] = defaultdict(list)

    for row in rows:
        if row.status not in selected_statuses:
            continue
        rows_by_status[row.status].append(row)
        rows_by_status_image[(row.status, row.image)].append(row)

    counts: dict[str, int] = {}
    copy_jobs: list[tuple[Path, Path]] = []
    for status, status_rows in rows_by_status.items():
        status_dir = output / status
        image_dir = status_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        write_compare_csv(status_rows, status_dir / "review.csv")
        for image_name in {row.image for row in status_rows}:
            src = image_by_name.get(image_name)
            if src is not None and src.exists():
                copy_jobs.append((src, image_dir / Path(image_name).name))
        counts[status] = len(status_rows)
    _copy_files(copy_jobs, workers=workers, progress=progress, progress_leave=progress_leave)
    return counts


def _copy_files(jobs: list[tuple[Path, Path]], *, workers: int, progress: bool, progress_leave: bool) -> None:
    worker_count = normalize_workers(workers)
    if worker_count == 1:
        for src, dst in iter_progress(jobs, enabled=progress, total=len(jobs), desc="review pack", leave=progress_leave):
            shutil.copy2(src, dst)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(shutil.copy2, src, dst) for src, dst in jobs]
        for future in iter_progress(as_completed(futures), enabled=progress, total=len(futures), desc="review pack", leave=progress_leave):
            future.result()
