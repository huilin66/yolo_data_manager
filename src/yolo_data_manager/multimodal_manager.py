from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

from yolo_data_manager.core.models import TASK_AUTO
from yolo_data_manager.core.multimodal import AlignmentReport, MultimodalYoloDataset
from yolo_data_manager.io.multimodal import load_multimodal_yolo_dataset
from yolo_data_manager.stats.multimodal import compute_multimodal_stats, write_multimodal_stats_plots
from yolo_data_manager.stats.report import write_json_report
from yolo_data_manager.vis.multimodal import crop_multimodal_dataset, render_multimodal_dataset


class MultiModalYoloManager:
    """Stateful entry point for a shared-label, multi-image YOLO dataset.

    The manager lazily loads and caches one :class:`MultimodalYoloDataset`.
    At present it intentionally exposes only multimodal-safe operations:
    alignment check, statistics, annotation rendering, and object crops.
    Other ``YoloManager`` operations need explicit all-modality write semantics
    before they can be added safely.
    """

    def __init__(
        self,
        root: str | Path,
        image_dirs: Sequence[str | Path],
        *,
        image_params: Mapping[str, Mapping[str, object]] | None = None,
        labels_dir: str | Path = "labels",
        label_params: Mapping[str, object] | None = None,
        class_file: str | Path | None = None,
        attribute_file: str | Path | None = None,
        task: str = TASK_AUTO,
        read_image_size: bool = True,
        init_load: bool = False,
        init_check: bool | str | Path = False,
        progress: bool = True,
        progress_leave: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_dirs = tuple(image_dirs)
        self.image_params = dict(image_params or {})
        self.labels_dir = labels_dir
        self.label_params = dict(label_params or {})
        self.class_file = class_file
        self.attribute_file = attribute_file
        self.task = task
        self.read_image_size = read_image_size
        self.progress = progress
        self.progress_leave = progress_leave
        self._dataset: MultimodalYoloDataset | None = None

        if init_load:
            self.load()
        if init_check:
            self.check(out=init_check if isinstance(init_check, (str, Path)) else None)

    @property
    def dataset(self) -> MultimodalYoloDataset:
        return self.load()

    @property
    def alignment_report(self) -> AlignmentReport:
        return self.load().alignment_report

    def load(
        self,
        *,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> MultimodalYoloDataset:
        """Load once, or explicitly rebuild the cached multimodal association."""

        if self._dataset is None or reload:
            self._dataset = load_multimodal_yolo_dataset(
                self.root,
                self.image_dirs,
                image_params=self.image_params,
                labels_dir=self.labels_dir,
                label_params=self.label_params,
                class_file=self.class_file,
                attribute_file=self.attribute_file,
                task=self.task,
                read_image_size=self.read_image_size,
                progress=self.progress if progress is None else progress,
                progress_leave=self.progress_leave if progress_leave is None else progress_leave,
            )
        return self._dataset

    def check(
        self,
        *,
        out: str | Path | None = None,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> dict[str, object]:
        """Return the multimodal association report without rereading cached data."""

        dataset = self.load(reload=reload, progress=progress, progress_leave=progress_leave)
        report = dataset.alignment_report
        payload = {
            "report_type": "multimodal_check",
            "ok": not any(issue.level == "error" for issue in report.issues),
            "scene_count": len(dataset.complete_scenes),
            **report.to_dict(),
        }
        report_path = Path(out) if out is not None else self.root / "multimodal_check_result.json"
        write_json_report(payload, report_path)
        _print_check_summary(payload, report_path)
        return payload

    def stats(
        self,
        *,
        out: str | Path | None = None,
        plots_dir: str | Path | None = None,
        stats_list: str | Sequence[str] | None = None,
        reload: bool = False,
        progress: bool | None = None,
        progress_leave: bool | None = None,
    ) -> dict[str, object]:
        """Compute shared annotation statistics plus per-modality image statistics."""

        dataset = self.load(reload=reload, progress=progress, progress_leave=progress_leave)
        payload = compute_multimodal_stats(dataset)
        if out is not None:
            write_json_report(payload, out)
        if plots_dir is not None:
            write_multimodal_stats_plots(dataset, plots_dir, stats_list=stats_list)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    def vis_draw(
        self,
        out: str | Path,
        *,
        modalities: Sequence[str] | None = None,
        limit: int | None = None,
        show_conf: bool = False,
        conf: float | None = None,
        mask_alpha: int = 64,
        fill_mask: bool = True,
        show_attrs: bool = False,
        show_id: bool = False,
        filter_no_attrs: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Render every selected modality from the shared cached annotations."""

        counts = render_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            out,
            modalities=modalities,
            limit=limit,
            show_confidence=show_conf,
            confidence_threshold=conf,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attributes=show_attrs,
            show_txt_id=show_id,
            filter_no_attributes=filter_no_attrs,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(json.dumps({"out": str(out), "modalities": counts}, indent=2, ensure_ascii=False))
        return counts

    def vis_crop(
        self,
        out: str | Path,
        *,
        modalities: Sequence[str] | None = None,
        keep_shape: bool = False,
        min_size: int = 1,
        conf: float | None = None,
        by_attr: bool = False,
        filter_no_attrs: bool = True,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Write object crops grouped by modality without reparsing labels."""

        counts = crop_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            out,
            modalities=modalities,
            keep_shape=keep_shape,
            min_size=min_size,
            confidence_threshold=conf,
            by_attribute=by_attr,
            filter_no_attributes=filter_no_attrs,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(json.dumps({"out": str(out), "crops": counts}, indent=2, ensure_ascii=False))
        return counts


def _print_check_summary(payload: dict[str, object], report_path: Path) -> None:
    summary = payload.get("summary", {})
    counts = summary if isinstance(summary, dict) else {}
    error_count = sum(count for key, count in counts.items() if str(key).startswith("error:"))
    warning_count = sum(count for key, count in counts.items() if str(key).startswith("warning:"))
    scene_count = payload.get("scene_count", 0)
    if error_count or warning_count:
        color = "\033[31m"
        reset = "\033[0m"
        print(
            f"{color}[MULTIMODAL CHECK WARNING] complete_scenes={scene_count}, "
            f"errors={error_count}, warnings={warning_count}. Full report: {report_path}{reset}",
            file=sys.stderr,
        )
        for key, count in sorted(counts.items()):
            print(f"{color}  {key}: {count}{reset}", file=sys.stderr)
        return
    print(
        f"\033[32m[MULTIMODAL CHECK OK] complete_scenes={scene_count}. "
        f"Full report: {report_path}\033[0m",
        file=sys.stderr,
    )
