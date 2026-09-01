from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from yolo_data_manager.core.models import TASK_AUTO
from yolo_data_manager.core.multimodal import AlignmentReport, MultimodalYoloDataset
from yolo_data_manager.io.image_conversion import convert_multimodal_images_to_uint8
from yolo_data_manager.io.multimodal import load_multimodal_yolo_dataset
from yolo_data_manager.io.output_paths import ydm_dir
from yolo_data_manager.stats.multimodal import (
    compute_multimodal_stats,
    write_multimodal_stats_plots,
)
from yolo_data_manager.stats.report import write_json_report
from yolo_data_manager.vis.multimodal import (
    crop_multimodal_dataset,
    render_multimodal_dataset,
)


class MultiModalYoloManager:
    """Modality-aware loader/cache for a shared-label YOLO dataset.

    Multimodality is a property of the dataset, not a separate business
    workflow. This adapter associates several image folders with one label set
    and reuses the same quality, statistics, visualization, and conversion
    output groups as :class:`YoloManager`. Operations without explicit
    all-modality write semantics are intentionally not exposed yet.
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
                progress_leave=self.progress_leave
                if progress_leave is None
                else progress_leave,
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

        dataset = self.load(
            reload=reload, progress=progress, progress_leave=progress_leave
        )
        report = dataset.alignment_report
        payload = {
            "report_type": "multimodal_check",
            "ok": not any(issue.level == "error" for issue in report.issues),
            "scene_count": len(dataset.complete_scenes),
            "image_type_summary": dataset.image_type_summary,
            **report.to_dict(),
        }
        report_path = Path(out) if out is not None else ydm_dir(self.root, "quality") / "multimodal_check.json"
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

        dataset = self.load(
            reload=reload, progress=progress, progress_leave=progress_leave
        )
        payload = compute_multimodal_stats(dataset)
        stats_dir = ydm_dir(self.root, "stats")
        report_path = Path(out) if out is not None else stats_dir / "multimodal_stats.json"
        plots_path = Path(plots_dir) if plots_dir is not None else stats_dir / "plots"
        write_json_report(payload, report_path)
        write_multimodal_stats_plots(dataset, plots_path, stats_list=stats_list)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    def convert_to_uint8(
        self,
        out: str | Path | None = None,
        *,
        modalities: Sequence[str] | None = None,
        stretch: bool = True,
        value_range: tuple[float, float] | None = None,
        preserve_zero: bool = True,
        overwrite: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, object]:
        """Write selected modalities as uint8 images without changing the source data.

        Existing uint8 files are copied unchanged. Other dtypes are written as
        PNG after optional linear stretching; use ``value_range`` for one fixed
        mapping across images, such as ``(0, 20000)`` for depth values.
        """

        output_path = Path(out) if out is not None else ydm_dir(self.root, "conversion") / "uint8"
        payload = convert_multimodal_images_to_uint8(
            self.load(
                reload=reload, progress=progress, progress_leave=progress_leave
            ),
            output_path,
            modalities=modalities,
            stretch=stretch,
            value_range=value_range,
            preserve_zero=preserve_zero,
            overwrite=overwrite,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    def vis_draw(
        self,
        out: str | Path | None = None,
        *,
        style: str = "cv2",
        modalities: Sequence[str] | None = None,
        limit: int | None = None,
        show_conf: bool = False,
        conf: float | None = None,
        mask_alpha: int = 64,
        fill_mask: bool = True,
        show_attrs: bool = False,
        show_id: bool = False,
        filter_no_attrs: bool = False,
        att_seperate: bool = False,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Render every selected modality from the shared cached annotations."""

        output_path = Path(out) if out is not None else ydm_dir(self.root, "vis") / "draw"
        counts = render_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            output_path,
            style=style,
            modalities=modalities,
            limit=limit,
            show_confidence=show_conf,
            confidence_threshold=conf,
            mask_alpha=mask_alpha,
            fill_mask=fill_mask,
            show_attributes=show_attrs,
            show_txt_id=show_id,
            filter_no_attributes=filter_no_attrs,
            att_seperate=att_seperate,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(
            json.dumps(
                {"out": str(output_path), "modalities": counts}, indent=2, ensure_ascii=False
            )
        )
        return counts

    def vis_crop(
        self,
        out: str | Path | None = None,
        *,
        style: str = "cv2",
        modalities: Sequence[str] | None = None,
        keep_shape: bool = False,
        min_size: int = 1,
        padding: int | float = 0,
        conf: float | None = None,
        by_attr: bool = False,
        filter_no_attrs: bool = True,
        workers: int = 8,
        progress: bool = True,
        progress_leave: bool = False,
        reload: bool = False,
    ) -> dict[str, int]:
        """Write object crops grouped by modality without reparsing labels."""

        output_path = Path(out) if out is not None else ydm_dir(self.root, "vis") / "crop"
        counts = crop_multimodal_dataset(
            self.load(reload=reload, progress=progress, progress_leave=progress_leave),
            output_path,
            style=style,
            modalities=modalities,
            keep_shape=keep_shape,
            min_size=min_size,
            padding=padding,
            confidence_threshold=conf,
            by_attribute=by_attr,
            filter_no_attributes=filter_no_attrs,
            workers=workers,
            progress=progress,
            progress_leave=progress_leave,
        )
        print(
            json.dumps({"out": str(output_path), "crops": counts}, indent=2, ensure_ascii=False)
        )
        return counts


def _print_check_summary(payload: dict[str, object], report_path: Path) -> None:
    summary = payload.get("summary", {})
    counts = summary if isinstance(summary, dict) else {}
    error_count = sum(
        count for key, count in counts.items() if str(key).startswith("error:")
    )
    warning_count = sum(
        count for key, count in counts.items() if str(key).startswith("warning:")
    )
    scene_count = payload.get("scene_count", 0)
    _print_image_type_summary(payload.get("image_type_summary"))
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


def _print_image_type_summary(value: object) -> None:
    if not isinstance(value, dict):
        return
    print("[MULTIMODAL IMAGE TYPES]", file=sys.stderr)
    for modality, raw_summary in value.items():
        if not isinstance(raw_summary, dict):
            continue
        image_count = raw_summary.get("image_count", 0)
        type_count = raw_summary.get("type_count", 0)
        print(
            f"  {modality}: {image_count} source image(s), {type_count} type(s)",
            file=sys.stderr,
        )
        raw_types = raw_summary.get("types", [])
        if not isinstance(raw_types, list):
            continue
        for raw_type in raw_types:
            if not isinstance(raw_type, dict):
                continue
            print(
                "    "
                f"{raw_type.get('count', 0)} x "
                f"{raw_type.get('format', 'unknown')}/"
                f"{raw_type.get('mode', 'unknown')}/"
                f"{raw_type.get('dtype', 'unknown')}/"
                f"{raw_type.get('channels', '?')}ch "
                f"{raw_type.get('width', '?')}x{raw_type.get('height', '?')}",
                file=sys.stderr,
            )
