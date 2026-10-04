from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

from yolo_data_manager.core.models import is_image_file
from yolo_data_manager.core.schema import find_class_source, read_dataset_class_schema
from yolo_data_manager.runtime import (
    ProgressCallback,
    count_matching_files,
    progress_stage,
)


@dataclass
class LayoutInfo:
    layout: str
    root: Path
    images_dir: Path | None = None
    labels_dir: Path | None = None
    split_files: list[Path] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    image_count: int = 0
    label_count: int = 0

    def to_dict(self) -> dict[str, object]:
        classes = read_dataset_class_schema(self.root)
        class_source = find_class_source(self.root)
        return {
            "report_type": "layout_detect",
            "message": "This is a layout detection result, not a dataset validation/check result.",
            "layout": self.layout,
            "root": str(self.root),
            "images_dir": str(self.images_dir) if self.images_dir else None,
            "labels_dir": str(self.labels_dir) if self.labels_dir else None,
            "split_files": [str(path) for path in self.split_files],
            "splits": self.splits,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "class_source": str(class_source) if class_source else None,
            "class_count": len(classes.names),
            "classes": classes.names,
        }


def detect_layout(
    root: str | Path,
    *,
    images_dir: str | Path = "images",
    progress: bool = False,
    progress_leave: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LayoutInfo:
    root_path = Path(root)
    split_files = [root_path / name for name in ("train.txt", "val.txt", "test.txt") if (root_path / name).exists()]
    if split_files:
        return _image_list_layout(
            root_path,
            split_files,
            images_dir=images_dir,
            progress=progress,
        )

    images_root = _resolve_under(root_path, images_dir)
    labels_root = root_path / "labels"
    if images_root.exists() and labels_root.exists():
        splits = [
            split for split in ("train", "val", "test")
            if (images_root / split).exists() or (labels_root / split).exists()
        ]
        image_count = _count_images(
            images_root,
            progress=progress,
            progress_leave=progress_leave,
            desc="layout scan images",
            progress_callback=progress_callback,
        )
        label_count = _count_labels(
            labels_root,
            progress=progress,
            progress_leave=progress_leave,
            desc="layout scan labels",
            progress_callback=progress_callback,
        )
        if splits:
            return LayoutInfo(
                layout="split_dirs",
                root=root_path,
                images_dir=images_root,
                labels_dir=labels_root,
                splits=splits,
                image_count=image_count,
                label_count=label_count,
            )
        return LayoutInfo(
            layout="flat",
            root=root_path,
            images_dir=images_root,
            labels_dir=labels_root,
            image_count=image_count,
            label_count=label_count,
        )

    image_count = _count_images(
        root_path,
        progress=progress,
        progress_leave=progress_leave,
        desc="layout scan images",
        progress_callback=progress_callback,
    )
    label_count = _count_labels(
        root_path,
        progress=progress,
        progress_leave=progress_leave,
        desc="layout scan labels",
        progress_callback=progress_callback,
    )
    if image_count or label_count:
        return LayoutInfo(
            layout="mixed",
            root=root_path,
            images_dir=root_path,
            labels_dir=root_path,
            image_count=image_count,
            label_count=label_count,
        )

    return LayoutInfo(layout="unknown", root=root_path)


def resolve_layout(
    root: str | Path,
    layout: str = "auto",
    images_dir: str | Path = "images",
    labels_dir: str | Path = "labels",
    progress: bool = False,
    progress_leave: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LayoutInfo:
    root_path = Path(root)
    if layout == "auto":
        return detect_layout(
            root_path,
            images_dir=images_dir,
            progress=progress,
            progress_leave=progress_leave,
            progress_callback=progress_callback,
        )
    if layout == "flat":
        image_root = _resolve_under(root_path, images_dir)
        label_root = _resolve_under(root_path, labels_dir)
        return LayoutInfo(
            layout="flat",
            root=root_path,
            images_dir=image_root,
            labels_dir=label_root,
            image_count=_count_images(
                image_root,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan images",
                progress_callback=progress_callback,
            ),
            label_count=_count_labels(
                label_root,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan labels",
                progress_callback=progress_callback,
            ),
        )
    if layout == "split_dirs":
        image_root = _resolve_under(root_path, images_dir)
        label_root = _resolve_under(root_path, labels_dir)
        splits = [
            split for split in ("train", "val", "test")
            if (image_root / split).exists() or (label_root / split).exists()
        ]
        return LayoutInfo(
            layout="split_dirs",
            root=root_path,
            images_dir=image_root,
            labels_dir=label_root,
            splits=splits,
            image_count=_count_images(
                image_root,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan images",
                progress_callback=progress_callback,
            ),
            label_count=_count_labels(
                label_root,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan labels",
                progress_callback=progress_callback,
            ),
        )
    if layout == "image_list":
        split_files = [root_path / name for name in ("train.txt", "val.txt", "test.txt") if (root_path / name).exists()]
        return _image_list_layout(
            root_path,
            split_files,
            images_dir=images_dir,
            progress=progress,
        )
    if layout == "mixed":
        return LayoutInfo(
            layout="mixed",
            root=root_path,
            images_dir=root_path,
            labels_dir=root_path,
            image_count=_count_images(
                root_path,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan images",
                progress_callback=progress_callback,
            ),
            label_count=_count_labels(
                root_path,
                progress=progress,
                progress_leave=progress_leave,
                desc="layout scan labels",
                progress_callback=progress_callback,
            ),
        )
    raise ValueError(f"unsupported YOLO layout: {layout}")


def read_image_list(
    paths: list[Path],
    root: Path,
    *,
    images_dir: str | Path = "images",
    progress: bool = False,
    desc: str = "read split image list",
) -> list[Path]:
    progress_stage(desc, enabled=progress)
    image_root = _resolve_under(root, images_dir)
    image_paths: list[Path] = []
    for list_path in paths:
        for line in list_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            image_paths.append(_resolve_image_list_entry(text, root, image_root))
    return image_paths


def infer_label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        if part == "image":
            parts[idx] = "label"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _image_list_layout(
    root: Path,
    split_files: list[Path],
    *,
    images_dir: str | Path = "images",
    progress: bool = False,
) -> LayoutInfo:
    image_paths = read_image_list(
        split_files,
        root,
        images_dir=images_dir,
        progress=progress,
        desc="layout read split image list",
    )
    label_paths = [infer_label_path_from_image(path) for path in image_paths]
    return LayoutInfo(
        layout="image_list",
        root=root,
        split_files=split_files,
        splits=[path.stem for path in split_files],
        image_count=len(image_paths),
        label_count=sum(1 for path in label_paths if path.exists()),
    )


def _count_images(
    root: Path,
    *,
    progress: bool = False,
    progress_leave: bool = False,
    desc: str = "scan images",
    progress_callback: ProgressCallback | None = None,
) -> int:
    return count_matching_files(
        root,
        lambda path: is_image_file(path),
        progress=progress,
        progress_leave=progress_leave,
        desc=desc,
        progress_callback=progress_callback,
    )


def _count_labels(
    root: Path,
    *,
    progress: bool = False,
    progress_leave: bool = False,
    desc: str = "scan labels",
    progress_callback: ProgressCallback | None = None,
) -> int:
    return count_matching_files(
        root,
        lambda path: path.suffix.lower() == ".txt",
        progress=progress,
        progress_leave=progress_leave,
        desc=desc,
        progress_callback=progress_callback,
    )


def _resolve_under(root: Path, child: str | Path) -> Path:
    child_path = Path(child)
    return child_path if child_path.is_absolute() else root / child_path


def _resolve_image_list_entry(text: str, root: Path, image_root: Path) -> Path:
    """Resolve a split-list image path, including paths copied from another host.

    Split files are often generated on a Linux training server and consumed on a
    Windows workstation (or the other way around).  A stale absolute path is
    still syntactically valid on the new host, so simply checking
    ``Path.is_absolute()`` would keep pointing at the old location.  If the
    path contains an ``images``/``image`` directory, rebase the suffix under
    the current dataset image directory when the original path is unavailable.
    """

    cleaned = text.strip().strip('"').strip("'")
    raw_path = Path(cleaned).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend((root / raw_path, image_root / raw_path))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    parts = _foreign_path_parts(cleaned)
    marker_names = {"images", "image"}
    if image_root.name:
        marker_names.add(image_root.name.casefold())
    marker_index = next(
        (
            index
            for index in range(len(parts) - 1, -1, -1)
            if parts[index].casefold() in marker_names
        ),
        None,
    )
    if marker_index is not None and marker_index + 1 < len(parts):
        return image_root.joinpath(*parts[marker_index + 1 :])

    if not raw_path.is_absolute():
        return image_root / raw_path

    # Keep valid external-image semantics for absolute paths that do not carry
    # a recognizable image-directory component.  The caller will report the
    # missing path if neither the original nor a rebase candidate exists.
    return raw_path


def _foreign_path_parts(text: str) -> list[str]:
    """Split POSIX and Windows path spellings independently of the host OS."""

    return [
        part
        for part in text.replace("\\", "/").split("/")
        if part and part not in {"."}
    ]
