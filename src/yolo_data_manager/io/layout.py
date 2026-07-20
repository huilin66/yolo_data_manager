from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from yolo_data_manager.core.models import is_image_file
from yolo_data_manager.core.schema import find_class_source, read_dataset_class_schema


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


def detect_layout(root: str | Path) -> LayoutInfo:
    root_path = Path(root)
    split_files = [root_path / name for name in ("train.txt", "val.txt", "test.txt") if (root_path / name).exists()]
    if split_files:
        return _image_list_layout(root_path, split_files)

    images_root = root_path / "images"
    labels_root = root_path / "labels"
    if images_root.exists() and labels_root.exists():
        splits = [
            split for split in ("train", "val", "test")
            if (images_root / split).exists() or (labels_root / split).exists()
        ]
        image_count = _count_images(images_root)
        label_count = _count_labels(labels_root)
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

    image_count = _count_images(root_path)
    label_count = _count_labels(root_path)
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
) -> LayoutInfo:
    root_path = Path(root)
    if layout == "auto":
        return detect_layout(root_path)
    if layout == "flat":
        image_root = _resolve_under(root_path, images_dir)
        label_root = _resolve_under(root_path, labels_dir)
        return LayoutInfo(
            layout="flat",
            root=root_path,
            images_dir=image_root,
            labels_dir=label_root,
            image_count=_count_images(image_root),
            label_count=_count_labels(label_root),
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
            image_count=_count_images(image_root),
            label_count=_count_labels(label_root),
        )
    if layout == "image_list":
        split_files = [root_path / name for name in ("train.txt", "val.txt", "test.txt") if (root_path / name).exists()]
        return _image_list_layout(root_path, split_files)
    if layout == "mixed":
        return LayoutInfo(
            layout="mixed",
            root=root_path,
            images_dir=root_path,
            labels_dir=root_path,
            image_count=_count_images(root_path),
            label_count=_count_labels(root_path),
        )
    raise ValueError(f"unsupported YOLO layout: {layout}")


def read_image_list(paths: list[Path], root: Path) -> list[Path]:
    image_paths: list[Path] = []
    for list_path in paths:
        for line in list_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            image_path = Path(text)
            if not image_path.is_absolute():
                image_path = root / image_path
            image_paths.append(image_path)
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


def _image_list_layout(root: Path, split_files: list[Path]) -> LayoutInfo:
    image_paths = read_image_list(split_files, root)
    label_paths = [infer_label_path_from_image(path) for path in image_paths]
    return LayoutInfo(
        layout="image_list",
        root=root,
        split_files=split_files,
        splits=[path.stem for path in split_files],
        image_count=len(image_paths),
        label_count=sum(1 for path in label_paths if path.exists()),
    )


def _count_images(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and is_image_file(path)) if root.exists() else 0


def _count_labels(root: Path) -> int:
    return sum(1 for path in root.rglob("*.txt") if path.is_file()) if root.exists() else 0


def _resolve_under(root: Path, child: str | Path) -> Path:
    child_path = Path(child)
    return child_path if child_path.is_absolute() else root / child_path
