from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from yolo_data_manager.core.errors import ClassNotFoundError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_AUTO = "auto"


@dataclass
class ClassSchema:
    names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return str(class_id)

    def id(self, value: int | str) -> int:
        if isinstance(value, int):
            if self.names and not 0 <= value < len(self.names):
                raise ClassNotFoundError(f"class id out of range: {value}")
            return value
        text = str(value).strip()
        if text.lstrip("-").isdigit():
            return self.id(int(text))
        if text not in self.names:
            raise ClassNotFoundError(f"class name not found: {text}")
        return self.names.index(text)

    def ensure(self, value: int | str) -> int:
        if isinstance(value, int) or str(value).strip().lstrip("-").isdigit():
            return self.id(value)
        text = str(value).strip()
        if text not in self.names:
            self.names.append(text)
        return self.names.index(text)

    def renamed(self, old: int | str, new_name: str) -> "ClassSchema":
        names = list(self.names)
        names[self.id(old)] = new_name
        return ClassSchema(names)


@dataclass
class AttributeSchema:
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.attributes.keys())

    def decode(self, values: Iterable[float | int | str]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for name, raw_value in zip(self.names, values):
            options = self.attributes.get(name)
            if isinstance(options, list):
                idx = int(float(raw_value))
                decoded[name] = options[idx] if 0 <= idx < len(options) else raw_value
            else:
                decoded[name] = raw_value
        return decoded


@dataclass
class Box:
    cx: float
    cy: float
    width: float
    height: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.cx, self.cy, self.width, self.height


@dataclass
class Polygon:
    points: list[tuple[float, float]]

    def flat(self) -> list[float]:
        values: list[float] = []
        for x, y in self.points:
            values.extend([x, y])
        return values


@dataclass
class YoloAnnotation:
    class_id: int
    box: Box | None = None
    polygon: Polygon | None = None
    attributes: list[float] = field(default_factory=list)
    confidence: float | None = None
    line_no: int | None = None
    source_line: str = ""

    @property
    def is_detection(self) -> bool:
        return self.box is not None

    @property
    def is_segmentation(self) -> bool:
        return self.polygon is not None

    def geometry_box(self) -> Box | None:
        if self.box is not None:
            return self.box
        if self.polygon is None:
            return None
        from yolo_data_manager.core.geometry import polygon_to_box

        cx, cy, width, height = polygon_to_box(self.polygon.points)
        return Box(cx, cy, width, height)

    def to_yolo_line(
        self,
        precision: int = 6,
        include_confidence: bool = False,
        include_attributes: bool = True,
    ) -> str:
        values: list[str] = [str(int(self.class_id))]
        if include_attributes and self.attributes:
            values.append(str(len(self.attributes)))
            values.extend(_format_number(v, precision) for v in self.attributes)
        if self.box is not None:
            values.extend(_format_number(v, precision) for v in self.box.as_tuple())
        elif self.polygon is not None:
            values.extend(_format_number(v, precision) for v in self.polygon.flat())
        if include_confidence and self.confidence is not None:
            values.append(_format_number(self.confidence, precision))
        return " ".join(values)


@dataclass
class YoloImage:
    path: Path
    label_path: Path | None = None
    width: int | None = None
    height: int | None = None
    annotations: list[YoloAnnotation] = field(default_factory=list)
    output_name: str | None = None

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def file_name(self) -> str:
        return self.output_name or self.path.name


@dataclass
class YoloDataset:
    root: Path
    images: list[YoloImage]
    classes: ClassSchema = field(default_factory=ClassSchema)
    attributes: AttributeSchema | None = None
    task: str = TASK_AUTO
    orphan_labels: list[Path] = field(default_factory=list)

    def class_id(self, value: int | str) -> int:
        return self.classes.id(value)

    def class_name(self, class_id: int) -> str:
        return self.classes.name(class_id)

    def labels(self) -> list[Path]:
        return [img.label_path for img in self.images if img.label_path is not None]

    def annotation_count(self) -> int:
        return sum(len(img.annotations) for img in self.images)

    def image_by_stem(self) -> dict[str, YoloImage]:
        return {img.stem: img for img in self.images}


def is_image_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _format_number(value: float | int, precision: int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"
