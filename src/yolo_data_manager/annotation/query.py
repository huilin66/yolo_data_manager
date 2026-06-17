from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from yolo_data_manager.core.models import YoloAnnotation, YoloDataset, YoloImage


@dataclass
class QueryMatch:
    image: YoloImage
    annotation: YoloAnnotation

    def to_row(self, dataset: YoloDataset) -> dict[str, object]:
        box = self.annotation.geometry_box()
        attrs = {}
        if dataset.attributes is not None and self.annotation.attributes:
            attrs = dataset.attributes.decode(self.annotation.attributes)
        return {
            "image": self.image.file_name,
            "image_path": str(self.image.path),
            "label_path": str(self.image.label_path) if self.image.label_path else "",
            "line_no": self.annotation.line_no or "",
            "class_id": self.annotation.class_id,
            "class_name": dataset.class_name(self.annotation.class_id),
            "task": "segment" if self.annotation.polygon else "detect",
            "cx": box.cx if box else "",
            "cy": box.cy if box else "",
            "width": box.width if box else "",
            "height": box.height if box else "",
            "points": len(self.annotation.polygon.points) if self.annotation.polygon else "",
            "confidence": self.annotation.confidence if self.annotation.confidence is not None else "",
            "attributes": attrs,
        }


@dataclass
class QueryResult:
    dataset: YoloDataset
    matches: list[QueryMatch]

    def __len__(self) -> int:
        return len(self.matches)

    def label_paths(self) -> list[Path]:
        paths = {
            match.image.label_path
            for match in self.matches
            if match.image.label_path is not None
        }
        return sorted(paths)

    def image_paths(self) -> list[Path]:
        return sorted({match.image.path for match in self.matches})

    def to_rows(self) -> list[dict[str, object]]:
        return [match.to_row(self.dataset) for match in self.matches]

    def write_csv(self, path: str | Path) -> None:
        rows = self.to_rows()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image",
            "image_path",
            "label_path",
            "line_no",
            "class_id",
            "class_name",
            "task",
            "cx",
            "cy",
            "width",
            "height",
            "points",
            "confidence",
            "attributes",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def query_by_class(dataset: YoloDataset, classes: Iterable[int | str]) -> QueryResult:
    class_ids = {dataset.class_id(value) for value in classes}
    matches: list[QueryMatch] = []
    for image in dataset.images:
        for annotation in image.annotations:
            if annotation.class_id in class_ids:
                matches.append(QueryMatch(image=image, annotation=annotation))
    return QueryResult(dataset=dataset, matches=matches)


def labels_containing_class(dataset: YoloDataset, classes: Iterable[int | str]) -> list[Path]:
    return query_by_class(dataset, classes).label_paths()

