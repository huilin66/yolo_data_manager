from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from yolo_data_manager.core.models import AttributeSchema, ClassSchema, YoloAnnotation, YoloDataset, YoloImage


@dataclass(frozen=True)
class ModalityConfig:
    """Configuration for one logical image modality in a multimodal dataset."""

    type: str
    path: Path
    suffix: str = ""
    required: bool = True


@dataclass(frozen=True)
class MultimodalImage:
    """One physical image associated with a scene and a modality."""

    path: Path
    relative_path: Path
    source_stem: str
    scene_stem: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class AlignmentIssue:
    level: str
    code: str
    message: str
    scene_stem: str | None = None
    modality: str | None = None
    path: str | None = None


@dataclass
class AlignmentReport:
    """Non-destructive report of image/label association results."""

    issues: list[AlignmentIssue] = field(default_factory=list)

    def add(
        self,
        level: str,
        code: str,
        message: str,
        *,
        scene_stem: str | None = None,
        modality: str | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.issues.append(
            AlignmentIssue(
                level=level,
                code=code,
                message=message,
                scene_stem=scene_stem,
                modality=modality,
                path=str(path) if path is not None else None,
            )
        )

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            key = f"{issue.level}:{issue.code}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "issues": [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    "scene_stem": issue.scene_stem,
                    "modality": issue.modality,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }


@dataclass
class MultimodalScene:
    """A scene with one shared YOLO label file and zero or more modality images."""

    stem: str
    label_path: Path | None = None
    annotations: list[YoloAnnotation] = field(default_factory=list)
    images: dict[str, MultimodalImage] = field(default_factory=dict)
    duplicate_modalities: set[str] = field(default_factory=set)


@dataclass
class MultimodalYoloDataset:
    """A scene-centric YOLO dataset with shared annotations across image modalities."""

    root: Path
    classes: ClassSchema
    attributes: AttributeSchema | None
    task: str
    modalities: dict[str, ModalityConfig]
    scenes: dict[str, MultimodalScene]
    alignment_report: AlignmentReport = field(default_factory=AlignmentReport)
    source_images: dict[str, list[MultimodalImage]] = field(default_factory=dict)
    image_type_summary: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def required_modalities(self) -> tuple[str, ...]:
        return tuple(name for name, config in self.modalities.items() if config.required)

    @property
    def complete_scenes(self) -> list[MultimodalScene]:
        required = self.required_modalities
        return [
            scene
            for scene in self.scenes.values()
            if scene.label_path is not None
            and not (scene.duplicate_modalities & set(required))
            and all(modality in scene.images for modality in required)
        ]

    def annotation_count(self, *, complete_only: bool = True) -> int:
        scenes = self.complete_scenes if complete_only else list(self.scenes.values())
        return sum(len(scene.annotations) for scene in scenes)

    def to_yolo_dataset(self, modality: str, *, complete_only: bool = True) -> YoloDataset:
        """Return an in-memory single-modality view without reparsing label files."""

        if modality not in self.modalities:
            raise KeyError(f"unknown modality: {modality}")
        scenes = self.complete_scenes if complete_only else list(self.scenes.values())
        images: list[YoloImage] = []
        for scene in scenes:
            source = scene.images.get(modality)
            if source is None:
                continue
            images.append(
                YoloImage(
                    path=source.path,
                    label_path=scene.label_path,
                    width=source.width,
                    height=source.height,
                    annotations=scene.annotations,
                    output_name=source.relative_path.as_posix(),
                )
            )
        return YoloDataset(
            root=self.root,
            images=images,
            classes=self.classes,
            attributes=self.attributes,
            task=self.task,
        )
