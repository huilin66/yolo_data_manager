"""FastAPI backend for the standalone YDM web workspace."""

from __future__ import annotations

import io
import mimetypes
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from yolo_data_manager.core.models import AttributeSchema, YoloDataset, YoloImage
from yolo_data_manager.core.schema import read_dataset_yaml
from yolo_data_manager.io.layout import LayoutInfo, read_image_list, resolve_layout
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.scripting import YoloManager
from yolo_data_manager.vis.renderer import render_image
from yolo_data_manager.vlm import (
    AssistantError,
    create_vlm_provider,
    execute_assistant_plan,
    load_vlm_config,
    run_assistant,
)
from yolo_data_manager.vlm.providers import VLMProviderError


SUPPORTED_LAYOUTS = ("auto", "flat", "split_dirs", "image_list", "mixed")
SUPPORTED_TASKS = ("auto", "detect", "segment")
_SPLIT_NAMES = ("train", "val", "test")
_COLOR_PALETTE = (
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#64748b",
    "#0ea5e9",
    "#ec4899",
)


def _package_version() -> str:
    try:
        from yolo_data_manager import __version__

        return __version__
    except (ImportError, AttributeError):
        pass
    try:
        return version("yolo-data-manager")
    except PackageNotFoundError:
        return "1.0.5"


class LoadDatasetRequest(BaseModel):
    root: str = Field(min_length=1)
    layout: Literal["auto", "flat", "split_dirs", "image_list", "mixed"] = "auto"
    task: Literal["auto", "detect", "segment"] = "auto"
    images_dir: str = "images"
    labels_dir: str = "labels"
    class_file: str | None = None
    attribute_file: str | None = None
    split_file: str | None = None
    workers: int = Field(default=8, ge=1, le=64)


class DatasetSessionStatus(BaseModel):
    loaded: bool
    root: str | None = None
    layout: str | None = None
    image_count: int = 0
    annotation_count: int = 0


class VLMAssistantRequest(BaseModel):
    """A natural-language request or a previously reviewed assistant plan."""

    intent: str | None = Field(default=None, min_length=1)
    plan: dict[str, Any] | None = None
    execute: bool = False
    confirm: bool = False


@dataclass
class DatasetSession:
    root: Path
    dataset: YoloDataset
    layout_info: LayoutInfo
    layout: str = "auto"
    task: str = "auto"
    images_dir: str = "images"
    labels_dir: str = "labels"
    class_file: str | None = None
    attribute_file: str | None = None
    split_file: str | None = None
    splits_by_image: dict[Path, str] = field(default_factory=dict)
    loaded_at: str = ""
    operations: list[dict[str, Any]] = field(default_factory=list)


_SESSION: DatasetSession | None = None
_SESSION_LOCK = threading.RLock()


def _resolve_root_and_yaml(root: str) -> tuple[Path, str | None]:
    candidate = Path(root).expanduser()
    if candidate.suffix.lower() not in {".yaml", ".yml"} or not candidate.is_file():
        return candidate, None

    data = read_dataset_yaml(candidate)
    path_value = data.get("path")
    if path_value:
        path = Path(str(path_value)).expanduser()
        dataset_root = path if path.is_absolute() else (candidate.parent / path).resolve()
    else:
        dataset_root = candidate.parent
    return dataset_root, str(candidate)


def _normalise_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _build_split_map(
    root: Path,
    layout_info: LayoutInfo,
    images_dir: str,
    explicit_split_file: str | None,
    images: list[YoloImage],
) -> dict[Path, str]:
    """Map loaded image paths to train/val/test without changing loader semantics."""

    split_map: dict[Path, str] = {}
    split_files = list(layout_info.split_files)
    if explicit_split_file:
        explicit = Path(explicit_split_file).expanduser()
        if explicit.is_file() and explicit not in split_files:
            split_files.append(explicit)

    for split_file in split_files:
        split_name = split_file.stem.lower()
        if split_name not in _SPLIT_NAMES or not split_file.is_file():
            continue
        try:
            paths = read_image_list([split_file], root, images_dir=images_dir)
        except (OSError, ValueError):
            continue
        for path in paths:
            split_map[_normalise_path(path)] = split_name

    image_root = layout_info.images_dir
    for image in images:
        normalised = _normalise_path(image.path)
        if normalised in split_map:
            continue
        parts = {part.casefold() for part in image.path.parts}
        for split_name in _SPLIT_NAMES:
            if split_name in parts:
                split_map[normalised] = split_name
                break

        if image_root is not None and normalised not in split_map:
            try:
                relative = image.path.relative_to(image_root)
            except ValueError:
                relative = None
            if relative is not None and relative.parts:
                first = relative.parts[0].casefold()
                if first in _SPLIT_NAMES:
                    split_map[normalised] = first

    return split_map


def _require_session() -> DatasetSession:
    with _SESSION_LOCK:
        if _SESSION is None:
            raise HTTPException(status_code=409, detail="Load a YOLO dataset first.")
        return _SESSION


def _relative_name(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _image_record(session: DatasetSession, image_id: int, image: YoloImage) -> dict[str, Any]:
    split = session.splits_by_image.get(_normalise_path(image.path))
    return {
        "id": image_id,
        "name": image.file_name,
        "relative_path": _relative_name(session.root, image.path),
        "split": split,
        "width": image.width,
        "height": image.height,
        "boxes": len(image.annotations),
        "label_exists": image.label_path is not None,
        "preview_url": f"/api/dataset/images/{image_id}/preview",
    }


def _summary(session: DatasetSession) -> dict[str, Any]:
    dataset = session.dataset
    class_counts = {name: 0 for name in dataset.classes.names}
    class_images: dict[str, set[int]] = defaultdict(set)
    attribute_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_data: dict[str, dict[str, Any]] = {
        name: {"name": name, "images": 0, "boxes": 0, "class_counts": {}}
        for name in _SPLIT_NAMES
    }
    unassigned_images = 0

    for image_id, image in enumerate(dataset.images):
        split = session.splits_by_image.get(_normalise_path(image.path))
        if split in split_data:
            split_data[split]["images"] += 1
        else:
            unassigned_images += 1
        for annotation in image.annotations:
            class_name = dataset.class_name(annotation.class_id)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            class_images[class_name].add(image_id)
            if split in split_data:
                split_data[split]["boxes"] += 1
                counts = split_data[split]["class_counts"]
                counts[class_name] = counts.get(class_name, 0) + 1
            attrs = dataset.annotation_attributes(annotation)
            for attr_name, attr_value in attrs.items():
                attribute_counts[attr_name][str(attr_value)] += 1

    class_rows = []
    for class_id, name in enumerate(dataset.classes.names):
        class_rows.append(
            {
                "id": class_id,
                "name": name,
                "boxes": class_counts.get(name, 0),
                "images": len(class_images.get(name, set())),
                "color": _COLOR_PALETTE[class_id % len(_COLOR_PALETTE)],
            }
        )

    split_rows = []
    for name in _SPLIT_NAMES:
        row = split_data[name]
        split_rows.append(
            {
                **row,
                "ratio": round(row["images"] / len(dataset.images), 4) if dataset.images else 0,
            }
        )
    if unassigned_images:
        split_rows.append(
            {
                "name": "unassigned",
                "images": unassigned_images,
                "boxes": 0,
                "class_counts": {},
                "ratio": round(unassigned_images / len(dataset.images), 4) if dataset.images else 0,
            }
        )

    attributes = [
        {"name": name, "values": dict(sorted(values.items()))}
        for name, values in sorted(attribute_counts.items())
    ]
    return {
        "root": str(session.root),
        "layout": session.layout_info.layout,
        "task": dataset.task,
        "loaded_at": session.loaded_at,
        "counts": {
            "images": len(dataset.images),
            "labels": len(dataset.labels()),
            "boxes": dataset.annotation_count(),
            "classes": len(dataset.classes.names),
            "attributes": len(attributes),
            "orphan_labels": len(dataset.orphan_labels),
            "empty_images": sum(1 for image in dataset.images if not image.annotations),
        },
        "classes": class_rows,
        "attributes": attributes,
        "splits": split_rows,
        "images": [_image_record(session, index, image) for index, image in enumerate(dataset.images[:12])],
        "operations": list(session.operations),
    }


def _image_matches(
    session: DatasetSession,
    image: YoloImage,
    *,
    split: str | None,
    class_name: str | None,
) -> bool:
    if split and session.splits_by_image.get(_normalise_path(image.path)) != split:
        return False
    if class_name:
        needle = class_name.strip().casefold()
        if not any(session.dataset.class_name(annotation.class_id).casefold() == needle for annotation in image.annotations):
            return False
    return True


_DEFAULT_VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _vlm_status_payload() -> dict[str, Any]:
    """Return a frontend-safe VLM configuration summary."""

    try:
        config = load_vlm_config()
    except Exception as exc:  # noqa: BLE001 - status must not break the web app
        return {
            "configured": False,
            "provider": "qwen",
            "model": "",
            "base_url": None,
            "dotenv_path": None,
            "error": str(exc),
        }

    configured = bool(
        config.api_key
        or config.model_path
        or (
            config.base_url
            and config.base_url.rstrip("/") != _DEFAULT_VLM_BASE_URL.rstrip("/")
        )
    )
    return {
        "configured": configured,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "dotenv_path": str(config.dotenv_path) if config.dotenv_path else None,
        "error": None,
    }


def _assistant_manager(session: DatasetSession) -> YoloManager:
    """Create a manager for one assistant call without repeating warm-up checks."""

    return YoloManager(
        session.root,
        layout=session.layout,
        task=session.task,
        images_dir=session.images_dir,
        labels_dir=session.labels_dir,
        class_file=session.class_file,
        attribute_file=session.attribute_file,
        split_file=session.split_file,
        init_layout=False,
        init_check=False,
    )


def _build_app() -> FastAPI:
    app = FastAPI(title="YDM Web", version=_package_version())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with _SESSION_LOCK:
            session = _SESSION
        return {
            "status": "ok",
            "service": "ydm-web",
            "version": _package_version(),
            "dataset_loaded": session is not None,
        }

    @app.get("/api/dataset/status", response_model=DatasetSessionStatus)
    def dataset_status() -> DatasetSessionStatus:
        with _SESSION_LOCK:
            session = _SESSION
        if session is None:
            return DatasetSessionStatus(loaded=False)
        return DatasetSessionStatus(
            loaded=True,
            root=str(session.root),
            layout=session.layout_info.layout,
            image_count=len(session.dataset.images),
            annotation_count=session.dataset.annotation_count(),
        )

    @app.get("/api/vlm/status")
    def vlm_status() -> dict[str, Any]:
        return _vlm_status_payload()

    @app.post("/api/vlm/assistant")
    def vlm_assistant(request: VLMAssistantRequest) -> dict[str, Any]:
        session = _require_session()
        if not request.intent and request.plan is None:
            raise HTTPException(
                status_code=422,
                detail="Provide an intent or a reviewed assistant plan.",
            )

        status = _vlm_status_payload()
        if request.plan is None and not status["configured"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "VLM is not configured. Add VLM_API_KEY (or a local "
                    "VLM_BASE_URL) to .env first."
                ),
            )

        try:
            manager = _assistant_manager(session)
            if request.plan is not None:
                result = execute_assistant_plan(
                    manager,
                    request.plan,
                    execute=request.execute,
                    confirm=request.confirm,
                )
            else:
                config = load_vlm_config()
                provider = create_vlm_provider(config)
                result = run_assistant(
                    manager,
                    request.intent or "",
                    provider,
                    execute=request.execute,
                    confirm=request.confirm,
                )
        except AssistantError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (VLMProviderError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if result.get("executed"):
            operation = {
                "name": f"Assistant: {result['plan']['method']}",
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "detail": "Executed from the VLM assistant",
            }
            with _SESSION_LOCK:
                if _SESSION is session:
                    session.operations.insert(0, operation)

        return jsonable_encoder({
            "provider": status["provider"],
            "model": status["model"],
            **result,
        })

    @app.post("/api/dataset/load")
    def load_dataset(request: LoadDatasetRequest) -> dict[str, Any]:
        global _SESSION
        started = time.perf_counter()
        root, yaml_path = _resolve_root_and_yaml(request.root)
        if not root.exists():
            raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {root}")
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset path is not a directory: {root}")

        class_file = request.class_file or yaml_path
        try:
            dataset = load_yolo_dataset(
                root,
                images_dir=request.images_dir,
                labels_dir=request.labels_dir,
                class_file=class_file,
                attribute_file=request.attribute_file,
                task=request.task,
                split_file=request.split_file,
                layout=request.layout,
                read_image_size=True,
                workers=request.workers,
                progress=False,
            )
            layout_info = resolve_layout(
                root,
                layout=request.layout,
                images_dir=request.images_dir,
                labels_dir=request.labels_dir,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - convert parser errors to an API response
            raise HTTPException(status_code=400, detail=f"Dataset load failed: {exc}") from exc

        split_map = _build_split_map(
            root,
            layout_info,
            request.images_dir,
            request.split_file,
            dataset.images,
        )
        loaded_at = datetime.now().astimezone().isoformat(timespec="seconds")
        operation = {
            "name": "Loaded dataset",
            "time": loaded_at,
            "detail": f"{len(dataset.images):,} images · {dataset.annotation_count():,} boxes",
        }
        session = DatasetSession(
            root=root,
            dataset=dataset,
            layout_info=layout_info,
            layout=request.layout,
            task=request.task,
            images_dir=request.images_dir,
            labels_dir=request.labels_dir,
            class_file=request.class_file or yaml_path,
            attribute_file=request.attribute_file,
            split_file=request.split_file,
            splits_by_image=split_map,
            loaded_at=loaded_at,
            operations=[operation],
        )
        with _SESSION_LOCK:
            _SESSION = session
        payload = _summary(session)
        payload["load_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return payload

    @app.post("/api/dataset/unload")
    def unload_dataset() -> dict[str, str]:
        global _SESSION
        with _SESSION_LOCK:
            _SESSION = None
        return {"status": "unloaded"}

    @app.get("/api/dataset/overview")
    def dataset_overview() -> dict[str, Any]:
        return _summary(_require_session())

    @app.get("/api/dataset/images")
    def dataset_images(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=24, ge=1, le=100),
        split: str | None = Query(default=None),
        class_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        session = _require_session()
        matches = [
            (index, image)
            for index, image in enumerate(session.dataset.images)
            if _image_matches(session, image, split=split, class_name=class_name)
        ]
        page = matches[offset : offset + limit]
        return {
            "total": len(matches),
            "offset": offset,
            "limit": limit,
            "items": [_image_record(session, index, image) for index, image in page],
        }

    @app.get("/api/dataset/images/{image_id}")
    def dataset_image(image_id: int) -> dict[str, Any]:
        session = _require_session()
        if image_id < 0 or image_id >= len(session.dataset.images):
            raise HTTPException(status_code=404, detail="Image not found.")
        return _image_record(session, image_id, session.dataset.images[image_id])

    @app.get("/api/dataset/images/{image_id}/preview")
    def dataset_image_preview(image_id: int):
        session = _require_session()
        if image_id < 0 or image_id >= len(session.dataset.images):
            raise HTTPException(status_code=404, detail="Image not found.")
        image = session.dataset.images[image_id]
        try:
            rendered = render_image(
                session.dataset,
                image,
                show_confidence=True,
                show_attributes=bool(session.dataset.attributes),
                show_txt_id=False,
            )
            buffer = io.BytesIO()
            rendered.save(buffer, format="JPEG", quality=88, optimize=True)
            buffer.seek(0)
        except Exception as exc:  # noqa: BLE001 - image codecs vary by dataset
            raise HTTPException(status_code=422, detail=f"Preview failed: {exc}") from exc
        return StreamingResponse(buffer, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/dataset/images/{image_id}/raw")
    def dataset_image_raw(image_id: int):
        session = _require_session()
        if image_id < 0 or image_id >= len(session.dataset.images):
            raise HTTPException(status_code=404, detail="Image not found.")
        image_path = session.dataset.images[image_id].path
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Image file not found.")
        media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        return StreamingResponse(image_path.open("rb"), media_type=media_type)

    return app


def create_app() -> FastAPI:
    """Return a fresh application object for embedding or standalone use."""

    return _build_app()


app = create_app()
