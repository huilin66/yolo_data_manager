"""VLM verification of error-analysis crops and correction-plan generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from yolo_data_manager.annotation.crop_correction import (
    correct_gt_attributes_from_error_crops,
    correct_gt_labels_from_error_crops,
)
from yolo_data_manager.core.models import YoloDataset
from yolo_data_manager.runtime import create_progress_bar, normalize_workers
from yolo_data_manager.vlm.prompts import error_verify_prompt
from yolo_data_manager.vlm.providers import VLMProvider, VLMResponse
from yolo_data_manager.vlm.schemas import VLMDecision, VLMOutputError, parse_error_decision


_CROP_RE = re.compile(
    r"^(?P<stem>.+)_pred(?P<pred>none|[1-9][0-9]*)_gt"
    r"(?P<gt>none|[1-9][0-9]*)(?:_(?P<attribute>.+))?$"
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ErrorCrop:
    path: Path
    kind: str
    group: str
    stem: str
    pred_index: int | None
    gt_index: int | None
    attribute: str | None


def verify_error_crops(
    dataset: YoloDataset,
    error_dir: str | Path,
    out_root: str | Path,
    provider: VLMProvider,
    *,
    mode: str = "all",
    workers: int = 4,
    confidence: float = 0.0,
    limit: int | None = None,
    progress: bool = True,
    progress_leave: bool = False,
) -> dict[str, Any]:
    """Ask a VLM to verify crops and write maps consumable by ann correction APIs."""

    if mode not in {"class", "attribute", "all"}:
        raise ValueError("mode must be class, attribute, or all")
    crops = collect_error_crops(error_dir, mode=mode)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        crops = crops[:limit]
    class_names = list(dataset.classes.names)
    attributes = dataset.attributes.attributes if dataset.attributes else {}

    def verify(crop: ErrorCrop) -> tuple[ErrorCrop, VLMDecision | None, str | None]:
        prompt = error_verify_prompt(
            mode="attribute" if crop.kind == "attribute" else "class",
            class_names=class_names,
            attributes=attributes,
            metadata={
                "kind": crop.kind,
                "group": crop.group,
                "file_name": crop.path.name,
                "stem": crop.stem,
                "pred_index": crop.pred_index,
                "gt_index": crop.gt_index,
                "attribute_from_crop": crop.attribute,
            },
        )
        try:
            response = provider.generate(prompt, images=[crop.path], json_mode=True)
            text = response.text if isinstance(response, VLMResponse) else str(response)
            decision = parse_error_decision(text)
            if decision.confidence is not None and decision.confidence < confidence:
                return crop, None, "below_confidence"
            return crop, decision, None
        except Exception as exc:
            return crop, None, str(exc)

    completed: list[tuple[ErrorCrop, VLMDecision | None, str | None] | None] = [None] * len(crops)
    worker_count = normalize_workers(workers)
    progress_bar = create_progress_bar(
        total=len(crops),
        desc="vlm error verification",
        enabled=progress,
        leave=progress_leave,
    )
    try:
        if worker_count == 1:
            for index, crop in enumerate(crops):
                completed[index] = verify(crop)
                progress_bar.update()
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(verify, crop): index for index, crop in enumerate(crops)}
                for future in as_completed(futures):
                    completed[futures[future]] = future.result()
                    progress_bar.update()
    finally:
        progress_bar.close()

    out = Path(out_root)
    class_map: dict[str, str | int | None] = {}
    attribute_map: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    copied = 0

    for result in completed:
        if result is None:
            continue
        crop, decision, error = result
        row: dict[str, Any] = {
            "crop": str(crop.path),
            "kind": crop.kind,
            "group": crop.group,
            "action": decision.action if decision else None,
            "confidence": decision.confidence if decision else None,
            "reason": decision.reason if decision else error,
            "target_class": decision.target_class if decision else None,
            "attribute_name": decision.attribute_name if decision else None,
            "attribute_value": decision.attribute_value if decision else None,
        }
        decisions.append(row)
        if error or decision is None or crop.gt_index is None:
            if error:
                errors.append({"crop": str(crop.path), "error": error})
            continue
        if decision.confidence is not None and decision.confidence < confidence:
            continue

        if crop.kind == "class":
            if decision.action not in {"correct_class", "delete_gt"}:
                continue
            target: str | int | None = None if decision.action == "delete_gt" else decision.target_class
            if target is None and decision.action == "correct_class":
                errors.append({"crop": str(crop.path), "error": "correct_class requires target_class"})
                continue
            if target is not None and not _class_exists(dataset, target):
                errors.append({"crop": str(crop.path), "error": f"unknown target class: {target}"})
                continue
            folder = out / "class_corrections" / ("__delete__" if target is None else _safe_name(str(target)))
            copied_path = _copy_unique(crop.path, folder)
            copied += int(copied_path is not None)
            class_map[str(folder)] = target
        elif crop.kind == "attribute":
            if decision.action != "correct_attribute":
                continue
            name = decision.attribute_name or crop.attribute
            if not name or decision.attribute_value is None:
                errors.append({"crop": str(crop.path), "error": "correct_attribute requires attribute_name and attribute_value"})
                continue
            if not _attribute_exists(dataset, name):
                errors.append({"crop": str(crop.path), "error": f"unknown attribute: {name}"})
                continue
            folder = out / "attribute_corrections" / _safe_name(name) / _safe_name(str(decision.attribute_value))
            copied_path = _copy_unique(crop.path, folder)
            copied += int(copied_path is not None)
            attribute_map[str(folder)] = {"name": name, "value": decision.attribute_value}

    plan: dict[str, Any] = {
        "mode": mode,
        "source": str(Path(error_dir)),
        "out": str(out),
        "crop_count": len(crops),
        "decision_count": len(decisions),
        "copied_count": copied,
        "error_count": len(errors),
        "class_crop_map": class_map,
        "attribute_crop_map": attribute_map,
        "decisions": decisions,
        "errors": errors,
    }
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "correction_plan.json"
    plan["plan"] = str(plan_path)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return plan


def collect_error_crops(error_dir: str | Path, *, mode: str = "all") -> list[ErrorCrop]:
    """Collect class and attribute crop files from an error-analysis review tree."""

    root = Path(error_dir)
    review = root / "review" if (root / "review").is_dir() else root
    records: list[ErrorCrop] = []
    for crop_dir in sorted(path for path in review.rglob("crops") if path.is_dir()):
        relative_parts = crop_dir.relative_to(review).parts
        kind = "attribute" if "attribute_error" in relative_parts else "class"
        if mode != "all" and kind != mode:
            continue
        group = str(crop_dir.relative_to(review).parent).replace("\\", "/")
        for path in sorted(crop_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            match = _CROP_RE.fullmatch(path.stem)
            if match is None:
                continue
            records.append(
                ErrorCrop(
                    path=path,
                    kind=kind,
                    group=group,
                    stem=match.group("stem"),
                    pred_index=_index(match.group("pred")),
                    gt_index=_index(match.group("gt")),
                    attribute=match.group("attribute"),
                )
            )
    return records


def apply_correction_plan(
    dataset: YoloDataset,
    plan: Mapping[str, Any],
    *,
    backup_dir: str | Path | None = None,
    pred_labels_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a generated plan through the existing annotation correction APIs."""

    result: dict[str, Any] = {}
    # Attribute crops point to the original GT line number. Apply them before
    # class deletions, because a class correction may change later line indexes.
    attribute_map = plan.get("attribute_crop_map") or {}
    if attribute_map:
        corrected, report = correct_gt_attributes_from_error_crops(
            dataset,
            {str(path): rule for path, rule in attribute_map.items()},
            backup_dir=backup_dir,
            dry_run=dry_run,
        )
        result["attribute"] = {
            "summary": corrected.__dict__,
            "report_rows": len(report.rows),
        }
    class_map = plan.get("class_crop_map") or {}
    if class_map:
        corrected, report = correct_gt_labels_from_error_crops(
            dataset,
            {str(path): target for path, target in class_map.items()},
            pred_labels_dir=pred_labels_dir,
            backup_dir=backup_dir,
            dry_run=dry_run,
        )
        result["class"] = {
            "summary": corrected.__dict__,
            "report_rows": len(report.rows),
        }
    return result


def _copy_unique(source: Path, folder: Path) -> Path | None:
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / source.name
    if destination.exists():
        return None
    shutil.copy2(source, destination)
    return destination


def _class_exists(dataset: YoloDataset, value: str | int) -> bool:
    try:
        dataset.classes.id(value)
        return True
    except Exception:
        return False


def _attribute_exists(dataset: YoloDataset, value: str) -> bool:
    if dataset.attributes is None:
        return False
    names = set(dataset.attributes.names)
    names.update(
        name
        for class_name in dataset.classes.names
        for name in dataset.attributes.names_for_class(class_name)
    )
    return value in names


def _index(value: str) -> int | None:
    return None if value == "none" else int(value)


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned.strip("_") or "value"


__all__ = [
    "ErrorCrop",
    "apply_correction_plan",
    "collect_error_crops",
    "verify_error_crops",
]
