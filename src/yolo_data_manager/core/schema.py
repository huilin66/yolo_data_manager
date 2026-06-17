from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yolo_data_manager.core.models import AttributeSchema, ClassSchema


def read_class_schema(path: Path | str | None) -> ClassSchema:
    if path is None:
        return ClassSchema([])
    class_path = Path(path)
    if not class_path.exists():
        return ClassSchema([])
    names = [line.strip() for line in class_path.read_text(encoding="utf-8").splitlines()]
    return ClassSchema([name for name in names if name])


def write_class_schema(schema: ClassSchema, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(schema.names) + ("\n" if schema.names else ""), encoding="utf-8")


def read_attribute_schema(path: Path | str | None) -> AttributeSchema | None:
    if path is None:
        return None
    attr_path = Path(path)
    if not attr_path.exists():
        return None
    data = yaml.safe_load(attr_path.read_text(encoding="utf-8")) or {}
    attributes = data.get("attributes", data)
    if not isinstance(attributes, dict):
        attributes = {}
    return AttributeSchema(attributes)


def write_attribute_schema(schema: AttributeSchema | None, path: Path | str) -> None:
    if schema is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"attributes": schema.attributes}, allow_unicode=True, sort_keys=False)
    Path(path).write_text(text, encoding="utf-8")


def write_dataset_yaml(
    class_schema: ClassSchema,
    path: Path | str,
    train: str = "images/train",
    val: str = "images/val",
    test: str | None = None,
) -> None:
    data = {
        "train": train,
        "val": val,
        "nc": len(class_schema.names),
        "names": class_schema.names,
    }
    if test is not None:
        data["test"] = test
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def find_class_file(root: Path) -> Path | None:
    for name in ("class.txt", "classes.txt"):
        path = root / name
        if path.exists():
            return path
    return None


def read_dataset_class_schema(root: Path) -> ClassSchema:
    class_file = find_class_file(root)
    if class_file is not None:
        return read_class_schema(class_file)
    dataset_yaml = root / "dataset.yaml"
    if not dataset_yaml.exists():
        return ClassSchema([])
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return ClassSchema([str(x) for x in names])
    if isinstance(names, dict):
        ordered = [str(names[k]) for k in sorted(names, key=lambda item: int(item))]
        return ClassSchema(ordered)
    return ClassSchema([])


def find_attribute_file(root: Path) -> Path | None:
    for name in ("attribute.yaml", "attributes.yaml"):
        path = root / name
        if path.exists():
            return path
    return None


def read_dataset_yaml(path: Path | str) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
