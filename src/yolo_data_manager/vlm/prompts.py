"""Prompt builders used by the built-in VLM workflows."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def auto_label_prompt(
    class_names: Sequence[str],
    *,
    attribute_schema: Mapping[str, Any] | None = None,
    image_name: str | None = None,
    extra_prompt: str | None = None,
) -> str:
    """Build a strict image-to-YOLO annotation prompt."""

    payload = {
        "classes": list(class_names),
        "attributes": dict(attribute_schema or {}),
        "image": image_name or "",
    }
    prompt = (
        "You are a precise object detection annotator. "
        "Return JSON only, with no markdown. Detect every visible object from the "
        "allowed classes. The bbox must be normalized YOLO xywh: "
        "[center_x, center_y, width, height], all values in [0,1]. "
        "Use class_name exactly as listed. If no object exists, return {\"boxes\":[]}.\n"
        "Schema: {\"boxes\":[{\"class_name\":\"...\",\"bbox\":[0,0,0,0],"
        "\"confidence\":0.0,\"attributes\":{}}]}\n"
        f"Context: {json.dumps(payload, ensure_ascii=False)}"
    )
    if extra_prompt:
        prompt += "\nAdditional instruction: " + str(extra_prompt)
    return prompt


def error_verify_prompt(
    *,
    mode: str,
    class_names: Sequence[str],
    attributes: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> str:
    """Build a correction-verification prompt for an error-analysis crop."""

    actions = {
        "class": ["keep", "correct_class", "delete_gt"],
        "attribute": ["keep", "correct_attribute"],
        "all": ["keep", "correct_class", "delete_gt", "correct_attribute"],
    }.get(mode, ["keep", "correct_class", "delete_gt", "correct_attribute"])
    context = {
        "allowed_classes": list(class_names),
        "attributes": dict(attributes or {}),
        "crop": dict(metadata),
    }
    return (
        "You are checking one object crop from YOLO error analysis. "
        "Decide whether the selected GT annotation is correct. Return JSON only, "
        "with no markdown. Do not invent a class outside allowed_classes. "
        f"Allowed actions: {actions}. For correct_class include target_class. "
        "For correct_attribute include attribute_name and attribute_value. "
        "Use keep when the crop does not justify a change. "
        "Schema: {\"action\":\"keep|correct_class|delete_gt|correct_attribute\","
        "\"target_class\":null,\"attribute_name\":null,\"attribute_value\":null,"
        "\"confidence\":0.0,\"reason\":\"...\"}\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}"
    )


def assistant_prompt(intent: str, methods: Mapping[str, str]) -> str:
    """Build a constrained intent-to-manager-method prompt."""

    return (
        "You are an assistant for yolo_data_manager. Convert the user's intent "
        "into exactly one safe manager method call. Return JSON only. Never create "
        "shell commands, Python code, or unknown methods. Use a method name from "
        "the supplied list and put keyword arguments in arguments. "
        "Set requires_confirmation=true for any annotation or dataset write.\n"
        "Schema: {\"method\":\"...\",\"arguments\":{},"
        "\"explanation\":\"...\",\"requires_confirmation\":false}\n"
        f"Available methods: {json.dumps(dict(methods), ensure_ascii=False)}\n"
        f"User intent: {intent}"
    )


__all__ = ["assistant_prompt", "auto_label_prompt", "error_verify_prompt"]
