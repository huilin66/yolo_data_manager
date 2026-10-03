"""Constrained natural-language assistant for YoloManager."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

from yolo_data_manager.vlm.prompts import assistant_prompt
from yolo_data_manager.vlm.providers import VLMProvider, VLMResponse
from yolo_data_manager.vlm.schemas import extract_json_object


class AssistantError(ValueError):
    """Raised when an assistant plan is invalid or unsafe."""


SAFE_METHODS: dict[str, str] = {
    "check": "validate dataset quality",
    "stats": "compute dataset statistics",
    "query_class": "query image and label names by class",
    "query_attr": "query image and label names by attribute",
    "vis_draw": "draw dataset annotations",
    "vis_crop": "crop dataset annotations",
    "eval_metrics": "compute detection metrics",
    "eval_error_analysis": "analyze prediction errors",
    "dataset_select": "copy a selected dataset subset",
    "dataset_split": "split dataset into train val test",
    "dataset_filter": "filter annotations by size or class",
    "ann_update_from_map": "update classes with a dictionary and backup",
    "ann_att_update_from_map": "update attributes with a dictionary and backup",
    "ann_correct_from_crops": "correct classes from selected crops",
    "ann_correct_from_error_crops": "correct classes from error-analysis crops",
    "ann_att_correct_from_crops": "correct attributes from selected crops",
    "ann_att_correct_from_error_crops": "correct attributes from error crops",
    "anno_update_by_size": "update annotations by object size",
}

DESTRUCTIVE_METHODS = {
    "dataset_select",
    "dataset_split",
    "dataset_filter",
    "ann_update_from_map",
    "ann_att_update_from_map",
    "ann_correct_from_crops",
    "ann_correct_from_error_crops",
    "ann_att_correct_from_crops",
    "ann_att_correct_from_error_crops",
    "anno_update_by_size",
}


def build_assistant_plan(
    manager: Any,
    intent: str,
    provider: VLMProvider,
) -> dict[str, Any]:
    """Ask the VLM for one constrained manager call without executing it."""

    methods = {
        name: f"{description}; signature {inspect.signature(getattr(manager, name))}"
        for name, description in SAFE_METHODS.items()
        if hasattr(manager, name)
    }
    response = provider.generate(
        assistant_prompt(intent, methods),
        json_mode=True,
    )
    text = response.text if isinstance(response, VLMResponse) else str(response)
    payload = extract_json_object(text)
    if not isinstance(payload, Mapping):
        raise AssistantError("assistant response must be a JSON object")
    method = str(payload.get("method", "")).strip()
    if method not in methods:
        raise AssistantError(f"assistant selected unsupported method: {method!r}")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise AssistantError("assistant arguments must be an object")
    clean_arguments = _validate_arguments(manager, method, arguments)
    return {
        "method": method,
        "arguments": clean_arguments,
        "explanation": str(payload.get("explanation", "")),
        "requires_confirmation": bool(
            payload.get("requires_confirmation", False)
            or method in DESTRUCTIVE_METHODS
        ),
    }


def execute_assistant_plan(
    manager: Any,
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """Validate and optionally execute a plan produced by the assistant."""

    method = str(plan.get("method", "")).strip()
    if method not in SAFE_METHODS or not hasattr(manager, method):
        raise AssistantError(f"unsupported manager method: {method!r}")
    arguments = plan.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise AssistantError("plan arguments must be an object")
    clean_arguments = _validate_arguments(manager, method, arguments)
    result: Any = None
    if execute:
        if method in DESTRUCTIVE_METHODS and not confirm:
            raise AssistantError(
                f"{method} changes dataset state; pass confirm=True to execute it"
            )
        result = getattr(manager, method)(**clean_arguments)
    return {
        "plan": {
            "method": method,
            "arguments": clean_arguments,
            "explanation": str(plan.get("explanation", "")),
            "requires_confirmation": method in DESTRUCTIVE_METHODS
            or bool(plan.get("requires_confirmation", False)),
        },
        "executed": execute,
        "result": result,
    }


def run_assistant(
    manager: Any,
    intent: str,
    provider: VLMProvider,
    *,
    execute: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """Build and optionally execute a single VLM assistant plan."""

    plan = build_assistant_plan(manager, intent, provider)
    return execute_assistant_plan(
        manager,
        plan,
        execute=execute,
        confirm=confirm,
    )


def _validate_arguments(
    manager: Any,
    method: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(getattr(manager, method))
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    result: dict[str, Any] = {}
    for key, value in arguments.items():
        name = str(key)
        if name in {"root", "command", "shell", "python", "code"}:
            raise AssistantError(f"assistant cannot override {name}")
        if name not in accepted and not has_kwargs:
            raise AssistantError(f"unsupported argument {name!r} for {method}")
        result[name] = value
    return result


__all__ = [
    "AssistantError",
    "DESTRUCTIVE_METHODS",
    "SAFE_METHODS",
    "build_assistant_plan",
    "execute_assistant_plan",
    "run_assistant",
]
