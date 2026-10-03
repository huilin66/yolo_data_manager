"""Extensible VLM integrations for YOLO Data Manager."""

from yolo_data_manager.vlm.assistant import (
    AssistantError,
    build_assistant_plan,
    execute_assistant_plan,
    run_assistant,
)
from yolo_data_manager.vlm.auto_label import AutoLabelSummary, generate_yolo_labels
from yolo_data_manager.vlm.config import VLMConfig, load_vlm_config
from yolo_data_manager.vlm.error_verify import (
    ErrorCrop,
    apply_correction_plan,
    collect_error_crops,
    verify_error_crops,
)
from yolo_data_manager.vlm.providers import (
    OpenAICompatibleVLMProvider,
    QwenVLMProvider,
    VLMProvider,
    VLMProviderError,
    VLMResponse,
    create_vlm_provider,
    register_vlm_provider,
)
from yolo_data_manager.vlm.schemas import (
    VLMBox,
    VLMDecision,
    VLMOutputError,
    extract_json_object,
    parse_detection_response,
    parse_error_decision,
)

__all__ = [
    "AssistantError",
    "AutoLabelSummary",
    "ErrorCrop",
    "OpenAICompatibleVLMProvider",
    "QwenVLMProvider",
    "VLMBox",
    "VLMConfig",
    "VLMDecision",
    "VLMOutputError",
    "VLMProvider",
    "VLMProviderError",
    "VLMResponse",
    "apply_correction_plan",
    "build_assistant_plan",
    "collect_error_crops",
    "create_vlm_provider",
    "execute_assistant_plan",
    "extract_json_object",
    "generate_yolo_labels",
    "load_vlm_config",
    "parse_detection_response",
    "parse_error_decision",
    "register_vlm_provider",
    "run_assistant",
    "verify_error_crops",
]
