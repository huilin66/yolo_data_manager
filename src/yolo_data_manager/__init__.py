"""YOLO Data Manager public API."""

__version__ = "1.2.1"

from yolo_data_manager.core.models import (
    AttributeSchema,
    Box,
    ClassSchema,
    Polygon,
    YoloAnnotation,
    YoloDataset,
    YoloImage,
)
from yolo_data_manager.core.multimodal import (
    AlignmentIssue,
    AlignmentReport,
    ModalityConfig,
    MultimodalImage,
    MultimodalScene,
    MultimodalYoloDataset,
)
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.io.image_conversion import convert_multimodal_images_to_uint8
from yolo_data_manager.io.multimodal import load_multimodal_yolo_dataset
from yolo_data_manager.evaluation.metrics import (
    SizeMetric,
    compute_detection_metrics,
    format_metrics_table,
)
from yolo_data_manager.evaluation.common_errors import extract_common_error_analysis
from yolo_data_manager.evaluation.error_analysis import (
    DEFAULT_CONF_CURVE_THRESHOLDS,
    write_confidence_curve,
)
from yolo_data_manager.annotation.crop_correction import (
    AttributeCropCorrectionResult,
    CropCorrectionResult,
    correct_gt_attributes_from_crops,
    correct_gt_attributes_from_error_crops,
    correct_gt_labels_from_error_crop_map,
    correct_gt_labels_from_error_crops,
    correct_labels_from_crop_map,
    correct_labels_from_crops,
)
from yolo_data_manager.annotation.remap import apply_class_map_data
from yolo_data_manager.annotation.edit import set_attributes_from_map
from yolo_data_manager.multimodal_manager import MultiModalYoloManager
from yolo_data_manager.scripting import YoloManager, build_task_argv, run_task
from yolo_data_manager.stats.multimodal import compute_multimodal_stats, write_multimodal_stats_plots
from yolo_data_manager.vis.multimodal import crop_multimodal_dataset, render_multimodal_dataset
from yolo_data_manager.vis.manual_box import ManualBoxResult, draw_manual_box, format_yolo_line
from yolo_data_manager.tools.image_resize import ResizeResult, resize_image, resize_yolo_dataset
from yolo_data_manager.tools.filename_remap import (
    FilenameRemapItem,
    FilenameRemapResult,
    filename_digits_for_count,
    remap_yolo_dataset_filenames,
)
from yolo_data_manager.vlm import (
    AssistantError,
    AutoLabelSummary,
    VLMBox,
    VLMConfig,
    VLMDecision,
    VLMOutputError,
    VLMProvider,
    VLMProviderError,
    VLMResponse,
    apply_correction_plan,
    build_assistant_plan,
    collect_error_crops,
    create_vlm_provider,
    execute_assistant_plan,
    generate_yolo_labels,
    load_vlm_config,
    run_assistant,
    verify_error_crops,
)

__all__ = [
    "AttributeSchema",
    "Box",
    "ClassSchema",
    "Polygon",
    "YoloAnnotation",
    "YoloDataset",
    "YoloImage",
    "AlignmentIssue",
    "AlignmentReport",
    "ModalityConfig",
    "MultimodalImage",
    "MultimodalScene",
    "MultimodalYoloDataset",
    "MultiModalYoloManager",
    "ManualBoxResult",
    "YoloManager",
    "compute_multimodal_stats",
    "convert_multimodal_images_to_uint8",
    "compute_detection_metrics",
    "SizeMetric",
    "extract_common_error_analysis",
    "DEFAULT_CONF_CURVE_THRESHOLDS",
    "write_confidence_curve",
    "AttributeCropCorrectionResult",
    "CropCorrectionResult",
    "correct_gt_attributes_from_crops",
    "correct_gt_attributes_from_error_crops",
    "correct_labels_from_crops",
    "correct_labels_from_crop_map",
    "correct_gt_labels_from_error_crop_map",
    "correct_gt_labels_from_error_crops",
    "apply_class_map_data",
    "set_attributes_from_map",
    "draw_manual_box",
    "crop_multimodal_dataset",
    "format_metrics_table",
    "format_yolo_line",
    "load_multimodal_yolo_dataset",
    "load_yolo_dataset",
    "render_multimodal_dataset",
    "write_multimodal_stats_plots",
    "build_task_argv",
    "run_task",
    "ResizeResult",
    "resize_image",
    "resize_yolo_dataset",
    "FilenameRemapItem",
    "FilenameRemapResult",
    "filename_digits_for_count",
    "remap_yolo_dataset_filenames",
    "AssistantError",
    "AutoLabelSummary",
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
    "generate_yolo_labels",
    "load_vlm_config",
    "run_assistant",
    "verify_error_crops",
]
