"""YOLO Data Manager public API."""

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
from yolo_data_manager.evaluation.metrics import compute_detection_metrics, format_metrics_table
from yolo_data_manager.multimodal_manager import MultiModalYoloManager
from yolo_data_manager.scripting import YoloManager, build_task_argv, run_task
from yolo_data_manager.stats.multimodal import compute_multimodal_stats, write_multimodal_stats_plots
from yolo_data_manager.vis.multimodal import crop_multimodal_dataset, render_multimodal_dataset

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
    "YoloManager",
    "compute_multimodal_stats",
    "convert_multimodal_images_to_uint8",
    "compute_detection_metrics",
    "crop_multimodal_dataset",
    "format_metrics_table",
    "load_multimodal_yolo_dataset",
    "load_yolo_dataset",
    "render_multimodal_dataset",
    "write_multimodal_stats_plots",
    "build_task_argv",
    "run_task",
]
