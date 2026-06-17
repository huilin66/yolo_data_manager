class YoloDataManagerError(Exception):
    """Base exception for YOLO Data Manager."""


class DatasetLoadError(YoloDataManagerError):
    """Raised when a dataset cannot be loaded."""


class ClassNotFoundError(YoloDataManagerError):
    """Raised when a class id or class name cannot be resolved."""

