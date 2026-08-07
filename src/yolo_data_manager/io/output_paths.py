"""Canonical default output locations for YOLO Data Manager tasks.

The helpers in this module deliberately only construct paths.  They do not
create directories, so callers can keep their existing write/overwrite
semantics and an explicitly supplied output path can always take precedence.
"""

from __future__ import annotations

from pathlib import Path


def ydm_dir(root: str | Path, group: str) -> Path:
    """Return a functional output group below a dataset root."""

    return Path(root) / f"ydm_{group}"


def ydm_operation_dir(root: str | Path, group: str, operation: str) -> Path:
    """Return ``<root>/ydm_<group>/<operation>``."""

    return ydm_dir(root, group) / operation


def ydm_output_path(
    root: str | Path,
    group: str,
    operation: str,
    filename: str,
) -> Path:
    """Return a file path in a canonical operation directory."""

    return ydm_operation_dir(root, group, operation) / filename


def default_dataset_output(root: str | Path, operation: str) -> Path:
    """Return the default dataset-producing output root."""

    return ydm_operation_dir(root, "dataset", operation)


def default_annotation_output(root: str | Path, operation: str) -> Path:
    """Return the default annotation-edit output root."""

    return ydm_operation_dir(root, "annotation", operation)


def default_visualization_output(root: str | Path, operation: str) -> Path:
    """Return the default visualization output directory."""

    return ydm_operation_dir(root, "vis", operation)


def default_evaluation_output(root: str | Path, operation: str) -> Path:
    """Return the default evaluation output directory."""

    return ydm_operation_dir(root, "evaluation", operation)


def default_stats_output(root: str | Path, operation: str) -> Path:
    """Return the default statistics output directory."""

    return ydm_operation_dir(root, "stats", operation)


def default_quality_output(root: str | Path, operation: str) -> Path:
    """Return the default data-quality output directory."""

    return ydm_operation_dir(root, "quality", operation)


def default_conversion_output(root: str | Path, operation: str) -> Path:
    """Return the default conversion output directory."""

    return ydm_operation_dir(root, "conversion", operation)

