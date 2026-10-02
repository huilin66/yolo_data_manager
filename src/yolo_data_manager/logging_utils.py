"""Timestamped operation logging for YOLO Data Manager tasks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import json
from pathlib import Path
import sys
from threading import Lock
import time
from typing import Any, Iterator

import yaml


LOG_DIR_NAME = "ydm_log"
_WRITE_LOCK = Lock()
_ACTIVE_LOG_ROOT: ContextVar[Path | None] = ContextVar(
    "ydm_active_log_root",
    default=None,
)


def now_text() -> str:
    """Return a local timestamp suitable for console and log output."""

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f%z")


def resolve_log_root(root: str | Path | None) -> Path:
    """Resolve a dataset YAML/root to the directory holding ``ydm_log``."""

    if root is None:
        return Path.cwd()

    root_path = Path(root).expanduser()
    if root_path.suffix.lower() not in {".yaml", ".yml"} or not root_path.is_file():
        return root_path

    try:
        data = yaml.safe_load(root_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return root_path.parent
    if not isinstance(data, dict):
        return root_path.parent

    configured_root = data.get("path")
    if configured_root is None or not str(configured_root).strip():
        return root_path.parent
    text = str(configured_root).strip()
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root_path.parent / candidate).resolve()


def log_file(root: str | Path | None, *, when: datetime | None = None) -> Path:
    """Return the daily log file path for a dataset root."""

    timestamp = when or datetime.now().astimezone()
    return (
        resolve_log_root(root)
        / LOG_DIR_NAME
        / f"{timestamp.strftime('%Y-%m-%d')}.log"
    )


def _json_default(value: Any) -> str:
    return str(value)


def summarize(values: dict[str, Any] | None) -> str:
    """Serialize operation details without allowing logging to break a task."""

    if not values:
        return ""
    filtered = {
        key: value
        for key, value in values.items()
        if key not in {"handler", "_output_operation"}
    }
    try:
        text = json.dumps(
            filtered,
            ensure_ascii=False,
            default=_json_default,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        text = repr(filtered)
    if len(text) > 4000:
        text = text[:3997] + "..."
    return text


def write_event(
    root: str | Path | None,
    level: str,
    message: str,
    *,
    when: datetime | None = None,
) -> Path:
    """Append one event to the dataset's current daily log file."""

    path = log_file(root, when=when)
    timestamp = (when or datetime.now().astimezone()).strftime(
        "%Y-%m-%d %H:%M:%S.%f%z"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return path
    line = f"{timestamp} [{level.upper()}] {message}\n"
    try:
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
    except OSError:
        return path
    return path


def console_event(
    level: str,
    message: str,
    *,
    stream=None,
    record: bool = True,
) -> None:
    """Print a timestamped event without changing structured stdout output.

    Events emitted inside an operation scope are also appended to that
    operation's daily log. Logging failures never interrupt the task.
    """

    target = sys.stderr if stream is None else stream
    print(f"[{now_text()}] [{level.upper()}] {message}", file=target, flush=True)
    if record:
        active_root = _ACTIVE_LOG_ROOT.get()
        if active_root is not None:
            write_event(active_root, level, message)


@contextmanager
def operation_scope(
    root: str | Path | None,
    operation: str,
    details: dict[str, Any] | None = None,
    *,
    announce: bool = True,
) -> Iterator[Path]:
    """Log and announce an operation, including duration and failures."""

    detail_text = summarize(details)
    suffix = f" params={detail_text}" if detail_text else ""
    started = time.perf_counter()
    active_root = resolve_log_root(root)
    token = _ACTIVE_LOG_ROOT.set(active_root)
    try:
        path = write_event(active_root, "INFO", f"START operation={operation}{suffix}")
        if announce:
            console_event(
                "INFO",
                f"START operation={operation} log={path}",
                record=False,
            )
        try:
            yield path
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            message = (
                f"END operation={operation} status=error elapsed={elapsed:.3f}s "
                f"exception={type(exc).__name__}: {exc}"
            )
            write_event(active_root, "ERROR", message)
            if announce:
                console_event("ERROR", message, record=False)
            raise
        else:
            elapsed = time.perf_counter() - started
            message = f"END operation={operation} status=ok elapsed={elapsed:.3f}s"
            write_event(active_root, "INFO", message)
            if announce:
                console_event("INFO", message, record=False)
    finally:
        _ACTIVE_LOG_ROOT.reset(token)


__all__ = [
    "LOG_DIR_NAME",
    "console_event",
    "log_file",
    "now_text",
    "operation_scope",
    "resolve_log_root",
    "summarize",
    "write_event",
]
