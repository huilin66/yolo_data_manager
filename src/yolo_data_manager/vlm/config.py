"""Environment-based configuration for vision-language model providers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


def _load_dotenv(
    dotenv_path: str | Path | None = None,
) -> tuple[Path | None, dict[str, str]]:
    """Read the nearest .env file without exposing its contents.

    Values are returned separately instead of being injected into
    ``os.environ``. This lets YDM give the project ``.env`` file a stable
    priority without mutating the host process configuration.
    """

    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "VLM support requires python-dotenv; install the project VLM dependencies"
        ) from exc

    resolved: str | None
    if dotenv_path is not None:
        resolved = str(Path(dotenv_path).expanduser())
    else:
        resolved = find_dotenv(usecwd=True) or None
    if resolved is None:
        return None, {}

    values = {
        str(key): str(value)
        for key, value in dotenv_values(resolved).items()
        if key and value is not None and str(value).strip()
    }
    return Path(resolved), values


def _get(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    overrides: Mapping[str, object] | None = None,
    dotenv_values: Mapping[str, object] | None = None,
    default: str | None = None,
) -> str | None:
    if overrides:
        for key in (name, *aliases):
            if key in overrides and overrides[key] is not None:
                return str(overrides[key])
    if dotenv_values:
        for key in (name, *aliases):
            if key in dotenv_values and dotenv_values[key] is not None:
                value = str(dotenv_values[key]).strip()
                if value:
                    return value
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _float(
    name: str,
    *,
    overrides: Mapping[str, object] | None,
    dotenv_values: Mapping[str, object] | None,
    default: float,
) -> float:
    value = _get(name, overrides=overrides, dotenv_values=dotenv_values)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _int(
    name: str,
    *,
    overrides: Mapping[str, object] | None,
    dotenv_values: Mapping[str, object] | None,
    default: int,
) -> int:
    value = _get(name, overrides=overrides, dotenv_values=dotenv_values)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bool(
    name: str,
    *,
    overrides: Mapping[str, object] | None,
    dotenv_values: Mapping[str, object] | None,
    default: bool,
) -> bool:
    value = _get(name, overrides=overrides, dotenv_values=dotenv_values)
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class VLMConfig:
    """Resolved VLM settings."""

    provider: str = "qwen"
    base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-vl-max"
    api_key: str | None = None
    model_path: str | None = None
    timeout: float = 120.0
    workers: int = 4
    temperature: float = 0.0
    max_tokens: int = 4096
    json_mode: bool = True
    dotenv_path: Path | None = None

    def to_dict(self, *, include_secret: bool = False) -> dict[str, object]:
        """Return a safe configuration view."""

        payload: dict[str, object] = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "model_path": self.model_path,
            "timeout": self.timeout,
            "workers": self.workers,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "dotenv_path": str(self.dotenv_path) if self.dotenv_path else None,
        }
        payload["api_key"] = self.api_key if include_secret else (
            "<configured>" if self.api_key else None
        )
        return payload


def load_vlm_config(
    dotenv_path: str | Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> VLMConfig:
    """Load VLM settings from .env and optional in-memory overrides."""

    loaded_path, dotenv_values = _load_dotenv(dotenv_path)
    provider = (
        _get(
            "VLM_PROVIDER",
            overrides=overrides,
            dotenv_values=dotenv_values,
            default="qwen",
        )
        or "qwen"
    ).lower()
    base_url = _get(
        "VLM_BASE_URL",
        overrides=overrides,
        dotenv_values=dotenv_values,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = _get(
        "VLM_MODEL",
        overrides=overrides,
        dotenv_values=dotenv_values,
        default="qwen-vl-max",
    ) or "qwen-vl-max"
    api_key = _get(
        "VLM_API_KEY",
        aliases=("DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
        overrides=overrides,
        dotenv_values=dotenv_values,
    )
    model_path = _get(
        "VLM_MODEL_PATH",
        overrides=overrides,
        dotenv_values=dotenv_values,
    )
    workers = _int(
        "VLM_WORKERS",
        overrides=overrides,
        dotenv_values=dotenv_values,
        default=4,
    )
    if workers < 1:
        raise ValueError("VLM_WORKERS must be at least 1")

    return VLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        model_path=model_path,
        timeout=_float(
            "VLM_TIMEOUT",
            overrides=overrides,
            dotenv_values=dotenv_values,
            default=120.0,
        ),
        workers=workers,
        temperature=_float(
            "VLM_TEMPERATURE",
            overrides=overrides,
            dotenv_values=dotenv_values,
            default=0.0,
        ),
        max_tokens=_int(
            "VLM_MAX_TOKENS",
            overrides=overrides,
            dotenv_values=dotenv_values,
            default=4096,
        ),
        json_mode=_bool(
            "VLM_JSON_MODE",
            overrides=overrides,
            dotenv_values=dotenv_values,
            default=True,
        ),
        dotenv_path=loaded_path,
    )


__all__ = ["VLMConfig", "load_vlm_config"]
