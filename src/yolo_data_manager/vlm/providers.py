"""Provider abstraction and OpenAI-compatible Qwen VLM implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, replace
import mimetypes
from pathlib import Path
from typing import Any, Callable, Sequence

from yolo_data_manager.vlm.config import VLMConfig, load_vlm_config


class VLMProviderError(RuntimeError):
    """Raised when a VLM provider cannot complete a request."""


@dataclass(frozen=True)
class VLMResponse:
    """Normalized provider response."""

    text: str
    raw: Any = None
    usage: dict[str, object] | None = None


class VLMProvider(ABC):
    """Common interface implemented by every VLM backend."""

    name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path] = (),
        json_mode: bool = True,
    ) -> VLMResponse:
        """Generate a response from text and optional images."""


class OpenAICompatibleVLMProvider(VLMProvider):
    """Provider for Qwen and other OpenAI-compatible VLM endpoints."""

    name = "openai_compatible"

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise VLMProviderError(
                "VLM support requires the openai package"
            ) from exc

        kwargs: dict[str, object] = {
            "api_key": config.api_key or "EMPTY",
            "timeout": config.timeout,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path] = (),
        json_mode: bool = True,
    ) -> VLMResponse:
        content: list[dict[str, object]] = [
            {"type": "text", "text": str(prompt)}
        ]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(image))},
                }
            )

        request: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode and self.config.json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if "response_format" not in request or not _looks_like_json_mode_error(exc):
                raise VLMProviderError(_safe_provider_error(exc)) from exc
            request.pop("response_format", None)
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as retry_exc:
                raise VLMProviderError(_safe_provider_error(retry_exc)) from retry_exc

        if not getattr(response, "choices", None):
            raise VLMProviderError("VLM response did not contain any choices")
        message = response.choices[0].message
        text = _content_to_text(getattr(message, "content", None))
        if not text:
            raise VLMProviderError("VLM response content was empty")

        usage = _usage_to_dict(getattr(response, "usage", None))
        return VLMResponse(text=text, raw=response, usage=usage)


class QwenVLMProvider(OpenAICompatibleVLMProvider):
    """Qwen-VL provider using either DashScope or a local compatible server."""

    name = "qwen"


ProviderFactory = Callable[[VLMConfig], VLMProvider]
_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "qwen": QwenVLMProvider,
    "qwen-vl": QwenVLMProvider,
    "openai": OpenAICompatibleVLMProvider,
    "openai-compatible": OpenAICompatibleVLMProvider,
    "openai_compatible": OpenAICompatibleVLMProvider,
}


def register_vlm_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory for applications and plugins."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("provider name must not be empty")
    _PROVIDER_FACTORIES[normalized] = factory


def create_vlm_provider(
    config: VLMConfig | None = None,
    *,
    provider: str | VLMProvider | None = None,
) -> VLMProvider:
    """Create a configured provider from the registry."""

    if isinstance(provider, VLMProvider):
        return provider
    resolved = config or load_vlm_config()
    if isinstance(provider, str) and provider.strip():
        resolved = replace(resolved, provider=provider.strip().lower())
    name = resolved.provider.strip().lower()
    factory = _PROVIDER_FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ValueError(f"unknown VLM provider {name!r}; available: {available}")
    return factory(resolved)


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"VLM image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip() if content is not None else ""


def _usage_to_dict(usage: object) -> dict[str, object] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        value = usage.model_dump()
        return value if isinstance(value, dict) else None
    if isinstance(usage, dict):
        return dict(usage)
    return None


def _looks_like_json_mode_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "response_format",
            "json_object",
            "structured output",
            "unsupported parameter",
        )
    )


def _safe_provider_error(exc: Exception) -> str:
    text = str(exc).strip()
    return f"VLM provider request failed: {text or type(exc).__name__}"


__all__ = [
    "OpenAICompatibleVLMProvider",
    "QwenVLMProvider",
    "VLMProvider",
    "VLMProviderError",
    "VLMResponse",
    "create_vlm_provider",
    "register_vlm_provider",
]
