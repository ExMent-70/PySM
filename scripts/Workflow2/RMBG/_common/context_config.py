"""Read and write a complete RMBG settings object through the PySM context API."""

from __future__ import annotations

import re
from typing import Any, Protocol

from .config_schema import RmbgSettings, default_settings, parse_settings


DEFAULT_CONFIG_VAR = "wf_rmbg_settings"
_CONTEXT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")


class ContextProtocol(Protocol):
    def get_structured(self, key: str, default: Any = None) -> Any: ...

    def set_structured(self, key_path: str, value: Any, commit: bool = False) -> None: ...


class ContextSettingsError(RuntimeError):
    """Base error for context configuration operations."""


class ContextSettingsMissingError(ContextSettingsError):
    """Raised when the requested context variable has not been configured."""


class ContextSettingsInvalidError(ContextSettingsError):
    """Raised when the stored JSON object does not satisfy the current schema."""


def validate_config_var(config_var: str) -> str:
    """Validate a context variable or dotted path before accessing it."""

    normalized = config_var.strip()
    if not _CONTEXT_KEY_RE.fullmatch(normalized):
        raise ValueError(
            "Имя переменной контекста должно состоять из латинских букв, "
            "цифр, подчёркиваний и необязательных dot-сегментов."
        )
    return normalized


def load_context_settings(
    context: ContextProtocol,
    config_var: str = DEFAULT_CONFIG_VAR,
    *,
    use_defaults_if_missing: bool = False,
) -> RmbgSettings:
    """Load and validate one complete settings object from context."""

    key = validate_config_var(config_var)
    raw_value = context.get_structured(key, default=None)
    if raw_value is None:
        if use_defaults_if_missing:
            return default_settings()
        raise ContextSettingsMissingError(
            f"Переменная '{key}' не настроена. Запустите RMBG Configurator."
        )
    try:
        return parse_settings(raw_value)
    except Exception as exc:
        raise ContextSettingsInvalidError(
            f"Переменная '{key}' содержит некорректную конфигурацию: {exc}"
        ) from exc


def save_context_settings(
    context: ContextProtocol,
    settings: RmbgSettings,
    config_var: str = DEFAULT_CONFIG_VAR,
    *,
    commit: bool = True,
) -> None:
    """Atomically store the validated settings object in context."""

    key = validate_config_var(config_var)
    validated = RmbgSettings.model_validate(settings)
    context.set_structured(key, validated.to_context_value(), commit=commit)
