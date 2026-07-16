"""Shared helpers for PySM context variable operations.

The helpers keep one contract for top-level and dotted context keys:
plain names operate on regular context variables, while dotted names operate on
nested values inside JSON context variables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

try:
    from .pysm_icons import icons
except Exception:  # pragma: no cover - fallback for standalone/import-limited runs.
    icons = None


@dataclass(frozen=True)
class ContextValue:
    """Result of reading a context key."""

    exists: bool
    value: Any = None
    var_type: Optional[str] = None
    read_only: bool = False
    is_top_level: bool = False


def _icon(name: str, fallback: str, size: int = 18) -> str:
    if icons is None:
        return fallback
    try:
        return getattr(icons, name)(size=size)
    except Exception:
        return fallback


def success_icon(size: int = 18) -> str:
    """Return the configured success icon as HTML."""

    return _icon("OK", "✅", size=size)


def error_icon(size: int = 18) -> str:
    """Return the configured error icon as HTML."""

    return _icon("ERROR", "❌", size=size)


def format_success(var_name: str, value: Any) -> str:
    """Format a standard context-write success log line."""

    return f"{success_icon()} <b>{var_name}</b> = <i>{value}</i>"


def format_error(message: str) -> str:
    """Format a standard context-operation error log line."""

    return f"{error_icon()} ОШИБКА: {message}"


def read_context_value(pysm_context: Any, key: str, default: Any = None) -> ContextValue:
    """Read a top-level or dotted context value."""

    if not pysm_context or not key:
        return ContextValue(False, default)

    variable_data = pysm_context.get_variable(key)
    if isinstance(variable_data, dict):
        return ContextValue(
            exists=True,
            value=variable_data.get("value", default),
            var_type=variable_data.get("type"),
            read_only=bool(variable_data.get("read_only", False)),
            is_top_level=True,
        )

    if not pysm_context.exists(key):
        return ContextValue(False, default)

    base_var_name = key.split(".", 1)[0]
    base_data = pysm_context.get_variable(base_var_name)
    read_only = bool(base_data.get("read_only", False)) if isinstance(base_data, dict) else False
    return ContextValue(
        exists=True,
        value=pysm_context.get_structured(key, default),
        var_type=None,
        read_only=read_only,
        is_top_level=False,
    )


def context_value_exists(pysm_context: Any, key: str) -> bool:
    """Return True when a top-level or dotted context key exists."""

    return bool(pysm_context and key and pysm_context.exists(key))


def write_context_value(
    pysm_context: Any,
    key: str,
    value: Any,
    *,
    var_type: Optional[str] = None,
    commit: bool = False,
) -> None:
    """Write a value to a top-level or dotted context key."""

    if not pysm_context:
        return

    if "." in key:
        pysm_context.set_structured(key, value, commit=commit)
        return

    pysm_context.set(key, value, var_type=var_type, commit=commit)


def remove_context_value(pysm_context: Any, key: Optional[str] = None, *, commit: bool = False) -> None:
    """Remove a top-level or dotted context key."""

    if not pysm_context:
        return
    pysm_context.remove(key, commit=commit)


def copy_context_value(pysm_context: Any, source_key: str, target_key: str) -> ContextValue:
    """Copy a top-level or dotted value and preserve top-level source type when possible."""

    source = read_context_value(pysm_context, source_key)
    if not source.exists:
        return source

    write_context_value(
        pysm_context,
        target_key,
        source.value,
        var_type=source.var_type if source.is_top_level else None,
    )
    return source


def initial_value_as_text(pysm_context: Any, key: str, default: str = "") -> str:
    """Return context value suitable for text inputs."""

    result = read_context_value(pysm_context, key)
    if not result.exists or result.value is None:
        return default
    if isinstance(result.value, (dict, list)):
        return json.dumps(result.value, ensure_ascii=False)
    return str(result.value)
