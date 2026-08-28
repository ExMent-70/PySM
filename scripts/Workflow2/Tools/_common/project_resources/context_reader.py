"""Stable read-only access to values stored in a PySM context JSON file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable


class ContextReadError(ValueError):
    """A context file cannot be read as one stable JSON object."""


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def read_context_values(
    path: Path,
    names: Iterable[str],
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    """Return selected unwrapped values without modifying the context file."""

    requested = tuple(dict.fromkeys(str(name) for name in names))
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            before = path.stat()
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (
                after.st_mtime_ns,
                after.st_size,
            ):
                raise ContextReadError("Файл изменился во время чтения.")
            if not isinstance(payload, dict):
                raise ContextReadError("Корень context.json должен быть JSON-объектом.")
            return {name: _unwrap_value(payload.get(name)) for name in requested}
        except FileNotFoundError as exc:
            raise ContextReadError(f"Файл контекста не найден: {path}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError, ContextReadError) as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(0.05)

    raise ContextReadError(
        f"Не удалось прочитать {path.name}: {last_error}"
    ) from last_error
