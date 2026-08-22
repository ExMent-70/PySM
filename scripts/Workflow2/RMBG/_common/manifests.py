"""Validated access to pinned upstream and model metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
UPSTREAM_LOCK_PATH = RESOURCE_DIR / "upstream.lock.json"
MODELS_LOCK_PATH = RESOURCE_DIR / "models.lock.json"
UPSTREAM_MAP_PATH = RESOURCE_DIR / "upstream_map.json"
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(RuntimeError):
    """Raised when a supplied lock file cannot be trusted."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Не удалось прочитать manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Manifest должен содержать JSON-объект: {path}")
    return value


def load_upstream_lock(path: Path = UPSTREAM_LOCK_PATH) -> dict[str, Any]:
    value = _read_json_object(path)
    if value.get("schema_version") != 1:
        raise ManifestError("Неподдерживаемая версия upstream lock.")
    if not _HEX40_RE.fullmatch(str(value.get("commit", ""))):
        raise ManifestError("upstream lock содержит некорректный commit SHA.")
    if not _HEX40_RE.fullmatch(str(value.get("tree", ""))):
        raise ManifestError("upstream lock содержит некорректный tree SHA.")
    return value


def load_models_lock(path: Path = MODELS_LOCK_PATH) -> dict[str, Any]:
    value = _read_json_object(path)
    if value.get("schema_version") != 1:
        raise ManifestError("Неподдерживаемая версия models lock.")
    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise ManifestError("models lock не содержит описаний моделей.")
    return value


def load_upstream_map(path: Path = UPSTREAM_MAP_PATH) -> dict[str, Any]:
    value = _read_json_object(path)
    if value.get("schema_version") != 1:
        raise ManifestError("Неподдерживаемая версия upstream map.")
    return value
