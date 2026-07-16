"""Storage backends for PySM runtime context."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict


class ContextStoreError(RuntimeError):
    """Base error for context storage failures."""


class FileContextStore:
    """Compatibility context store backed by the existing JSON file."""

    backend_name = "file"

    def __init__(self, path: pathlib.Path):
        self.path = path

    @property
    def generation(self) -> int:
        try:
            return int(self.path.stat().st_mtime_ns)
        except OSError:
            return 0

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.{id(data)}.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.path)

    def close(self) -> None:
        return None

    def unlink(self) -> None:
        return None
