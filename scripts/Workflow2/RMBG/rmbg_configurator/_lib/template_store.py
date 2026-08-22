"""Persistent named RMBG templates stored beside the configurator script."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common.config_schema import RmbgSettings


TEMPLATE_FILE_TYPE = "pysm_rmbg_templates"
TEMPLATE_SCHEMA_VERSION = 1


class TemplateStoreError(RuntimeError):
    """Raised when the user template collection cannot be read or updated."""


@dataclass(frozen=True, slots=True)
class RmbgTemplate:
    template_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    settings: RmbgSettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.template_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "settings": self.settings.to_context_value(),
        }


class TemplateStore:
    """Manage one atomically written collection of validated named templates."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")

    def ensure_exists(self) -> None:
        if not self.path.exists():
            self._write_templates(())

    def list_templates(self) -> tuple[RmbgTemplate, ...]:
        self.ensure_exists()
        document = self._read_document()
        templates = tuple(self._parse_template(item) for item in document["templates"])
        return tuple(sorted(templates, key=lambda item: item.name.casefold()))

    def get(self, template_id: str) -> RmbgTemplate:
        for item in self.list_templates():
            if item.template_id == template_id:
                return item
        raise TemplateStoreError("Выбранный шаблон больше не существует.")

    def create(
        self,
        *,
        name: str,
        description: str,
        settings: RmbgSettings,
    ) -> RmbgTemplate:
        templates = list(self.list_templates())
        normalized_name = self._validate_name(name)
        self._assert_unique_name(templates, normalized_name)
        timestamp = _utc_now()
        item = RmbgTemplate(
            template_id=uuid.uuid4().hex,
            name=normalized_name,
            description=description.strip(),
            created_at=timestamp,
            updated_at=timestamp,
            settings=settings,
        )
        templates.append(item)
        self._write_templates(templates)
        return item

    def update(
        self,
        template_id: str,
        *,
        name: str,
        description: str,
        settings: RmbgSettings | None = None,
    ) -> RmbgTemplate:
        templates = list(self.list_templates())
        normalized_name = self._validate_name(name)
        self._assert_unique_name(
            templates,
            normalized_name,
            excluding=template_id,
        )
        for index, current in enumerate(templates):
            if current.template_id != template_id:
                continue
            updated = RmbgTemplate(
                template_id=current.template_id,
                name=normalized_name,
                description=description.strip(),
                created_at=current.created_at,
                updated_at=_utc_now(),
                settings=settings or current.settings,
            )
            templates[index] = updated
            self._write_templates(templates)
            return updated
        raise TemplateStoreError("Выбранный шаблон больше не существует.")

    def delete(self, template_id: str) -> None:
        templates = list(self.list_templates())
        remaining = [item for item in templates if item.template_id != template_id]
        if len(remaining) == len(templates):
            raise TemplateStoreError("Выбранный шаблон больше не существует.")
        self._write_templates(remaining)

    @staticmethod
    def _validate_name(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise TemplateStoreError("Введите название шаблона.")
        if len(normalized) > 120:
            raise TemplateStoreError("Название шаблона не должно превышать 120 символов.")
        return normalized

    @staticmethod
    def _assert_unique_name(
        templates: list[RmbgTemplate],
        name: str,
        *,
        excluding: str | None = None,
    ) -> None:
        if any(
            item.template_id != excluding and item.name.casefold() == name.casefold()
            for item in templates
        ):
            raise TemplateStoreError(f"Шаблон с названием «{name}» уже существует.")

    def _read_document(self) -> dict[str, Any]:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemplateStoreError(
                f"Не удалось прочитать файл шаблонов {self.path}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise TemplateStoreError("Файл шаблонов должен содержать JSON-объект.")
        if document.get("file_type") != TEMPLATE_FILE_TYPE:
            raise TemplateStoreError("Файл не является коллекцией шаблонов RMBG.")
        if document.get("schema_version") != TEMPLATE_SCHEMA_VERSION:
            raise TemplateStoreError("Версия файла шаблонов пока не поддерживается.")
        if not isinstance(document.get("templates"), list):
            raise TemplateStoreError("В файле шаблонов отсутствует массив templates.")
        return document

    @staticmethod
    def _parse_template(value: Any) -> RmbgTemplate:
        if not isinstance(value, dict):
            raise TemplateStoreError("Запись шаблона должна быть JSON-объектом.")
        try:
            return RmbgTemplate(
                template_id=str(value["id"]),
                name=str(value["name"]),
                description=str(value.get("description", "")),
                created_at=str(value["created_at"]),
                updated_at=str(value["updated_at"]),
                settings=RmbgSettings.model_validate(value["settings"]),
            )
        except Exception as exc:
            raise TemplateStoreError(f"Некорректная запись шаблона: {exc}") from exc

    def _write_templates(self, templates: tuple[RmbgTemplate, ...] | list[RmbgTemplate]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "file_type": TEMPLATE_FILE_TYPE,
            "schema_version": TEMPLATE_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "templates": [item.to_dict() for item in templates],
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if self.path.is_file():
                shutil.copy2(self.path, self.backup_path)
            os.replace(temporary, self.path)
        except OSError as exc:
            raise TemplateStoreError(
                f"Не удалось сохранить файл шаблонов {self.path}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
