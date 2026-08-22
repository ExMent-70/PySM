"""On-disk test sessions, mask sets and reusable intermediate mask caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common.config_schema import RmbgSettings


SESSION_FILE_TYPE = "pysm_rmbg_mask_session"
SET_FILE_TYPE = "pysm_rmbg_mask_set"
PREVIEW_SCHEMA_VERSION = 1
_SAFE_CHARS = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._-]+")


class PreviewStoreError(RuntimeError):
    """Raised when a managed test session is missing, invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class TestSource:
    source_id: str
    filename: str
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class TestSession:
    session_id: str
    name: str
    path: Path
    created_at: str
    next_set_number: int
    sources: tuple[TestSource, ...]


@dataclass(frozen=True, slots=True)
class TestMaskSet:
    set_id: str
    number: int
    name: str
    path: Path
    created_at: str
    status: str
    settings: RmbgSettings
    source_masks: tuple[dict[str, Any], ...]
    base_cache_keys: tuple[str, ...]
    refined_cache_keys: tuple[str, ...]
    error: str | None = None


class PreviewStore:
    """Own all paths below one RMBG preview root and validate every mutation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> tuple[TestSession, ...]:
        if not self.root.is_dir():
            return ()
        sessions: list[TestSession] = []
        for child in self.root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = child / "session.json"
            if not manifest.is_file():
                continue
            try:
                sessions.append(self._read_session(manifest))
            except PreviewStoreError:
                continue
        return tuple(sorted(sessions, key=lambda item: item.created_at, reverse=True))

    def create_session(self, name: str, source_paths: tuple[Path, ...]) -> TestSession:
        normalized_name = name.strip()
        if not normalized_name:
            raise PreviewStoreError("Введите название тестовой сессии.")
        unique_sources = tuple(dict.fromkeys(path.resolve() for path in source_paths))
        if not unique_sources:
            raise PreviewStoreError("Выберите хотя бы одно тестовое изображение.")
        for source in unique_sources:
            if not source.is_file():
                raise PreviewStoreError(f"Тестовое изображение не найдено: {source}")

        self.ensure_root()
        session_id = uuid.uuid4().hex
        folder = self.root / f"{_timestamp_slug()}_{_safe_slug(normalized_name)}"
        suffix = 2
        while folder.exists():
            folder = self.root / (
                f"{_timestamp_slug()}_{_safe_slug(normalized_name)}_{suffix}"
            )
            suffix += 1
        sources_dir = folder / "Sources"
        sources_dir.mkdir(parents=True)
        (folder / "Sets").mkdir()
        (folder / "Cache" / "Base").mkdir(parents=True)
        (folder / "Cache" / "Refined").mkdir(parents=True)

        copied: list[TestSource] = []
        used_names: set[str] = set()
        for index, source in enumerate(unique_sources, start=1):
            filename = _unique_filename(source.name, used_names, index)
            destination = sources_dir / filename
            shutil.copy2(source, destination)
            copied.append(
                TestSource(
                    source_id=uuid.uuid4().hex,
                    filename=filename,
                    relative_path=f"Sources/{filename}",
                    sha256=_sha256_file(destination),
                    size=destination.stat().st_size,
                )
            )

        document = {
            "file_type": SESSION_FILE_TYPE,
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "id": session_id,
            "name": normalized_name,
            "created_at": _utc_now(),
            "next_set_number": 1,
            "sources": [
                {
                    "source_id": source.source_id,
                    "filename": source.filename,
                    "relative_path": source.relative_path,
                    "sha256": source.sha256,
                    "size": source.size,
                }
                for source in copied
            ],
        }
        _write_json_atomic(folder / "session.json", document)
        return self._read_session(folder / "session.json")

    def get_session(self, session_id: str) -> TestSession:
        for session in self.list_sessions():
            if session.session_id == session_id:
                return session
        raise PreviewStoreError("Выбранная тестовая сессия больше не существует.")

    def list_sets(self, session_id: str) -> tuple[TestMaskSet, ...]:
        session = self.get_session(session_id)
        sets_dir = session.path / "Sets"
        results: list[TestMaskSet] = []
        if sets_dir.is_dir():
            for folder in sets_dir.iterdir():
                manifest = folder / "set.json"
                if folder.is_dir() and manifest.is_file():
                    results.append(self._read_set(manifest))
        return tuple(sorted(results, key=lambda item: item.number))

    def get_set(self, session_id: str, set_id: str) -> TestMaskSet:
        for mask_set in self.list_sets(session_id):
            if mask_set.set_id == set_id:
                return mask_set
        raise PreviewStoreError("Выбранный тестовый набор больше не существует.")

    def create_set(
        self,
        session_id: str,
        *,
        name: str,
        settings: RmbgSettings,
    ) -> TestMaskSet:
        session = self.get_session(session_id)
        normalized_name = name.strip() or f"Набор {session.next_set_number:03d}"
        document = json.loads((session.path / "session.json").read_text(encoding="utf-8"))
        number = int(document["next_set_number"])
        document["next_set_number"] = number + 1
        _write_json_atomic(session.path / "session.json", document)

        set_id = uuid.uuid4().hex
        folder = session.path / "Sets" / f"{number:03d}_{_safe_slug(normalized_name)}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "Masks").mkdir()
        set_document = {
            "file_type": SET_FILE_TYPE,
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "id": set_id,
            "number": number,
            "name": normalized_name,
            "created_at": _utc_now(),
            "status": "generating",
            "settings": settings.to_context_value(),
            "source_masks": [],
            "base_cache_keys": [],
            "refined_cache_keys": [],
            "error": None,
        }
        _write_json_atomic(folder / "set.json", set_document)
        return self._read_set(folder / "set.json")

    def complete_set(
        self,
        session_id: str,
        set_id: str,
        *,
        source_masks: list[dict[str, Any]],
        base_cache_keys: list[str],
        refined_cache_keys: list[str],
    ) -> TestMaskSet:
        mask_set = self.get_set(session_id, set_id)
        document = json.loads((mask_set.path / "set.json").read_text(encoding="utf-8"))
        document.update(
            status="complete",
            source_masks=source_masks,
            base_cache_keys=sorted(set(base_cache_keys)),
            refined_cache_keys=sorted(set(refined_cache_keys)),
            error=None,
        )
        _write_json_atomic(mask_set.path / "set.json", document)
        return self._read_set(mask_set.path / "set.json")

    def fail_set(self, session_id: str, set_id: str, error: str) -> None:
        mask_set = self.get_set(session_id, set_id)
        document = json.loads((mask_set.path / "set.json").read_text(encoding="utf-8"))
        document.update(status="failed", error=error)
        _write_json_atomic(mask_set.path / "set.json", document)

    def delete_sets(self, session_id: str, set_ids: tuple[str, ...]) -> None:
        session = self.get_session(session_id)
        targets = [self.get_set(session_id, set_id).path for set_id in set_ids]
        for target in targets:
            self._remove_managed_directory(target, session.path / "Sets")
        self.cleanup_unreferenced_cache(session_id)

    def delete_all_sets(self, session_id: str) -> None:
        self.delete_sets(
            session_id,
            tuple(item.set_id for item in self.list_sets(session_id)),
        )

    def clear_cache(self, session_id: str) -> None:
        session = self.get_session(session_id)
        cache_root = session.path / "Cache"
        for name in ("Base", "Refined"):
            target = cache_root / name
            if target.exists():
                self._remove_managed_directory(target, cache_root)
            target.mkdir(parents=True, exist_ok=True)

    def cleanup_unreferenced_cache(self, session_id: str) -> None:
        session = self.get_session(session_id)
        sets = self.list_sets(session_id)
        referenced_base = {key for item in sets for key in item.base_cache_keys}
        referenced_refined = {key for item in sets for key in item.refined_cache_keys}
        for folder_name, referenced in (
            ("Base", referenced_base),
            ("Refined", referenced_refined),
        ):
            folder = session.path / "Cache" / folder_name
            if not folder.is_dir():
                continue
            for path in folder.glob("*.npy"):
                if path.stem not in referenced:
                    path.unlink(missing_ok=True)

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        self._remove_managed_directory(session.path, self.root)

    def cache_path(self, session_id: str, layer: str, key: str) -> Path:
        if layer not in {"Base", "Refined"} or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise PreviewStoreError("Некорректный ключ промежуточного кэша.")
        session = self.get_session(session_id)
        return session.path / "Cache" / layer / f"{key}.npy"

    def _read_session(self, manifest: Path) -> TestSession:
        value = _read_json(manifest)
        if value.get("file_type") != SESSION_FILE_TYPE:
            raise PreviewStoreError(f"Некорректная тестовая сессия: {manifest.parent}")
        try:
            sources = tuple(TestSource(**item) for item in value["sources"])
            return TestSession(
                session_id=str(value["id"]),
                name=str(value["name"]),
                path=manifest.parent.resolve(),
                created_at=str(value["created_at"]),
                next_set_number=int(value["next_set_number"]),
                sources=sources,
            )
        except Exception as exc:
            raise PreviewStoreError(f"Повреждён session.json: {exc}") from exc

    @staticmethod
    def _read_set(manifest: Path) -> TestMaskSet:
        value = _read_json(manifest)
        if value.get("file_type") != SET_FILE_TYPE:
            raise PreviewStoreError(f"Некорректный набор масок: {manifest.parent}")
        try:
            return TestMaskSet(
                set_id=str(value["id"]),
                number=int(value["number"]),
                name=str(value["name"]),
                path=manifest.parent.resolve(),
                created_at=str(value["created_at"]),
                status=str(value["status"]),
                settings=RmbgSettings.model_validate(value["settings"]),
                source_masks=tuple(value.get("source_masks", [])),
                base_cache_keys=tuple(value.get("base_cache_keys", [])),
                refined_cache_keys=tuple(value.get("refined_cache_keys", [])),
                error=value.get("error"),
            )
        except Exception as exc:
            raise PreviewStoreError(f"Повреждён set.json: {exc}") from exc

    @staticmethod
    def _remove_managed_directory(target: Path, parent: Path) -> None:
        resolved_target = target.resolve()
        resolved_parent = parent.resolve()
        if resolved_target.parent != resolved_parent:
            raise PreviewStoreError(f"Отказано в удалении неожиданного пути: {target}")
        if not resolved_target.is_dir():
            raise PreviewStoreError(f"Папка для удаления не найдена: {target}")
        _assert_no_reparse_points(resolved_target)
        shutil.rmtree(resolved_target)


def _assert_no_reparse_points(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if path.is_symlink() or attributes & reparse_flag:
                raise PreviewStoreError(
                    f"Удаление остановлено: внутри управляемой папки есть ссылка {path}"
                )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreviewStoreError(f"Не удалось прочитать {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PREVIEW_SCHEMA_VERSION:
        raise PreviewStoreError(f"Неподдерживаемый JSON-файл: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_slug(value: str) -> str:
    slug = _SAFE_CHARS.sub("_", value.strip()).strip("._-")
    return slug[:80] or "Session"


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unique_filename(original: str, used: set[str], number: int) -> str:
    path = Path(original)
    stem = _safe_slug(path.stem)
    candidate = f"{stem}{path.suffix.lower()}"
    stem_key = f"stem:{stem.casefold()}"
    if candidate.casefold() in used or stem_key in used:
        stem = f"{stem}_{number:03d}"
        candidate = f"{stem}{path.suffix.lower()}"
    used.add(candidate.casefold())
    used.add(f"stem:{stem.casefold()}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
