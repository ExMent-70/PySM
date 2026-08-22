"""Download pinned RMBG model files with progress and atomic publication."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .config_schema import ModelName, SDMatteVariant
from .manifests import load_models_lock
from .model_registry import ModelDescriptor
from .model_store import (
    ModelFiles,
    ModelStoreError,
    SDMatteFiles,
    resolve_model_files,
    resolve_sdmatte_files,
    sha256_file,
    verify_model_files,
)
from .progress import ProgressFactory


class ModelDownloadError(RuntimeError):
    """Raised when a pinned model cannot be downloaded safely."""


@dataclass(frozen=True, slots=True)
class ModelEnsureResult:
    files: ModelFiles
    downloaded: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SDMatteEnsureResult:
    files: SDMatteFiles
    downloaded: tuple[str, ...]


HttpGet = Callable[..., Any]


def ensure_model_files(
    model_id: ModelName,
    descriptor: ModelDescriptor,
    model_store: Path,
    *,
    progress_factory: ProgressFactory | None = None,
    http_get: HttpGet | None = None,
) -> ModelEnsureResult:
    """Download missing files and return a fully checksum-verified model set."""

    lock = load_models_lock()["models"][model_id.value]
    files = resolve_model_files(model_store.resolve(), descriptor)
    by_name = _files_by_name(files)
    try:
        verify_model_files(model_id, files, require_all=False)
    except ModelStoreError as exc:
        raise ModelDownloadError(
            "Существующие файлы модели повреждены или относятся к другой версии. "
            "Автоматическая перезапись отключена; проверьте model store вручную."
        ) from exc
    missing = tuple(name for name, path in by_name.items() if not path.is_file())
    if not missing:
        return ModelEnsureResult(files=files, downloaded=())

    download = lock.get("download")
    if not isinstance(download, dict):
        raise ModelDownloadError(
            f"Для модели '{model_id.value}' отсутствует проверенный download manifest."
        )
    repository = str(lock.get("repository") or "")
    revision = str(download.get("revision") or "")
    metadata_by_name = download.get("files")
    if not repository or not revision or not isinstance(metadata_by_name, dict):
        raise ModelDownloadError(
            f"Download manifest модели '{model_id.value}' заполнен не полностью."
        )

    if http_get is None:
        try:
            import requests
        except ImportError as exc:
            raise ModelDownloadError(
                "Для автоматического скачивания моделей требуется пакет requests."
            ) from exc
        http_get = requests.get

    downloaded: list[str] = []
    files.model_dir.mkdir(parents=True, exist_ok=True)
    for filename in missing:
        metadata = metadata_by_name.get(filename)
        if not isinstance(metadata, dict):
            raise ModelDownloadError(
                f"В download manifest отсутствует файл '{filename}'."
            )
        expected_hash = str(metadata.get("sha256") or "").casefold()
        expected_size = int(metadata.get("size") or 0)
        if len(expected_hash) != 64 or expected_size <= 0:
            raise ModelDownloadError(
                f"Для файла '{filename}' не зафиксированы SHA-256 и размер."
            )
        url = (
            f"https://huggingface.co/{repository}/resolve/{revision}/"
            f"{quote(filename)}"
        )
        _download_file(
            url=url,
            destination=by_name[filename],
            expected_hash=expected_hash,
            expected_size=expected_size,
            progress_factory=progress_factory,
            http_get=http_get,
        )
        downloaded.append(filename)

    try:
        verify_model_files(model_id, files)
    except ModelStoreError as exc:
        raise ModelDownloadError(
            "Скачанные файлы не прошли итоговую проверку модели."
        ) from exc
    return ModelEnsureResult(files=files, downloaded=tuple(downloaded))


def ensure_sdmatte_files(
    variant: SDMatteVariant,
    model_store: Path,
    *,
    progress_factory: ProgressFactory | None = None,
    http_get: HttpGet | None = None,
) -> SDMatteEnsureResult:
    """Download and verify the selected SDMatte checkpoint and pinned runtime."""

    lock = load_models_lock().get("refiners", {}).get("sdmatte")
    if not isinstance(lock, dict):
        raise ModelDownloadError("models.lock.json не содержит SDMatte manifest.")
    files = resolve_sdmatte_files(model_store.resolve(), variant.value)
    metadata = lock.get("files")
    variants = lock.get("variants")
    if not isinstance(metadata, dict) or not isinstance(variants, dict):
        raise ModelDownloadError("SDMatte manifest заполнен не полностью.")
    weight_name = variants.get(variant.value)
    if not isinstance(weight_name, str):
        raise ModelDownloadError(f"Не описан вариант SDMatte: {variant.value}")
    required_metadata = dict(metadata)
    weight_metadata = lock.get("weights", {}).get(weight_name)
    if not isinstance(weight_metadata, dict):
        raise ModelDownloadError(f"Не описаны веса SDMatte: {weight_name}")
    required_metadata[weight_name] = weight_metadata

    if http_get is None:
        try:
            import requests
        except ImportError as exc:
            raise ModelDownloadError(
                "Для автоматического скачивания SDMatte требуется requests."
            ) from exc
        http_get = requests.get

    by_name = files.by_relative_path()
    _verify_pinned_paths(by_name, required_metadata, require_all=False)
    missing = tuple(name for name, path in by_name.items() if not path.is_file())
    downloaded: list[str] = []
    repository = str(lock.get("repository") or "")
    revision = str(lock.get("revision") or "")
    for relative_path in missing:
        file_metadata = required_metadata.get(relative_path)
        if not isinstance(file_metadata, dict):
            raise ModelDownloadError(
                f"В SDMatte manifest отсутствует '{relative_path}'."
            )
        url = str(file_metadata.get("url") or "")
        if not url:
            url = (
                f"https://huggingface.co/{repository}/resolve/{revision}/"
                f"{quote(relative_path)}"
            )
        _download_file(
            url=url,
            destination=by_name[relative_path],
            expected_hash=str(file_metadata.get("sha256") or "").casefold(),
            expected_size=int(file_metadata.get("size") or 0),
            progress_factory=progress_factory,
            http_get=http_get,
        )
        downloaded.append(relative_path)
    _verify_pinned_paths(by_name, required_metadata, require_all=True)
    return SDMatteEnsureResult(files=files, downloaded=tuple(downloaded))


def _verify_pinned_paths(
    paths: dict[str, Path],
    metadata: dict[str, Any],
    *,
    require_all: bool,
) -> None:
    mismatches: list[str] = []
    for name, path in paths.items():
        expected = metadata.get(name)
        if not isinstance(expected, dict):
            mismatches.append(f"{name}: отсутствует manifest")
            continue
        if not path.is_file():
            if require_all:
                mismatches.append(f"{name}: файл отсутствует")
            continue
        expected_hash = str(expected.get("sha256") or "").casefold()
        expected_size = int(expected.get("size") or 0)
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            mismatches.append(f"{name}: размер или SHA-256 не совпадает")
    if mismatches:
        raise ModelDownloadError(
            "Проверка SDMatte не пройдена; существующие файлы не перезаписаны:\n- "
            + "\n- ".join(mismatches)
        )


def _download_file(
    *,
    url: str,
    destination: Path,
    expected_hash: str,
    expected_size: int,
    progress_factory: ProgressFactory | None,
    http_get: HttpGet,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.download"
    )
    progress = None
    unit_size = 1024 * 1024 if expected_size >= 1024 * 1024 else 1
    total_units = math.ceil(expected_size / unit_size)
    unit = "MiB" if unit_size > 1 else "B"
    try:
        progress = (
            progress_factory(
                total=total_units,
                desc=f"Загрузка модели: {destination.name}",
                unit=unit,
            )
            if progress_factory is not None
            else None
        )
        digest = hashlib.sha256()
        written = 0
        reported_units = 0
        with http_get(url, stream=True, timeout=(15, 300)) as response:
            response.raise_for_status()
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        current_units = min(written // unit_size, total_units)
                        if current_units > reported_units:
                            progress.update(current_units - reported_units)
                            reported_units = current_units
        if written != expected_size:
            raise ModelDownloadError(
                f"Размер '{destination.name}' после скачивания: {written}; "
                f"ожидалось {expected_size}."
            )
        actual_hash = digest.hexdigest()
        if actual_hash.casefold() != expected_hash:
            raise ModelDownloadError(
                f"SHA-256 '{destination.name}' не совпадает с models.lock.json."
            )
        if progress is not None and reported_units < total_units:
            progress.update(total_units - reported_units)
        _publish_without_overwrite(temporary, destination, expected_hash)
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise ModelDownloadError(
            f"Не удалось скачать '{destination.name}': {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        if progress is not None:
            progress.close()


def _files_by_name(files: ModelFiles) -> dict[str, Path]:
    return {
        files.weights.name: files.weights,
        files.model_script.name: files.model_script,
        files.config_script.name: files.config_script,
        files.config_json.name: files.config_json,
    }


def _publish_without_overwrite(
    temporary: Path,
    destination: Path,
    expected_hash: str,
) -> None:
    """Atomically publish a same-volume hard link without replacing user data."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.is_file() and sha256_file(destination) == expected_hash:
            return
        raise ModelDownloadError(
            f"Файл '{destination}' появился во время загрузки и отличается "
            "от ожидаемого. Он не был перезаписан."
        )
    except OSError as exc:
        raise ModelDownloadError(
            f"Не удалось атомарно опубликовать '{destination.name}': {exc}"
        ) from exc
