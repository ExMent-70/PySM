"""Resolve and verify model files without mutating the PySM model cache."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .config_schema import ModelName
from .manifests import load_models_lock
from .model_registry import ModelDescriptor


@dataclass(frozen=True, slots=True)
class ModelFiles:
    model_dir: Path
    weights: Path
    model_script: Path
    config_script: Path
    config_json: Path


@dataclass(frozen=True, slots=True)
class SDMatteFiles:
    """Pinned runtime, component configs and one selected SDMatte checkpoint."""

    model_dir: Path
    weights: Path
    required: tuple[tuple[str, Path], ...]

    def by_relative_path(self) -> dict[str, Path]:
        return dict(self.required)


class ModelStoreError(RuntimeError):
    """Raised when a local model set is incomplete or does not match its lock."""


def default_model_store(project_root: Path) -> Path:
    """Use the PySM model directory with an explicit environment override."""

    override = os.environ.get("PYSM_RMBG_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root / "_BIN" / "models" / "RMBG"


def locate_model_files(
    model_store: Path,
    descriptor: ModelDescriptor,
) -> ModelFiles:
    """Locate one model using the directory layout inherited from ComfyUI-RMBG."""

    files = resolve_model_files(model_store, descriptor)
    required_paths = (
        files.weights,
        files.model_script,
        files.config_script,
        files.config_json,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ModelStoreError(
            "Не найдены обязательные файлы модели:\n- " + "\n- ".join(missing)
        )
    return files


def resolve_model_files(
    model_store: Path,
    descriptor: ModelDescriptor,
) -> ModelFiles:
    """Return expected paths without requiring the files to exist yet."""

    if descriptor.model_id == ModelName.RMBG_2_0:
        directory_name = "RMBG-2.0"
    else:
        directory_name = "BiRefNet"
    model_dir = (
        model_store
        if model_store.name.casefold() == directory_name.casefold()
        else model_store / directory_name
    )
    files = ModelFiles(
        model_dir=model_dir,
        weights=model_dir / descriptor.weights_file,
        model_script=model_dir / descriptor.model_script_file,
        config_script=model_dir / "BiRefNet_config.py",
        config_json=model_dir / "config.json",
    )
    return files


def resolve_sdmatte_files(model_store: Path, variant: str) -> SDMatteFiles:
    """Return the expected SDMatte layout without touching the model store."""

    root = (
        model_store
        if model_store.name.casefold() == "sdmatte"
        else model_store / "SDMatte"
    )
    weight_name = {
        "sdmatte": "SDMatte.safetensors",
        "sdmatte_plus": "SDMatte_plus.safetensors",
    }.get(variant)
    if weight_name is None:
        raise ValueError(f"Неизвестный вариант SDMatte: {variant}")
    relative_paths = (
        weight_name,
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "vae/config.json",
        "__init__.py",
        "modeling/__init__.py",
        "modeling/SDMatte/__init__.py",
        "modeling/SDMatte/meta_arch.py",
        "utils/__init__.py",
        "utils/utils.py",
        "utils/replace.py",
    )
    return SDMatteFiles(
        model_dir=root,
        weights=root / weight_name,
        required=tuple((name, root / Path(name)) for name in relative_paths),
    )


def verify_model_files(
    model_id: ModelName,
    files: ModelFiles,
    *,
    require_all: bool = True,
) -> None:
    """Check every checksum that is currently pinned in models.lock.json."""

    lock = load_models_lock()["models"][model_id.value]
    by_name = {
        files.weights.name: files.weights,
        files.model_script.name: files.model_script,
        files.config_script.name: files.config_script,
        files.config_json.name: files.config_json,
    }
    mismatches: list[str] = []
    for filename, expected in lock["files"].items():
        accepted = _accepted_hashes(lock, filename, expected)
        if not accepted:
            continue
        path = by_name.get(filename)
        if path is None or not path.is_file():
            if require_all:
                mismatches.append(f"{filename}: файл отсутствует")
            continue
        actual = sha256_file(path)
        if actual.casefold() not in accepted:
            mismatches.append(
                f"{filename}: SHA-256 {actual}, ожидался один из "
                + ", ".join(sorted(accepted))
            )
    if mismatches:
        raise ModelStoreError(
            "Проверка файлов модели не пройдена:\n- " + "\n- ".join(mismatches)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accepted_hashes(
    lock: dict[str, object],
    filename: str,
    primary: object,
) -> set[str]:
    values: list[object]
    if isinstance(primary, list):
        values = primary
    else:
        values = [primary]
    download = lock.get("download")
    if isinstance(download, dict):
        download_files = download.get("files")
        if isinstance(download_files, dict):
            metadata = download_files.get(filename)
            if isinstance(metadata, dict):
                values.append(metadata.get("sha256"))
    return {
        str(value).casefold()
        for value in values
        if isinstance(value, str) and value
    }
