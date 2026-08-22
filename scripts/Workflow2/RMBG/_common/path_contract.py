"""Portable path contracts shared by RMBG configuration and runtime code."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


PYSM_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = "_BIN/models/RMBG"


def normalize_model_dir_value(value: str) -> str:
    """Store model paths relative to the PySM root."""

    normalized = value.strip().strip('"')
    if not normalized:
        raise ValueError("Укажите папку хранения моделей RMBG.")

    path = Path(normalized).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(PYSM_PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(
                "Папка моделей должна находиться внутри корня PySM."
            ) from exc

    portable = PurePosixPath(normalized.replace("\\", "/"))
    if (
        portable.is_absolute()
        or ".." in portable.parts
        or any(":" in part for part in portable.parts)
    ):
        raise ValueError(
            "Относительная папка моделей должна находиться внутри корня PySM."
        )
    return portable.as_posix()


def resolve_model_dir_value(
    value: str,
    *,
    project_root: Path = PYSM_PROJECT_ROOT,
) -> Path:
    """Resolve one stored model path against the explicit PySM project root."""

    normalized = normalize_model_dir_value(value)
    path = Path(normalized)
    root = project_root.resolve()
    resolved = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError("Папка моделей выходит за пределы корня PySM.")
    return resolved
