"""Read-only image discovery used before a processing run starts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
)


def discover_images(input_dir: str | Path, *, recursive: bool) -> tuple[Path, ...]:
    """Return deterministic input paths without modifying the source directory."""

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Папка исходных изображений не найдена: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Путь исходных изображений не является папкой: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = (
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    return tuple(sorted(files, key=lambda item: item.as_posix().casefold()))


def resolve_background_image(
    background_dir: str | Path | None,
    relative_path: str,
) -> Path:
    """Resolve one configured image without allowing escape from background_dir."""

    if background_dir is None:
        raise ValueError(
            "Для composite с изображением необходимо указать background_dir."
        )
    if not relative_path:
        raise ValueError(
            "В профиле не выбрано фоновое изображение. "
            "Откройте RMBG Configurator и выберите файл из background_dir."
        )
    root = Path(background_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Папка фоновых изображений не найдена: {root}")

    portable = PurePosixPath(relative_path.replace("\\", "/"))
    candidate = root.joinpath(*portable.parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Фоновое изображение находится вне background_dir.")
    if not candidate.is_file():
        raise FileNotFoundError(
            "Выбранное фоновое изображение не найдено: "
            f"{relative_path} (background_dir: {root})"
        )
    if candidate.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат фонового изображения: {candidate}")
    return candidate
