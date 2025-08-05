# pysm_lib/path_utils.py
import os
import pathlib
from typing import Optional

from .app_constants import APPLICATION_ROOT_DIR



def find_best_relative_path(
    target_path_str: Optional[str], base_dir: pathlib.Path
) -> Optional[str]:
    """
    Находит наилучший (кратчайший) относительный путь от base_dir до target_path_str.
    В отличие от to_relative_if_possible, эта функция ПОЗВОЛЯЕТ использование "../".

    :param target_path_str: Строка с абсолютным путем для преобразования.
    :param base_dir: Базовая директория, от которой строится относительный путь.
    :return: Строка с относительным или абсолютным путем.
    """
    if not target_path_str:
        return None

    try:
        target_path = pathlib.Path(target_path_str).resolve()
        # Пытаемся построить относительный путь. os.path.relpath умеет работать с "../"
        # Проверяем, что пути находятся на одном диске (для Windows)
        if target_path.drive.lower() == base_dir.drive.lower():
            relative_path_str = os.path.relpath(target_path, base_dir)
            # Возвращаем путь в POSIX-формате для кросс-платформенности
            return pathlib.Path(relative_path_str).as_posix()

        # Если на разных дисках или произошла ошибка, возвращаем абсолютный путь
        return str(target_path)
    except (TypeError, ValueError):
        # В случае ошибки возвращаем как есть
        return target_path_str


def to_relative_if_possible(
    absolute_path_str: Optional[str], base_dir: pathlib.Path
) -> Optional[str]:
    """
    Преобразует абсолютный путь в относительный, используя приоритеты.

    Приоритеты:
    1. Относительно корня приложения (APPLICATION_ROOT_DIR).
    2. Относительно указанной базовой директории (base_dir, обычно папка коллекции).
    3. Если ни один из вариантов не подходит, возвращается абсолютный путь.

    :param absolute_path_str: Строка с абсолютным путем для преобразования.
    :param base_dir: Второстепенная базовая директория (например, папка коллекции).
    :return: Строка с относительным или абсолютным путем.
    """
    if not absolute_path_str:
        return None

    try:
        absolute_path = pathlib.Path(absolute_path_str).resolve()

        # --- ИЗМЕНЕНИЕ: Логика с приоритетами ---

        # Приоритет 1: Попробовать сделать путь относительным к корню приложения
        try:
            if absolute_path.drive.lower() == APPLICATION_ROOT_DIR.drive.lower():
                app_relative_path = os.path.relpath(absolute_path, APPLICATION_ROOT_DIR)
                if not app_relative_path.startswith(".."):
                    return pathlib.Path(app_relative_path).as_posix()
        except Exception:
            # Игнорируем ошибки, если пути, например, на разных дисках в Windows
            pass

        # Приоритет 2: Попробовать сделать путь относительным к base_dir (папке коллекции)
        try:
            if absolute_path.drive.lower() == base_dir.drive.lower():
                base_relative_path = os.path.relpath(absolute_path, base_dir)
                if not base_relative_path.startswith(".."):
                    return pathlib.Path(base_relative_path).as_posix()
        except Exception:
            pass

        # Fallback: Если ничего не подошло, вернуть абсолютный путь
        return str(absolute_path)

    except (TypeError, ValueError):
        # В случае ошибки (например, некорректный путь) возвращаем как есть
        return absolute_path_str


def resolve_path(path_str: Optional[str], base_dir: pathlib.Path) -> Optional[str]:
    """
    Преобразует потенциально относительный путь в абсолютный.
    Если путь уже абсолютный, возвращает его без изменений.
    Если путь относительный, он сначала проверяется относительно корня приложения,
    а затем относительно указанной base_dir.

    :param path_str: Строка с относительным или абсолютным путем.
    :param base_dir: Базовая директория для разрешения относительных путей.
    :return: Строка с разрешенным абсолютным путем.
    """
    if not path_str:
        return None

    path = pathlib.Path(path_str)
    if path.is_absolute():
        return str(path.resolve(strict=False))

    # --- ИЗМЕНЕНИЕ: Сначала проверяем относительно корня приложения ---
    # Это позволяет путям вроде "scripts/my_script" работать из любой коллекции
    app_based_path = APPLICATION_ROOT_DIR / path
    if app_based_path.exists():
        return str(app_based_path.resolve())

    # Если не нашли относительно приложения, ищем относительно base_dir (папки коллекции)
    collection_based_path = base_dir / path
    return str(collection_based_path.resolve(strict=False))
