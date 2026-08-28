"""Диалог выбора файла или папки с сохранением результата в контекст PySM."""

import argparse
import logging
import os
import sys
from pathlib import Path


IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import (
        format_error,
        format_success,
        read_context_value,
        write_context_value,
    )
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"

    def format_success(var_name: str, value: object) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"


try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    print("Пожалуйста, установите его: pip install PySide6", file=sys.stderr)
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

LIMIT_MODE_ONLY_IN_INITIAL_DIR = "only_in_initial_dir"
LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS = (
    "only_in_initial_dir_and_subfolders"
)
LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS = "initial_dir_and_subfolders"
LIMIT_MODE_ALL = "all"
LIMIT_MODES = {
    LIMIT_MODE_ONLY_IN_INITIAL_DIR,
    LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS,
    LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS,
    LIMIT_MODE_ALL,
}


def _normalize_limit_mode(value: object) -> str:
    """Нормализует строковое значение режима после ConfigResolver."""

    value_str = str(value).strip()
    if value_str in LIMIT_MODES:
        return value_str

    # ConfigResolver считает имя параметра path-like из-за слова "dir" и
    # превращает choice-значение в абсолютный путь. Восстанавливаем последний
    # компонент только тогда, когда он точно совпадает с допустимым режимом.
    path_name = Path(value_str).name
    if path_name in LIMIT_MODES:
        return path_name

    raise ValueError(f"Неизвестный режим ограничения области выбора: {value}")


def _parse_limit_mode(value: str) -> str:
    """Преобразует CLI-значение режима в формат argparse."""

    try:
        return _normalize_limit_mode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def get_config() -> argparse.Namespace:
    """Определяет аргументы и возвращает полностью обработанную конфигурацию."""

    parser = argparse.ArgumentParser(
        description="Показывает диалог выбора файла/папки и сохраняет результат в контекст.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dlg_open_var",
        type=str,
        default="dlg_open_user_var",
        help=(
            "Имя переменной для сохранения результата. Поддерживается точечная "
            "нотация, например project.output_path."
        ),
    )
    parser.add_argument(
        "--dlg_open_type",
        type=str,
        choices=["file", "directory"],
        default="file",
        help="Тип диалога: 'file' для выбора файла, 'directory' для выбора папки.",
    )
    parser.add_argument(
        "--dlg_open_title",
        type=str,
        default="Выберите путь",
        help="Текст заголовка диалогового окна.",
    )
    parser.add_argument(
        "--dlg_open_filter",
        type=str,
        default="Все файлы (*.*)",
        help=(
            "Фильтр файлов, например 'Изображения (*.png *.jpg)'. "
            "Используется только при выборе файла."
        ),
    )
    parser.add_argument(
        "--dlg_open_result_mode",
        type=str,
        choices=["full_path", "name"],
        default="full_path",
        help=(
            "Формат результата: 'full_path' сохраняет полный путь, "
            "'name' — только имя выбранного файла или папки."
        ),
    )
    parser.add_argument(
        "--dlg_open_initial_dir",
        type=str,
        default="",
        help=(
            "Начальная папка диалога. Может содержать шаблоны контекста, "
            "например {photo_session_dir}."
        ),
    )
    parser.add_argument(
        "--dlg_open_limit_to_initial_dir",
        type=_parse_limit_mode,
        choices=[
            "only_in_initial_dir",
            "only_in_initial_dir_and_subfolders",
            "initial_dir_and_subfolders",
            "all",
        ],
        default="all",
        help=(
            "Ограничение области выбора: только непосредственно в начальной "
            "папке; внутри неё на любом уровне без выбора самой папки; внутри "
            "неё с выбором самой папки; либо без ограничений."
        ),
    )

    if IS_MANAGED_RUN:
        config = ConfigResolver(parser).resolve_all()
    else:
        config = parser.parse_args()

    try:
        config.dlg_open_limit_to_initial_dir = _normalize_limit_mode(
            config.dlg_open_limit_to_initial_dir
        )
    except ValueError as error:
        parser.error(str(error))
    return config


def _normalized_absolute_path(path: str) -> str:
    """Возвращает нормализованный абсолютный путь без разрешения ссылок."""

    return os.path.normpath(os.path.abspath(path))


def _is_selection_allowed(candidate: str, root: str, limit_mode: str) -> bool:
    """Проверяет выбранный путь по режиму, с учётом ссылок и junction."""

    candidate_path = Path(candidate).resolve(strict=False)
    root_path = Path(root).resolve(strict=False)

    if limit_mode == LIMIT_MODE_ALL:
        return True
    if limit_mode == LIMIT_MODE_ONLY_IN_INITIAL_DIR:
        return candidate_path.parent == root_path
    if limit_mode == LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS:
        return candidate_path != root_path and candidate_path.is_relative_to(
            root_path
        )
    if limit_mode == LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS:
        return candidate_path == root_path or candidate_path.is_relative_to(
            root_path
        )

    raise ValueError(f"Неизвестный режим ограничения области выбора: {limit_mode}")


def _determine_initial_directory(config: argparse.Namespace) -> tuple[str, bool]:
    """Возвращает начальную папку и признак её явного задания параметром."""

    configured_dir = str(config.dlg_open_initial_dir or "")
    if configured_dir.strip():
        initial_dir = _normalized_absolute_path(configured_dir)
        if not os.path.isdir(initial_dir):
            raise ValueError(f"Начальная папка не существует: {initial_dir}")
        return initial_dir, True

    if config.dlg_open_result_mode == "full_path":
        existing_path = read_context_value(pysm_context, config.dlg_open_var)
        existing_path_str = existing_path.value if existing_path.exists else None
        if isinstance(existing_path_str, str) and os.path.exists(existing_path_str):
            if os.path.isfile(existing_path_str):
                return _normalized_absolute_path(os.path.dirname(existing_path_str)), False
            if os.path.isdir(existing_path_str):
                return _normalized_absolute_path(existing_path_str), False

    collection_dir_value = read_context_value(pysm_context, "pysm_info.collection_dir")
    collection_dir = collection_dir_value.value if collection_dir_value.exists else None
    if isinstance(collection_dir, str) and os.path.isdir(collection_dir):
        return _normalized_absolute_path(collection_dir), False

    return _normalized_absolute_path("."), False


def _show_outside_initial_dir_warning(
    selected_path: str,
    initial_dir: str,
    limit_mode: str,
) -> None:
    """Сообщает пользователю, что выбранный путь находится вне разрешённой папки."""

    if limit_mode == LIMIT_MODE_ONLY_IN_INITIAL_DIR:
        rule_text = "Выберите объект непосредственно в начальной папке."
    elif limit_mode == LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS:
        rule_text = (
            "Выберите объект внутри начальной папки или её подпапок.\n"
            "Саму начальную папку выбирать нельзя."
        )
    else:
        rule_text = "Выберите начальную папку либо объект внутри неё."

    message_box = QMessageBox(
        QMessageBox.Icon.Warning,
        "Выбор недоступен",
        (
            f"{rule_text}\n\n"
            f"Начальная папка:\n{initial_dir}\n\n"
            f"Выбранный объект:\n{selected_path}"
        ),
    )
    message_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    message_box.exec()


def _open_selection_dialog(config: argparse.Namespace, initial_dir: str) -> str:
    """Открывает диалог и возвращает допустимый выбранный путь либо пустую строку."""

    limit_mode = _normalize_limit_mode(config.dlg_open_limit_to_initial_dir)
    while True:
        if config.dlg_open_type == "file":
            selected_path, _ = QFileDialog.getOpenFileName(
                parent=None,
                caption=config.dlg_open_title,
                dir=initial_dir,
                filter=config.dlg_open_filter,
            )
        else:
            selected_path = QFileDialog.getExistingDirectory(
                parent=None,
                caption=config.dlg_open_title,
                dir=initial_dir,
            )

        if not selected_path:
            return ""

        selected_path = _normalized_absolute_path(selected_path)
        if _is_selection_allowed(selected_path, initial_dir, limit_mode):
            return selected_path

        _show_outside_initial_dir_warning(selected_path, initial_dir, limit_mode)


def _prepare_context_result(
    selected_path: str,
    result_mode: str,
    dialog_type: str,
) -> tuple[str, str]:
    """Формирует сохраняемое значение и тип переменной контекста."""

    if result_mode == "name":
        selected = Path(selected_path)
        return selected.name or selected.anchor or str(selected), "string"

    path_type = "dir_path" if dialog_type == "directory" else "file_path"
    return selected_path, path_type


def main() -> None:
    """Запускает диалог и сохраняет подтверждённый результат в контекст PySM."""

    config = get_config()
    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical(format_error("Этот скрипт может быть запущен только в среде PySM"))
        sys.exit(1)

    q_app = QApplication.instance() or QApplication(sys.argv)

    try:
        initial_dir, is_configured_initial_dir = _determine_initial_directory(config)
    except ValueError as error:
        logger.critical(format_error(str(error)))
        sys.exit(1)

    limit_mode = _normalize_limit_mode(config.dlg_open_limit_to_initial_dir)
    if limit_mode != LIMIT_MODE_ALL and not is_configured_initial_dir:
        logger.critical(
            format_error(
                "Для ограничения области выбора задайте параметр dlg_open_initial_dir."
            )
        )
        sys.exit(1)

    logger.info(f"<b>{config.dlg_open_title}</b>")
    selected_path = _open_selection_dialog(config, initial_dir)
    if not selected_path:
        logger.critical(format_error("Операция отменена пользователем<br>"))
        sys.exit(1)

    result_value, result_type = _prepare_context_result(
        selected_path,
        config.dlg_open_result_mode,
        config.dlg_open_type,
    )
    try:
        write_context_value(
            pysm_context,
            config.dlg_open_var,
            result_value,
            var_type=result_type,
        )
    except Exception as error:
        logger.critical(format_error(f"Критическая ошибка при записи в контекст: {error}"))
        sys.exit(1)

    logger.info(format_success(config.dlg_open_var, result_value))
    sys.exit(0)


if __name__ == "__main__":
    main()
