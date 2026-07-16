# run_py_path_if_else.py

"""Read-only файловый шлюз If-Then-Else для наборов PySM.

Скрипт проверяет существование и тип пути, пустоту папки или наличие файлов по
glob-шаблонам. По результату он запрашивает переход к ветке Then/Else через
``pysm_context.set_next_script``. Файловая система при этом не изменяется.
"""

import argparse
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass
from html import escape
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional, Sequence


IS_MANAGED_RUN = False

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[3]
    system_scripts_dir = current_script_path.parents[1]

    for import_path in (project_root, system_scripts_dir):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = True
except ImportError as exc:
    pysm_context = None
    ConfigResolver = None
    print(f"Критическая ошибка импорта: {exc}", file=sys.stderr)
    print(
        "Убедитесь, что структура папок верна и все зависимости установлены.",
        file=sys.stderr,
    )
    sys.exit(1)

from _common import icon_error, icon_info, icon_ok, icon_play, icon_warning


logger = logging.getLogger(__name__)


COND_PATH_EXISTS = "путь существует"
COND_PATH_NOT_EXISTS = "путь не существует"
COND_FILE_EXISTS = "файл существует"
COND_FILE_NOT_EXISTS = "файл не существует"
COND_DIR_EXISTS = "папка существует"
COND_DIR_NOT_EXISTS = "папка не существует"
COND_DIR_EMPTY = "папка пуста"
COND_DIR_NOT_EMPTY = "папка не пуста"
COND_DIR_CONTAINS_FILES = "папка содержит файлы"
COND_DIR_NOT_CONTAINS_FILES = "папка не содержит файлы"

CONDITIONS = [
    COND_PATH_EXISTS,
    COND_PATH_NOT_EXISTS,
    COND_FILE_EXISTS,
    COND_FILE_NOT_EXISTS,
    COND_DIR_EXISTS,
    COND_DIR_NOT_EXISTS,
    COND_DIR_EMPTY,
    COND_DIR_NOT_EMPTY,
    COND_DIR_CONTAINS_FILES,
    COND_DIR_NOT_CONTAINS_FILES,
]

FILE_SEARCH_CONDITIONS = {
    COND_DIR_CONTAINS_FILES,
    COND_DIR_NOT_CONTAINS_FILES,
}

MAX_MATCH_SAMPLES = 10


@dataclass(frozen=True)
class PathConditionResult:
    """Результат файловой проверки и данные для диагностического отчёта."""

    is_true: bool
    path_exists: bool
    is_file: bool
    is_dir: bool
    warning: Optional[str] = None
    matched_files: tuple[Path, ...] = ()


def get_config() -> Namespace:
    """Определяет CLI-параметры и разрешает конфигурацию через PySM."""
    parser = argparse.ArgumentParser(
        description="Проверяет файл или папку и выполняет условный переход."
    )
    parser.add_argument(
        "--checked-path",
        type=str,
        required=True,
        help=(
            "Проверяемый путь. Поддерживает шаблоны переменных PySM вида "
            "'{variable}'."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=CONDITIONS,
        help="Условие проверки файла или папки.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=["*"],
        help=(
            "Glob-шаблоны для условий поиска файлов. "
            "По умолчанию проверяются все файлы."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Искать подходящие файлы во вложенных папках.",
    )
    parser.add_argument(
        "--then-instance-id",
        type=str,
        required=True,
        help="ID экземпляра скрипта при истинном результате.",
    )
    parser.add_argument(
        "--else-instance-id",
        type=str,
        default=None,
        help=(
            "ID экземпляра скрипта при ложном результате. Если не задан, "
            "выполнение продолжается по штатному порядку."
        ),
    )

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def ensure_required_text(value: Any, field_name: str) -> str:
    """Возвращает непустое строковое значение или поднимает ValueError."""
    normalized = "" if value is None else str(value).strip()
    if not normalized:
        raise ValueError(f"Параметр '{field_name}' не задан.")
    return normalized


def normalize_optional_instance_id(value: Optional[str]) -> Optional[str]:
    """Нормализует необязательный ID ветки Else."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized if normalized else None


def normalize_patterns(values: Any) -> tuple[str, ...]:
    """Нормализует CLI-, GUI- или context-представление glob-шаблонов."""
    if values is None:
        return ()

    if isinstance(values, str):
        raw_values: Sequence[Any] = values.splitlines()
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raw_values = (values,)

    patterns = tuple(str(value).strip() for value in raw_values if str(value).strip())
    return patterns


def validate_pattern(pattern: str) -> None:
    """Запрещает glob-шаблону выходить за пределы проверяемой папки."""
    windows_pattern = PureWindowsPath(pattern)
    posix_pattern = PurePosixPath(pattern)

    if windows_pattern.drive or windows_pattern.root or posix_pattern.is_absolute():
        raise ValueError(f"Glob-шаблон должен быть относительным: '{pattern}'.")

    if ".." in windows_pattern.parts or ".." in posix_pattern.parts:
        raise ValueError(
            f"Glob-шаблон не может выходить за пределы проверяемой папки: '{pattern}'."
        )


def validate_patterns(patterns: Sequence[str]) -> None:
    """Проверяет непустой список glob-шаблонов."""
    if not patterns:
        raise ValueError("Для файлового поиска необходимо задать хотя бы один шаблон.")
    for pattern in patterns:
        validate_pattern(pattern)


def find_matching_files(
    directory: Path,
    patterns: Sequence[str],
    recursive: bool,
) -> tuple[Path, ...]:
    """Возвращает уникальные файлы, подходящие под glob-шаблоны."""
    matched: set[Path] = set()

    for pattern in patterns:
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        for candidate in iterator:
            if candidate.is_file():
                matched.add(candidate)

    return tuple(sorted(matched, key=lambda item: str(item).casefold()))


def evaluate_path_condition(
    checked_path: Path,
    condition: str,
    patterns: Sequence[str] = ("*",),
    recursive: bool = False,
) -> PathConditionResult:
    """Вычисляет файловое условие без логирования и маршрутизации."""
    try:
        path_exists = checked_path.exists()
        is_file = checked_path.is_file()
        is_dir = checked_path.is_dir()
    except OSError as exc:
        return PathConditionResult(
            is_true=False,
            path_exists=False,
            is_file=False,
            is_dir=False,
            warning=f"Не удалось проверить путь: {exc}",
        )

    state = {
        "path_exists": path_exists,
        "is_file": is_file,
        "is_dir": is_dir,
    }

    if condition == COND_PATH_EXISTS:
        return PathConditionResult(is_true=path_exists, **state)
    if condition == COND_PATH_NOT_EXISTS:
        return PathConditionResult(is_true=not path_exists, **state)
    if condition == COND_FILE_EXISTS:
        return PathConditionResult(is_true=is_file, **state)
    if condition == COND_FILE_NOT_EXISTS:
        return PathConditionResult(is_true=not is_file, **state)
    if condition == COND_DIR_EXISTS:
        return PathConditionResult(is_true=is_dir, **state)
    if condition == COND_DIR_NOT_EXISTS:
        return PathConditionResult(is_true=not is_dir, **state)

    if not is_dir:
        warning = (
            "Проверяемая папка не существует."
            if not path_exists
            else "Проверяемый путь не является папкой."
        )
        return PathConditionResult(is_true=False, warning=warning, **state)

    if condition in {COND_DIR_EMPTY, COND_DIR_NOT_EMPTY}:
        try:
            is_empty = next(checked_path.iterdir(), None) is None
        except OSError as exc:
            return PathConditionResult(
                is_true=False,
                warning=f"Не удалось прочитать содержимое папки: {exc}",
                **state,
            )

        expected_empty = condition == COND_DIR_EMPTY
        return PathConditionResult(is_true=is_empty == expected_empty, **state)

    if condition in FILE_SEARCH_CONDITIONS:
        try:
            validate_patterns(patterns)
            matched_files = find_matching_files(checked_path, patterns, recursive)
        except (OSError, ValueError, RuntimeError) as exc:
            return PathConditionResult(
                is_true=False,
                warning=f"Не удалось выполнить поиск файлов: {exc}",
                **state,
            )

        contains_files = bool(matched_files)
        expected_files = condition == COND_DIR_CONTAINS_FILES
        return PathConditionResult(
            is_true=contains_files == expected_files,
            matched_files=matched_files,
            **state,
        )

    return PathConditionResult(
        is_true=False,
        warning=f"Неизвестное файловое условие: {condition}",
        **state,
    )


def log_condition_report(
    checked_path: Path,
    condition: str,
    patterns: Sequence[str],
    recursive: bool,
    result: PathConditionResult,
    target_id: Optional[str],
) -> None:
    """Выводит HTML-безопасный отчёт о файловой проверке."""
    logger.info("<b>ПРОВЕРКА ФАЙЛОВОГО УСЛОВИЯ...</b>")
    logger.info("%s Путь: <i>%s</i>", icon_info, escape(str(checked_path)))
    logger.info("%s Условие: <b>%s</b>", icon_info, escape(condition))
    logger.info(
        "%s Состояние: существует=<b>%s</b>, файл=<b>%s</b>, папка=<b>%s</b>",
        icon_info,
        "да" if result.path_exists else "нет",
        "да" if result.is_file else "нет",
        "да" if result.is_dir else "нет",
    )

    if condition in FILE_SEARCH_CONDITIONS:
        logger.info(
            "%s Шаблоны: <i>%s</i>",
            icon_info,
            escape(", ".join(patterns)),
        )
        logger.info(
            "%s Рекурсивный поиск: <b>%s</b>",
            icon_info,
            "да" if recursive else "нет",
        )
        logger.info(
            "%s Найдено файлов: <b>%d</b>",
            icon_info,
            len(result.matched_files),
        )
        for matched_file in result.matched_files[:MAX_MATCH_SAMPLES]:
            logger.info("%s <i>%s</i>", icon_info, escape(str(matched_file)))
        if len(result.matched_files) > MAX_MATCH_SAMPLES:
            logger.info(
                "%s Показаны первые %d из %d совпадений.",
                icon_info,
                MAX_MATCH_SAMPLES,
                len(result.matched_files),
            )

    if result.warning:
        logger.warning("%s %s", icon_warning, escape(result.warning))

    logger.info(
        "%s Результат условия: <b>%s</b>",
        icon_ok if result.is_true else icon_error,
        "TRUE" if result.is_true else "FALSE",
    )

    if result.is_true:
        branch_name = "THEN"
    elif target_id:
        branch_name = "ELSE"
    else:
        branch_name = "DEFAULT"

    logger.info(
        "%s Выбранная ветка: <b>%s</b>",
        icon_ok if result.is_true else icon_warning,
        branch_name,
    )
    if target_id:
        logger.info(
            "%s Целевой instance_id: <i>%s</i>",
            icon_play,
            escape(target_id),
        )


def select_target_id(
    is_true: bool,
    then_instance_id: str,
    else_instance_id: Optional[str],
) -> Optional[str]:
    """Выбирает целевой instance_id для результата условия."""
    return then_instance_id if is_true else else_instance_id


def main() -> None:
    """Получает конфигурацию, вычисляет условие и выбирает следующую ветку."""
    log_level = (
        pysm_context.get("sys_log_level", "INFO")
        if IS_MANAGED_RUN and pysm_context
        else "INFO"
    )
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )

    config = get_config()

    try:
        checked_path_text = ensure_required_text(config.checked_path, "checked-path")
        then_instance_id = ensure_required_text(
            config.then_instance_id,
            "then-instance-id",
        )
        patterns = normalize_patterns(config.patterns)
        if config.condition in FILE_SEARCH_CONDITIONS:
            validate_patterns(patterns)
    except ValueError as exc:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: %s", exc)
        sys.exit(1)

    checked_path = Path(checked_path_text)
    else_instance_id = normalize_optional_instance_id(config.else_instance_id)
    result = evaluate_path_condition(
        checked_path=checked_path,
        condition=config.condition,
        patterns=patterns,
        recursive=bool(config.recursive),
    )
    target_id = select_target_id(
        is_true=result.is_true,
        then_instance_id=then_instance_id,
        else_instance_id=else_instance_id,
    )

    log_condition_report(
        checked_path=checked_path,
        condition=config.condition,
        patterns=patterns,
        recursive=bool(config.recursive),
        result=result,
        target_id=target_id,
    )

    if target_id:
        try:
            pysm_context.set_next_script(target_id)
            logger.info(
                "\n%s Порядок выполнения скриптов изменён",
                icon_play,
            )
        except Exception as exc:
            logger.error(
                "КРИТИЧЕСКАЯ ОШИБКА при отправке команды перехода: %s",
                exc,
            )
            sys.exit(1)
    elif not result.is_true:
        logger.info(
            "%s Ветка 'Else' не определена. "
            "Выполнение будет продолжено по умолчанию.",
            icon_info,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
