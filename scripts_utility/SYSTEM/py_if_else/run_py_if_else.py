# run_py_if_else.py

"""
Скрипт-шлюз для условных переходов в PyScriptManager.

Назначение:
    Реализует логику If-Then-Else для управления потоком выполнения набора
    скриптов PySM.

Принцип работы:
    1. Скрипт получает имя переменной, оператор сравнения, значение сравнения
       и ID целевых экземпляров скриптов.
    2. Читает значение переменной из контекста PySM через get_structured().
    3. Проверяет условие.
    4. Если условие истинно, направляет выполнение на then-instance-id.
    5. Если условие ложно, направляет выполнение на else-instance-id.
    6. Если else-instance-id не задан, выполнение продолжается штатно.
    7. При необходимости удаляет проверенную переменную из контекста.

Особенности:
    - Поддерживает dot-notation для доступа к вложенным значениям.
    - Поддерживает проверку существования переменной.
    - Поддерживает строковое, числовое, булево, JSON и auto-сравнение.
    - Строковые сравнения выполняются регистронезависимо:
      "Yes" == "yes" считается истинным.
    - HTML-значения в логах экранируются.
"""

# ==============================================================================
# 1. Импорты и настройка окружения
# ==============================================================================

import argparse
import json
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Optional


IS_MANAGED_RUN = False

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[3]
    system_scripts_dir = current_script_path.parents[1]

    for path in (project_root, system_scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import context_value_exists, remove_context_value
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = True

except ImportError as e:
    pysm_context = None
    context_value_exists = None
    remove_context_value = None
    ConfigResolver = None

    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    print(
        "Убедитесь, что структура папок верна и все зависимости установлены.",
        file=sys.stderr,
    )
    sys.exit(1)


from _common import (
    icon_ok,
    icon_warning,
    icon_error,
    icon_info,
    icon_delete,
    icon_play,
)


logger = logging.getLogger(__name__)


# ==============================================================================
# 2. Константы операторов и типов сравнения
# ==============================================================================

OP_EQ = "равно"
OP_NE = "не равно"
OP_GT = "больше"
OP_LT = "меньше"
OP_GE = "больше или равно"
OP_LE = "меньше или равно"
OP_CONTAINS = "содержит"
OP_NOT_CONTAINS = "не содержит"
OP_EMPTY = "пусто"
OP_NOT_EMPTY = "не пусто"
OP_EXISTS = "существует"
OP_NOT_EXISTS = "не существует"

OPERATORS = [
    OP_EQ,
    OP_NE,
    OP_GT,
    OP_LT,
    OP_GE,
    OP_LE,
    OP_CONTAINS,
    OP_NOT_CONTAINS,
    OP_EMPTY,
    OP_NOT_EMPTY,
    OP_EXISTS,
    OP_NOT_EXISTS,
]

OPERATORS_WITHOUT_COMPARISON_VALUE = {
    OP_EXISTS,
    OP_NOT_EXISTS,
    OP_EMPTY,
    OP_NOT_EMPTY,
}

VALUE_TYPE_AUTO = "auto"
VALUE_TYPE_STRING = "string"
VALUE_TYPE_NUMBER = "number"
VALUE_TYPE_BOOL = "bool"
VALUE_TYPE_JSON = "json"

VALUE_TYPES = [
    VALUE_TYPE_AUTO,
    VALUE_TYPE_STRING,
    VALUE_TYPE_NUMBER,
    VALUE_TYPE_BOOL,
    VALUE_TYPE_JSON,
]


# ==============================================================================
# 3. Модели результата проверки
# ==============================================================================

@dataclass(frozen=True)
class ConditionResult:
    """
    Результат вычисления условия.

    Attributes:
        is_true:
            Итоговый результат условия.
        warning:
            Текст предупреждения, если условие вычислено штатно, но обнаружена
            потенциальная проблема, например невозможность числового сравнения.
    """

    is_true: bool
    warning: Optional[str] = None


# ==============================================================================
# 4. Получение конфигурации
# ==============================================================================

def get_config() -> Namespace:
    """
    Определяет аргументы командной строки и возвращает итоговую конфигурацию.

    В managed-режиме PySM значения разрешаются через ConfigResolver:
        CLI -> context -> default.

    В автономном режиме используются аргументы командной строки argparse.
    """
    parser = argparse.ArgumentParser(
        description="Выполняет условный переход к другому скрипту."
    )

    parser.add_argument(
        "--if-variable-name",
        type=str,
        required=True,
        help=(
            "Имя переменной в контексте для проверки. "
            "Поддерживается dot-notation, например 'a.b.c'."
        ),
    )

    parser.add_argument(
        "--if-operator",
        type=str,
        required=True,
        choices=OPERATORS,
        help="Оператор для проверки условия.",
    )

    parser.add_argument(
        "--if-comparison-value",
        type=str,
        default="",
        help=(
            "Значение, с которым будет сравниваться переменная. "
            "Для операторов 'существует', 'не существует', 'пусто', "
            "'не пусто' параметр может быть пустым."
    ),
)

    parser.add_argument(
        "--if-value-type",
        type=str,
        default=VALUE_TYPE_AUTO,
        choices=VALUE_TYPES,
        help=(
            "Тип сравнения: auto, string, number, bool, json. "
            "В режиме auto тип выбирается по фактическому значению переменной."
        ),
    )

    parser.add_argument(
        "--then-instance-id",
        type=str,
        required=True,
        help="ID экземпляра скрипта, если условие истинно.",
    )

    parser.add_argument(
        "--else-instance-id",
        type=str,
        default=None,
        help=(
            "ID экземпляра скрипта, если условие ложно. "
            "Если не задан, выполнение продолжается по штатному порядку."
        ),
    )

    parser.add_argument(
        "--clear-variable",
        action="store_true",
        help=(
            "Если указано, удалить проверенную переменную из контекста "
            "после вычисления условия."
        ),
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()

    return parser.parse_args()


# ==============================================================================
# 5. Вспомогательные функции нормализации и форматирования
# ==============================================================================

def normalize_optional_instance_id(value: Optional[str]) -> Optional[str]:
    """
    Нормализует необязательный ID экземпляра.

    Пустая строка и None трактуются как отсутствие целевой ветки.
    """
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized if normalized else None


def ensure_required_text(value: Any, field_name: str) -> str:
    """
    Проверяет обязательное строковое поле и возвращает очищенную строку.

    Raises:
        ValueError: если значение отсутствует или является пустой строкой.
    """
    normalized = "" if value is None else str(value).strip()

    if not normalized:
        raise ValueError(f"Параметр '{field_name}' не задан.")

    return normalized


def html_value(value: Any) -> str:
    """
    Безопасно форматирует значение для HTML-лога PySM.

    Для dict/list используется JSON-представление.
    Для остальных типов используется str(value).
    """
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    return escape(text)


def is_empty_value(value: Any) -> bool:
    """
    Проверяет значение на пустоту.

    Правила:
        None -> пусто
        "" -> пусто
        [] -> пусто
        {} -> пусто
        0 -> не пусто
        False -> не пусто
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value == ""

    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0

    return False



# ==============================================================================
# 7. Преобразование типов для сравнения
# ==============================================================================

def parse_bool(value: Any) -> bool:
    """
    Преобразует значение в bool.

    Поддерживаемые истинные значения:
        true, 1, yes, y, on, да

    Поддерживаемые ложные значения:
        false, 0, no, n, off, нет
    """
    if isinstance(value, bool):
        return value

    text = str(value).strip().casefold()

    if text in {"true", "1", "yes", "y", "on", "да"}:
        return True

    if text in {"false", "0", "no", "n", "off", "нет"}:
        return False

    raise ValueError(f"Значение '{value}' невозможно преобразовать в bool.")


def parse_json_value(value: Any) -> Any:
    """
    Преобразует строку JSON в Python-значение.

    Если значение уже является dict/list/int/float/bool/None, оно возвращается
    без преобразования.
    """
    if not isinstance(value, str):
        return value

    return json.loads(value)


def resolve_effective_value_type(actual_value: Any, configured_type: str) -> str:
    """
    Определяет фактический тип сравнения.

    Если configured_type != auto, возвращается заданный тип.
    В режиме auto:
        bool -> bool
        int/float -> number
        dict/list -> json
        str/прочее -> string
    """
    if configured_type != VALUE_TYPE_AUTO:
        return configured_type

    if isinstance(actual_value, bool):
        return VALUE_TYPE_BOOL

    if isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool):
        return VALUE_TYPE_NUMBER

    if isinstance(actual_value, (dict, list)):
        return VALUE_TYPE_JSON

    return VALUE_TYPE_STRING


def convert_pair_for_comparison(
    actual_value: Any,
    comparison_value: str,
    value_type: str,
) -> tuple[Any, Any]:
    """
    Преобразует фактическое и эталонное значение к единому типу сравнения.
    """
    if value_type == VALUE_TYPE_STRING:
        return str(actual_value), str(comparison_value)

    if value_type == VALUE_TYPE_NUMBER:
        return float(actual_value), float(comparison_value)

    if value_type == VALUE_TYPE_BOOL:
        return parse_bool(actual_value), parse_bool(comparison_value)

    if value_type == VALUE_TYPE_JSON:
        return parse_json_value(actual_value), parse_json_value(comparison_value)

    raise ValueError(f"Неизвестный тип сравнения: {value_type}")


# ==============================================================================
# 8. Функции сравнения
# ==============================================================================

def string_equals(left: Any, right: Any) -> bool:
    """
    Регистронезависимое строковое равенство.

    Пример:
        "Yes" == "yes" -> True
    """
    return str(left).casefold() == str(right).casefold()


def string_contains(container: Any, item: Any) -> bool:
    """
    Регистронезависимая проверка вхождения для строк.
    """
    return str(item).casefold() in str(container).casefold()


def object_contains(container: Any, item: Any, value_type: str) -> bool:
    """
    Проверяет оператор 'содержит' для разных типов данных.

    Поведение:
        str:
            регистронезависимое вхождение подстроки.
        list/tuple/set:
            проверка наличия элемента.
        dict:
            проверка наличия ключа.
        прочее:
            строковое регистронезависимое вхождение.
    """
    if isinstance(container, str):
        return string_contains(container, item)

    if isinstance(container, dict):
        item_text = str(item).casefold()
        return any(str(key).casefold() == item_text for key in container.keys())

    if isinstance(container, (list, tuple, set)):
        if value_type == VALUE_TYPE_STRING:
            return any(string_equals(element, item) for element in container)

        return item in container

    return string_contains(container, item)


def evaluate_condition(
    actual_value: Any,
    variable_exists: bool,
    operator: str,
    comparison_value: str,
    configured_value_type: str,
) -> ConditionResult:
    """
    Вычисляет условие и возвращает результат без побочных эффектов.

    Логирование и маршрутизация выполняются вне этой функции.
    """
    if operator == OP_EXISTS:
        return ConditionResult(is_true=variable_exists)

    if operator == OP_NOT_EXISTS:
        return ConditionResult(is_true=not variable_exists)

    if not variable_exists:
        return ConditionResult(
            is_true=False,
            warning="Переменная или вложенный ключ не найдены.",
        )

    if operator == OP_EMPTY:
        return ConditionResult(is_true=is_empty_value(actual_value))

    if operator == OP_NOT_EMPTY:
        return ConditionResult(is_true=not is_empty_value(actual_value))

    effective_type = resolve_effective_value_type(
        actual_value=actual_value,
        configured_type=configured_value_type,
    )

    try:
        converted_actual, converted_comparison = convert_pair_for_comparison(
            actual_value=actual_value,
            comparison_value=comparison_value,
            value_type=effective_type,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return ConditionResult(
            is_true=False,
            warning=f"Не удалось привести значения к типу '{effective_type}': {e}",
        )

    try:
        if operator == OP_EQ:
            if effective_type == VALUE_TYPE_STRING:
                return ConditionResult(
                    is_true=string_equals(converted_actual, converted_comparison)
                )
            return ConditionResult(is_true=converted_actual == converted_comparison)

        if operator == OP_NE:
            if effective_type == VALUE_TYPE_STRING:
                return ConditionResult(
                    is_true=not string_equals(converted_actual, converted_comparison)
                )
            return ConditionResult(is_true=converted_actual != converted_comparison)

        if operator == OP_GT:
            return ConditionResult(is_true=converted_actual > converted_comparison)

        if operator == OP_LT:
            return ConditionResult(is_true=converted_actual < converted_comparison)

        if operator == OP_GE:
            return ConditionResult(is_true=converted_actual >= converted_comparison)

        if operator == OP_LE:
            return ConditionResult(is_true=converted_actual <= converted_comparison)

        if operator == OP_CONTAINS:
            return ConditionResult(
                is_true=object_contains(
                    container=converted_actual,
                    item=converted_comparison,
                    value_type=effective_type,
                )
            )

        if operator == OP_NOT_CONTAINS:
            return ConditionResult(
                is_true=not object_contains(
                    container=converted_actual,
                    item=converted_comparison,
                    value_type=effective_type,
                )
            )

    except TypeError as e:
        return ConditionResult(
            is_true=False,
            warning=f"Оператор '{operator}' неприменим к указанным значениям: {e}",
        )

    return ConditionResult(
        is_true=False,
        warning=f"Неизвестный оператор условия: {operator}",
    )


# ==============================================================================
# 9. Логирование результата
# ==============================================================================

def log_condition_report(
    variable_name: str,
    actual_value: Any,
    variable_exists: bool,
    operator: str,
    comparison_value: str,
    value_type: str,
    result: ConditionResult,
    target_id: Optional[str],
) -> None:
    """
    Выводит подробный отчёт о проверке условия в лог PySM.
    """
    logger.info("<b>ПРОВЕРКА УСЛОВИЯ ОПРЕДЕЛЕННОГО ПОЛЬЗОВАТЕЛЕМ...</b>")

    logger.info(
        "%s Переменная: <b>%s</b>",
        icon_info,
        escape(variable_name),
    )

    logger.info(
        "%s Существует: <b>%s</b>",
        icon_info,
        "да" if variable_exists else "нет",
    )

    logger.info(
        "%s Фактическое значение: <i>%s</i>",
        icon_info,
        html_value(actual_value),
    )

    logger.info(
        "%s Оператор: <b>%s</b>",
        icon_info,
        escape(operator),
    )

    logger.info(
        "%s Значение сравнения: <i>%s</i>",
        icon_info,
        html_value(comparison_value),
    )

    logger.info(
        "%s Тип сравнения: <b>%s</b>",
        icon_info,
        escape(value_type),
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


# ==============================================================================
# 10. Основная логика
# ==============================================================================

def main() -> None:
    """
    Основная функция-оркестратор.

    Последовательность:
        1. Настроить логирование.
        2. Получить конфигурацию.
        3. Проверить обязательные параметры.
        4. Проверить существование переменной.
        5. Получить значение переменной.
        6. Вычислить условие.
        7. Опционально удалить переменную.
        8. Установить следующий скрипт или продолжить штатно.
    """
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
    if (
        config.if_operator not in OPERATORS_WITHOUT_COMPARISON_VALUE
        and not str(config.if_comparison_value).strip()
    ):
        logger.error(
            "КРИТИЧЕСКАЯ ОШИБКА: для оператора '%s' необходимо указать "
            "параметр if-comparison-value.",
            config.if_operator,
        )
        sys.exit(1)

    try:
        variable_name = ensure_required_text(
            config.if_variable_name,
            "if-variable-name",
        )
        then_instance_id = ensure_required_text(
            config.then_instance_id,
            "then-instance-id",
        )
    except ValueError as e:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: %s", e)
        sys.exit(1)

    else_instance_id = normalize_optional_instance_id(config.else_instance_id)

    variable_exists = context_value_exists(pysm_context, variable_name)
    actual_value = pysm_context.get_structured(variable_name)

    condition_result = evaluate_condition(
        actual_value=actual_value,
        variable_exists=variable_exists,
        operator=config.if_operator,
        comparison_value=config.if_comparison_value,
        configured_value_type=config.if_value_type,
    )

    target_id = then_instance_id if condition_result.is_true else else_instance_id

    log_condition_report(
        variable_name=variable_name,
        actual_value=actual_value,
        variable_exists=variable_exists,
        operator=config.if_operator,
        comparison_value=config.if_comparison_value,
        value_type=config.if_value_type,
        result=condition_result,
        target_id=target_id,
    )

    if config.clear_variable:
        if variable_name.strip():
            logger.info(
                "%s Очистка переменной: <b>%s</b>",
                icon_delete,
                escape(variable_name),
            )
            remove_context_value(pysm_context, variable_name)
        else:
            logger.warning(
                "%s Очистка переменной пропущена: имя переменной пустое.",
                icon_warning,
            )

    if target_id:
        try:
            pysm_context.set_next_script(target_id)
            logger.info(
                "\n%s Порядок выполнения скриптов изменён",
                icon_play,
            )
        except Exception as e:
            logger.error(
                "КРИТИЧЕСКАЯ ОШИБКА при отправке команды перехода: %s",
                e,
            )
            sys.exit(1)

    elif not condition_result.is_true:
        logger.info(
            "%s Ветка 'Else' не определена. "
            "Выполнение будет продолжено по умолчанию.",
            icon_info,
        )

    sys.exit(0)


# ==============================================================================
# 11. Точка входа
# ==============================================================================

if __name__ == "__main__":
    main()
