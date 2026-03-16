# run_py_if_else.py

"""
Скрипт-шлюз для условных переходов в PyScriptManager.

Назначение:
Реализует логику "If-Then-Else" для потока выполнения скриптов.
Он проверяет значение переменной в контексте по заданному условию и,
в зависимости от результата, устанавливает следующий для выполнения скрипт.

Принцип работы:
- Скрипт является неинтерактивным. Вся его работа определяется параметрами.
- Он получает имя переменной, оператор, значение для сравнения и ID целевых
  скриптов для "Then" и "Else" веток.
- Сравнивает фактическое значение переменной из контекста с эталонным.
- Если условие истинно, устанавливает "Then" скрипт как следующий.
- Если условие ложно, устанавливает "Else" скрипт как следующий. Если "Else"
  не указан, выполнение продолжается по стандартному порядку.
- Опционально может удалить проверенную переменную из контекста.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import logging
import sys
from pathlib import Path
from argparse import Namespace
from typing import Any

IS_MANAGED_RUN = False
# Добавляем корневую папку 'analize' в sys.path
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
     
    # Импорт PySM
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder
    IS_MANAGED_RUN = True
except ImportError as e:
    pysm_context = None
    ConfigResolver = None
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    print("Убедитесь, что структура папок верна и все зависимости установлены.", file=sys.stderr)
    sys.exit(1)


from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_delete,
    icon_play,
    icon_save,
    icon_arrow_sub,
    icon_save_warning,
    icon_save_error
)


# Инициализируем глобальный логгер
logger = logging.getLogger(__name__)

# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Выполняет условный переход к другому скрипту."
    )
    # --- Аргументы для условия ---
    parser.add_argument(
        "--if-variable-name",
        type=str,
        required=True,
        help="Имя переменной в контексте для проверки (можно использовать 'a.b.c')."
    )
    parser.add_argument(
        "--if-operator",
        type=str,
        required=True,
        choices=["равно", "не равно", "больше", "меньше", "содержит", "не содержит"],
        help="Оператор для сравнения."
    )
    parser.add_argument(
        "--if-comparison-value",
        type=str,
        required=True,
        help="Значение, с которым будет сравниваться переменная."
    )
    # --- Аргументы для переходов ---
    parser.add_argument(
        "--then-instance-id",
        type=str,
        required=True,
        help="ID экземпляра скрипта, если условие истинно."
    )
    parser.add_argument(
        "--else-instance-id",
        type=str,
        default=None,
        help="ID экземпляра скрипта, если условие ложно (необязательно)."
    )
    # --- Дополнительные опции ---
    parser.add_argument(
        "--clear-variable",
        action="store_true",
        help="Если указано, удалить проверенную переменную из контекста после сравнения."
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        # Для автономного запуска потребуются все аргументы
        return parser.parse_args()


# 3. БЛОК: Вспомогательная функция для сравнения
# ==============================================================================
def evaluate_condition(actual_value: Any, operator: str, comparison_value: str) -> bool:
    """Выполняет сравнение на основе оператора."""
    # Приводим фактическое значение к строке для текстовых операций
    actual_str = str(actual_value)
    
    # 1. Текстовые и общие сравнения
    if operator == "равно":
        return actual_str == comparison_value
    if operator == "не равно":
        return actual_str != comparison_value
    if operator == "содержит":
        return comparison_value in actual_str
    if operator == "не содержит":
        return comparison_value not in actual_str

    # 2. Числовые сравнения
    if operator in ["больше", "меньше"]:
        try:
            # Пытаемся преобразовать оба значения в числа
            num_actual = float(actual_value)
            num_comparison = float(comparison_value)
            if operator == "больше":
                return num_actual > num_comparison
            if operator == "меньше":
                return num_actual < num_comparison
        except (ValueError, TypeError):
            # Если преобразование не удалось, числовое сравнение невозможно
            tqdm.write(
                f"ПРЕДУПРЕЖДЕНИЕ: Не удалось выполнить числовое сравнение. "
                f"Значение '{actual_str}' или '{comparison_value}' не является числом."
            )
            return False
    return False


# 4. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN and pysm_context else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    config = get_config()

    # 1. Получаем фактическое значение переменной из контекста
    actual_value = pysm_context.get_structured(config.if_variable_name)
    logger.info("<b>ПРОВЕРКА УСЛОВИЯ...</b>")

    # 2. Вычисляем результат условия
    is_true = evaluate_condition(actual_value, config.if_operator, config.if_comparison_value)

    target_id = None
    branch = icon_warning
    if is_true:
        target_id = config.then_instance_id
        branch = icon_ok
        #logger.info("Результат: <b>ИСТИНА</b>")
    else:
        target_id = config.else_instance_id
        branch = icon_error
        #logger.info(f"{icon_error}Результат: <b>ЛОЖЬ</b>")

    logger.info(f"{branch} Значение переменной <b>{config.if_variable_name.upper()}</b> [<i>{str(actual_value).upper()}</i>] <b>{config.if_operator}</b> [<i>{config.if_comparison_value.upper()}</i>]")

    # 4. Опционально очищаем переменную
    if config.clear_variable:
        logger.info(f"{icon_delete} Очистка переменной <b>{config.if_variable_name.upper()}</b>")
        pysm_context.remove(config.if_variable_name)

    logger.info(f"{icon_info} Порядок выполнения скриптов изменён, следующий скрипт:")

    # 3. Устанавливаем следующий скрипт или продолжаем по умолчанию
    if target_id:
        try:
            # Получаем имя целевого скрипта для красивого лога
            all_instances = {inst['id']: inst['name'] for inst in pysm_context.list_instances()}
            target_name = all_instances.get(target_id, "Неизвестное имя")
            
            pysm_context.set_next_script(target_id)
            logger.info(f" {icon_arrow_sub} <b>{target_name.upper()}</b> (ID: <i>{target_id}</i>)<br>")
        except Exception as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА при установке следующего скрипта: {e}")
            sys.exit(1)
    elif not is_true:
        # Это случай, когда условие ложно и `else_instance_id` не указан
        logger.info("Ветка 'Else' не определена. Выполнение будет продолжено по умолчанию.")
    
    sys.exit(0)


# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()