#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cli_remove_var.py
=====================
Утилита для удаления переменных из контекста PyScriptManager.

Функциональность:
    - Если имя переменной == "all" (регистронезависимо), удаляет ВСЕ переменные.
    - Иначе удаляет конкретную переменную по имени.

Зависимости:
    - pysm_lib (для взаимодействия с контекстом).
"""

# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import logging
import sys
from typing import Optional, Dict, Any

try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import (
        context_value_exists,
        format_error,
        remove_context_value,
        success_icon,
    )
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    def success_icon() -> str:
        return "✅"
    IS_MANAGED_RUN = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format="%(message)s", 
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# 2. БЛОК: Конфигурация
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Удаляет переменную или очищает контекст."
    )
    
    parser.add_argument(
        "--remove_var_name", 
        type=str, 
        required=True,
        help="Имя переменной, dotted-путь или 'all' для полной очистки."
    )
    
    # Флаг строгого режима
    parser.add_argument(
        "--remove_fail_if_missing", 
        nargs='?',
        const="True",
        default="False",
        help="Завершать с ошибкой, если переменная не найдена."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    
    return parser.parse_args()


def str_to_bool(value: Any) -> bool:
    """Надежное преобразование в bool."""
    return str(value).lower() in ('true', 'yes', '1', 't', 'on')


# 3. БЛОК: Основная логика
# ==============================================================================
def main() -> None:
    """
    Оркестратор процесса удаления.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        logger.warning("Скрипт запущен вне среды PySM. Операция невозможна.")
        sys.exit(0)

    config = get_config()
    
    # Получаем имя и очищаем от пробелов
    var_name_input: str = config.remove_var_name.strip()
    fail_if_missing: bool = str_to_bool(config.remove_fail_if_missing)

    if not var_name_input:
        logger.error(format_error("Имя переменной не может быть пустым."))
        sys.exit(1)

    # --- СЦЕНАРИЙ 1: Полная очистка (ключевое слово 'all') ---
    if var_name_input.lower() == 'all':
        logger.warning("Получена команда 'all'. ЗАПУЩЕН РЕЖИМ ПОЛНОЙ ОЧИСТКИ КОНТЕКСТА.")
        try:
            # Вызов remove() без аргументов удаляет всё
            remove_context_value(pysm_context)
            logger.info(f"{success_icon()} Все пользовательские переменные удалены.")
            sys.exit(0)
        except Exception as e:
            logger.critical(format_error(f"Ошибка при очистке: {e}"))
            sys.exit(1)

    # --- СЦЕНАРИЙ 2: Удаление конкретной переменной ---
    logger.info(f"Запрос на удаление конкретной переменной: '{var_name_input}'")

    # Проверка существования
    variable_exists = context_value_exists(pysm_context, var_name_input)

    if not variable_exists:
        msg = f"Переменная '{var_name_input}' не найдена в контексте."
        if fail_if_missing:
            logger.error(format_error(msg))
            sys.exit(1)
        else:
            logger.warning(f"ПРЕДУПРЕЖДЕНИЕ: {msg} Удалять нечего.")
            sys.exit(0)

    # Удаление
    try:
        # Дополнительная защита для обычных переменных и базовых JSON-объектов.
        base_var_name = var_name_input.split(".", 1)[0]
        existing_data: Optional[Dict[str, Any]] = pysm_context.get_variable(base_var_name)
        if existing_data and existing_data.get("read_only", False):
            logger.error(format_error(f"Переменная '{var_name_input}' защищена от удаления (read_only)."))
            sys.exit(1)

        remove_context_value(pysm_context, var_name_input)
        
        # Верификация
        if not context_value_exists(pysm_context, var_name_input):
            logger.info(f"{success_icon()} Переменная '{var_name_input}' удалена.")
        else:
            raise RuntimeError("Метод remove отработал, но переменная осталась.")
            
    except Exception as e:
        logger.critical(format_error(f"Ошибка удаления: {e}"))
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
