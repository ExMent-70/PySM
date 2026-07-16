#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cli_set_var.py
====================
Утилита для установки значения переменной контекста без GUI.

Особенности:
- Принимает имя переменной и значение через аргументы командной строки.
- Выполняет валидацию значения перед сохранением (используя ту же логику, что и GUI-версия).
- Завершается с кодом 1, если валидация не пройдена.
"""

# 1. БЛОК: Импорты и настройки
# ==============================================================================
import argparse
import logging
import sys

# Попытка импорта зависимостей PySM
try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.input_processor import InputProcessor
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    InputProcessor = None
    pysm_context = None
    ConfigResolver = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"
    IS_MANAGED_RUN = False

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

SET_VALIDATION_PRESETS = [
    "not_empty",
    "filename_txt",
    "integer",
    "positive_integer",
    "float",
    "email",
]


# 2. БЛОК: Конфигурация и Main
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Устанавливает значение переменной контекста без GUI."
    )
    
    parser.add_argument(
        "--set_var_name", type=str, required=True,
        help="Имя переменной контекста. Поддерживается точечная нотация, например project.name."
    )
    parser.add_argument(
        "--set_var_value", type=str, default="",
        help="Значение для сохранения."
    )
    parser.add_argument(
        "--set_valid_type", type=str, default="none",
        choices=["none", "custom"] + SET_VALIDATION_PRESETS,
        help="Тип валидации."
    )
    parser.add_argument(
        "--set_custom_regexp", type=str,
        help="Regex шаблон (для custom)."
    )
    parser.add_argument(
        "--set_custom_regexp_desc", type=str,
        help="Описание ошибки (для custom)."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    return parser.parse_args()


def main():
    """Оркестратор процесса."""
    # 1. Получение конфигурации
    config = get_config()

    if InputProcessor is None:
        logger.critical(format_error("InputProcessor недоступен. Скрипт должен запускаться в окружении PySM."))
        sys.exit(1)

    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)

    try:
        result = processor.process(
            raw_value=config.set_var_value,
            var_name=config.set_var_name,
            value_type="string",
        )
        logger.info(format_success(config.set_var_name, result) + "\n")
    except Exception as e:
        logger.error(format_error(f"Ошибка валидации: {e}"))
        logger.error(f"Полученное значение: '{config.set_var_value}'")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
