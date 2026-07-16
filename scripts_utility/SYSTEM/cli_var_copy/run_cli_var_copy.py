#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cli_copy_value.py
=====================
Утилита для копирования переменных внутри контекста PyScriptManager.
"""

# 1. БЛОК: Импорты
import argparse
import logging
import sys

# Попытка импорта зависимостей PySM
try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import (
        copy_context_value,
        format_error,
        format_success,
    )
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
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


# 2. БЛОК: Конфигурация (ИСПРАВЛЕНО)
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Копирует значение и тип одной переменной контекста в другую."
    )
    
    parser.add_argument(
        "--copy_source_var", 
        type=str, 
        required=True,
        help="Имя исходной переменной. Поддерживается точечная нотация, например project.name."
    )
    
    parser.add_argument(
        "--copy_target_var", 
        type=str, 
        required=True,
        help="Имя целевой переменной. Поддерживается точечная нотация, например project.backup_name."
    )
    
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # Используем nargs='?', чтобы аргумент мог работать и как флаг (без значения),
    # и как параметр со значением.
    parser.add_argument(
        "--copy_fail_if_missing", 
        nargs='?',        # 0 или 1 аргумент
        const="True",     # Если значение не передано (просто флаг), считаем True
        default="True",   # Значение по умолчанию, если флага нет вообще
        help="Завершать с ошибкой, если исходная переменная не найдена."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    
    return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main() -> None:
    # 1. Проверка среды
    if not IS_MANAGED_RUN or not pysm_context:
        logger.warning("Скрипт запущен вне среды PyScriptManager. Пропуск.")
        sys.exit(0)

    # 2. Получение конфигурации
    config = get_config()
    source_key: str = config.copy_source_var
    target_key: str = config.copy_target_var
    
    # Ручное преобразование в bool (надежнее, чем type=bool в argparse)
    raw_bool_val = str(config.copy_fail_if_missing).lower()
    fail_if_missing = raw_bool_val in ('true', 'yes', '1', 't', 'on')

    logger.info(f"Запуск копирования: '{source_key}' -> '{target_key}'")

    # 3. Получение данных
    try:
        source = copy_context_value(pysm_context, source_key, target_key)
    except Exception as e:
        logger.critical(format_error(f"Ошибка записи: {e}"))
        sys.exit(1)

    if not source.exists:
        msg = f"Исходная переменная '{source_key}' отсутствует в контексте."
        if fail_if_missing:
            logger.error(format_error(msg))
            sys.exit(1)
        else:
            logger.warning(f"ПРЕДУПРЕЖДЕНИЕ: {msg} Пропуск операции.")
            sys.exit(0)

    # 4. Копирование
    val_str = str(source.value)
    val_preview = (val_str[:47] + "...") if len(val_str) > 50 else val_str
    
    logger.info(f"Найдено значение: {val_preview} (Тип: {source.var_type})")
    logger.info(format_success(target_key, source.value))

    sys.exit(0)


if __name__ == "__main__":
    main()
