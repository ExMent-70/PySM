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
from typing import Optional, Dict, Any

# Попытка импорта зависимостей PySM
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
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
        help="Имя исходной переменной."
    )
    
    parser.add_argument(
        "--copy_target_var", 
        type=str, 
        required=True,
        help="Имя целевой переменной."
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
    source_data: Optional[Dict[str, Any]] = pysm_context.get_variable(source_key)

    if source_data is None:
        msg = f"Исходная переменная '{source_key}' отсутствует в контексте."
        if fail_if_missing:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {msg}")
            sys.exit(1)
        else:
            logger.warning(f"ПРЕДУПРЕЖДЕНИЕ: {msg} Пропуск операции.")
            sys.exit(0)

    # 4. Копирование
    value = source_data.get("value")
    var_type = source_data.get("type")
    
    val_str = str(value)
    val_preview = (val_str[:47] + "...") if len(val_str) > 50 else val_str
    
    logger.info(f"Найдено значение: {val_preview} (Тип: {var_type})")

    try:
        pysm_context.set(target_key, value, var_type=var_type)
        logger.info(f"УСПЕХ: Значение скопировано в '{target_key}'.")
    except Exception as e:
        logger.critical(f"ОШИБКА ЗАПИСИ: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()