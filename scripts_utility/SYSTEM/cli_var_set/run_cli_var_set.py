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
import re
import sys
from typing import Optional, Dict, Tuple

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


# 2. БЛОК: Константы
# ==============================================================================
VALIDATION_PRESETS: Dict[str, Dict[str, str]] = {
    "not_empty": {
        "pattern": r".+",
        "description": "Требуется любой непустой текст.",
    },
    "integer": {
        "pattern": r"^-?\d+$",
        "description": "Требуется целое число.",
    },
    "positive_integer": {
        "pattern": r"^\d+$",
        "description": "Требуется положительное целое число или ноль.",
    },
    "float": {
        "pattern": r"^-?\d+(\.\d+)?$",
        "description": "Требуется число с плавающей точкой.",
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "description": "Требуется корректный адрес электронной почты.",
    },
    "filename_txt": {
        "pattern": r"^[^\\/:*?\"<>|]+\.txt$",
        "description": "Требуется имя файла с расширением .txt.",
    },
}


# 3. БЛОК: Сервисные классы
# ==============================================================================
class ContextHandler:
    """
    Отвечает за сохранение данных в контекст PyScriptManager.
    """
    def __init__(self, var_name: str):
        self.var_name = var_name

    def save_result(self, value: str) -> None:
        """Сохраняет результат в контекст."""
        logger.debug(f"Попытка сохранить значение в '{self.var_name}': '{value}'")
        
        if IS_MANAGED_RUN and pysm_context:
            try:
                pysm_context.set(self.var_name, value)
                #logger.info("Переменная контекста успешно сохранена.")
                logger.info(f"✅ <b>{self.var_name}</b> = <i>{value}</i>\n")
            except Exception as e:
                logger.critical(f"Ошибка при сохранении данных в контекст: {e}")
                sys.exit(1)
        else:
            logger.info("Запуск в автономном режиме, запись в контекст эмулирована.")


class Validator:
    """
    Отвечает за проверку данных по регулярным выражениям.
    Полностью идентичен валидатору из GUI-версии.
    """
    def __init__(self, config: argparse.Namespace):
        self.pattern: Optional[str] = None
        self.error_desc: str = "Неизвестная ошибка валидации."
        self._setup(config)

    def _setup(self, config: argparse.Namespace) -> None:
        """Настраивает паттерн на основе аргументов CLI."""
        # Маппинг имен аргументов отличается от GUI версии,
        # поэтому здесь используем set_ префиксы
        if config.set_valid_type == "custom":
            self.pattern = config.set_custom_regexp
            self.error_desc = (
                config.set_custom_regexp_desc
                or "Значение не соответствует заданному формату."
            )
            if not self.pattern:
                logger.warning("Тип 'custom' выбран, но шаблон пуст. Валидация отключена.")
        elif config.set_valid_type in VALIDATION_PRESETS:
            preset = VALIDATION_PRESETS[config.set_valid_type]
            self.pattern = preset["pattern"]
            self.error_desc = preset["description"]

    def validate(self, text: str) -> Tuple[bool, str]:
        """Returns: (Valid?, ErrorMessage)"""
        if not self.pattern:
            return True, ""

        try:
            if re.fullmatch(self.pattern, text, re.IGNORECASE):
                return True, ""
            return False, self.error_desc
        except re.error as e:
            err_msg = f"Ошибка в Regex шаблоне: {e}"
            logger.error(err_msg)
            return False, err_msg


# 4. БЛОК: Конфигурация и Main
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Устанавливает значение переменной контекста без GUI."
    )
    
    parser.add_argument(
        "--set_var_name", type=str, required=True,
        help="Имя переменной контекста."
    )
    parser.add_argument(
        "--set_var_value", type=str, default="",
        help="Значение для сохранения."
    )
    parser.add_argument(
        "--set_valid_type", type=str, default="none",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
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
    
    # 2. Подготовка значения и сервисов
    target_value = config.set_var_value
    validator = Validator(config)
    context_handler = ContextHandler(config.set_var_name)
    
    # 3. Валидация
    is_valid, error_msg = validator.validate(target_value)
    
    if not is_valid:
        logger.error(f"ОШИБКА ВАЛИДАЦИИ: {error_msg}")
        logger.error(f"Полученное значение: '{target_value}'")
        # Завершаем с кодом 1, чтобы цепочка скриптов остановилась (если так настроено)
        sys.exit(1)
        
    # 4. Сохранение
    context_handler.save_result(target_value)
    
    # 5. Успешный выход
    sys.exit(0)


if __name__ == "__main__":
    main()