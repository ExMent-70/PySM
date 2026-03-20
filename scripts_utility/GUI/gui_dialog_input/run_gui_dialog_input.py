#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_gui_dialog_input.py
=======================
Утилита для отображения модального диалогового окна ввода текста.

Особенности:
- Поддержка валидации (Regex) с пресетами и кастомными шаблонами.
- Интеграция с PyScriptManager (PySM) для чтения/записи контекста.
- Использование PySide6 для GUI.
"""

# 1. БЛОК: Импорты и настройки
# ==============================================================================
import argparse
import logging
import re
import sys
from typing import Optional, Dict, Any, Tuple

# Попытка импорта зависимостей PySM
try:
    from pysm_lib import pysm_context
    from pysm_lib import theme_api
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    IS_MANAGED_RUN = False
    # Заглушка для theme_api, чтобы код не падал в автономном режиме
    class MockThemeApi:
        @staticmethod
        def apply_theme_to_app(app):
            pass
    theme_api = MockThemeApi()

# Импорт GUI библиотеки
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox
except ImportError:
    print("❌ Критическая ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# 2. БЛОК: Константы и Данные
# ==============================================================================
VALIDATION_PRESETS: Dict[str, Dict[str, str]] = {
    "not_empty": {
        "pattern": r".+",
        "description": "Требуется любой непустой текст.",
    },
    "integer": {
        "pattern": r"^-?\d+$",
        "description": "Требуется целое число (например, -10, 0, 123).",
    },
    "positive_integer": {
        "pattern": r"^\d+$",
        "description": "Требуется положительное целое число или ноль (например, 0, 5, 100).",
    },
    "float": {
        "pattern": r"^-?\d+(\.\d+)?$",
        "description": "Требуется число с плавающей точкой (например, -3.14, 10, 99.9).",
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "description": "Требуется корректный адрес электронной почты.",
    },
    "filename_txt": {
        "pattern": r"^[^\\/:*?\"<>|]+\.txt$",
        "description": "Требуется имя файла с расширением .txt, без запрещенных символов.",
    },
}


# 3. БЛОК: Сервисные классы (Бизнес-логика)
# ==============================================================================
class ContextHandler:
    """
    Абстракция для работы с контекстом приложения (PySM).
    Скрывает различия между управляемым и автономным запуском.
    """

    def __init__(self, var_name: str):
        self.var_name = var_name

    def get_initial_value(self, default: str) -> str:
        """Получает значение из контекста или возвращает дефолтное."""
        if IS_MANAGED_RUN and pysm_context:
            context_val = pysm_context.get(self.var_name)
            if context_val is not None:
                logger.debug(f"ℹ️ Текущее значение: <b>{self.var_name}</b> = <i>{context_val}</i>")
                return str(context_val)
        return default

    def save_result(self, value: str) -> None:
        """Сохраняет результат в контекст, если это возможно."""
        logger.debug(f"Пользователь ввел: '{value}'")
        
        if IS_MANAGED_RUN and pysm_context:
            try:
                pysm_context.set(self.var_name, value)
                logger.debug("Переменная контекста успешно сохранена.")
                logger.info(f"✅ <b>{self.var_name}</b> = <i>{value}</i>\n")
            except Exception as e:
                logger.critical(f"❌ Ошибка при сохранении данных в контекст: {e}")
                sys.exit(1)
            #finally:
                #logger.info(f"<br>")
            
        else:
            logger.info("⚠️ Запуск в автономном режиме, результат в контекст не сохраняется.")


class Validator:
    """
    Отвечает за логику проверки введенных данных.
    """
    def __init__(self, config: argparse.Namespace):
        self.pattern: Optional[str] = None
        self.error_desc: str = "❌ Неизвестная ошибка валидации."
        self._setup(config)

    def _setup(self, config: argparse.Namespace) -> None:
        """Настраивает паттерн на основе конфигурации."""
        if config.dlg_input_valid_type == "custom":
            self.pattern = config.dlg_input_custom_regexp
            self.error_desc = (
                config.dlg_input_custom_regexp_desc
                or "⚠️ Значение не соответствует заданному формату."
            )
            if not self.pattern:
                logger.warning(
                    "⚠️ Выбран тип валидации 'custom', но не задан шаблон. Валидация отключена."
                )
        elif config.dlg_input_valid_type in VALIDATION_PRESETS:
            preset = VALIDATION_PRESETS[config.dlg_input_valid_type]
            self.pattern = preset["pattern"]
            self.error_desc = preset["description"]

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет текст.
        Returns:
            Tuple[bool, str]: (Valid?, ErrorMessage)
        """
        if not self.pattern:
            return True, ""

        try:
            if re.fullmatch(self.pattern, text, re.IGNORECASE):
                return True, ""
            return False, self.error_desc
        except re.error as e:
            err_msg = f"❌ Ошибка в шаблоне регулярного выражения: {e}"
            logger.error(err_msg)
            return False, err_msg


# 4. БЛОК: UI Логика
# ==============================================================================
def run_dialog_loop(
    title: str, 
    msg: str, 
    initial_value: str, 
    validator: Validator
) -> Optional[str]:
    """
    Запускает цикл отображения диалога.
    
    Returns:
        str: Введенное значение.
        None: Если пользователь отменил ввод.
    """
    # Инициализация QApplication (синглтон)
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    current_value = initial_value

    while True:
        text, ok = QInputDialog.getText(
            None,
            title,
            msg,
            QLineEdit.EchoMode.Normal,
            current_value,
        )

        if not ok:
            return None

        current_value = text
        is_valid, error_msg = validator.validate(current_value)

        if is_valid:
            return current_value
        
        # Показ ошибки
        msg_box = QMessageBox(
            QMessageBox.Icon.Warning, "Неверный формат", error_msg
        )
        msg_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        msg_box.exec()


# 5. БЛОК: Конфигурация и Точка входа
# ==============================================================================
def get_config() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог для ввода значения с гибкой валидацией."
    )
    parser.add_argument(
        "--dlg_input_var", type=str, default="dlg_input_user_var",
        help="Имя переменной контекста для сохранения результата."
    )
    parser.add_argument(
        "--dlg_input_msg", type=str, default="Введите значение:",
        help="Текст-приглашение для ввода."
    )
    parser.add_argument(
        "--dlg_input_title", type=str, default="Ввод значения",
        help="Заголовок диалогового окна."
    )
    parser.add_argument(
        "--dlg_input_dvalue", type=str, default="",
        help="Значение по умолчанию."
    )
    parser.add_argument(
        "--dlg_input_valid_type", type=str, default="none",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
        help="Тип валидации."
    )
    parser.add_argument(
        "--dlg_input_custom_regexp", type=str,
        help="Пользовательский шаблон регулярного выражения."
    )
    parser.add_argument(
        "--dlg_input_custom_regexp_desc", type=str,
        help="Описание ошибки для пользовательского шаблона."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    return parser.parse_args()


def main():
    """Оркестратор процесса."""
    # 1. Конфигурация
    config = get_config()
    
    #print(f"<b>Инициализация переменной контекста <i>{config.dlg_input_var}</i></b><br>")
    # 2. Подготовка сервисов
    context_handler = ContextHandler(config.dlg_input_var)
    validator = Validator(config)
    
    # 3. Получение начальных данных
    initial_val = context_handler.get_initial_value(config.dlg_input_dvalue)
    
    # 4. Запуск GUI
    result = run_dialog_loop(
        title=config.dlg_input_title,
        msg=config.dlg_input_msg,
        initial_value=initial_val,
        validator=validator
    )
    
    # 5. Обработка результата
    if result is None:
        logger.debug("Операция отменена пользователем. Выполнение прервано.")
        sys.exit(1)
        
    context_handler.save_result(result)
    logger.debug("Скрипт успешно завершен.")
    sys.exit(0)


if __name__ == "__main__":
    main()