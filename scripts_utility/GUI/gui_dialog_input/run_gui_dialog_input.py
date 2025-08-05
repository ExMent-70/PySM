#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_gui_dialog_input.py
=======================
Утилита для отображения модального диалогового окна ввода текста.

Этот скрипт позволяет запрашивать у пользователя ввод значения через GUI,
поддерживает валидацию с использованием предустановленных или кастомных
регулярных выражений и сохраняет результат в контекст PyScriptManager.
"""

# 1. БЛОК: Импорты и константы
# ==============================================================================
import argparse
import logging
import re
import sys

# Определяем, запущен ли скрипт под управлением PySM
IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None

# Импортируем PySide6
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox
except ImportError:
    print("Критическая ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)

# Настройка логирования для вывода информации в stdout
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# Словарь с предустановленными шаблонами валидации.
# Шаблоны не содержат флагов (?i), так как используется флаг re.IGNORECASE.
VALIDATION_PRESETS = {
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


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Определяет, парсит и возвращает аргументы командной строки.

    Использует ConfigResolver из pysm_lib для получения значений из контекста,
    если скрипт запущен под управлением PyScriptManager.

    Returns:
        argparse.Namespace: Объект с параметрами конфигурации.
    """
    parser = argparse.ArgumentParser(
        description="Показывает диалог для ввода значения с гибкой валидацией."
    )

    parser.add_argument(
        "--dlg_input_var",
        type=str,
        help="Имя переменной контекста для сохранения результата.",
        default="dlg_input_user_var",
    )
    parser.add_argument(
        "--dlg_input_msg",
        type=str,
        help="Текст-приглашение для ввода в диалоговом окне.",
        default="Введите значение:",
    )
    parser.add_argument(
        "--dlg_input_title",
        type=str,
        help="Заголовок диалогового окна.",
        default="Ввод значения",
    )
    parser.add_argument(
        "--dlg_input_dvalue",
        type=str,
        help="Значение, отображаемое в поле ввода по умолчанию.",
        default="",
    )
    parser.add_argument(
        "--dlg_input_valid_type",
        type=str,
        help="Тип валидации вводимого значения из предустановленных.",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
        default="none",
    )
    parser.add_argument(
        "--dlg_input_custom_regexp",
        type=str,
        help="Пользовательский шаблон регулярного выражения (используется, если --dlg_input_valid_type='custom').",
    )
    parser.add_argument(
        "--dlg_input_custom_regexp_desc",
        type=str,
        help="Описание для пользовательского шаблона (используется, если --dlg_input_valid_type='custom').",
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    """
    Основная функция-оркестратор.

    Выполняет следующие шаги:
    1. Получает конфигурацию.
    2. Определяет параметры валидации.
    3. Определяет начальное значение для поля ввода.
    4. Запускает цикл GUI для ввода и валидации.
    5. Сохраняет результат в контекст (если применимо) и завершает работу.
    """
    # 3.1. Получение конфигурации
    config = get_config()

    # 3.2. Определение шаблона и описания для валидации
    validation_pattern: Optional[str] = None
    error_description = "Неизвестная ошибка валидации."

    # --- НАЧАЛО ИЗМЕНЕНИЙ: Восстанавливаем логику для 'custom' ---
    if config.dlg_input_valid_type == "custom":
        validation_pattern = config.dlg_input_custom_regexp
        error_description = (
            config.dlg_input_custom_regexp_desc
            or "Значение не соответствует заданному формату."
        )
        if not validation_pattern:
            logger.warning(
                "Выбран тип валидации 'custom', но не задан шаблон 'dlg_input_custom_regexp'. Валидация отключена."
            )
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    elif config.dlg_input_valid_type in VALIDATION_PRESETS:
        preset = VALIDATION_PRESETS[config.dlg_input_valid_type]
        validation_pattern = preset["pattern"]
        error_description = preset["description"]

    # 3.3. Определение начального значения для поля ввода
    initial_value = config.dlg_input_dvalue
    if IS_MANAGED_RUN and pysm_context:
        context_value = pysm_context.get(config.dlg_input_var)
        if context_value is not None:
            initial_value = str(context_value)
            logger.info(
                f"Начальное значение из контекста '{config.dlg_input_var}': '{initial_value}'"
            )

    # 3.4. Инициализация GUI и цикл ввода/валидации
    q_app = QApplication.instance() or QApplication(sys.argv)
    user_input = initial_value
    ok_pressed = False

    while True:
        text, ok = QInputDialog.getText(
            None,
            config.dlg_input_title,
            config.dlg_input_msg,
            QLineEdit.EchoMode.Normal,
            user_input,
        )

        if not ok:
            ok_pressed = False
            break

        user_input = text

        if not validation_pattern:
            ok_pressed = True
            break

        is_valid = False
        try:
            if re.fullmatch(validation_pattern, user_input, re.IGNORECASE):
                is_valid = True
        except re.error as e:
            error_description = f"Ошибка в шаблоне регулярного выражения: {e}"
            logger.error(error_description)

        if is_valid:
            ok_pressed = True
            break
        else:
            msg_box = QMessageBox(
                QMessageBox.Icon.Warning, "Неверный формат", error_description
            )
            msg_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msg_box.exec()

    # 3.5. Обработка результата и завершение
    if not ok_pressed:
        logger.warning("Операция отменена пользователем. Выполнение прервано.")
        sys.exit(1)

    logger.info(f"Пользователь ввел: '{user_input}'")

    if IS_MANAGED_RUN and pysm_context:
        try:
            pysm_context.set(config.dlg_input_var, user_input)
            logger.info("Переменная контекста успешно сохранена.")
            logger.info(f"{config.dlg_input_var} = {user_input}")
        except Exception as e:
            logger.critical(f"Ошибка при сохранении данных в контекст: {e}")
            sys.exit(1)
    else:
        logger.info("Запуск в автономном режиме, результат в контекст не сохраняется.")

    # 3.6. Успешное завершение
    logger.info("Скрипт успешно завершен.")
    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()