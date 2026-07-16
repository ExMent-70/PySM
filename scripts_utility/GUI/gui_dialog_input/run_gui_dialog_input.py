#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_gui_dialog_input.py
=======================

GUI ввод значения с поддержкой:
- dot-notation
- типов данных
- единого InputProcessor
"""

# ==============================================================================
# 1. ИМПОРТЫ
# ==============================================================================
import argparse
import logging
import sys

try:
    from pysm_lib import pysm_context
    from pysm_lib import theme_api
    from pysm_lib.context_variable_ops import format_success
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.input_processor import InputProcessor, VALIDATION_PRESETS
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    InputProcessor = None
    VALIDATION_PRESETS = {}
    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"
    IS_MANAGED_RUN = False

    class MockThemeApi:
        @staticmethod
        def apply_theme_to_app(app):
            pass

    theme_api = MockThemeApi()

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox
except ImportError:
    print("❌ Требуется PySide6", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# ==============================================================================
# 2. CONFIG
# ==============================================================================
def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dlg_input_var", type=str, default="dlg_input_user_var")
    parser.add_argument("--dlg_input_msg", type=str, default="Введите значение:")
    parser.add_argument("--dlg_input_title", type=str, default="Ввод")
    parser.add_argument("--dlg_input_dvalue", type=str, default="")

    parser.add_argument(
        "--dlg_input_value_type",
        type=str,
        default="auto",
        choices=["auto", "string", "int", "float", "bool", "json"],
    )

    parser.add_argument(
        "--dlg_input_valid_type",
        type=str,
        default="none",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
    )

    parser.add_argument("--dlg_input_custom_regexp", type=str)
    parser.add_argument("--dlg_input_custom_regexp_desc", type=str)

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()

    return parser.parse_args()


# ==============================================================================
# 3. UI
# ==============================================================================
def run_dialog(title, msg, initial):
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    text, ok = QInputDialog.getText(
        None,
        title,
        msg,
        QLineEdit.EchoMode.Normal,
        initial,
    )

    if not ok:
        return None

    return text


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    config = get_config()

    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)

    # 👉 Получаем initial value через unified API
    initial_val = processor.get_initial_value(
        config.dlg_input_var,
        config.dlg_input_dvalue,
    )

    while True:
        result = run_dialog(
            config.dlg_input_title,
            config.dlg_input_msg,
            initial_val,
        )

        if result is None:
            sys.exit(1)

        try:
            processor.process(
                raw_value=result,
                var_name=config.dlg_input_var,
                value_type=config.dlg_input_value_type,
            )
            break

        except Exception as e:
            QMessageBox.warning(None, "Ошибка", str(e))
            initial_val = result  # повторный ввод

    logger.info(format_success(config.dlg_input_var, result))

    sys.exit(0)


if __name__ == "__main__":
    main()
