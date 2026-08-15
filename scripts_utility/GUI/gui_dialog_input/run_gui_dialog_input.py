#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Request, validate, convert, and save a value through an HTML dialog."""

from __future__ import annotations

import argparse
import html
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional


GUI_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(GUI_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

IS_MANAGED_RUN = False

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.input_processor import InputProcessor, VALIDATION_PRESETS
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = pysm_context.is_managed
except ImportError as import_error:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    InputProcessor = None
    VALIDATION_PRESETS = {}

    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"

    logger.debug("PySM API недоступен: %s", import_error)

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QLineEdit,
        QMessageBox,
        QVBoxLayout,
    )
    from _common.html_dialog import (
        build_html_document,
        create_html_browser,
        log_html_to_console,
        validate_html_layout,
    )
except ImportError:
    print("❌ Требуется PySide6", file=sys.stderr)
    sys.exit(1)


OUTPUT_DIALOG = "dialog"
OUTPUT_CONSOLE_DIALOG = "console_dialog"


def get_config() -> Namespace:
    """Resolve command-line and PySM Collection Context parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Показывает HTML-диалог для ввода, проверки и сохранения значения."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dlg_input_var",
        type=str,
        default="dlg_input_user_var",
        help="Переменная Контекста Коллекции для сохранения результата.",
    )
    parser.add_argument(
        "--dlg_input_title",
        type=str,
        default="Ввод значения",
        help="Заголовок диалогового окна.",
    )
    parser.add_argument(
        "--html_content",
        type=str,
        default="<b>Введите значение:</b>",
        help="HTML-текст приглашения над полем ввода.",
    )
    parser.add_argument(
        "--html_output",
        type=str,
        choices=[OUTPUT_DIALOG, OUTPUT_CONSOLE_DIALOG],
        default=OUTPUT_DIALOG,
        help="Показывать HTML только в диалоге или также в консоли PySM.",
    )
    parser.add_argument(
        "--html_align",
        type=str,
        choices=["left", "center", "right"],
        default="left",
        help="Горизонтальное выравнивание HTML-контента.",
    )
    parser.add_argument(
        "--html_margin",
        type=int,
        default=0,
        help="Вертикальный внешний отступ HTML-блока в пикселях.",
    )
    parser.add_argument(
        "--html_padding",
        type=int,
        default=10,
        help="Внутренний отступ HTML-блока в пикселях.",
    )
    parser.add_argument(
        "--html_style",
        type=str,
        default="script_description",
        help="Имя HTML-стиля из активной темы PySM.",
    )
    parser.add_argument(
        "--dlg_input_dvalue",
        type=str,
        default="",
        help="Начальное значение, если переменная ещё отсутствует в контексте.",
    )
    parser.add_argument(
        "--dlg_input_value_type",
        type=str,
        default="auto",
        choices=["auto", "string", "int", "float", "bool", "json"],
        help="Тип преобразования введённого значения.",
    )
    parser.add_argument(
        "--dlg_input_valid_type",
        type=str,
        default="not_empty",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
        help="Встроенный или пользовательский режим валидации.",
    )
    parser.add_argument(
        "--dlg_input_custom_regexp",
        type=str,
        help="Пользовательское регулярное выражение.",
    )
    parser.add_argument(
        "--dlg_input_custom_regexp_desc",
        type=str,
        help="Пояснение формата для ошибки пользовательской валидации.",
    )
    parser.add_argument(
        "--dlg_input_text_ok",
        type=str,
        default="",
        help="Пользовательская подпись кнопки OK.",
    )
    parser.add_argument(
        "--dlg_input_text_cancel",
        type=str,
        default="",
        help="Пользовательская подпись кнопки Cancel.",
    )
    parser.add_argument(
        "--dlg_input_size_width",
        type=int,
        default=700,
        help="Начальная ширина диалогового окна в пикселях.",
    )
    parser.add_argument(
        "--dlg_input_size_height",
        type=int,
        default=500,
        help="Начальная высота диалогового окна в пикселях.",
    )

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()

    return parser.parse_args()


def validate_config(config: Namespace) -> None:
    """Validate fields needed before the interactive loop starts."""
    validate_html_layout(
        align=config.html_align,
        margin=config.html_margin,
        padding=config.html_padding,
    )
    if not config.dlg_input_var:
        raise ValueError("Необходимо указать dlg_input_var.")
    if not config.html_content or not config.html_content.strip():
        raise ValueError("Параметр html_content не может быть пустым.")


class HtmlInputDialog(QDialog):
    """Resizable input dialog with a shared themed HTML content view."""

    def __init__(
        self,
        *,
        title: str,
        html_document: str,
        initial_value: str,
        width: int,
        height: int,
        ok_text: str = "",
        cancel_text: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(width, height)

        layout = QVBoxLayout(self)
        browser = create_html_browser(
            parent=self,
            html_document=html_document,
        )
        layout.addWidget(browser)

        self.input_edit = QLineEdit(self)
        self.input_edit.setText(initial_value)
        layout.addWidget(self.input_edit)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        if ok_text and ok_button is not None:
            ok_button.setText(ok_text)
        if cancel_text and cancel_button is not None:
            cancel_button.setText(cancel_text)
        if ok_button is not None:
            ok_button.setDefault(True)

        self.input_edit.selectAll()
        self.input_edit.setFocus()

    def input_text(self) -> str:
        """Return the current unmodified line-edit value."""
        return self.input_edit.text()


def show_input_dialog(
    config: Namespace,
    html_document: str,
    initial_value: str,
) -> Optional[str]:
    """Show one input attempt and return None when the user cancels."""
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    dialog = HtmlInputDialog(
        title=config.dlg_input_title,
        html_document=html_document,
        initial_value=initial_value,
        width=config.dlg_input_size_width,
        height=config.dlg_input_size_height,
        ok_text=config.dlg_input_text_ok,
        cancel_text=config.dlg_input_text_cancel,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.input_text()


def main() -> int:
    """Run the HTML input, validation, and context-save workflow."""
    config = get_config()

    if not IS_MANAGED_RUN or pysm_context is None or InputProcessor is None:
        logger.error(format_error("Скрипт предназначен для запуска в среде PySM."))
        return 1

    try:
        validate_config(config)
        blocks = [config.html_content]
        html_document = build_html_document(
            blocks,
            theme_api=theme_api,
            style_name=config.html_style,
            align=config.html_align,
            margin=config.html_margin,
            padding=config.html_padding,
        )

        if config.html_output == OUTPUT_CONSOLE_DIALOG:
            log_html_to_console(
                blocks,
                pysm_context=pysm_context,
                theme_api=theme_api,
                style_name=config.html_style,
                align=config.html_align,
                margin=config.html_margin,
                padding=config.html_padding,
            )

        processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)
        initial_value = processor.get_initial_value(
            config.dlg_input_var,
            config.dlg_input_dvalue,
        )

        while True:
            result = show_input_dialog(config, html_document, initial_value)
            if result is None:
                return 1

            try:
                processor.process(
                    raw_value=result,
                    var_name=config.dlg_input_var,
                    value_type=config.dlg_input_value_type,
                )
                break
            except Exception as error:
                QMessageBox.warning(None, "Ошибка", str(error))
                initial_value = result

        safe_var_name = html.escape(config.dlg_input_var)
        safe_result = html.escape(result)
        logger.info(format_success(safe_var_name, safe_result))
        return 0
    except Exception as error:
        logger.error(format_error(str(error)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
