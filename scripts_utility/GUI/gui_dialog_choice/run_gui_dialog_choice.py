#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select and save one value through a themed HTML dialog."""

from __future__ import annotations

import argparse
import html
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Optional


GUI_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(GUI_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

IS_MANAGED_RUN = False

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.input_processor import InputProcessor
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = pysm_context.is_managed
except ImportError as import_error:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    InputProcessor = None

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"

    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

    logger.debug("PySM API недоступен: %s", import_error)

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QVBoxLayout,
    )
    from _common.html_dialog import (
        build_html_document,
        create_html_browser,
        log_html_to_console,
        validate_html_layout,
    )
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)


OUTPUT_DIALOG = "dialog"
OUTPUT_CONSOLE_DIALOG = "console_dialog"
DEFAULT_CHOICES = [
    "PORTRAIT",
    "SCHOOL",
    "SEPTEMBER1",
    "STREET",
    "STUDIO",
]


def get_config() -> Namespace:
    """Resolve command-line and PySM Collection Context parameters."""
    parser = argparse.ArgumentParser(
        description="Показывает HTML-диалог для выбора значения из списка.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dlg_choice_var",
        type=str,
        default="dlg_choice_user_var",
        help="Переменная Контекста Коллекции для сохранения результата.",
    )
    parser.add_argument(
        "--dlg_choice_title",
        type=str,
        default="Выбор опции",
        help="Заголовок диалогового окна.",
    )
    parser.add_argument(
        "--html_content",
        type=str,
        default="<b>Выберите один из вариантов:</b>",
        help="HTML-текст приглашения над выпадающим списком.",
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
        "--dlg_choice_list",
        type=str,
        nargs="+",
        default=DEFAULT_CHOICES,
        help="Список доступных вариантов.",
    )
    parser.add_argument(
        "--dlg_choice_dvalue",
        type=str,
        default="",
        help="Начальный вариант, если переменная отсутствует в контексте.",
    )
    parser.add_argument(
        "--dlg_choice_text_ok",
        type=str,
        default="ОК",
        help="Пользовательская подпись кнопки OK.",
    )
    parser.add_argument(
        "--dlg_choice_text_cancel",
        type=str,
        default="Отмена",
        help="Пользовательская подпись кнопки Cancel.",
    )
    parser.add_argument(
        "--dlg_choice_size_width",
        type=int,
        default=320,
        help="Начальная ширина диалогового окна в пикселях.",
    )
    parser.add_argument(
        "--dlg_choice_size_height",
        type=int,
        default=150,
        help="Начальная высота диалогового окна в пикселях.",
    )

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()

    return parser.parse_args()


def normalize_choices(raw_choices: Optional[Iterable[str] | str]) -> list[str]:
    """Return non-empty choice labels while preserving their order."""
    if isinstance(raw_choices, str):
        candidates: Iterable[str] = raw_choices.splitlines()
    else:
        candidates = raw_choices or []

    choices: list[str] = []
    for item in candidates:
        if item is None:
            continue
        normalized = str(item).strip()
        if normalized:
            choices.append(normalized)
    return choices


def validate_config(config: Namespace, choices: list[str]) -> None:
    """Validate fields needed before opening the dialog."""
    validate_html_layout(
        align=config.html_align,
        margin=config.html_margin,
        padding=config.html_padding,
    )
    if not config.dlg_choice_var:
        raise ValueError("Необходимо указать dlg_choice_var.")
    if not config.html_content or not config.html_content.strip():
        raise ValueError("Параметр html_content не может быть пустым.")
    if not choices:
        raise ValueError("Список для выбора dlg_choice_list пуст.")


class HtmlChoiceDialog(QDialog):
    """Resizable choice dialog with a shared themed HTML content view."""

    def __init__(
        self,
        *,
        title: str,
        html_document: str,
        choices: list[str],
        current_index: int,
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

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(choices)
        self.combo_box.setCurrentIndex(current_index)
        layout.addWidget(self.combo_box)

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

        self.combo_box.setFocus()

    def selected_item(self) -> str:
        """Return the currently selected choice label."""
        return self.combo_box.currentText()


def show_choice_dialog(
    config: Namespace,
    html_document: str,
    choices: list[str],
    current_index: int,
) -> Optional[str]:
    """Show the choice dialog and return None when the user cancels."""
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    dialog = HtmlChoiceDialog(
        title=config.dlg_choice_title,
        html_document=html_document,
        choices=choices,
        current_index=current_index,
        width=config.dlg_choice_size_width,
        height=config.dlg_choice_size_height,
        ok_text=config.dlg_choice_text_ok,
        cancel_text=config.dlg_choice_text_cancel,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.selected_item()


def main() -> int:
    """Run the HTML choice and context-save workflow."""
    config = get_config()

    if not IS_MANAGED_RUN or pysm_context is None or InputProcessor is None:
        logger.error(format_error("Скрипт предназначен для запуска в среде PySM."))
        return 1

    try:
        choices = normalize_choices(config.dlg_choice_list)
        validate_config(config, choices)

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
            config.dlg_choice_var,
            config.dlg_choice_dvalue or "",
        )
        current_index = (
            choices.index(initial_value) if initial_value in choices else 0
        )

        result = show_choice_dialog(
            config,
            html_document,
            choices,
            current_index,
        )
        if result is None:
            logger.error(format_error("Операция отменена пользователем."))
            return 1

        processor.process(
            raw_value=result,
            var_name=config.dlg_choice_var,
            value_type="string",
        )

        safe_title = html.escape(config.dlg_choice_title)
        safe_var_name = html.escape(config.dlg_choice_var)
        safe_result = html.escape(result)
        logger.info(f"<b>{safe_title}</b>")
        logger.info(format_success(safe_var_name, safe_result))
        return 0
    except Exception as error:
        logger.error(format_error(html.escape(str(error))))
        return 1


if __name__ == "__main__":
    sys.exit(main())
