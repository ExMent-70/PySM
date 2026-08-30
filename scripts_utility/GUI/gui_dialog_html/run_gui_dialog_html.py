"""Show HTML in the PySM console, a modal dialog, both, or nowhere."""

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
    from pysm_lib.input_processor import InputProcessor
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = pysm_context.is_managed
except ImportError as import_error:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    InputProcessor = None
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
        QVBoxLayout,
    )
    from _common.html_dialog import (
        build_html_document,
        create_html_browser,
        log_html_to_console,
        validate_html_layout,
    )
except ImportError:
    print(
        "Ошибка: для работы этого скрипта требуется PySide6.",
        file=sys.stderr,
    )
    sys.exit(1)


OUTPUT_CONSOLE = "console"
OUTPUT_DIALOG = "dialog"
OUTPUT_CONSOLE_DIALOG = "console_dialog"
OUTPUT_NONE = "none"
DIALOG_OUTPUT_MODES = {OUTPUT_DIALOG, OUTPUT_CONSOLE_DIALOG}
CONSOLE_OUTPUT_MODES = {OUTPUT_CONSOLE, OUTPUT_CONSOLE_DIALOG}


def get_config() -> Namespace:
    """Resolve command-line and PySM collection-context parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Выводит HTML-текст или HTML-файл в консоль PySM, "
            "диалоговое окно, одновременно в оба места либо не выводит."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--html_content",
        type=str,
        default="",
        help="Строка с HTML-разметкой для вывода.",
    )
    parser.add_argument(
        "--html_file",
        type=str,
        help="Путь к UTF-8 HTML-файлу, содержимое которого нужно вывести.",
    )
    parser.add_argument(
        "--html_output",
        type=str,
        choices=[
            OUTPUT_CONSOLE,
            OUTPUT_DIALOG,
            OUTPUT_CONSOLE_DIALOG,
            OUTPUT_NONE,
        ],
        default=OUTPUT_CONSOLE_DIALOG,
        help="Куда выводить HTML-контент; none завершает скрипт без вывода.",
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
        "--dlg_msg_var",
        type=str,
        default="dlg_go_var",
        help=(
            "Переменная Контекста Коллекции для результата диалога. "
            "Поддерживается точечная нотация."
        ),
    )
    parser.add_argument(
        "--dlg_msg_type",
        type=str,
        choices=["ok", "yes_no", "yes_no_cancel"],
        default="yes_no",
        help="Набор кнопок диалогового окна.",
    )
    parser.add_argument(
        "--dlg_msg_text_ok",
        type=str,
        default="Продолжить",
        help="Пользовательская подпись утвердительной кнопки OK или Yes.",
    )
    parser.add_argument(
        "--dlg_msg_text_no",
        type=str,
        default="Остановить",
        help="Пользовательская подпись кнопки No.",
    )
    parser.add_argument(
        "--dlg_msg_text_cancel",
        type=str,
        default="Отменить",
        help="Пользовательская подпись кнопки Cancel.",
    )
    parser.add_argument(
        "--dlg_msg_title",
        type=str,
        default="Информационное сообщение",
        help="Заголовок диалогового окна.",
    )
    parser.add_argument(
        "--dlg_msg_size_width",
        type=int,
        default=700,
        help="Начальная ширина диалогового окна в пикселях.",
    )
    parser.add_argument(
        "--dlg_msg_size_height",
        type=int,
        default=500,
        help="Начальная высота диалогового окна в пикселях.",
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser, force_path_args=["html_file"])
        return resolver.resolve_all()

    return parser.parse_args()


def validate_config(config: Namespace) -> None:
    """Reject values that cannot be expressed safely by the UI."""
    validate_html_layout(
        align=config.html_align,
        margin=config.html_margin,
        padding=config.html_padding,
    )
    if config.html_output in DIALOG_OUTPUT_MODES and not config.dlg_msg_var:
        raise ValueError("Для вывода в диалог нужно указать dlg_msg_var.")


def load_html_sources(
    html_content: Optional[str],
    html_file: Optional[str],
) -> tuple[list[str], Optional[Path]]:
    """Load non-empty HTML sources in their display order."""
    blocks: list[str] = []
    base_dir: Optional[Path] = None

    if html_content and html_content.strip():
        blocks.append(html_content)

    if html_file:
        file_path = Path(html_file)
        if not file_path.is_file():
            raise ValueError(f"HTML-файл не найден: {file_path}")

        try:
            file_content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(
                f"Не удалось прочитать HTML-файл '{file_path}': {error}"
            ) from error

        if file_content.strip():
            blocks.append(file_content)
            base_dir = file_path.resolve().parent

    if not blocks:
        raise ValueError(
            "Необходимо указать непустой html_content или непустой html_file."
        )

    return blocks, base_dir


class HtmlMessageDialog(QDialog):
    """Resizable rich-text dialog with deterministic standard-button results."""

    BUTTONS = {
        "ok": QDialogButtonBox.StandardButton.Ok,
        "yes_no": (
            QDialogButtonBox.StandardButton.Yes
            | QDialogButtonBox.StandardButton.No
        ),
        "yes_no_cancel": (
            QDialogButtonBox.StandardButton.Yes
            | QDialogButtonBox.StandardButton.No
            | QDialogButtonBox.StandardButton.Cancel
        ),
    }
    RESULT_NAMES = {
        QDialogButtonBox.StandardButton.Ok: "ok",
        QDialogButtonBox.StandardButton.Yes: "yes",
        QDialogButtonBox.StandardButton.No: "no",
        QDialogButtonBox.StandardButton.Cancel: "cancel",
    }

    def __init__(
        self,
        title: str,
        html_document: str,
        dialog_type: str,
        width: int,
        height: int,
        base_dir: Optional[Path] = None,
        button_texts: Optional[dict[str, str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.choice = "unknown"
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(width, height)

        layout = QVBoxLayout(self)
        browser = create_html_browser(
            parent=self,
            html_document=html_document,
            base_dir=base_dir,
        )
        layout.addWidget(browser)

        self.button_box = QDialogButtonBox(self.BUTTONS[dialog_type], self)
        self.button_box.clicked.connect(self._handle_button)
        self._apply_custom_button_texts(button_texts or {})
        layout.addWidget(self.button_box)

        default_button = (
            QDialogButtonBox.StandardButton.Ok
            if dialog_type == "ok"
            else QDialogButtonBox.StandardButton.Yes
        )
        button = self.button_box.button(default_button)
        if button is not None:
            button.setDefault(True)
            button.setFocus()

    def _apply_custom_button_texts(self, button_texts: dict[str, str]) -> None:
        """Replace visible labels without changing standard-button semantics."""
        affirmative_text = button_texts.get("ok")
        if affirmative_text:
            for standard_button in (
                QDialogButtonBox.StandardButton.Ok,
                QDialogButtonBox.StandardButton.Yes,
            ):
                button = self.button_box.button(standard_button)
                if button is not None:
                    button.setText(affirmative_text)

        standard_buttons = {
            "no": QDialogButtonBox.StandardButton.No,
            "cancel": QDialogButtonBox.StandardButton.Cancel,
        }
        for result_name, standard_button in standard_buttons.items():
            custom_text = button_texts.get(result_name)
            button = self.button_box.button(standard_button)
            if custom_text and button is not None:
                button.setText(custom_text)

    def _handle_button(self, button) -> None:
        standard_button = self.button_box.standardButton(button)
        self.choice = self.RESULT_NAMES.get(standard_button, "unknown")
        if self.choice in {"ok", "yes"}:
            self.accept()
        else:
            self.reject()


def show_html_dialog(
    html_document: str,
    config: Namespace,
    base_dir: Optional[Path],
) -> str:
    """Display the modal dialog and return its stable string result."""
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    dialog = HtmlMessageDialog(
        title=config.dlg_msg_title,
        html_document=html_document,
        dialog_type=config.dlg_msg_type,
        width=config.dlg_msg_size_width,
        height=config.dlg_msg_size_height,
        base_dir=base_dir,
        button_texts={
            "ok": config.dlg_msg_text_ok,
            "no": config.dlg_msg_text_no,
            "cancel": config.dlg_msg_text_cancel,
        },
    )
    dialog.exec()
    return dialog.choice


def save_dialog_choice(config: Namespace, result: str) -> None:
    """Persist a dialog result in the Collection Context."""
    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)
    processor.process(
        raw_value=result,
        var_name=config.dlg_msg_var,
        value_type="string",
    )


def log_dialog_choice(config: Namespace, result: str) -> None:
    """Report the saved choice without interpreting user-controlled HTML."""
    safe_result = html.escape(result.upper())
    safe_var_name = html.escape(config.dlg_msg_var)
    message = "<b>Выбор пользователя сохранён:</b>"

    logger.info(message)
    logger.info(format_success(safe_var_name, safe_result))


def determine_exit_code(dialog_type: str, result: str) -> int:
    """Map a user choice to the PySM chain-control exit code."""
    if result in {"ok", "yes"}:
        return 0
    if result == "no" and dialog_type == "yes_no_cancel":
        return 0
    return 1


def main() -> int:
    """Run the configured output flow and return its process exit code."""
    config = get_config()

    if config.html_output == OUTPUT_NONE:
        return 0

    if not IS_MANAGED_RUN or pysm_context is None:
        logger.error(format_error("Скрипт предназначен для запуска в среде PySM."))
        return 1

    try:
        validate_config(config)
        blocks, base_dir = load_html_sources(
            config.html_content,
            config.html_file,
        )
        html_document = build_html_document(
            blocks,
            theme_api=theme_api,
            style_name=config.html_style,
            align=config.html_align,
            margin=config.html_margin,
            padding=config.html_padding,
        )

        if config.html_output in CONSOLE_OUTPUT_MODES:
            log_html_to_console(
                blocks,
                pysm_context=pysm_context,
                theme_api=theme_api,
                style_name=config.html_style,
                align=config.html_align,
                margin=config.html_margin,
                padding=config.html_padding,
            )

        if config.html_output == OUTPUT_CONSOLE:
            return 0

        result = show_html_dialog(html_document, config, base_dir)
        save_dialog_choice(config, result)

        if config.html_output in CONSOLE_OUTPUT_MODES:
            log_dialog_choice(config, result)

        return determine_exit_code(config.dlg_msg_type, result)
    except Exception as error:
        logger.error(format_error(str(error)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
