"""Shared HTML rendering helpers for PySM GUI dialog scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


HTML_ALIGNMENTS = ("left", "center", "right")
logger = logging.getLogger(__name__)


def validate_html_layout(*, align: str, margin: int, padding: int) -> None:
    """Validate HTML layout values shared by dialog scripts."""
    if align not in HTML_ALIGNMENTS:
        raise ValueError(f"Неизвестное выравнивание HTML: {align}")
    if margin < 0:
        raise ValueError("Параметр html_margin не может быть отрицательным.")
    if padding < 0:
        raise ValueError("Параметр html_padding не может быть отрицательным.")


def normalize_html_line_breaks(value: str) -> str:
    """Apply the same line-break conversion as ``pysm_context.log_html``."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "<br>")


def load_html_sources(
    html_content: Optional[str],
    html_file: Optional[str],
) -> tuple[list[str], Optional[Path]]:
    """Load non-empty inline and UTF-8 file sources in display order."""
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


def _theme_style_string(theme_api: Any, style_name: Optional[str]) -> str:
    """Return a CSS declaration string for a named PySM theme style."""
    selected_style = style_name or "script_description"
    style_dict = theme_api.get_parsed_style(
        selected_style,
        default="color: #adbac7;",
    )

    # QTextBrowser ignores CSS padding on a div. Padding is represented once
    # through table cellpadding so the PySM console and Qt render it equally.
    return " ".join(
        f"{key}: {value};"
        for key, value in style_dict.items()
        if key.strip().lower() != "padding"
    )


def build_html_document(
    blocks: Iterable[str],
    *,
    theme_api: Any,
    style_name: Optional[str],
    align: str,
    margin: int,
    padding: int,
) -> str:
    """Normalize and wrap HTML sources for identical console/dialog output."""
    style_string = _theme_style_string(theme_api, style_name)
    table_style = (
        f"margin-top: {margin}px; "
        "margin-right: 0px; "
        f"margin-bottom: {margin}px; "
        "margin-left: 0px;"
    )
    cell_style = f"text-align: {align}; {style_string}"

    return "".join(
        '<table width="100%" cellspacing="0" '
        f'cellpadding="{padding}" border="0" '
        f'style="{table_style}"><tr><td align="{align}" '
        f'style="{cell_style}">'
        f"{normalize_html_line_breaks(block)}"
        "</td></tr></table>"
        for block in blocks
    )


def log_html_to_console(
    blocks: Iterable[str],
    *,
    pysm_context: Any,
    theme_api: Any,
    style_name: Optional[str],
    align: str,
    margin: int,
    padding: int,
) -> None:
    """Render each source as an independent HTML block in the PySM console."""
    block_list = list(blocks)
    if not block_list:
        return

    logger.info("")
    for block in block_list:
        pysm_context.log_html(
            html_content=build_html_document(
                [block],
                theme_api=theme_api,
                style_name=style_name,
                align=align,
                margin=margin,
                padding=padding,
            ),
            align=align,
            margin=margin,
            padding=padding,
        )


def create_html_browser(
    *,
    parent: Optional[QWidget],
    html_document: str,
    base_dir: Optional[Path] = None,
) -> QTextBrowser:
    """Create the shared transparent, frameless rich-text browser."""
    browser = QTextBrowser(parent)
    browser.setFrameShape(QFrame.Shape.NoFrame)
    browser.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Ignored,
    )
    browser.setContentsMargins(0, 0, 0, 0)
    browser.setAutoFillBackground(False)
    browser.viewport().setContentsMargins(0, 0, 0, 0)
    browser.viewport().setAutoFillBackground(False)
    browser.document().setDocumentMargin(0)
    browser.setStyleSheet(
        "QTextBrowser { "
        "background-color: transparent; "
        "border: none; "
        "margin: 0px; "
        "padding: 0px; "
        "}"
    )
    browser.viewport().setStyleSheet("background-color: transparent;")
    browser.setOpenExternalLinks(True)

    if base_dir is not None:
        browser.document().setBaseUrl(QUrl.fromLocalFile(f"{base_dir}/"))

    browser.setHtml(html_document)
    return browser


class HtmlMessageDialog(QDialog):
    """Resizable HTML dialog with stable standard-button result names."""

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
        *,
        title: str,
        html_document: str,
        dialog_type: str,
        width: int,
        height: int,
        base_dir: Optional[Path] = None,
        button_texts: Optional[dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.choice = "unknown"
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(width, height)

        layout = QVBoxLayout(self)
        layout.addWidget(
            create_html_browser(
                parent=self,
                html_document=html_document,
                base_dir=base_dir,
            )
        )

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
        """Replace labels without changing standard-button semantics."""
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

    def _handle_button(self, button: QAbstractButton) -> None:
        standard_button = self.button_box.standardButton(button)
        self.choice = self.RESULT_NAMES.get(standard_button, "unknown")
        if self.choice in {"ok", "yes"}:
            self.accept()
        else:
            self.reject()


def show_html_message_dialog(
    *,
    theme_api: Any,
    title: str,
    html_document: str,
    dialog_type: str,
    width: int,
    height: int,
    base_dir: Optional[Path] = None,
    button_texts: Optional[dict[str, str]] = None,
) -> str:
    """Show a themed modal HTML dialog and return its stable result name."""
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    dialog = HtmlMessageDialog(
        title=title,
        html_document=html_document,
        dialog_type=dialog_type,
        width=width,
        height=height,
        base_dir=base_dir,
        button_texts=button_texts,
    )
    dialog.exec()
    return dialog.choice
