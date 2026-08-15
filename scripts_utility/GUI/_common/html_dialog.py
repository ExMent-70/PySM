"""Shared HTML rendering helpers for PySM GUI dialog scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QFrame, QSizePolicy, QTextBrowser, QWidget


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
