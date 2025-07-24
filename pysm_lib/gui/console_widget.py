# pysm_lib/gui/console_widget.py

import logging
import re
from typing import Optional, Any, Dict
from PySide6.QtCore import Slot, QUrl
from PySide6.QtGui import QPalette, QColor, QDesktopServices, QMouseEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QTextBrowser,
    QProgressBar,
)
from ..config_manager import ConfigManager
from ..locale_manager import LocaleManager
from .gui_utils import resolve_themed_text


class ClickableConsoleBrowser(QTextBrowser):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)

    def mouseReleaseEvent(self, event: QMouseEvent):
        anchor = self.anchorAt(event.pos())
        if anchor:
            QDesktopServices.openUrl(QUrl(anchor))
        else:
            super().mouseReleaseEvent(event)


class ConsoleWidget(QWidget):
    def __init__(self, config_manager: ConfigManager, locale_manager: LocaleManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"PyScriptManager.{self.__class__.__name__}")
        self.config_manager = config_manager
        self.locale_manager = locale_manager
        self._init_ui()
        self.apply_theme()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        console_groupbox = QGroupBox(
            self.locale_manager.get("console_widget.group_title")
        )
        console_layout = QVBoxLayout(console_groupbox)

        self.text_console_output = ClickableConsoleBrowser()

        self.text_console_output.setAutoFillBackground(True)
        font = self.text_console_output.font()
        font.setPointSize(10)
        self.text_console_output.setFont(font)
        console_layout.addWidget(self.text_console_output)
        main_layout.addWidget(console_groupbox, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat(
            self.locale_manager.get("console_widget.progress_bar.default_format")
        )
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        main_layout.addWidget(self.progress_bar)

    def apply_theme(self):
        """Применяет стили из активной темы к виджету."""
        active_theme = self.config_manager.get_active_theme()
        styles = active_theme.get_styles_as_dict()
        
        bg_style_str = styles.get("console_background", "background-color: #ffffff;")
        match = re.search(r"background-color:\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", bg_style_str)
        bg_color_hex = "#FFFFFF"
        if match:
            bg_color_hex = match.group(1).strip()
            
        palette = self.text_console_output.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(bg_color_hex))
        self.text_console_output.setPalette(palette)

    def _wrap_padded_html_in_table(self, html: str) -> str:
        """
        Находит теги <div> с padding/margin, заменяет их на <td>
        и оборачивает в табличную структуру.
        """
        pattern = re.compile(
            r"<div([^>]*?style\s*=\s*['\"].*?(padding|margin):.*?)>(.*?)</div>",
            re.IGNORECASE | re.DOTALL
        )

        def replacer(match):
            attributes = match.group(1)
            content = match.group(3)
            return f'<table width="100%" cellspacing="0" cellpadding="0" border="0"><tr><td{attributes}>{content}</td></tr></table>'

        return pattern.sub(replacer, html)

    @Slot()
    def clear_console(self):
        self.text_console_output.clear()

    @Slot(str, str)
    def append_to_console(self, msg_type: str, text: str):
        processed_text = resolve_themed_text(text, self.config_manager)

        if msg_type.lower() == "html_block":
            cursor = self.text_console_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)

            final_html = self._wrap_padded_html_in_table(processed_text)
            
            cursor.insertHtml(final_html)
            return

        if msg_type.upper() == "EMPTY_LINE":
            self.text_console_output.append("")
            return
        
        active_theme = self.config_manager.get_active_theme()
        styles = active_theme.get_styles_as_dict()
        style = styles.get(msg_type.lower(), "color: black;")
        escaped_text = processed_text.replace("&", "&").replace("<", "<").replace(">", ">")
        html_content = escaped_text.replace("\n", "<br>")
        is_block_style = "_block" in msg_type.lower() or "header" in msg_type.lower()

        if is_block_style:
            html_line = f"""
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr><td style="padding: 2px 5px; {style}">{html_content}</td></tr>
            </table>
            """
            cursor = self.text_console_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertHtml(html_line)
        else:
            html_line = f'<span style="{style}">{html_content}</span>'
            self.text_console_output.append(html_line)

        self.text_console_output.ensureCursorVisible()

    @Slot(str, int, int, object)
    def update_progress_bar(
        self, instance_id: str, current: int, total: int, text_obj: Optional[Any]
    ):
        if total > 0 and current >= 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            progress_text = text_obj if isinstance(text_obj, str) else ""
            self.progress_bar.setFormat(
                self.locale_manager.get(
                    "console_widget.progress_bar.active_format", text=progress_text
                )
            )
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(
                self.locale_manager.get("console_widget.progress_bar.default_format")
            )