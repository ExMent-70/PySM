# pysm_lib/gui/widgets/list_editor.py

from typing import Optional, List
from PySide6.QtWidgets import (
    QGridLayout, QLineEdit, QPushButton, QDialog, QVBoxLayout, QLabel,
    QPlainTextEdit, QDialogButtonBox, QSizePolicy
)

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("list")
class ListEditorWidget(BaseEditor):
    def __init__(self, value: Optional[List[str]], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        self.current_value = value or []
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(True)
        self.line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._update_display_text()
        button = QPushButton("...")
        button.setFixedWidth(30)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.line_edit, 0, 0)
        layout.addWidget(button, 0, 1)
        layout.setColumnStretch(0, 1)
        button.clicked.connect(self.on_button_click)
    def on_button_click(self):
        dialog = QDialog(self)
        dialog.setObjectName("MultilineTextDialog")
        dialog.setWindowTitle(self.context.locale_manager.get("dialogs.context_editor.list_editor_title"))
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout(dialog)
        label = QLabel(self.context.locale_manager.get("dialogs.context_editor.list_editor_label"))
        layout.addWidget(label)
        editor = QPlainTextEdit()
        editor.setPlainText("\n".join(self.current_value))
        layout.addWidget(editor)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        if dialog.exec():
            text = editor.toPlainText()
            new_list = [line.strip() for line in text.splitlines() if line.strip()]
            self.current_value = new_list
            self._update_display_text()
            self.valueChanged.emit(self.current_value)
    def _update_display_text(self):
        display_text = ", ".join(self.current_value)
        self.line_edit.setText(display_text)
        self.line_edit.setToolTip(display_text)
        self.line_edit.setCursorPosition(0)