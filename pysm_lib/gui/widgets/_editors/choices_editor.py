# pysm_lib/gui/widgets/choices_editor.py

from typing import Optional, List
from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import (
    QGridLayout, QComboBox, QPushButton, QDialog, QVBoxLayout,
    QLabel, QPlainTextEdit, QDialogButtonBox, QSizePolicy
)

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("choice")
class ChoicesEditorWidget(BaseEditor):
    choicesChanged = Signal(list)

    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        self._choices = self.options.get("choices") or []
        
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.combo = QComboBox()
        self.combo.setProperty("isEditor", True)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.combo.addItems(self._choices)
        if value is not None:
            self.combo.setCurrentText(str(value))
            
        self.edit_btn = QPushButton("...")
        self.edit_btn.setFixedWidth(30)
        self.edit_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.edit_btn.setToolTip(self.context.locale_manager.get("dialogs.context_editor.edit_choices_tooltip"))
        
        layout.addWidget(self.combo, 0, 0)
        layout.addWidget(self.edit_btn, 0, 1)
        layout.setColumnStretch(0, 1)
        
        self.combo.currentTextChanged.connect(self.valueChanged.emit)
        self.edit_btn.clicked.connect(self._edit_choices)

    def _edit_choices(self):
        dialog = QDialog(self)
        dialog.setObjectName("MultilineTextDialog")
        dialog.setWindowTitle(self.context.locale_manager.get("dialogs.context_editor.edit_choices_title"))
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout(dialog)
        label = QLabel(self.context.locale_manager.get("dialogs.context_editor.edit_choices_label"))
        layout.addWidget(label)
        editor = QPlainTextEdit()
        editor.setPlainText("\n".join(self._choices))
        layout.addWidget(editor)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        if dialog.exec():
            text = editor.toPlainText()
            new_choices = [line.strip() for line in text.splitlines() if line.strip()]
            self._choices = new_choices
            self.choicesChanged.emit(self._choices)
            current_text = self.combo.currentText()
            with QSignalBlocker(self.combo):
                self.combo.clear()
                self.combo.addItems(self._choices)
                if current_text in self._choices:
                    self.combo.setCurrentText(current_text)
                elif self._choices:
                    self.combo.setCurrentIndex(0)