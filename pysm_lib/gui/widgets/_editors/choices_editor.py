# pysm_lib/gui/widgets/_editors/choices_editor.py

from typing import Optional, List

from PySide6.QtCore import Signal, QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QGridLayout, QComboBox, QPushButton, QDialog, QVBoxLayout,
    QLabel, QPlainTextEdit, QDialogButtonBox, QSizePolicy, QSplitter, QWidget
)

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext
from .template_widget import TemplateManagerWidget
from ....app_constants import COLLECTION_DEFAULT_FOLDER


@register_editor("choice")
class ChoicesEditorWidget(BaseEditor):
    choicesChanged = Signal(list)

    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        self._choices = self.options.get("choices") or[]
        
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
        dialog.setMinimumSize(850, 500)
        
        main_layout = QVBoxLayout(dialog)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        editor = QPlainTextEdit()
        
        template_path = COLLECTION_DEFAULT_FOLDER / "_text_template" / "choice_templates.json"
        template_manager = TemplateManagerWidget(
            templates_file_path=template_path,
            get_current_text_func=lambda: editor.toPlainText(),
            parent=dialog
        )
        splitter.addWidget(template_manager)
        
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(self.context.locale_manager.get("dialogs.context_editor.edit_choices_label"))
        right_layout.addWidget(label)
        
        editor.setPlainText("\n".join(self._choices))
        right_layout.addWidget(editor, 1)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        right_layout.addWidget(button_box)
        
        splitter.addWidget(right_container)
        splitter.setSizes([250, 600])
        
        # --- Обработчик применения шаблона ---
        def on_template_applied(text: str, replace: bool):
            if replace:
                editor.setPlainText(text)
            else:
                current = editor.toPlainText()
                editor.setPlainText(f"{current}\n{text}" if current else text)
                
        template_manager.template_applied.connect(on_template_applied)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec():
            text = editor.toPlainText()
            new_choices =[line.strip() for line in text.splitlines() if line.strip()]
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