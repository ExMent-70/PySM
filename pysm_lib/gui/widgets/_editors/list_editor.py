# pysm_lib/gui/widgets/_editors/list_editor.py

from typing import Optional, List, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QLineEdit, QPushButton, QDialog, QVBoxLayout, QLabel,
    QPlainTextEdit, QDialogButtonBox, QSizePolicy, QSplitter, QWidget
)

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext
from .template_widget import TemplateManagerWidget
from ....app_constants import COLLECTION_DEFAULT_FOLDER


@register_editor("list")
class ListEditorWidget(BaseEditor):
    def __init__(self, value: Any, context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        
        # --- ЗАЩИТА ТИПОВ ---
        # Если из БД пришла старая одиночная строка (из-за ошибки в паспорте) 
        # или многострочный текст, корректно преобразуем в список строк
        if isinstance(value, list):
            self.current_value =[str(v) for v in value]
        elif isinstance(value, str):
            self.current_value =[line.strip() for line in value.splitlines() if line.strip()]
        else:
            self.current_value =[]
            
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
        dialog.setMinimumSize(850, 500)
        
        main_layout = QVBoxLayout(dialog)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        template_path = COLLECTION_DEFAULT_FOLDER / "_text_template" / "list_templates.json"
        
        editor = QPlainTextEdit()
        
        template_manager = TemplateManagerWidget(
            templates_file_path=template_path,
            get_current_text_func=lambda: editor.toPlainText(),
            parent=dialog
        )
        splitter.addWidget(template_manager)
        
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(self.context.locale_manager.get("dialogs.context_editor.list_editor_label"))
        right_layout.addWidget(label)
        
        editor.setPlainText("\n".join(self.current_value))
        right_layout.addWidget(editor, 1)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        right_layout.addWidget(button_box)
        
        splitter.addWidget(right_container)
        splitter.setSizes([300, 550])
        
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
            new_list =[line.strip() for line in text.splitlines() if line.strip()]
            self.current_value = new_list
            self._update_display_text()
            self.valueChanged.emit(self.current_value)

    def _update_display_text(self):
        display_text = ", ".join(self.current_value)
        self.line_edit.setText(display_text)
        self.line_edit.setToolTip(display_text)
        self.line_edit.setCursorPosition(0)