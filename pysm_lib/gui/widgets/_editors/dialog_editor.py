# pysm_lib/gui/widgets/_editors/dialog_editor.py

import json
import pathlib
from typing import Any, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QLineEdit, QPushButton, QSizePolicy, QDialog, QVBoxLayout,
    QPlainTextEdit, QDialogButtonBox, QMessageBox, QFileDialog, 
    QInputDialog, QSplitter, QWidget
)

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext
from .template_widget import TemplateManagerWidget
from ....app_constants import COLLECTION_DEFAULT_FOLDER


class DialogEditorWidget(BaseEditor):
    def __init__(self, value: Any, context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        self.var_type = self.options.get("var_type", "string")
        
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        
        self.line_edit = QLineEdit(str(value) if value is not None else "")
        self.line_edit.setToolTip(self.line_edit.text())
        self.line_edit.setCursorPosition(0)
        self.line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        if self.var_type in ("string_multiline", "json"):
            self.line_edit.setReadOnly(True)
        else:
            self.line_edit.editingFinished.connect(self._on_editing_finished)

        button = QPushButton("...")
        button.setFixedWidth(30)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        layout.addWidget(self.line_edit, 0, 0)
        layout.addWidget(button, 0, 1)
        layout.setColumnStretch(0, 1)
        
        button.clicked.connect(self.on_button_click)

    def _on_editing_finished(self):
        new_value = self.line_edit.text()
        self.value = new_value
        self.line_edit.setToolTip(new_value)
        self.valueChanged.emit(new_value)

    def on_button_click(self):
        new_value, changed = self._handle_button_dialogs()
        if changed:
            self.value = new_value 
            display_text = str(new_value) if new_value is not None else ""
            self.line_edit.setText(display_text)
            self.line_edit.setToolTip(display_text)
            self.line_edit.setCursorPosition(0)
            self.valueChanged.emit(new_value)

    def _handle_button_dialogs(self) -> Tuple[Any, bool]:
        current_value = self.value
        locale = self.context.locale_manager
        
        if self.var_type == "file_path":
            start_path = str(current_value) if current_value else str(pathlib.Path.home())
            new_path, _ = QFileDialog.getOpenFileName(
                self, locale.get("dialogs.script_properties.select_file_title"), start_path
            )
            return (new_path, True) if new_path else (current_value, False)
            
        if self.var_type == "dir_path":
            start_path = str(current_value) if current_value else str(pathlib.Path.home())
            new_path = QFileDialog.getExistingDirectory(
                self, locale.get("dialogs.script_properties.select_dir_title"), start_path
            )
            return (new_path, True) if new_path else (current_value, False)
            
        if self.var_type == "password":
            new_pass, ok = QInputDialog.getText(
                self, "Enter Password", "Password:", QLineEdit.EchoMode.Password, current_value
            )
            return (new_pass, ok)
            
        if self.var_type in ("string_multiline", "json"):
            is_json = self.var_type == "json"
            title = locale.get(
                "dialogs.context_editor.json_editor_title" if is_json 
                else "dialogs.context_editor.multiline_editor_title"
            )
            
            dialog = QDialog(self)
            dialog.setObjectName("JsonEditorDialog" if is_json else "MultilineTextDialog")
            dialog.setWindowTitle(title)
            dialog.setMinimumSize(850, 500) 
            
            main_layout = QVBoxLayout(dialog)
            
            splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(splitter, 1)
            
            template_path = COLLECTION_DEFAULT_FOLDER / "_text_template" / "dialog_templates.json"
            
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
            
            text_to_edit = ""
            if is_json:
                try:
                    text_to_edit = json.dumps(current_value, indent=2, ensure_ascii=False) if current_value is not None else ""
                except TypeError:
                    text_to_edit = str(current_value or "")
            else:
                text_to_edit = str(current_value or "")
                
            editor.setPlainText(text_to_edit)
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
                if not text.strip() and is_json:
                    return None, True
                    
                if is_json:
                    try:
                        return json.loads(text), True
                    except json.JSONDecodeError:
                        QMessageBox.warning(
                            self, locale.get("general.error_title"), 
                            locale.get("dialogs.context_editor.json_invalid_error")
                        )
                        return current_value, False
                else:
                    return text, True
                    
        return current_value, False


register_editor("file_path")(DialogEditorWidget)
register_editor("dir_path")(DialogEditorWidget)
register_editor("password")(DialogEditorWidget)
register_editor("string_multiline")(DialogEditorWidget)
register_editor("json")(DialogEditorWidget)