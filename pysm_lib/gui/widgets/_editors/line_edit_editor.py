# pysm_lib/gui/widgets/line_edit_editor.py

from typing import Optional
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QSizePolicy
from PySide6.QtGui import QValidator

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("string")
class LineEditEditor(BaseEditor):
    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        validator: Optional[QValidator] = self.options.get("validator")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_edit = QLineEdit(str(value) if value is not None else "")
        self.line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        if validator:
            self.line_edit.setValidator(validator)
            
        layout.addWidget(self.line_edit)
        
        self.line_edit.editingFinished.connect(lambda: self.valueChanged.emit(self.line_edit.text()))