# pysm_lib/gui/widgets/checkbox_editor.py

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QCheckBox

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("bool")
class CheckBoxEditor(BaseEditor):
    def __init__(self, value: Optional[bool], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        if value is not None:
            self.checkbox.setChecked(bool(value))
        layout.addWidget(self.checkbox)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox.toggled.connect(self.valueChanged.emit)