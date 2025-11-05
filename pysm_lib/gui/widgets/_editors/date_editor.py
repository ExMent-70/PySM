# pysm_lib/gui/widgets/date_editor.py

from typing import Optional
from datetime import date
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QDateEdit

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("date")
class DateEditor(BaseEditor):
    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = QDateEdit(self)
        self.editor.setCalendarPopup(True)
        self.editor.setDisplayFormat("yyyy-MM-dd")
        if value:
            try: self.editor.setDate(date.fromisoformat(value))
            except (ValueError, TypeError): self.editor.setDate(date.today())
        else:
            self.editor.setDate(date.today())
        layout.addWidget(self.editor)
        self.editor.dateChanged.connect(lambda d: self.valueChanged.emit(d.toString(Qt.DateFormat.ISODate)))