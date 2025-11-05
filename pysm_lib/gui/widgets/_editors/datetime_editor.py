# pysm_lib/gui/widgets/datetime_editor.py

from typing import Optional
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QDateTimeEdit

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext

@register_editor("datetime")
class DateTimeEditor(BaseEditor):
    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = QDateTimeEdit(self)
        self.editor.setCalendarPopup(True)
        self.editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        if value:
            try: self.editor.setDateTime(datetime.fromisoformat(value))
            except (ValueError, TypeError): self.editor.setDateTime(datetime.now())
        else:
            self.editor.setDateTime(datetime.now())
        layout.addWidget(self.editor)
        self.editor.dateTimeChanged.connect(lambda dt: self.valueChanged.emit(dt.toString(Qt.DateFormat.ISODateWithMs)))