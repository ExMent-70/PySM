# pysm_lib/gui/widgets/instance_editor.py

from typing import Optional
from PySide6.QtWidgets import QGridLayout, QLineEdit, QPushButton, QSizePolicy, QDialog

from .._registry import BaseEditor, register_editor
from ..editor_context import EditorContext
from ..instance_selection_dialog import InstanceSelectionDialog

@register_editor("instance")
class InstanceEditorWidget(BaseEditor):
    def __init__(self, value: Optional[str], context: EditorContext, **kwargs):
        super().__init__(value, context, **kwargs)
        self.current_value = value
        
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
        """Открывает диалог выбора экземпляра."""
        dialog = InstanceSelectionDialog(
            title=self.context.locale_manager.get("dialogs.instance_editor.title"),
            script_entries=self.context.script_entries, # Передается новый тип List[Tuple[str, Model]]
            get_script_name_func=self.context.get_script_info_func,
            locale_manager=self.context.locale_manager,
            theme_manager=self.context.theme_manager,
            current_value=self.current_value,
            forbidden_instance_id=self.context.current_instance_id,
            parent=self,
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_id = dialog.get_selected_instance_id()
            # Проверяем явно на None, чтобы пустая строка "" считалась успешной очисткой
            if selected_id is not None:
                self.current_value = selected_id
                self._update_display_text()
                self.valueChanged.emit(self.current_value)

    def _update_display_text(self):
        """Обновляет текст в QLineEdit для отображения [Имя набора] Имя экземпляра или списка."""
        display_text = self.context.locale_manager.get("dialogs.instance_editor.none_selected")
        
        if self.current_value:
            ids =[x.strip() for x in self.current_value.split(",") if x.strip()]
            
            if len(ids) > 1:
                display_text = f"[Макрос] Выбрано скриптов: {len(ids)}"
            elif len(ids) == 1:
                display_text = ids[0] # Fallback
                for set_name, entry in self.context.script_entries:
                    if entry.instance_id == ids[0]:
                        script_info = self.context.get_script_info_func(entry.id)
                        script_name = script_info.name if script_info else "Unknown Script"
                        display_name = entry.name or script_name
                        display_text = f"[{set_name}] {display_name}"
                        break
                        
        self.line_edit.setText(display_text)
        self.line_edit.setToolTip(self.current_value or display_text)
        self.line_edit.setCursorPosition(0)