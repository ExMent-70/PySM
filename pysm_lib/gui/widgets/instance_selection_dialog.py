# pysm_lib/gui/widgets/instance_selection_dialog.py

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Slot 
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QVBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QWidget,
    QStyle,
)

from ...models import ScriptSetEntryModel, ScriptInfoModel
from ...locale_manager import LocaleManager
from ...theme_manager import ThemeManager
from ..tooltip_generator import generate_instance_tooltip_html


class InstanceSelectionDialog(QDialog):
    """
    Диалог для выбора экземпляра скрипта из предоставленного списка.
    """

    def __init__(
        self,
        title: str,
        script_entries: List[ScriptSetEntryModel],
        get_script_name_func: callable,
        locale_manager: LocaleManager,
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        theme_manager: ThemeManager,
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---        
        current_value: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        self.theme_manager = theme_manager
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---        
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        self.setObjectName("InstanceSelectionDialog")

        main_layout = QVBoxLayout(self)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(
            self.locale_manager.get("dialogs.context_editor.search_placeholder")
        )
        main_layout.addWidget(self.search_bar)

        self.list_widget = QListWidget()
        main_layout.addWidget(self.list_widget, 1)

        # Заполняем список
        for entry in script_entries:
            script_name = "Unknown Script"
            script_info: Optional[ScriptInfoModel] = None
            try:
                script_info = get_script_name_func(entry.id)
                if script_info:
                    script_name = script_info.name
            except Exception:
                pass
            
            display_name = entry.name or script_name
            
            item_text = f"{display_name})"
            #item_text = f"{display_name}  ({entry.instance_id})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, entry.instance_id)
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon))
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            # Используем tooltip_generator для создания подсказки
            tooltip_html = generate_instance_tooltip_html(
                script_info=script_info,
                instance_entry=entry,
                locale_manager=self.locale_manager,
                theme_manager=self.theme_manager,
            )
            item.setToolTip(tooltip_html)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            
            self.list_widget.addItem(item)
            
            if entry.instance_id == current_value:
                self.list_widget.setCurrentItem(item)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        self.search_bar.textChanged.connect(self._filter_list)
        self.list_widget.itemDoubleClicked.connect(self.accept)


    @Slot(str)
    def _filter_list(self, text: str):
        """Фильтрует список по введенному тексту."""
        search_text = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(search_text not in item.text().lower())

    def get_selected_instance_id(self) -> Optional[str]:
        """
        Возвращает ID выбранного экземпляра, если диалог был подтвержден.
        """
        if self.result() == QDialog.DialogCode.Accepted:
            selected_item = self.list_widget.currentItem()
            if selected_item:
                return selected_item.data(Qt.ItemDataRole.UserRole)
        return None