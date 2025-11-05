# pysm_lib/gui/widgets/context_editor_widget.py

import pathlib
import re
from typing import Optional, Dict, List

from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QInputDialog,
    QAbstractItemView,
    QLineEdit,
)

from ...models import (
    ContextVariableModel,
    ScriptArgMetaDetailModel,
    ScriptSetEntryModel,
)
from ...locale_manager import LocaleManager
from ...theme_manager import ThemeManager
from ...app_enums import EditMode
from .parameter_editor_widget import ParameterEditorWidget
from .arg_selection_dialog import ArgSelectionDialog


class ContextEditorWidget(QWidget):
    data_changed = Signal()

    def __init__(
        self,
        collection_path: Optional[pathlib.Path],
        known_args_meta: Dict[str, ScriptArgMetaDetailModel],
        locale_manager: LocaleManager,
        theme_manager: ThemeManager,
        script_entries: List[ScriptSetEntryModel],
        get_script_name_func: callable,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.theme_manager = theme_manager
        self.known_args_meta = known_args_meta
        self.script_entries = script_entries
        self.get_script_name_func = get_script_name_func
        self._data: Dict[str, ContextVariableModel] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(
            self.locale_manager.get("dialogs.context_editor.search_placeholder")
        )
        layout.addWidget(self.search_bar)

        self.editor = ParameterEditorWidget(
            mode=EditMode.CONTEXT_VARS,
            locale_manager=self.locale_manager,
            theme_manager=self.theme_manager,
            script_entries=self.script_entries,
            get_script_name_func=self.get_script_name_func,
        )
        
        layout.addWidget(self.editor, 1)
        self.editor.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        buttons_layout = QHBoxLayout()
        self.btn_create_var = QPushButton(self.locale_manager.get("dialogs.context_editor.create_variable"))
        self.btn_add_from_script = QPushButton(self.locale_manager.get("dialogs.context_editor.add_variable"))
        self.remove_btn = QPushButton(self.locale_manager.get("dialogs.context_editor.remove_variable"))
        self.remove_btn.setEnabled(False)
        buttons_layout.addWidget(self.btn_create_var)
        buttons_layout.addWidget(self.btn_add_from_script)
        buttons_layout.addWidget(self.remove_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

    def _connect_signals(self):
        self.btn_create_var.clicked.connect(self._on_create_custom_variable)
        self.btn_add_from_script.clicked.connect(self._on_add_from_script)
        self.remove_btn.clicked.connect(self._on_remove_variable)
        self.editor.table.itemSelectionChanged.connect(self._update_button_states)
        self.editor.data_changed.connect(self.data_changed.emit)
        self.search_bar.textChanged.connect(self._filter_table)
        self.editor.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

    @Slot(str)
    def _filter_table(self, text: str):
        search_text = text.lower()
        for row in range(self.editor.table.rowCount()):
            item = self.editor.table.item(row, 0)
            if item:
                self.editor.table.setRowHidden(row, search_text not in item.text().lower())

    @Slot(int, int)
    def _on_cell_double_clicked(self, row, column):
        if column == 0:
            name_item = self.editor.table.item(row, 0)
            if not name_item: return
            var_name = name_item.text()
            if var_name in self.known_args_meta:
                dialog = ArgSelectionDialog(
                    self.locale_manager.get("dialogs.context_editor.view_title"),
                    self.locale_manager.get("dialogs.context_editor.add_label"),
                    {var_name: self.known_args_meta[var_name]},
                    self.locale_manager,
                    self,
                    read_only=True,
                )
                dialog.exec()
        else:
            self.editor.on_cell_double_clicked(row, column)

    def set_data(self, data: Dict[str, ContextVariableModel]):
        self._data = {k: v.model_copy(deep=True) for k, v in data.items()}
        self.editor.set_data(self._data)

    def get_data(self) -> Dict[str, ContextVariableModel]:
        return self.editor.get_updated_context_vars()

    @Slot()
    def _on_create_custom_variable(self):
        name, ok = QInputDialog.getText(self, self.locale_manager.get("dialogs.context_editor.create_title"), self.locale_manager.get("dialogs.context_editor.add_label"))
        if not (ok and name.strip()): return
        name = name.strip()
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            QMessageBox.warning(self, self.locale_manager.get("general.error_title"), self.locale_manager.get("dialogs.context_editor.add_invalid_name_error", name=name))
            return
        if name in self._data:
            QMessageBox.warning(self, self.locale_manager.get("general.error_title"), self.locale_manager.get("dialogs.context_editor.add_duplicate_error", name=name))
            return
        self._data[name] = ContextVariableModel(type="string")
        self.set_data(self._data)
        self.data_changed.emit()

    @Slot()
    def _on_add_from_script(self):
        dialog = ArgSelectionDialog(
            self.locale_manager.get("dialogs.context_editor.add_title"),
            self.locale_manager.get("dialogs.context_editor.add_label"),
            self.known_args_meta, self.locale_manager, self
        )
        selected_meta, selected_name = dialog.get_selected_arg_meta()
        if not selected_meta: return
        name = selected_name
        if name in self._data:
            QMessageBox.warning(self, self.locale_manager.get("general.error_title"), self.locale_manager.get("dialogs.context_editor.add_duplicate_error", name=name))
            return
        new_var = ContextVariableModel(
            type=selected_meta.type,
            value=selected_meta.default,
            description=selected_meta.description,
            choices=selected_meta.choices,
        )
        self._data[name] = new_var
        self.set_data(self._data)
        self.data_changed.emit()

    @Slot()
    def _on_remove_variable(self):
        selected_rows = self.editor.table.selectionModel().selectedRows()
        if not selected_rows: return
        names_to_remove = [self.editor.table.item(index.row(), 0).text() for index in selected_rows if self.editor.table.item(index.row(), 0)]
        for name in names_to_remove:
            if name in self._data: del self._data[name]
        self.set_data(self._data)
        self.data_changed.emit()

    def _update_button_states(self):
        self.remove_btn.setEnabled(len(self.editor.table.selectionModel().selectedRows()) > 0)