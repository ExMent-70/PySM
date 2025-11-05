# pysm_lib/gui/widgets/parameter_editor_widget.py

from typing import Dict, Optional, Any, List, Union

from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer, QEvent, QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QPushButton,
)

from ...models import (
    ScriptArgMetaDetailModel,
    ScriptSetEntryValueEnabled,
    ContextVariableType,
    ContextVariableModel,
    ScriptSetEntryModel,
)
from ...locale_manager import LocaleManager
from ...theme_manager import ThemeManager
from ...app_enums import EditMode
from .editor_factory import EditorFactory
from .editor_context import EditorContext


class ParamTableColumn:
    INSTANCE_ENABLE = 0
    INSTANCE_NAME = 1
    INSTANCE_VALUE = 2
    INSTANCE_ACTIONS = 3

    PASSPORT_REQUIRED = 0
    PASSPORT_NAME = 1
    PASSPORT_TYPE = 2
    PASSPORT_DEFAULT = 3
    PASSPORT_DESCRIPTION = 4

    CONTEXT_NAME = 0
    CONTEXT_TYPE = 1
    CONTEXT_VALUE = 2
    CONTEXT_READONLY = 3
    CONTEXT_DESCRIPTION = 4


class ParameterEditorWidget(QWidget):
    data_changed = Signal()

    def __init__(
        self,
        mode: EditMode,
        locale_manager: LocaleManager,
        theme_manager: ThemeManager,
        script_entries: Optional[List[ScriptSetEntryModel]] = None,
        get_script_name_func: Optional[callable] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.mode = mode
        
        self.editor_context = EditorContext(
            theme_manager=theme_manager,
            locale_manager=locale_manager,
            get_script_info_func=get_script_name_func or (lambda x: None),
            script_entries=script_entries or [],
        )

        self._args_meta: Dict[str, ScriptArgMetaDetailModel] = {}
        self._args_values: Dict[str, ScriptSetEntryValueEnabled] = {}
        self._context_vars: Dict[str, ContextVariableModel] = {}

        self.prefix_color_palette = [
            QColor("#e8f0fe"), QColor("#eaf5ea"), QColor("#fff5e6"),
            QColor("#fdecf0"), QColor("#f0eefc"),
        ]
        self._prefix_to_color_map: Dict[str, QColor] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

    def _connect_signals(self):
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)

    def on_cell_double_clicked(self, row: int, column: int):
        allowed_columns = []
        if self.mode == EditMode.PASSPORT:
            allowed_columns = [ParamTableColumn.PASSPORT_DEFAULT, ParamTableColumn.PASSPORT_DESCRIPTION]
        elif self.mode == EditMode.INSTANCE:
            allowed_columns = [ParamTableColumn.INSTANCE_VALUE]
        elif self.mode == EditMode.CONTEXT_VARS:
            allowed_columns = [ParamTableColumn.CONTEXT_VALUE, ParamTableColumn.CONTEXT_DESCRIPTION]

        if column in allowed_columns:
            widget = self.table.cellWidget(row, column)
            if hasattr(widget, "on_button_click"):
                widget.on_button_click()
            else:
                item = self.table.item(row, column)
                if item and item.flags() & Qt.ItemFlag.ItemIsEditable:
                    self.table.editItem(item)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QComboBox):
            return True
        return super().eventFilter(watched, event)

    def set_data(
        self,
        data: Dict[str, Any],
        instance_values: Optional[Dict[str, ScriptSetEntryValueEnabled]] = None,
    ):
        if self.mode == EditMode.PASSPORT:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
        elif self.mode == EditMode.INSTANCE:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
            self._args_values = {k: v.model_copy(deep=True) for k, v in (instance_values or {}).items()}
        elif self.mode == EditMode.CONTEXT_VARS:
            self._context_vars = {k: v.model_copy(deep=True) for k, v in data.items()}
        self._populate_table()

    def get_updated_meta(self) -> Dict[str, ScriptArgMetaDetailModel]: return self._args_meta
    def get_updated_values(self) -> Dict[str, ScriptSetEntryValueEnabled]: return self._args_values
    def get_updated_context_vars(self) -> Dict[str, ContextVariableModel]: return self._context_vars

    def _populate_table(self):
        self._prefix_to_color_map.clear()
        with QSignalBlocker(self.table):
            self.table.clear(); self.table.setRowCount(0)
            if self.mode == EditMode.PASSPORT: self._setup_passport_mode()
            elif self.mode == EditMode.INSTANCE: self._setup_instance_mode()
            else: self._setup_context_mode()
            data_source = self._args_meta if self.mode != EditMode.CONTEXT_VARS else self._context_vars
            for row, (name, model) in enumerate(sorted(data_source.items())):
                self.table.insertRow(row)
                if self.mode == EditMode.PASSPORT: self._create_passport_row(row, name, model)
                elif self.mode == EditMode.INSTANCE: self._create_instance_row(row, name, model)
                else: self._create_context_row(row, name, model)
        QTimer.singleShot(0, self._adjust_table_columns)


    def _setup_passport_mode(self):
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_required"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_name"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_type"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_default"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_description"),
        ])

    # 2. БЛОК: Метод _setup_instance_mode (ИСПРАВЛЕН)
    # ==============================================================================
    def _setup_instance_mode(self):
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_enable"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_name"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_value"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_actions"),
        ])

    # 3. БЛОК: Метод _setup_context_mode (ИСПРАВЛЕН)
    # ==============================================================================
    def _setup_context_mode(self):
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.context_editor.header_name"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_type"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_value"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_readonly"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_description"),
        ])


    def _create_context_row(self, row: int, name: str, var: ContextVariableModel):
        prefix = name.split("_", 1)[0] if "_" in name else None
        bg_color = self._get_color_for_prefix(prefix)
        def create_item(text=""):
            item = QTableWidgetItem(text);
            if bg_color: item.setBackground(bg_color)
            return item
        name_item = create_item(name); name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, ParamTableColumn.CONTEXT_NAME, name_item)
        self.table.setItem(row, ParamTableColumn.CONTEXT_TYPE, create_item())
        type_combo = QComboBox(); type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self); type_combo.setCurrentText(var.type)
        type_combo.currentTextChanged.connect(lambda t, r=row: self._on_type_changed(r, t))
        self.table.setCellWidget(row, ParamTableColumn.CONTEXT_TYPE, type_combo)
        self.table.setItem(row, ParamTableColumn.CONTEXT_VALUE, create_item())
        self._create_value_editor(row, name, var)
        self.table.setItem(row, ParamTableColumn.CONTEXT_READONLY, create_item())
        ro_check = QCheckBox(); ro_check.setChecked(var.read_only)
        ro_check.toggled.connect(lambda s, r=row: self._on_readonly_changed(r, s))
        self.table.setCellWidget(row, ParamTableColumn.CONTEXT_READONLY, self._center_widget(ro_check))
        desc_item = create_item(var.description or ""); desc_item.setToolTip(var.description)
        self.table.setItem(row, ParamTableColumn.CONTEXT_DESCRIPTION, desc_item)

    def _create_instance_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        entry = self._args_values.get(name)
        chk_enable = QCheckBox(); chk_enable.setChecked(entry.enabled if entry else False)
        chk_enable.setEnabled(not meta.required)
        chk_enable.toggled.connect(lambda state, r=row: self._on_enable_toggled(r, state))
        self.table.setCellWidget(row, ParamTableColumn.INSTANCE_ENABLE, self._center_widget(chk_enable))
        name_item = QTableWidgetItem(name); name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        name_item.setToolTip(meta.description or name)
        self.table.setItem(row, ParamTableColumn.INSTANCE_NAME, name_item)
        self.table.setItem(row, ParamTableColumn.INSTANCE_VALUE, QTableWidgetItem())
        self._create_value_editor(row, name, meta)
        reset_btn = QPushButton("Reset")
        
        
        reset_btn = QPushButton(
            self.editor_context.locale_manager.get("dialogs.script_properties.reset_button")
        )        
        
        
        reset_btn.clicked.connect(lambda checked=False, r=row, n=name: self._on_reset_to_default(r, n))
        self.table.setCellWidget(row, ParamTableColumn.INSTANCE_ACTIONS, self._center_widget(reset_btn))

    def _create_passport_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        chk_required = QCheckBox(); chk_required.setChecked(meta.required)
        chk_required.toggled.connect(lambda state, r=row: self._on_required_toggled(r, state))
        self.table.setCellWidget(row, ParamTableColumn.PASSPORT_REQUIRED, self._center_widget(chk_required))
        name_item = QTableWidgetItem(name)
        self.table.setItem(row, ParamTableColumn.PASSPORT_NAME, name_item)
        type_combo = QComboBox(); type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self); type_combo.setCurrentText(meta.type)
        type_combo.currentTextChanged.connect(lambda text, r=row: self._on_type_changed(r, text))
        self.table.setCellWidget(row, ParamTableColumn.PASSPORT_TYPE, type_combo)
        self.table.setItem(row, ParamTableColumn.PASSPORT_DEFAULT, QTableWidgetItem())
        self._create_value_editor(row, name, meta)
        desc_item = QTableWidgetItem(meta.description or "")
        self.table.setItem(row, ParamTableColumn.PASSPORT_DESCRIPTION, desc_item)

    def _create_value_editor(self, row: int, name: str, model: Union[ScriptArgMetaDetailModel, ContextVariableModel]):
        is_passport, is_context = self.mode == EditMode.PASSPORT, self.mode == EditMode.CONTEXT_VARS
        target_col, value, var_type, choices = -1, None, "string", None
        
        if is_context:
            target_col, value, var_type, choices = ParamTableColumn.CONTEXT_VALUE, model.value, model.type, model.choices
        elif is_passport:
            target_col, value, var_type, choices = ParamTableColumn.PASSPORT_DEFAULT, model.default, model.type, model.choices
        else: # INSTANCE_ARGS
            target_col, entry = ParamTableColumn.INSTANCE_VALUE, self._args_values.get(name)
            value, var_type, choices = (entry.value if entry else None), model.type, model.choices
        
        if self.table.cellWidget(row, target_col): self.table.removeCellWidget(row, target_col)
        
        options = {"choices": choices}
        editor = EditorFactory.create_editor(var_type, value, self.editor_context, options)
        
        if editor:
            if is_passport: editor.valueChanged.connect(lambda v, r=row: self._on_default_value_changed(r, v))
            elif is_context: editor.valueChanged.connect(lambda v, r=row: self._on_context_value_changed(r, v))
            else: editor.valueChanged.connect(lambda v, r=row: self._on_instance_value_changed(r, v))
            
            if (is_passport or is_context) and hasattr(editor, "choicesChanged"):
                editor.choicesChanged.connect(lambda c, r=row: self._on_choices_changed(r, c))

            entry = self._args_values.get(name) if not (is_passport or is_context) else None
            is_enabled = is_passport or is_context or (entry and entry.enabled)

            if is_enabled:
                self.table.setCellWidget(row, target_col, editor)
            else:
                item = QTableWidgetItem("N/A")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, target_col, item)

    def _get_color_for_prefix(self, prefix: Optional[str]) -> Optional[QColor]:
        if not prefix: return None
        if prefix not in self._prefix_to_color_map:
            self._prefix_to_color_map[prefix] = self.prefix_color_palette[len(self._prefix_to_color_map) % len(self.prefix_color_palette)]
        return self._prefix_to_color_map[prefix]

    def _get_name_from_row(self, row: int) -> Optional[str]:
        col_map = {EditMode.CONTEXT_VARS: 0, EditMode.PASSPORT: 1, EditMode.INSTANCE: 1}
        item = self.table.item(row, col_map[self.mode])
        return item.text() if item else None

    @Slot(int, bool)
    def _on_required_toggled(self, row: int, state: bool):
        if name := self._get_name_from_row(row): self._args_meta[name].required = state; self.data_changed.emit()
    @Slot(int, str)
    def _on_type_changed(self, row: int, new_type: str):
        if name := self._get_name_from_row(row):
            model = self._context_vars[name] if self.mode == EditMode.CONTEXT_VARS else self._args_meta[name]
            if model.type != new_type:
                if self.mode == EditMode.CONTEXT_VARS: model.value = None
                else: setattr(model, 'default', None)
                model.type = new_type; model.choices = [] if new_type == "choice" else None
                self._create_value_editor(row, name, model); self.data_changed.emit()
    @Slot(int, object)
    def _on_default_value_changed(self, row: int, value: Any):
        if name := self._get_name_from_row(row): self._args_meta[name].default = value; self.data_changed.emit()
    @Slot(int, object)
    def _on_context_value_changed(self, row: int, value: Any):
        if name := self._get_name_from_row(row): self._context_vars[name].value = value; self.data_changed.emit()
    @Slot(int, list)
    def _on_choices_changed(self, row: int, choices: List[str]):
        if name := self._get_name_from_row(row):
            (self._context_vars[name] if self.mode == EditMode.CONTEXT_VARS else self._args_meta[name]).choices = choices; self.data_changed.emit()
    @Slot(int, bool)
    def _on_enable_toggled(self, row: int, state: bool):
        if name := self._get_name_from_row(row):
            self._args_values[name].enabled = state
            self._create_value_editor(row, name, self._args_meta[name])
            self.data_changed.emit()
    @Slot(int, object)
    def _on_instance_value_changed(self, row: int, value: Any):
        if name := self._get_name_from_row(row): self._args_values[name].value = value; self.data_changed.emit()
    @Slot(int, str)
    def _on_reset_to_default(self, row: int, name: str):
        if meta := self._args_meta.get(name):
            self._args_values[name].value = meta.default
            self._create_value_editor(row, name, meta)
            self.data_changed.emit()
    @Slot(int, bool)
    def _on_readonly_changed(self, row: int, checked: bool):
        if name := self._get_name_from_row(row): self._context_vars[name].read_only = checked; self.data_changed.emit()
    def _center_widget(self, widget: QWidget) -> QWidget:
        cell = QWidget(); layout = QHBoxLayout(cell)
        layout.addWidget(widget); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0); return cell
    def _adjust_table_columns(self):
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        if self.mode == EditMode.PASSPORT: header.setSectionResizeMode(ParamTableColumn.PASSPORT_DESCRIPTION, QHeaderView.ResizeMode.Stretch)
        elif self.mode == EditMode.INSTANCE: header.setSectionResizeMode(ParamTableColumn.INSTANCE_VALUE, QHeaderView.ResizeMode.Stretch)
        elif self.mode == EditMode.CONTEXT_VARS:
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_NAME, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_VALUE, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_DESCRIPTION, QHeaderView.ResizeMode.Stretch)