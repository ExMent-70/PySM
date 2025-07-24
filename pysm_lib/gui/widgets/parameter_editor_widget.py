# pysm_lib/gui/widgets/parameter_editor_widget.py

from typing import Dict, Optional, Any, List, Union
from enum import Enum

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
)
from ...locale_manager import LocaleManager
from .editor_factory import EditorFactory


class EditorMode(Enum):
    PASSPORT_ARGS = 1
    INSTANCE_ARGS = 2
    CONTEXT_VARS = 3


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


class ParameterEditorWidget(QWidget):
    data_changed = Signal()

    def __init__(
        self,
        mode: EditorMode,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.locale_manager = locale_manager

        self._args_meta: Dict[str, ScriptArgMetaDetailModel] = {}
        self._args_values: Dict[str, ScriptSetEntryValueEnabled] = {}
        self._context_vars: Dict[str, ContextVariableModel] = {}

        # Палитра для цветовой группировки префиксов
        self.prefix_color_palette = [
            QColor("#e8f0fe"),
            QColor("#eaf5ea"),
            QColor("#fff5e6"),
            QColor("#fdecf0"),
            QColor("#f0eefc"),
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
        # Этот сигнал остается для внутреннего использования
        self.table.cellDoubleClicked.connect(self._on_internal_cell_double_clicked)

    # --- БЛОК 1: Новый публичный метод ---
    def on_cell_double_clicked(self, row, column):
        """Публичный метод, вызываемый извне для имитации двойного клика."""
        self._on_internal_cell_double_clicked(row, column)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QComboBox):
            return True
        return super().eventFilter(watched, event)

    def set_data(
        self,
        data: Union[
            Dict[str, ScriptArgMetaDetailModel], Dict[str, ContextVariableModel]
        ],
        instance_values: Optional[Dict[str, ScriptSetEntryValueEnabled]] = None,
    ):
        if self.mode == EditorMode.PASSPORT_ARGS:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
        elif self.mode == EditorMode.INSTANCE_ARGS:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
            self._args_values = {
                k: v.model_copy(deep=True) for k, v in instance_values.items()
            }
        elif self.mode == EditorMode.CONTEXT_VARS:
            self._context_vars = {k: v.model_copy(deep=True) for k, v in data.items()}

        self._populate_table()

    def get_updated_meta(self) -> Dict[str, ScriptArgMetaDetailModel]:
        return self._args_meta

    def get_updated_values(self) -> Dict[str, ScriptSetEntryValueEnabled]:
        return self._args_values

    def get_updated_context_vars(self) -> Dict[str, ContextVariableModel]:
        return self._context_vars

    def _populate_table(self):
        self._prefix_to_color_map.clear()

        with QSignalBlocker(self.table):
            self.table.clear()
            self.table.setRowCount(0)

            if self.mode == EditorMode.PASSPORT_ARGS:
                self._setup_passport_mode()
                data_source = self._args_meta
            elif self.mode == EditorMode.INSTANCE_ARGS:
                self._setup_instance_mode()
                data_source = self._args_meta
            else:
                self._setup_context_mode()
                data_source = self._context_vars

            for row, (name, model) in enumerate(sorted(data_source.items())):
                self.table.insertRow(row)
                if self.mode == EditorMode.PASSPORT_ARGS:
                    self._create_passport_row(row, name, model)
                elif self.mode == EditorMode.INSTANCE_ARGS:
                    self._create_instance_row(row, name, model)
                else:
                    self._create_context_row(row, name, model)

        QTimer.singleShot(0, self._adjust_table_columns)

    def _setup_passport_mode(self):
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_required"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_name"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_type"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_default"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_description"
                ),
            ]
        )

    def _setup_instance_mode(self):
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_enable"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_name"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_value"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.args_tab.header_actions"
                ),
            ]
        )

    def _setup_context_mode(self):
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                self.locale_manager.get("dialogs.context_editor.header_name"),
                self.locale_manager.get("dialogs.context_editor.header_type"),
                self.locale_manager.get("dialogs.context_editor.header_value"),
                self.locale_manager.get("dialogs.context_editor.header_readonly"),
                self.locale_manager.get("dialogs.context_editor.header_description"),
            ]
        )

    # 1. БЛОК: Метод _create_context_row (ПЕРЕРАБОТАН)
    def _create_context_row(self, row: int, name: str, var: ContextVariableModel):
        prefix = name.split("_", 1)[0] if "_" in name else None
        background_color = self._get_color_for_prefix(prefix)

        # Функция-помощник для создания и раскраски ячеек
        def create_and_paint_item(text=""):
            item = QTableWidgetItem(text)
            if background_color:
                item.setBackground(background_color)
            return item

        # 1. Имя
        name_item = create_and_paint_item(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 0, name_item)

        # 2. Тип (ячейка красится, виджет - нет)
        self.table.setItem(row, 1, create_and_paint_item())
        type_combo = QComboBox()
        type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self)
        type_combo.setCurrentText(var.type)
        type_combo.currentTextChanged.connect(
            lambda t, r=row: self._on_type_changed(r, t)
        )
        self.table.setCellWidget(row, 1, type_combo)

        # 3. Значение (ячейка красится, виджет - нет)
        self.table.setItem(row, 2, create_and_paint_item())
        self._create_value_editor(row, name, var)

        # 4. Только чтение (ячейка красится, виджет - нет)
        self.table.setItem(row, 3, create_and_paint_item())
        ro_check = QCheckBox()
        ro_check.setChecked(var.read_only)
        ro_check.toggled.connect(lambda s, r=row: self._on_readonly_changed(r, s))
        cell_widget_ro = self._center_widget(ro_check)
        self.table.setCellWidget(row, 3, cell_widget_ro)

        # 5. Описание (ячейка красится, виджет - нет)
        self.table.setItem(row, 4, create_and_paint_item())
        self._create_description_editor(row, var)

    def _create_instance_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        current_entry = self._args_values.get(name)

        chk_enable = QCheckBox()
        chk_enable.setChecked(current_entry.enabled)
        chk_enable.setEnabled(not meta.required)
        chk_enable.toggled.connect(
            lambda state, r=row: self._on_enable_toggled(r, state)
        )
        self.table.setCellWidget(
            row, ParamTableColumn.INSTANCE_ENABLE, self._center_widget(chk_enable)
        )

        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        name_item.setToolTip(meta.description or name)
        self.table.setItem(row, ParamTableColumn.INSTANCE_NAME, name_item)

        self._create_value_editor(row, name, meta)

        reset_btn = QPushButton(
            self.locale_manager.get("dialogs.script_properties.reset_button")
        )
        reset_btn.clicked.connect(
            lambda checked=False, r=row, n=name: self._on_reset_to_default(r, n)
        )
        self.table.setCellWidget(
            row, ParamTableColumn.INSTANCE_ACTIONS, self._center_widget(reset_btn)
        )

    def _create_passport_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        chk_required = QCheckBox()
        chk_required.setChecked(meta.required)
        chk_required.toggled.connect(
            lambda state, r=row: self._on_required_toggled(r, state)
        )
        self.table.setCellWidget(
            row, ParamTableColumn.PASSPORT_REQUIRED, self._center_widget(chk_required)
        )

        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        name_item.setToolTip(name)
        self.table.setItem(row, ParamTableColumn.PASSPORT_NAME, name_item)

        type_combo = QComboBox()
        type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self)
        type_combo.setCurrentText(meta.type)
        type_combo.currentTextChanged.connect(
            lambda text, r=row: self._on_type_changed(r, text)
        )
        self.table.setCellWidget(row, ParamTableColumn.PASSPORT_TYPE, type_combo)

        self._create_value_editor(row, name, meta)
        self._create_description_editor(row, meta)

    def _create_value_editor(
        self,
        row: int,
        arg_name: str,
        model: Union[ScriptArgMetaDetailModel, ContextVariableModel],
    ):
        is_passport_mode = self.mode == EditorMode.PASSPORT_ARGS
        is_context_mode = self.mode == EditorMode.CONTEXT_VARS

        target_column = -1
        if is_passport_mode:
            target_column = ParamTableColumn.PASSPORT_DEFAULT
        elif is_context_mode:
            target_column = 2
        else:
            target_column = ParamTableColumn.INSTANCE_VALUE

        if self.table.cellWidget(row, target_column):
            self.table.removeCellWidget(row, target_column)

        entry = (
            self._args_values.get(arg_name)
            if self.mode == EditorMode.INSTANCE_ARGS
            else None
        )

        if isinstance(model, ContextVariableModel):
            value = model.value
            choices = model.choices
            var_type = model.type
        else:
            value = (
                model.default if is_passport_mode else (entry.value if entry else None)
            )
            choices = model.choices
            var_type = model.type

        options = {"value": value, "choices": choices}
        editor = EditorFactory.create_editor(var_type, options, self.locale_manager)

        if editor:
            slot = None
            if is_passport_mode:
                slot = self._on_default_value_changed
            elif is_context_mode:
                slot = self._on_context_value_changed
            else:
                slot = self._on_instance_value_changed

            if hasattr(editor, "valueChanged"):
                editor.valueChanged.connect(lambda v, r=row: slot(r, v))

            if (is_passport_mode or is_context_mode) and hasattr(
                editor, "choicesChanged"
            ):
                editor.choicesChanged.connect(
                    lambda c, r=row: self._on_choices_changed(r, c)
                )

            is_enabled = (
                is_passport_mode or is_context_mode or (entry and entry.enabled)
            )

            if is_enabled:
                self.table.setCellWidget(row, target_column, editor)
            else:
                item = QTableWidgetItem(
                    self.locale_manager.get("general.not_applicable")
                )
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("Argument is disabled")
                self.table.setItem(row, target_column, item)

    # 2. БЛОК: Метод _create_description_editor (ИСПРАВЛЕН)
    def _create_description_editor(
        self, row: int, model: Union[ScriptArgMetaDetailModel, ContextVariableModel]
    ):
        target_column = 4
        value = model.description
        editor = EditorFactory.create_editor(
            "string", {"value": value}, self.locale_manager
        )

        if editor and hasattr(editor, "valueChanged"):
            slot = (
                self._on_context_description_changed
                if self.mode == EditorMode.CONTEXT_VARS
                else self._on_description_changed
            )
            editor.valueChanged.connect(lambda v, r=row: slot(r, v))
            editor.setToolTip(value or "")

            # --- ИЗМЕНЕНИЕ: Убираем прямое применение стиля к редактору ---
            # if background_color:
            #     editor.setStyleSheet(f"background-color: {background_color.name()};")

            self.table.setCellWidget(row, target_column, editor)

    def _get_color_for_prefix(self, prefix: Optional[str]) -> Optional[QColor]:
        if not prefix:
            return None

        if prefix not in self._prefix_to_color_map:
            next_color_index = len(self._prefix_to_color_map) % len(
                self.prefix_color_palette
            )
            self._prefix_to_color_map[prefix] = self.prefix_color_palette[
                next_color_index
            ]

        return self._prefix_to_color_map[prefix]

    def _get_name_from_row(self, row: int) -> Optional[str]:
        target_column = 0
        if self.mode != EditorMode.CONTEXT_VARS:
            target_column = (
                ParamTableColumn.PASSPORT_NAME
                if self.mode == EditorMode.PASSPORT_ARGS
                else ParamTableColumn.INSTANCE_NAME
            )

        name_item = self.table.item(row, target_column)
        return name_item.text() if name_item else None

    @Slot(int, bool)
    def _on_required_toggled(self, row: int, state: bool):
        name = self._get_name_from_row(row)
        if name and name in self._args_meta:
            self._args_meta[name].required = state
            self.data_changed.emit()

    @Slot(int, str)
    def _on_type_changed(self, row: int, new_type: str):
        name = self._get_name_from_row(row)

        if self.mode == EditorMode.CONTEXT_VARS:
            if name and name in self._context_vars:
                var = self._context_vars[name]
                if var.type != new_type:
                    var.type = new_type
                    var.value = None
                    var.choices = [] if new_type == "choice" else None
                    self._create_value_editor(row, name, var)
                    self.data_changed.emit()
        else:  # PASSPORT_ARGS
            if name and name in self._args_meta:
                meta = self._args_meta[name]
                if meta.type != new_type:
                    meta.type = new_type
                    meta.default = None
                    self._create_value_editor(row, name, meta)
                    self.data_changed.emit()

    @Slot(int, object)
    def _on_default_value_changed(self, row: int, value: Any):
        name = self._get_name_from_row(row)
        if name and name in self._args_meta:
            self._args_meta[name].default = value
            self.data_changed.emit()

    @Slot(int, object)
    def _on_description_changed(self, row: int, value: Any):
        name = self._get_name_from_row(row)
        if name and name in self._args_meta:
            self._args_meta[name].description = str(value) or None
            self.data_changed.emit()

    @Slot(int, list)
    def _on_choices_changed(self, row: int, choices: List[str]):
        name = self._get_name_from_row(row)
        if self.mode == EditorMode.CONTEXT_VARS:
            if name in self._context_vars:
                self._context_vars[name].choices = choices
        else:
            if name in self._args_meta:
                self._args_meta[name].choices = choices
        self.data_changed.emit()

    @Slot(int, bool)
    def _on_enable_toggled(self, row: int, state: bool):
        name = self._get_name_from_row(row)
        if name and name in self._args_values:
            self._args_values[name].enabled = state
            meta = self._args_meta[name]
            self._create_value_editor(row, name, meta)
            self.data_changed.emit()

    @Slot(int, object)
    def _on_instance_value_changed(self, row: int, value: Any):
        name = self._get_name_from_row(row)
        if name and name in self._args_values:
            self._args_values[name].value = value
            self.data_changed.emit()

    @Slot(int, str)
    def _on_reset_to_default(self, row: int, name: str):
        meta = self._args_meta.get(name)
        if not meta:
            return
        self._args_values[name].value = meta.default
        self._create_value_editor(row, name, meta)
        self.data_changed.emit()

    @Slot(int, bool)
    def _on_readonly_changed(self, row: int, checked: bool):
        name = self._get_name_from_row(row)
        if name and name in self._context_vars:
            self._context_vars[name].read_only = checked
            self.data_changed.emit()

    @Slot(int, object)
    def _on_context_value_changed(self, row: int, value: Any):
        name = self._get_name_from_row(row)
        if name and name in self._context_vars:
            self._context_vars[name].value = value
            self.data_changed.emit()

    @Slot(int, object)
    def _on_context_description_changed(self, row: int, value: Any):
        name = self._get_name_from_row(row)
        if name and name in self._context_vars:
            self._context_vars[name].description = str(value) or None
            self.data_changed.emit()

    # --- БЛОК 2: Слот переименован в приватный ---
    @Slot(int, int)
    def _on_internal_cell_double_clicked(self, row: int, column: int):
        widget = self.table.cellWidget(row, column)
        if hasattr(widget, "on_button_click"):
            widget.on_button_click()
        elif hasattr(widget, "line_edit"):
            widget.line_edit.setFocus()
        else:
            # Для ячеек без виджета (например, имя в контексте),
            # делаем их редактируемыми по двойному клику
            item = self.table.item(row, column)
            if item and item.flags() & Qt.ItemFlag.ItemIsEditable:
                self.table.editItem(item)

    def _center_widget(self, widget: QWidget) -> QWidget:
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return cell

    def _adjust_table_columns(self):
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        if self.mode == EditorMode.PASSPORT_ARGS:
            header.setSectionResizeMode(
                ParamTableColumn.PASSPORT_DESCRIPTION, QHeaderView.ResizeMode.Stretch
            )
        elif self.mode == EditorMode.INSTANCE_ARGS:
            header.setSectionResizeMode(
                ParamTableColumn.INSTANCE_VALUE, QHeaderView.ResizeMode.Stretch
            )
            header.setSectionResizeMode(
                ParamTableColumn.INSTANCE_ACTIONS,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        elif self.mode == EditorMode.CONTEXT_VARS:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
