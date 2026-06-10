#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import html
import json
import logging
import sys
from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_icons import ICON_CATEGORIES, icons

    IS_MANAGED_RUN = getattr(pysm_context, "_context_file_path", None) is not None
except ImportError:
    pysm_context = None
    icons = None
    ICON_CATEGORIES = {}
    IS_MANAGED_RUN = False

    class MockThemeApi:
        @staticmethod
        def apply_theme_to_app(app):
            return None

    theme_api = MockThemeApi()


SCHEMA_SUFFIX = "__schema"
DEFAULT_ICON_BY_TYPE = {
    "dict": "SETTINGS",
    "list": "LIST",
    "str": "FILE_TXT",
    "int": "SLIDERS",
    "float": "SLIDERS",
    "bool": "OK",
    "NoneType": "INFO",
}
SCHEMA_FIELD_DEFAULTS = {
    "label": "",
    "description": "",
    "widget": "string",
    "icon": "FILE_TXT",
    "min": 0.0,
    "max": 1.0,
    "step": 0.01,
    "decimals": 4,
}
WIDGET_TYPES = ["string", "int", "float", "bool", "object", "array", "null"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config():
    parser = argparse.ArgumentParser(
        description="Visual editor for PySM context variables of type json."
    )
    parser.add_argument("--var_name", required=True)
    parser.add_argument("--title", default="Редактор JSON")
    parser.add_argument("--msg", default="")

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def json_type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


def default_value_for_type(type_name: str) -> Any:
    return {
        "string": "",
        "int": 0,
        "float": 0.0,
        "bool": False,
        "object": {},
        "array": [],
        "null": None,
    }[type_name]


def get_raw_context_data() -> dict[str, Any]:
    if not pysm_context:
        return {}
    reader = getattr(pysm_context, "_read_data", None)
    if callable(reader):
        return reader()
    getter = getattr(pysm_context, "get_all", None)
    if callable(getter):
        return getter()
    return {}


def load_context_variable(var_name: str) -> tuple[Optional[dict[str, Any]], Any]:
    variable_data = get_raw_context_data().get(var_name)
    if not isinstance(variable_data, dict):
        return None, None
    return variable_data, variable_data.get("value")


def schema_var_name_for(var_name: str) -> str:
    return f"{var_name}{SCHEMA_SUFFIX}"


def target_name_for_schema(schema_var_name: str) -> str:
    return schema_var_name[: -len(SCHEMA_SUFFIX)]


def is_schema_variable(var_name: str) -> bool:
    return var_name.endswith(SCHEMA_SUFFIX)


def build_default_schema(title: str, data: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "version": 1,
        "title": title,
        "fields": {},
    }

    def walk(value: Any, path: str = ""):
        if isinstance(value, dict):
            for key, child_value in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                schema["fields"][child_path] = {
                    "label": str(key),
                    "description": "",
                    "widget": json_type_name(child_value),
                    "icon": DEFAULT_ICON_BY_TYPE.get(type(child_value).__name__, "FILE_CODE"),
                }
                walk(child_value, child_path)
        elif isinstance(value, list):
            for i, child_value in enumerate(value):
                walk(child_value, f"{path}.{i}" if path else str(i))

    walk(data)
    return schema


class SchemaIndex:
    def __init__(self, schema: Optional[dict[str, Any]]):
        self.schema = schema if isinstance(schema, dict) else {}
        fields = self.schema.get("fields")
        self.fields = fields if isinstance(fields, dict) else {}

    def field(self, node: "Node") -> dict[str, Any]:
        for key in (node.path, str(node.key)):
            value = self.fields.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def label_for(self, node: "Node") -> str:
        label = self.field(node).get("label")
        return str(label) if label else ""

    def has_meaningful_labels(self) -> bool:
        for path, field in self.fields.items():
            if not isinstance(field, dict):
                continue
            label = field.get("label")
            if label and str(label) != str(path).split(".")[-1]:
                return True
        return False

    def icon_for(self, node: "Node") -> str:
        icon_name = self.field(node).get("icon")
        if icon_name:
            return str(icon_name)
        return DEFAULT_ICON_BY_TYPE.get(type(node.value).__name__, "FILE_CODE")


class Node:
    def __init__(self, key: Any, value: Any, parent: Optional["Node"] = None):
        self.key = key
        self.value = value
        self.parent = parent
        self.children: list[Node] = []
        self.path = self._build_path()
        self.rebuild_children()

    def _build_path(self) -> str:
        if not self.parent or self.parent.key == "root":
            return "" if self.key == "root" else str(self.key)
        return f"{self.parent.path}.{self.key}" if self.parent.path else str(self.key)

    def rebuild_children(self):
        self.children = []
        if isinstance(self.value, dict):
            for key, value in self.value.items():
                self.children.append(Node(key, value, self))
        elif isinstance(self.value, list):
            for i, value in enumerate(self.value):
                self.children.append(Node(i, value, self))

    def refresh_paths(self):
        self.path = self._build_path()
        for child in self.children:
            child.refresh_paths()

    def is_leaf(self) -> bool:
        return not isinstance(self.value, (dict, list))


class JsonModel(QAbstractItemModel):
    def __init__(
        self,
        data: Any,
        schema: Optional[dict[str, Any]] = None,
        schema_mode: bool = False,
    ):
        super().__init__()
        self.root = Node("root", data)
        self.schema = SchemaIndex(schema)
        self.schema_mode = schema_mode

    def columnCount(self, parent=QModelIndex()):
        return 4

    def rowCount(self, parent=QModelIndex()):
        return len(self.get_node(parent).children)

    def index(self, row, column, parent=QModelIndex()):
        parent_node = self.get_node(parent)
        if 0 <= row < len(parent_node.children):
            return self.createIndex(row, column, parent_node.children[row])
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if not node or not node.parent or node.parent == self.root:
            return QModelIndex()
        parent = node.parent
        grand = parent.parent
        if not grand:
            return QModelIndex()
        return self.createIndex(grand.children.index(parent), 0, parent)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        node = index.internalPointer()
        column = index.column()

        if role == Qt.DecorationRole and column == 0 and icons:
            icon_name = self.schema.icon_for(node)
            return icons.get_qicon(icon_name, size=24)

        if role in (Qt.DisplayRole, Qt.EditRole):
            if column == 0:
                return str(node.key)
            if column == 1:
                if isinstance(node.value, dict):
                    return "<object>"
                if isinstance(node.value, list):
                    return f"<array: {len(node.children)}>"
                if node.value is None:
                    return "null"
                return str(node.value)
            if column == 2:
                return json_type_name(node.value)
            if column == 3:
                return self.schema.label_for(node)

        if role == Qt.ToolTipRole:
            field = self.schema.field(node)
            description = field.get("description") or field.get("help")
            if description:
                return str(description)
            return node.path

        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        node = index.internalPointer()
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 1 and node.is_leaf():
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False

        node = index.internalPointer()
        if index.column() != 1 or not node.is_leaf():
            return False

        try:
            node.value = self.convert_value(value, node.value)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(None, "Ошибка", f"Некорректное значение: {exc}")
            return False

        self.dataChanged.emit(index, index)
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return ["Ключ", "Значение", "Тип", "Метка"][section]
        return None

    def get_node(self, index):
        return index.internalPointer() if index.isValid() else self.root

    def index_for_node(self, node: Node, column: int = 0) -> QModelIndex:
        if node == self.root or not node.parent:
            return QModelIndex()
        return self.createIndex(node.parent.children.index(node), column, node)

    def convert_value(self, value: Any, old_value: Any) -> Any:
        if isinstance(old_value, bool):
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes", "да")
        if isinstance(old_value, int) and not isinstance(old_value, bool):
            return int(value)
        if isinstance(old_value, float):
            return float(value)
        if old_value is None:
            if str(value).strip().lower() in ("", "none", "null"):
                return None
            return value
        return str(value)

    def rename_key(self, node: Node, new_key: str) -> bool:
        parent = node.parent
        if not parent or not isinstance(parent.value, dict):
            return False
        if not new_key:
            QMessageBox.warning(None, "Ошибка", "Имя ключа не может быть пустым.")
            return False
        if new_key != node.key and new_key in parent.value:
            QMessageBox.warning(None, "Ошибка", f"Ключ '{new_key}' уже существует.")
            return False

        old_key = node.key
        node.key = new_key
        items = []
        for child in parent.children:
            key = new_key if child is node else child.key
            items.append((key, self.to_python(child)))
        parent.value.clear()
        parent.value.update(items)
        parent.refresh_paths()

        row = parent.children.index(node)
        parent_index = self.index_for_node(parent)
        self.dataChanged.emit(
            self.index(row, 0, parent_index),
            self.index(row, 3, parent_index),
        )
        logger.info("Renamed JSON key %s -> %s", old_key, new_key)
        return True

    def insert_child(self, parent_node: Node, key: Any, value: Any) -> bool:
        if not isinstance(parent_node.value, (dict, list)):
            return False

        if isinstance(parent_node.value, dict):
            key = str(key)
            if not key:
                QMessageBox.warning(None, "Ошибка", "Имя ключа не может быть пустым.")
                return False
            if key in parent_node.value:
                QMessageBox.warning(None, "Ошибка", f"Ключ '{key}' уже существует.")
                return False
        else:
            key = len(parent_node.children)

        parent_index = self.index_for_node(parent_node)
        row = len(parent_node.children)
        self.beginInsertRows(parent_index, row, row)
        if isinstance(parent_node.value, dict):
            parent_node.value[key] = value
        else:
            parent_node.value.append(value)
        parent_node.children.append(Node(key, value, parent_node))
        parent_node.refresh_paths()
        self.endInsertRows()
        return True

    def remove_node(self, node: Node) -> bool:
        parent = node.parent
        if not parent:
            return False
        row = parent.children.index(node)
        parent_index = self.index_for_node(parent)

        self.beginRemoveRows(parent_index, row, row)
        if isinstance(parent.value, dict):
            parent.value.pop(node.key, None)
        elif isinstance(parent.value, list):
            parent.value.pop(row)
        parent.children.pop(row)
        if isinstance(parent.value, list):
            for i, child in enumerate(parent.children):
                child.key = i
        parent.refresh_paths()
        self.endRemoveRows()
        return True

    def to_python(self, node: Optional[Node] = None) -> Any:
        node = node or self.root
        if isinstance(node.value, dict):
            return {child.key: self.to_python(child) for child in node.children}
        if isinstance(node.value, list):
            return [self.to_python(child) for child in node.children]
        return node.value


class ValueDelegate(QStyledItemDelegate):
    ROW_HEIGHT = 30

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height(), self.ROW_HEIGHT))

    def createEditor(self, parent, option, index):
        node = index.internalPointer()
        model = index.model()
        field = model.schema.field(node)

        if model.schema_mode and index.column() == 1 and node.parent:
            if node.key == "widget":
                editor = QComboBox(parent)
                editor.addItems(WIDGET_TYPES)
                editor.setMinimumHeight(self.ROW_HEIGHT - 2)
                return editor

            if node.key == "icon":
                editor = QComboBox(parent)
                for icon_name in self.icon_names():
                    if icons:
                        editor.addItem(icons.get_qicon(icon_name, size=20), icon_name)
                    else:
                        editor.addItem(icon_name)
                editor.setMinimumHeight(self.ROW_HEIGHT - 2)
                return editor

        if isinstance(node.value, bool):
            editor = QCheckBox(parent)
            editor.setMinimumHeight(self.ROW_HEIGHT - 2)
            return editor

        if isinstance(node.value, int) and not isinstance(node.value, bool):
            editor = QSpinBox(parent)
            editor.setMinimum(int(field.get("min", -(10**9))))
            editor.setMaximum(int(field.get("max", 10**9)))
            editor.setSingleStep(int(field.get("step", 1)))
            editor.setMinimumHeight(self.ROW_HEIGHT - 2)
            return editor

        if isinstance(node.value, float):
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(int(field.get("decimals", 4)))
            editor.setMinimum(float(field.get("min", -(10**9))))
            editor.setMaximum(float(field.get("max", 10**9)))
            editor.setSingleStep(float(field.get("step", 0.01)))
            editor.setMinimumHeight(self.ROW_HEIGHT - 2)
            return editor

        editor = QLineEdit(parent)
        editor.setMinimumHeight(self.ROW_HEIGHT - 2)
        return editor

    def icon_names(self) -> list[str]:
        names: list[str] = []
        for category_names in ICON_CATEGORIES.values():
            names.extend(category_names)
        return sorted(set(names))

    def setEditorData(self, editor, index):
        value = index.internalPointer().value
        if isinstance(editor, QComboBox):
            text = "" if value is None else str(value)
            pos = editor.findText(text)
            if pos >= 0:
                editor.setCurrentIndex(pos)
            else:
                editor.insertItem(0, text)
                editor.setCurrentIndex(0)
        elif isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.setValue(value)
        else:
            editor.setText("" if value is None else str(value))

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.EditRole)
        elif isinstance(editor, QCheckBox):
            model.setData(index, editor.isChecked(), Qt.EditRole)
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            model.setData(index, editor.value(), Qt.EditRole)
        else:
            model.setData(index, editor.text(), Qt.EditRole)


class NewItemDialog(QDialog):
    def __init__(self, parent_node: Node, parent=None):
        super().__init__(parent)
        self.parent_node = parent_node
        self.setWindowTitle("Новый элемент")

        layout = QFormLayout(self)
        self.key_edit = QLineEdit()
        self.type_box = QComboBox()
        self.type_box.addItems(["string", "int", "float", "bool", "object", "array", "null"])

        if isinstance(parent_node.value, list):
            self.key_edit.setEnabled(False)
            self.key_edit.setPlaceholderText("Индекс будет назначен автоматически")

        layout.addRow("Ключ:", self.key_edit)
        layout.addRow("Тип:", self.type_box)

        buttons = QHBoxLayout()
        btn_create = QPushButton("Создать")
        btn_cancel = QPushButton("Отмена")
        btn_create.clicked.connect(self.validate)
        btn_cancel.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(btn_create)
        buttons.addWidget(btn_cancel)
        layout.addRow(buttons)

    def validate(self):
        if isinstance(self.parent_node.value, dict) and not self.key_edit.text().strip():
            QMessageBox.warning(self, "Ошибка", "Имя ключа не может быть пустым.")
            return
        self.accept()

    def get_data(self):
        key = self.key_edit.text().strip()
        return key, default_value_for_type(self.type_box.currentText())


class JsonEditor(QMainWindow):
    def __init__(
        self,
        var_name: str,
        data: Any,
        variable_data: dict[str, Any],
        schema: Optional[dict[str, Any]],
        title: str,
        msg: str = "",
        schema_mode: bool = False,
    ):
        super().__init__()
        self.var_name = var_name
        self.variable_data = variable_data
        self.schema_mode = schema_mode
        self.msg = msg
        self.saved = False
        self.force_close = False
        self.dirty = False
        self.original_data = copy.deepcopy(data)
        self._adjusting_columns = False
        self.description_label: Optional[QLabel] = None

        self.setWindowTitle(self.build_window_title(title))
        self.resize(700, 850)

        self.model = JsonModel(copy.deepcopy(data), schema, schema_mode=schema_mode)
        self.model.dataChanged.connect(self.mark_dirty)
        self.model.rowsInserted.connect(self.mark_dirty)
        self.model.rowsRemoved.connect(self.mark_dirty)

        self.view = QTreeView()
        self.view.setModel(self.model)
        self.view.expandAll()
        self.view.setRootIsDecorated(True)
        self.view.setItemsExpandable(True)
        self.view.setExpandsOnDoubleClick(True)
        self.view.setIndentation(28)
        self.view.setIconSize(QSize(24, 24))
        self.view.setAlternatingRowColors(True)
        self.view.setUniformRowHeights(False)
        self.view.setStyleSheet(
            "QTreeView::item { min-height: 25px; padding-top: 3px; padding-bottom: 3px; }"
        )
        self.view.setItemDelegateForColumn(1, ValueDelegate())
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.open_context_menu)

        header = self.view.header()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(42)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        self.view.setColumnHidden(3, not self.model.schema.has_meaningful_labels())
        self.resize_non_value_columns_to_contents()
        header.sectionResized.connect(self.on_section_resized)
        self.view.selectionModel().currentChanged.connect(self.update_description_panel)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.build_info_label())
        layout.addWidget(self.view, 1)
        layout.addWidget(self.build_description_panel())
        layout.addLayout(self.build_buttons())
        self.setCentralWidget(central)
        self.update_description_panel(self.view.currentIndex(), QModelIndex())
        QTimer.singleShot(0, self.adjust_value_column_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.adjust_value_column_width)

    def resize_non_value_columns_to_contents(self):
        for column in (0, 2, 3):
            if not self.view.isColumnHidden(column):
                self.view.resizeColumnToContents(column)

    def on_section_resized(self, logical_index: int, old_size: int, new_size: int):
        if self._adjusting_columns or logical_index == 1:
            return
        QTimer.singleShot(0, self.adjust_value_column_width)

    def adjust_value_column_width(self):
        if self._adjusting_columns:
            return

        viewport_width = self.view.viewport().width()
        fixed_width = 0
        for column in (0, 2, 3):
            if not self.view.isColumnHidden(column):
                fixed_width += self.view.columnWidth(column)

        value_width = max(140, viewport_width - fixed_width - 8)
        self._adjusting_columns = True
        try:
            self.view.setColumnWidth(1, value_width)
        finally:
            self._adjusting_columns = False

    def build_window_title(self, title: str) -> str:
        if self.schema_mode:
            target = target_name_for_schema(self.var_name)
            return f"{title}: схема для {target}"
        return f"{title}: {self.var_name}"

    def build_info_label(self) -> QLabel:
        if self.msg:
            text = self.msg
        elif self.schema_mode:
            text = (
                f"Редактируется служебная схема {self.var_name}. "
                "Для нее не загружается дополнительная __schema-переменная."
            )
        else:
            schema_name = schema_var_name_for(self.var_name)
            text = f"Редактируется JSON-переменная {self.var_name}. Схема отображения: {schema_name}."
        label = QLabel(text)
        label.setTextFormat(Qt.RichText if self.msg else Qt.PlainText)
        label.setOpenExternalLinks(True)
        label.setWordWrap(True)
        return label

    def build_description_panel(self) -> QLabel:
        self.description_label = QLabel()
        self.description_label.setTextFormat(Qt.RichText)
        self.description_label.setOpenExternalLinks(True)
        self.description_label.setWordWrap(True)
        self.description_label.setMinimumHeight(44)
        self.description_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.description_label.setContentsMargins(0, 4, 0, 4)
        return self.description_label

    def update_description_panel(self, current: QModelIndex, previous: QModelIndex):
        if self.description_label is None:
            return
        if not current.isValid():
            self.description_label.setText("Описание: выберите элемент JSON.")
            return

        node = current.internalPointer()
        field = self.model.schema.field(node)
        label = field.get("label")
        description = field.get("description") or field.get("help")
        path = node.path or "<root>"

        parts = []
        if label:
            parts.append(f"<b>{html.escape(str(label))}</b>")
        else:
            parts.append(f"<b>{html.escape(str(node.key))}</b>")
        parts.append(f"<b>Путь:</b> {html.escape(path)}")
        if description:
            parts.append(html.escape(str(description)))
        else:
            parts.append("Описание для этого параметра не задано в схеме.")

        self.description_label.setText("<br>".join(parts))

    def build_buttons(self) -> QHBoxLayout:
        buttons = QHBoxLayout()

        btn_add = QPushButton("Добавить")
        btn_expand = QPushButton("Развернуть все")
        btn_collapse = QPushButton("Свернуть все")
        btn_save = QPushButton("Сохранить")
        btn_cancel = QPushButton("Отмена")

        if icons:
            btn_add.setIcon(icons.get_qicon("ADD"))
            btn_save.setIcon(icons.get_qicon("SAVE"))
            btn_cancel.setIcon(icons.get_qicon("CLOSE"))

        btn_add.clicked.connect(lambda: self.create_node(self.model.root))
        btn_expand.clicked.connect(self.view.expandAll)
        btn_collapse.clicked.connect(self.view.collapseAll)
        btn_save.clicked.connect(self.save)
        btn_cancel.clicked.connect(self.cancel)

        buttons.addWidget(btn_add)

        if not self.schema_mode:
            btn_make_schema = QPushButton("Создать схему")
            if icons:
                btn_make_schema.setIcon(icons.get_qicon("SETTINGS"))
            btn_make_schema.clicked.connect(self.create_default_schema)
            buttons.addWidget(btn_make_schema)
        else:
            btn_fill_schema = QPushButton("Заполнить свойства")
            if icons:
                btn_fill_schema.setIcon(icons.get_qicon("SETTINGS"))
            btn_fill_schema.clicked.connect(self.fill_schema_properties)
            buttons.addWidget(btn_fill_schema)

        buttons.addWidget(btn_expand)
        buttons.addWidget(btn_collapse)
        buttons.addStretch()
        buttons.addWidget(btn_save)
        buttons.addWidget(btn_cancel)
        return buttons

    def mark_dirty(self, *args):
        self.dirty = True

    def current_data(self) -> Any:
        return self.model.to_python()

    def has_changes(self) -> bool:
        return self.current_data() != self.original_data

    def print_saved_value(self, value: Any):
        s=json.dumps(value, ensure_ascii=False, indent=2)
        print(f"✅ <b>{self.var_name}</b> {s}")
        print()

    def open_context_menu(self, pos):
        index = self.view.indexAt(pos)
        if not index.isValid():
            return

        node = index.internalPointer()
        menu = QMenu(self)

        if isinstance(node.value, (dict, list)):
            add_action = QAction("Добавить в узел...", self)
            if icons:
                add_action.setIcon(icons.get_qicon("ADD"))
            add_action.triggered.connect(lambda: self.create_node(node))
            menu.addAction(add_action)

        schema_node = self.get_schema_field_node(node)
        if schema_node:
            schema_menu = menu.addMenu("Свойства схемы")
            if icons:
                schema_menu.setIcon(icons.get_qicon("SETTINGS"))

            for prop_name in SCHEMA_FIELD_DEFAULTS:
                action = QAction(f"Добавить {prop_name}", self)
                action.setEnabled(prop_name not in schema_node.value)
                action.triggered.connect(
                    lambda checked=False, n=schema_node, p=prop_name: self.add_schema_property(n, p)
                )
                schema_menu.addAction(action)

            schema_menu.addSeparator()
            fill_action = QAction("Добавить все недостающие", self)
            fill_action.triggered.connect(lambda: self.fill_schema_node_properties(schema_node))
            schema_menu.addAction(fill_action)

        if node.parent and isinstance(node.parent.value, dict):
            rename_action = QAction("Переименовать ключ...", self)
            if icons:
                rename_action.setIcon(icons.get_qicon("SETTINGS"))
            rename_action.triggered.connect(lambda: self.rename_node_key(node))
            menu.addAction(rename_action)

        delete_action = QAction("Удалить", self)
        if icons:
            delete_action.setIcon(icons.get_qicon("DELETE"))
        delete_action.triggered.connect(lambda: self.delete_node(node))
        menu.addAction(delete_action)

        menu.exec(self.view.viewport().mapToGlobal(pos))

    def get_schema_field_node(self, node: Node) -> Optional[Node]:
        if not self.schema_mode:
            return None

        candidate = node if isinstance(node.value, dict) else node.parent
        if not candidate or not isinstance(candidate.value, dict):
            return None
        if candidate.parent and candidate.parent.key == "fields":
            return candidate
        return None

    def default_schema_property_value(self, node: Node, prop_name: str) -> Any:
        if prop_name == "label":
            return str(node.key)
        if prop_name == "widget":
            widget = node.value.get("widget")
            return widget if widget else "string"
        if prop_name == "icon":
            widget = str(node.value.get("widget", "string"))
            return {
                "object": "SETTINGS",
                "array": "LIST",
                "string": "FILE_TXT",
                "int": "SLIDERS",
                "float": "SLIDERS",
                "bool": "OK",
                "null": "INFO",
            }.get(widget, "FILE_TXT")
        return copy.deepcopy(SCHEMA_FIELD_DEFAULTS[prop_name])

    def add_schema_property(self, node: Node, prop_name: str):
        if prop_name in node.value:
            return
        self.model.insert_child(node, prop_name, self.default_schema_property_value(node, prop_name))
        self.view.expand(self.model.index_for_node(node))

    def fill_schema_node_properties(self, node: Node):
        for prop_name in SCHEMA_FIELD_DEFAULTS:
            if prop_name not in node.value:
                self.add_schema_property(node, prop_name)

    def fill_schema_properties(self):
        fields_node = next(
            (
                child
                for child in self.model.root.children
                if child.key == "fields" and isinstance(child.value, dict)
            ),
            None,
        )
        if not fields_node:
            QMessageBox.warning(self, "Ошибка", "В схеме нет объекта fields.")
            return

        for field_node in list(fields_node.children):
            if isinstance(field_node.value, dict):
                self.fill_schema_node_properties(field_node)

        QMessageBox.information(self, "OK", "Недостающие свойства схемы добавлены.")

    def rename_node_key(self, node: Node):
        parent = node.parent
        if not parent or not isinstance(parent.value, dict):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Переименование ключа")
        layout = QFormLayout(dialog)
        key_edit = QLineEdit(str(node.key))
        layout.addRow("Новый ключ:", key_edit)

        buttons = QHBoxLayout()
        btn_ok = QPushButton("Переименовать")
        btn_cancel = QPushButton("Отмена")
        buttons.addStretch()
        buttons.addWidget(btn_ok)
        buttons.addWidget(btn_cancel)
        layout.addRow(buttons)

        btn_ok.clicked.connect(dialog.accept)
        btn_cancel.clicked.connect(dialog.reject)

        if not dialog.exec():
            return

        new_key = key_edit.text().strip()
        if new_key == node.key:
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Переименовать ключ '{node.key}' в '{new_key}'?\n\n"
            "Это может нарушить работу скриптов, которые ожидают старое имя.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.model.rename_key(node, new_key)

    def create_node(self, node: Node):
        if not isinstance(node.value, (dict, list)):
            return
        dialog = NewItemDialog(node, self)
        if not dialog.exec():
            return
        key, value = dialog.get_data()
        if self.model.insert_child(node, key, value):
            self.view.expand(self.model.index_for_node(node))

    def delete_node(self, node: Node):
        if node == self.model.root:
            return
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Удалить элемент '{node.key}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.model.remove_node(node)

    def create_default_schema(self):
        schema_name = schema_var_name_for(self.var_name)
        schema_data, _ = load_context_variable(schema_name)
        if schema_data and QMessageBox.question(
            self,
            "Схема уже существует",
            f"Перезаписать {schema_name} автоматически созданной схемой?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        schema = build_default_schema(self.var_name, self.model.to_python())
        pysm_context.set_structured(schema_name, schema, commit=True)
        QMessageBox.information(self, "OK", f"Схема сохранена в {schema_name}.")

    def save(self):
        try:
            data = self.current_data()
            changed = data != self.original_data
            if changed:
                pysm_context.set_structured(self.var_name, data, commit=True)
                self.print_saved_value(data)
                self.original_data = copy.deepcopy(data)
            self.saved = True
            self.dirty = False
            #message = "Сохранено." if changed else "Изменений нет, сохранять нечего."
            #QMessageBox.information(self, "OK", message)
            if not changed:
                QMessageBox.information(self, f"Сохранение переменной {self.var_name}", f"Значение переменной {self.var_name} не изменилось.\nСохранение отменено")

            QApplication.exit(0)
            self.close()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def cancel(self):
        if self.confirm_discard():
            self.force_close = True
            QApplication.exit(0)
            self.close()

    def confirm_discard(self) -> bool:
        if self.saved or not self.has_changes():
            return True
        reply = QMessageBox.question(
            self,
            "Есть несохраненные изменения",
            "Закрыть редактор без сохранения?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event: QCloseEvent):
        if self.saved or self.force_close:
            event.accept()
            return
        if self.confirm_discard():
            QApplication.exit(0)
            event.accept()
        else:
            event.ignore()


def validate_variable(var_name: str) -> tuple[dict[str, Any], Any]:
    variable_data, value = load_context_variable(var_name)
    if not variable_data:
        raise ValueError(f"Переменная '{var_name}' не найдена в контексте.")
    if variable_data.get("type") != "json":
        raise ValueError(f"Переменная '{var_name}' имеет тип '{variable_data.get('type')}', а нужен 'json'.")
    if variable_data.get("read_only", False):
        raise ValueError(f"Переменная '{var_name}' защищена от записи.")
    if not isinstance(value, (dict, list)):
        raise ValueError(f"Значение переменной '{var_name}' должно быть объектом или массивом JSON.")
    return variable_data, value


def load_schema(var_name: str) -> Optional[dict[str, Any]]:
    if is_schema_variable(var_name):
        return None

    schema_name = schema_var_name_for(var_name)
    variable_data, value = load_context_variable(schema_name)
    if not variable_data:
        return None
    if variable_data.get("type") != "json" or not isinstance(value, dict):
        logger.warning("Ignoring invalid schema variable %s", schema_name)
        return None
    return value


def show_startup_error(message: str):
    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)
    QMessageBox.critical(None, "Ошибка", message)


def main():
    config = get_config()

    if not IS_MANAGED_RUN:
        print("Этот скрипт предназначен для запуска внутри PySM.", file=sys.stdout)
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    try:
        variable_data, data = validate_variable(config.var_name)
        schema_mode = is_schema_variable(config.var_name)
        schema = load_schema(config.var_name)
    except Exception as exc:
        show_startup_error(str(exc))
        sys.exit(1)

    window = JsonEditor(
        var_name=config.var_name,
        data=data,
        variable_data=variable_data,
        schema=schema,
        title=config.title,
        msg=config.msg,
        schema_mode=schema_mode,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
