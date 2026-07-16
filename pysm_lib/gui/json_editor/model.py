"""Qt item model and data helpers for the visual JSON editor."""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt
from PySide6.QtWidgets import QMessageBox

from ...pysm_icons import icons


logger = logging.getLogger(__name__)

DEFAULT_ICON_BY_TYPE = {
    "dict": "SETTINGS",
    "list": "LIST",
    "str": "FILE_TXT",
    "int": "SLIDERS",
    "float": "SLIDERS",
    "bool": "OK",
    "NoneType": "INFO",
}


def json_type_name(value: Any) -> str:
    """Return the editor type name for a Python JSON value."""

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
    """Return a new empty value for an editor type."""

    return {
        "string": "",
        "int": 0,
        "float": 0.0,
        "bool": False,
        "object": {},
        "array": [],
        "null": None,
    }[type_name]


def build_default_schema(title: str, data: Any) -> dict[str, Any]:
    """Build a basic display schema by walking JSON data."""

    schema: dict[str, Any] = {
        "version": 1,
        "title": title,
        "fields": {},
    }

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child_value in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                schema["fields"][child_path] = {
                    "label": str(key),
                    "description": "",
                    "widget": json_type_name(child_value),
                    "icon": DEFAULT_ICON_BY_TYPE.get(
                        type(child_value).__name__,
                        "FILE_CODE",
                    ),
                }
                walk(child_value, child_path)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                child_path = f"{path}.{index}" if path else str(index)
                walk(child_value, child_path)

    walk(data)
    return schema


class SchemaIndex:
    """Lookup adapter for optional field display metadata."""

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
    """Mutable tree node that mirrors one JSON value."""

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

    def rebuild_children(self) -> None:
        self.children = []
        if isinstance(self.value, dict):
            for key, value in self.value.items():
                self.children.append(Node(key, value, self))
        elif isinstance(self.value, list):
            for index, value in enumerate(self.value):
                self.children.append(Node(index, value, self))

    def refresh_paths(self) -> None:
        self.path = self._build_path()
        for child in self.children:
            child.refresh_paths()

    def is_leaf(self) -> bool:
        return not isinstance(self.value, (dict, list))


class JsonModel(QAbstractItemModel):
    """Editable Qt tree model for dictionaries and lists."""

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
            for index, child in enumerate(parent.children):
                child.key = index
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
