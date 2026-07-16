"""Window and public orchestration API for the visual JSON editor."""

import copy
import html
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from PySide6.QtCore import (
    QEventLoop,
    QModelIndex,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ... import pysm_context, theme_api
from ...context_variable_ops import read_context_value, write_context_value
from ...pysm_icons import icons
from .delegates import NewItemDialog, ValueDelegate
from .model import JsonModel, Node, build_default_schema


__all__ = [
    "JsonEditor",
    "JsonEditorResult",
    "JsonEditorStatus",
    "create_json_editor",
    "edit_json_variable",
    "show_json_editor_error",
]


SCHEMA_SUFFIX = "__schema"
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

logger = logging.getLogger(__name__)


class JsonEditorStatus(str, Enum):
    """How the editor session finished."""

    SAVED = "saved"
    UNCHANGED = "unchanged"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JsonEditorResult:
    """Result returned to a script that opened the visual editor."""

    status: JsonEditorStatus
    var_name: str
    value: Any
    changed: bool

    @property
    def saved(self) -> bool:
        """Return whether the user confirmed the value."""

        return self.status in (JsonEditorStatus.SAVED, JsonEditorStatus.UNCHANGED)


def load_context_variable(
    var_name: str,
    context: Any = None,
) -> tuple[Optional[dict[str, Any]], Any]:
    """Load a top-level or dotted context variable with its metadata."""

    context = pysm_context if context is None else context
    variable_data = context.get_variable(var_name) if context else None
    if not isinstance(variable_data, dict):
        result = read_context_value(context, var_name)
        if not result.exists or "." not in var_name:
            return None, None

        base_var_name = var_name.split(".", 1)[0]
        base_variable_data = context.get_variable(base_var_name)
        if not isinstance(base_variable_data, dict):
            return None, None

        value = result.value
        nested_variable_data = {
            "type": "json" if isinstance(value, (dict, list)) else base_variable_data.get("type"),
            "value": value,
            "description": base_variable_data.get("description"),
            "read_only": result.read_only,
            "choices": base_variable_data.get("choices"),
        }
        return nested_variable_data, value

    return variable_data, variable_data.get("value")


def schema_var_name_for(var_name: str) -> str:
    return f"{var_name}{SCHEMA_SUFFIX}"


def target_name_for_schema(schema_var_name: str) -> str:
    return schema_var_name[: -len(SCHEMA_SUFFIX)]


def is_schema_variable(var_name: str) -> bool:
    return var_name.endswith(SCHEMA_SUFFIX)


class JsonEditor(QMainWindow):
    """Window for editing one JSON context variable."""

    finished = Signal(object)

    def __init__(
        self,
        var_name: str,
        data: Any,
        variable_data: dict[str, Any],
        schema: Optional[dict[str, Any]],
        title: str,
        msg: str = "",
        schema_mode: bool = False,
        context: Any = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.var_name = var_name
        self.variable_data = variable_data
        self.schema_mode = schema_mode
        self.msg = msg
        self.context = pysm_context if context is None else context
        self.saved = False
        self.force_close = False
        self.dirty = False
        self.result: Optional[JsonEditorResult] = None
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
        schema_data, _ = load_context_variable(schema_name, self.context)
        if schema_data and QMessageBox.question(
            self,
            "Схема уже существует",
            f"Перезаписать {schema_name} автоматически созданной схемой?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        schema = build_default_schema(self.var_name, self.model.to_python())
        write_context_value(self.context, schema_name, schema, commit=True)
        QMessageBox.information(self, "OK", f"Схема сохранена в {schema_name}.")

    def save(self):
        try:
            data = self.current_data()
            changed = data != self.original_data
            if changed:
                write_context_value(self.context, self.var_name, data, commit=True)
                self.original_data = copy.deepcopy(data)
            self.saved = True
            self.dirty = False
            if not changed:
                QMessageBox.information(
                    self,
                    f"Сохранение переменной {self.var_name}",
                    f"Значение переменной {self.var_name} не изменилось.\n"
                    "Сохранение отменено",
                )

            status = JsonEditorStatus.SAVED if changed else JsonEditorStatus.UNCHANGED
            self._complete(status, data, changed)
            self.close()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def cancel(self):
        if self.confirm_discard():
            self.force_close = True
            self._complete(
                JsonEditorStatus.CANCELLED,
                copy.deepcopy(self.original_data),
                False,
            )
            self.close()

    def _complete(self, status: JsonEditorStatus, value: Any, changed: bool) -> None:
        """Publish the session result exactly once."""

        if self.result is not None:
            return
        self.result = JsonEditorResult(
            status,
            self.var_name,
            copy.deepcopy(value),
            changed,
        )
        self.finished.emit(self.result)

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
            self.force_close = True
            self._complete(
                JsonEditorStatus.CANCELLED,
                copy.deepcopy(self.original_data),
                False,
            )
            event.accept()
        else:
            event.ignore()


def validate_variable(
    var_name: str,
    context: Any = None,
) -> tuple[dict[str, Any], Any]:
    """Validate that a context path can be edited as JSON."""

    variable_data, value = load_context_variable(var_name, context)
    if not variable_data:
        raise ValueError(f"Переменная '{var_name}' не найдена в контексте.")
    if variable_data.get("type") != "json":
        raise ValueError(f"Переменная '{var_name}' имеет тип '{variable_data.get('type')}', а нужен 'json'.")
    if variable_data.get("read_only", False):
        raise ValueError(f"Переменная '{var_name}' защищена от записи.")
    if not isinstance(value, (dict, list)):
        raise ValueError(f"Значение переменной '{var_name}' должно быть объектом или массивом JSON.")
    return variable_data, value


def load_schema(var_name: str, context: Any = None) -> Optional[dict[str, Any]]:
    """Load the optional ``<var_name>__schema`` context variable."""

    if is_schema_variable(var_name):
        return None

    schema_name = schema_var_name_for(var_name)
    variable_data, value = load_context_variable(schema_name, context)
    if not variable_data:
        return None
    if variable_data.get("type") != "json" or not isinstance(value, dict):
        logger.warning("Ignoring invalid schema variable %s", schema_name)
        return None
    return value


def show_json_editor_error(message: str, *, apply_theme: bool = True) -> None:
    """Show a themed startup error without terminating the caller."""

    app = QApplication.instance() or QApplication([])
    if apply_theme:
        theme_api.apply_theme_to_app(app)
    QMessageBox.critical(None, "Ошибка", message)


def create_json_editor(
    var_name: str,
    *,
    title: str = "Редактор JSON",
    message: str = "",
    context: Any = None,
    parent: Optional[QWidget] = None,
) -> JsonEditor:
    """Create a validated editor window without starting an event loop."""

    context = pysm_context if context is None else context
    variable_data, data = validate_variable(var_name, context)
    schema_mode = is_schema_variable(var_name)
    schema = load_schema(var_name, context)
    return JsonEditor(
        var_name=var_name,
        data=data,
        variable_data=variable_data,
        schema=schema,
        title=title,
        msg=message,
        schema_mode=schema_mode,
        context=context,
        parent=parent,
    )


def edit_json_variable(
    var_name: str,
    *,
    title: str = "Редактор JSON",
    message: str = "",
    context: Any = None,
    parent: Optional[QWidget] = None,
    apply_theme: bool = True,
) -> JsonEditorResult:
    """Open the visual editor and return the user's save/cancel decision.

    A local event loop is used so the function neither starts nor terminates the
    host application's main Qt event loop.
    """

    app = QApplication.instance() or QApplication([])
    if apply_theme:
        theme_api.apply_theme_to_app(app)

    window = create_json_editor(
        var_name,
        title=title,
        message=message,
        context=context,
        parent=parent,
    )
    event_loop = QEventLoop()
    result_holder: list[JsonEditorResult] = []

    def finish(result: JsonEditorResult) -> None:
        result_holder.append(result)
        event_loop.quit()

    window.finished.connect(finish)
    window.show()
    event_loop.exec()

    if result_holder:
        return result_holder[0]
    return JsonEditorResult(
        JsonEditorStatus.CANCELLED,
        var_name,
        copy.deepcopy(window.current_data()),
        False,
    )
