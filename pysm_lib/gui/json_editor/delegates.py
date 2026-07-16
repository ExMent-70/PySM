"""Qt delegates and small dialogs used by the visual JSON editor."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
)

from ...pysm_icons import ICON_CATEGORIES, icons
from .model import Node, default_value_for_type


WIDGET_TYPES = ["string", "int", "float", "bool", "object", "array", "null"]


class ValueDelegate(QStyledItemDelegate):
    """Create type-aware editors for leaf JSON values."""

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
    """Collect the key and type for a new JSON child."""

    def __init__(self, parent_node: Node, parent=None):
        super().__init__(parent)
        self.parent_node = parent_node
        self.setWindowTitle("Новый элемент")

        layout = QFormLayout(self)
        self.key_edit = QLineEdit()
        self.type_box = QComboBox()
        self.type_box.addItems(WIDGET_TYPES)

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
