#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTreeView, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox,
    QMenu, QDialog, QFormLayout, QLineEdit, QComboBox,
    QStyledItemDelegate, QSpinBox, QDoubleSpinBox, QCheckBox, QHeaderView
)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_icons import icons
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    IS_MANAGED_RUN = False

    class MockThemeApi:
        @staticmethod
        def apply_theme_to_app(app): pass

    theme_api = MockThemeApi()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIG
# ==============================================================================
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var_name", required=True)

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


# ==============================================================================
# NODE
# ==============================================================================
class Node:
    def __init__(self, key, value, parent=None):
        self.key = key
        self.value = value
        self.parent = parent
        self.children = []

        if isinstance(value, dict):
            for k, v in value.items():
                self.children.append(Node(k, v, self))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.children.append(Node(i, v, self))

    def is_leaf(self):
        return not isinstance(self.value, (dict, list))


# ==============================================================================
# MODEL
# ==============================================================================
class JsonModel(QAbstractItemModel):
    def __init__(self, data):
        super().__init__()
        self.root = Node("root", data)

    def columnCount(self, parent):
        return 3

    def rowCount(self, parent):
        node = self.get_node(parent)
        return len(node.children)

    def index(self, row, col, parent):
        node = self.get_node(parent)
        if 0 <= row < len(node.children):
            return self.createIndex(row, col, node.children[row])
        return QModelIndex()

    def parent(self, index):
        node = index.internalPointer()
        if not node or not node.parent or node.parent == self.root:
            return QModelIndex()

        parent = node.parent
        grand = parent.parent

        if grand:
            row = grand.children.index(parent)
            return self.createIndex(row, 0, parent)

        return QModelIndex()

    def data(self, index, role):
        if not index.isValid():
            return None

        node = index.internalPointer()


        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return str(node.key)

            if index.column() == 1:
                if isinstance(node.value, dict):
                    return "<object>"
                if isinstance(node.value, list):
                    return "<array>"
                return str(node.value)

            if index.column() == 2:
                return type(node.value).__name__

        return None

    def flags(self, index):
        node = index.internalPointer()

        if index.column() == 1 and node.is_leaf():
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role):
        if role != Qt.EditRole:
            return False

        node = index.internalPointer()
        old = node.value

        try:
            if isinstance(old, bool):
                node.value = value.lower() in ("true", "1", "yes")
            elif isinstance(old, int):
                node.value = int(value)
            elif isinstance(old, float):
                node.value = float(value)
            else:
                node.value = value

            self.dataChanged.emit(index, index)
            return True

        except:
            QMessageBox.warning(None, "Ошибка", "Некорректное значение")
            return False

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        return ["Ключ", "Значение", "Тип"][section]

    def get_node(self, index):
        return index.internalPointer() if index.isValid() else self.root

    def to_dict(self, node=None):
        node = node or self.root

        if isinstance(node.value, dict):
            return {child.key: self.to_dict(child) for child in node.children}

        if isinstance(node.value, list):
            return [self.to_dict(child) for child in node.children]

        return node.value


# ==============================================================================
# DELEGATE
# ==============================================================================
class ValueDelegate(QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        node = index.internalPointer()

        if isinstance(node.value, bool):
            return QCheckBox(parent)

        if isinstance(node.value, int):
            w = QSpinBox(parent)
            w.setMaximum(10**9)
            return w

        if isinstance(node.value, float):
            w = QDoubleSpinBox(parent)
            w.setDecimals(4)
            w.setMaximum(10**9)
            return w

        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        val = index.internalPointer().value

        if isinstance(editor, QCheckBox):
            editor.setChecked(val)
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.setValue(val)
        else:
            editor.setText(str(val))

    def setModelData(self, editor, model, index):
        node = index.internalPointer()

        if isinstance(editor, QCheckBox):
            node.value = editor.isChecked()
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            node.value = editor.value()
        else:
            node.value = editor.text()

        model.dataChanged.emit(index, index)


# ==============================================================================
# DIALOG
# ==============================================================================
class NewItemDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Новый элемент")

        layout = QFormLayout()

        self.key_edit = QLineEdit()
        self.type_box = QComboBox()
        self.type_box.addItems(["string", "int", "float", "bool", "object", "array"])

        layout.addRow("Имя:", self.key_edit)
        layout.addRow("Тип:", self.type_box)

        btn = QPushButton("Создать")
        btn.clicked.connect(self.validate)

        layout.addWidget(btn)
        self.setLayout(layout)

    def validate(self):
        if not self.key_edit.text().strip():
            QMessageBox.warning(self, "Ошибка", "Имя ключа не может быть пустым")
            return
        self.accept()

    def get_data(self):
        key = self.key_edit.text().strip()
        t = self.type_box.currentText()

        default = {
            "string": "",
            "int": 0,
            "float": 0.0,
            "bool": False,
            "object": {},
            "array": []
        }[t]

        return key, default


# ==============================================================================
# MAIN WINDOW
# ==============================================================================
class JsonEditor(QMainWindow):
    def __init__(self, var_name, data):
        super().__init__()

        self.var_name = var_name
        self.setWindowTitle(f"Редактирование переменной {var_name}")
        self.resize(700, 600)

        self.model = JsonModel(data)

        self.view = QTreeView()
        self.view.setModel(self.model)
        self.view.expandAll()
        
        header = self.view.header()

        # режимы ресайза
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Ключ
        header.setSectionResizeMode(1, QHeaderView.Interactive)      # Значение
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Тип

        # начальные ширины
        self.view.setColumnWidth(0, 150)  # ~15-20%
        self.view.setColumnWidth(1, 400)   # минимально под тип        

        self.view.setAlternatingRowColors(True)
        self.view.setUniformRowHeights(True)
        self.view.setStyleSheet("QTreeView::item { height: 30px; }")
        self.view.setItemDelegateForColumn(1, ValueDelegate())

        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.menu)

        central = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.view)

        btns = QHBoxLayout()

        btn_add_root = QPushButton("Добавить")
        btn_add_root.clicked.connect(lambda: self.create_node(self.model.root))

        btn_save = QPushButton("Сохранить")
        btn_cancel = QPushButton("Отмена")

        btn_save.clicked.connect(self.save)
        btn_cancel.clicked.connect(self.close)

        btns.addWidget(btn_add_root)
        btns.addStretch()
        btns.addWidget(btn_save)
        btns.addWidget(btn_cancel)

        layout.addLayout(btns)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def menu(self, pos):
        index = self.view.indexAt(pos)
        if not index.isValid():
            return

        node = index.internalPointer()
        menu = QMenu()

        if isinstance(node.value, (dict, list)):
            menu.addAction("Добавить в узел...", lambda: self.create_node(node))

        menu.addAction("Удалить", lambda: self.delete_node(node))

        menu.exec(self.view.viewport().mapToGlobal(pos))

    def create_node(self, node):
        dlg = NewItemDialog()
        if not dlg.exec():
            return

        key, value = dlg.get_data()

        parent_index = self.model.createIndex(
            node.parent.children.index(node), 0, node
        ) if node.parent else QModelIndex()

        row = len(node.children)

        self.model.beginInsertRows(parent_index, row, row)

        if isinstance(node.value, dict):
            node.value[key] = value
            new_node = Node(key, value, node)

        elif isinstance(node.value, list):
            node.value.append(value)
            new_node = Node(len(node.children), value, node)

        else:
            self.model.endInsertRows()
            return

        node.children.append(new_node)

        self.model.endInsertRows()

    def delete_node(self, node):
        if node == self.model.root:
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Удалить элемент?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        parent = node.parent
        if not parent:
            return

        try:
            row = parent.children.index(node)
        except ValueError:
            return

        parent_index = self.model.createIndex(
            parent.parent.children.index(parent), 0, parent
        ) if parent.parent else QModelIndex()

        self.model.beginRemoveRows(parent_index, row, row)

        if isinstance(parent.value, dict):
            parent.value.pop(node.key, None)

        elif isinstance(parent.value, list):
            parent.value.pop(row)

        parent.children.pop(row)

        if isinstance(parent.value, list):
            for i, child in enumerate(parent.children):
                child.key = i

        self.model.endRemoveRows()

    def save(self):
        data = self.model.to_dict()

        try:
            pysm_context.set_structured(self.var_name, data)
            QMessageBox.information(self, "OK", "Сохранено")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    config = get_config()

    if not IS_MANAGED_RUN:
        print("Только внутри PySM")
        sys.exit(1)

    data = pysm_context.get_structured(config.var_name)

    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    win = JsonEditor(config.var_name, data)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()