#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTreeView,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
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
    parser.add_argument("--var_name", type=str, required=True)
    parser.add_argument("--title", type=str, default="JSON Editor")

    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()

    return parser.parse_args()


# ==============================================================================
# NODE MODEL
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
# MODEL (QAbstractItemModel)
# ==============================================================================
class JsonModel(QAbstractItemModel):
    def __init__(self, data):
        super().__init__()
        self.root = Node("root", data)

    def columnCount(self, parent):
        return 3  # key, value, type

    def rowCount(self, parent):
        node = self.get_node(parent)
        return len(node.children)

    def index(self, row, column, parent):
        node = self.get_node(parent)
        if 0 <= row < len(node.children):
            return self.createIndex(row, column, node.children[row])
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

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == 0:
                return str(node.key)
            elif index.column() == 1:
                if isinstance(node.value, dict):
                    return "<object>"
                if isinstance(node.value, list):
                    return "<array>"
                return str(node.value)
            elif index.column() == 2:
                return type(node.value).__name__

        return None

    def flags(self, index):
        node = index.internalPointer()
        if index.column() == 1 and node.is_leaf():
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def setData(self, index, value, role):
        if role != Qt.EditRole:
            return False

        node = index.internalPointer()

        try:
            old = node.value

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
            return False

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        headers = ["Ключ", "Значение", "Тип"]
        return headers[section]

    def get_node(self, index):
        if index.isValid():
            return index.internalPointer()
        return self.root

    # ==============================================================================
    # BUILD JSON BACK
    # ==============================================================================
    def to_dict(self, node=None):
        if node is None:
            node = self.root

        if isinstance(node.value, dict):
            return {child.key: self.to_dict(child) for child in node.children}

        elif isinstance(node.value, list):
            return [self.to_dict(child) for child in node.children]

        else:
            return node.value


# ==============================================================================
# MAIN WINDOW
# ==============================================================================
class JsonEditor(QMainWindow):
    def __init__(self, var_name, data):
        super().__init__()

        self.var_name = var_name
        self.setWindowTitle(var_name)
        self.resize(900, 600)

        self.model = JsonModel(data)

        self.view = QTreeView()
        self.view.setModel(self.model)
        self.view.expandAll()

        central = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.view)

        btns = QHBoxLayout()
        btn_save = QPushButton("Сохранить")
        btn_cancel = QPushButton("Отмена")

        btn_save.clicked.connect(self.save)
        btn_cancel.clicked.connect(self.close)

        btns.addWidget(btn_save)
        btns.addWidget(btn_cancel)

        layout.addLayout(btns)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def save(self):
        try:
            data = self.model.to_dict()
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

    if data is None:
        logger.error("Переменная не найдена")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

    win = JsonEditor(config.var_name, data)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()