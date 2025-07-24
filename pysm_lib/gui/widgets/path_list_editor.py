# pysm_lib/gui/widgets/path_list_editor.py

import pathlib
import os  # <--- НОВЫЙ ИМПОРТ
import sys  # <--- НОВЫЙ ИМПОРТ
import subprocess  # <--- НОВЫЙ ИМПОРТ
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QLineEdit,
    QListWidgetItem,
    QStyle,
)

# --- ИЗМЕНЕНИЕ: Добавлен импорт Signal ---
from PySide6.QtCore import Slot, Signal

from ...locale_manager import LocaleManager


class PathListEditor(QWidget):
    # --- ИЗМЕНЕНИЕ: Добавлен новый сигнал ---
    data_changed = Signal()

    def __init__(
        self,
        locale_manager: LocaleManager,
        dialog_title: str,
        allow_editing: bool = True,
        default_dialog_path: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.dialog_title = dialog_title
        self._allow_editing = allow_editing
        self._default_dialog_path = default_dialog_path or str(pathlib.Path.home())

        self._init_ui()
        self._connect_signals()
        self._update_buttons_state()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        main_layout.addWidget(self.list_widget, 1)
        buttons_layout = QHBoxLayout()
        self.btn_add = QPushButton(
            self.locale_manager.get("path_list_editor.buttons.add")
        )
        self.btn_edit = QPushButton(
            self.locale_manager.get("path_list_editor.buttons.edit")
        )
        self.btn_remove = QPushButton(
            self.locale_manager.get("path_list_editor.buttons.remove")
        )
        self.btn_edit.setVisible(self._allow_editing)
        buttons_layout.addWidget(self.btn_add)
        if self._allow_editing:
            buttons_layout.addWidget(self.btn_edit)
        buttons_layout.addWidget(self.btn_remove)
        buttons_layout.addStretch()
        main_layout.addLayout(buttons_layout)

    def _connect_signals(self):
        self.btn_add.clicked.connect(self._add_path)
        self.btn_edit.clicked.connect(self._edit_path)
        self.btn_remove.clicked.connect(self._remove_path)
        self.list_widget.itemSelectionChanged.connect(self._update_buttons_state)

        # Если редактирование разрешено, двойной клик открывает редактор.
        # Иначе - открывает папку в проводнике.
        if self._allow_editing:
            self.list_widget.doubleClicked.connect(self._edit_path)
        else:
            self.list_widget.doubleClicked.connect(self._on_item_double_clicked)

    # --- БЛОК 1: Новый вспомогательный метод для создания элемента списка ---
    def _create_list_item(self, path_str: str) -> QListWidgetItem:
        """Создает QListWidgetItem с текстом и стандартной иконкой папки."""
        item = QListWidgetItem(path_str)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirClosedIcon)
        item.setIcon(icon)
        return item

    # --- БЛОК 2: Новый слот для обработки двойного клика в режиме read-only ---
    @Slot()
    def _on_item_double_clicked(self):
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            return
        self._open_folder_in_explorer(selected_item.text())

    # --- БЛОК 3: Новый вспомогательный метод для открытия папки ---
    def _open_folder_in_explorer(self, path: str):
        try:
            if not os.path.isdir(path):
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Путь не является директорией или не существует:\n{path}",
                )
                return

            if sys.platform == "win32":
                os.startfile(os.path.realpath(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось открыть папку '{path}':\n{e}"
            )

    def _update_buttons_state(self):
        has_selection = bool(self.list_widget.selectedItems())
        self.btn_edit.setEnabled(has_selection)
        self.btn_remove.setEnabled(has_selection)

    @Slot()
    def _add_path(self):
        directory = QFileDialog.getExistingDirectory(
            self, self.dialog_title, self._default_dialog_path
        )
        if directory:
            items = self.get_paths()
            if directory not in items:
                # Создаем элемент с иконкой
                item = self._create_list_item(directory)
                self.list_widget.addItem(item)
                self.data_changed.emit()
            else:
                QMessageBox.information(
                    self,
                    self.locale_manager.get("path_list_editor.duplicate_dialog.title"),
                    self.locale_manager.get(
                        "path_list_editor.duplicate_dialog.text", path=directory
                    ),
                )

    @Slot()
    def _edit_path(self):
        if not self._allow_editing:
            return
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            return
        old_text = selected_item.text()
        text, ok = QInputDialog.getText(
            self,
            self.locale_manager.get("path_list_editor.edit_dialog.title"),
            self.locale_manager.get("path_list_editor.edit_dialog.label"),
            QLineEdit.EchoMode.Normal,
            old_text,
        )
        if ok and text.strip() and text.strip() != old_text:
            current_row = self.list_widget.row(selected_item)
            items = [
                self.list_widget.item(i).text()
                for i in range(self.list_widget.count())
                if i != current_row
            ]
            if text.strip() not in items:
                selected_item.setText(text.strip())
                self.data_changed.emit()  # --- ИЗМЕНЕНИЕ ---
            else:
                QMessageBox.warning(
                    self,
                    self.locale_manager.get("path_list_editor.duplicate_dialog.title"),
                    self.locale_manager.get(
                        "path_list_editor.duplicate_dialog.text", path=text.strip()
                    ),
                )

    @Slot()
    def _remove_path(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.list_widget.takeItem(self.list_widget.row(item))
        self.data_changed.emit()  # --- ИЗМЕНЕНИЕ ---

    def get_paths(self) -> List[str]:
        return [
            self.list_widget.item(i).text().strip()
            for i in range(self.list_widget.count())
        ]

    # --- БЛОК 3: Метод set_paths изменен для использования нового хелпера ---
    def set_paths(self, paths: List[str]):
        self.list_widget.clear()
        for path in paths:
            # Создаем и добавляем каждый элемент с иконкой
            item = self._create_list_item(path)
            self.list_widget.addItem(item)
