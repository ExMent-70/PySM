# pysm_lib/gui/widgets/_editors/template_widget.py

import json
import pathlib
from typing import Callable, Optional, Dict, Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget,
    QTreeWidgetItem, QMessageBox, QAbstractItemView,
    QDialog, QLabel, QLineEdit, QDialogButtonBox, QCheckBox
)

from ....pysm_icons import icons


class CustomInputDialog(QDialog):
    """
    Кастомный диалог ввода текста для полной поддержки QSS-тем.
    """
    def __init__(self, title: str, label: str, parent: Optional[QWidget] = None, initial_text: str = ""):
        super().__init__(parent)
        self.setObjectName("TemplateInputDialog")
        self.setWindowTitle(title)
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        
        self.label = QLabel(label)
        layout.addWidget(self.label)
        
        self.line_edit = QLineEdit()
        self.line_edit.setText(initial_text)
        if initial_text:
            self.line_edit.selectAll()
        layout.addWidget(self.line_edit)
        
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.button_box)
        
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def get_text(self) -> str:
        return self.line_edit.text()


class TemplateManagerWidget(QWidget):
    """
    Виджет для управления текстовыми шаблонами (сохранение/загрузка из JSON).
    Поддерживает иерархическую структуру категорий.
    """
    # Сигнал передает текст шаблона (str) и флаг замены текста (bool)
    template_applied = Signal(str, bool)

    def __init__(
        self,
        templates_file_path: pathlib.Path,
        get_current_text_func: Callable[[], str],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.templates_file_path = templates_file_path
        self.get_current_text_func = get_current_text_func

        self.templates_file_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_ui()
        self._load_templates()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. ДЕРЕВО ИЕРАРХИИ (Сверху, занимает всё доступное место)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.tree, 1)

        # 2. ЧЕКБОКС РЕЖИМА ВСТАВКИ
        self.chk_replace = QCheckBox("Заменять существующий текст")
        self.chk_replace.setToolTip("Если отмечено, шаблон полностью заменит текст. Иначе - добавится в конец.")
        self.chk_replace.setChecked(False) # По умолчанию - добавлять в конец
        layout.addWidget(self.chk_replace)

        # 3. ПАНЕЛЬ ИНСТРУМЕНТОВ (В самом низу виджета)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(5)
        
        self.btn_add_folder = QPushButton("")
        self.btn_add_folder.setIcon(icons.get_qicon("FOLDER"))
        self.btn_add_folder.setToolTip("Создать новую категорию")

        self.btn_add_template = QPushButton("")
        self.btn_add_template.setIcon(icons.get_qicon("FILE"))
        self.btn_add_template.setToolTip("Сохранить текущий текст как шаблон")
        
        self.btn_rename = QPushButton("")
        self.btn_rename.setIcon(icons.get_qicon("SETTINGS"))
        self.btn_rename.setToolTip("Переименовать выбранный элемент")

        self.btn_delete = QPushButton("")
        self.btn_delete.setIcon(icons.get_qicon("DELETE"))
        self.btn_delete.setToolTip("Удалить выбранный элемент")

        btn_layout.addWidget(self.btn_add_folder)
        btn_layout.addWidget(self.btn_add_template)
        btn_layout.addWidget(self.btn_rename)
        btn_layout.addWidget(self.btn_delete)
        
        layout.addLayout(btn_layout)

        # Сигналы
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_add_template.clicked.connect(self._on_add_template)
        self.btn_rename.clicked.connect(self._on_rename_item)
        self.btn_delete.clicked.connect(self._on_delete_item)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemSelectionChanged.connect(self._update_buttons_state)
        
        self._update_buttons_state()

    def _update_buttons_state(self):
        has_selection = bool(self.tree.selectedItems())
        self.btn_rename.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)

    def _load_templates(self):
        self.tree.clear()
        if not self.templates_file_path.exists():
            return
        try:
            with open(self.templates_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._populate_tree(self.tree.invisibleRootItem(), data)
            self.tree.expandAll()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить шаблоны:\n{e}")

    def _save_templates(self):
        data = self._build_dict_from_tree(self.tree.invisibleRootItem())
        try:
            with open(self.templates_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить шаблоны:\n{e}")

    def _populate_tree(self, parent_item: QTreeWidgetItem, data_dict: Dict[str, Any]):
        for key, value in data_dict.items():
            item = QTreeWidgetItem(parent_item)
            item.setText(0, key)
            if isinstance(value, dict):
                item.setData(0, Qt.ItemDataRole.UserRole, "folder")
                item.setIcon(0, icons.get_qicon("FOLDER"))
                self._populate_tree(item, value)
            else:
                item.setData(0, Qt.ItemDataRole.UserRole, "template")
                item.setData(0, Qt.ItemDataRole.UserRole + 1, str(value))
                item.setIcon(0, icons.get_qicon("FILE_TXT"))
                preview = str(value)
                if len(preview) > 300:
                    preview = preview[:300] + "..."
                item.setToolTip(0, preview)

    def _build_dict_from_tree(self, parent_item: QTreeWidgetItem) -> Dict[str, Any]:
        result = {}
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            key = child.text(0)
            node_type = child.data(0, Qt.ItemDataRole.UserRole)
            if node_type == "folder":
                result[key] = self._build_dict_from_tree(child)
            else:
                result[key] = child.data(0, Qt.ItemDataRole.UserRole + 1)
        return result

    @Slot()
    def _on_add_folder(self):
        dialog = CustomInputDialog("Новая категория", "Имя категории:", self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.get_text().strip()
        if not name:
            return
        parent_item = self._get_target_parent_for_new_item()
        for i in range(parent_item.childCount()):
            if parent_item.child(i).text(0) == name:
                QMessageBox.warning(self, "Ошибка", "Элемент с таким именем уже существует!")
                return
        new_item = QTreeWidgetItem(parent_item)
        new_item.setText(0, name)
        new_item.setData(0, Qt.ItemDataRole.UserRole, "folder")
        new_item.setIcon(0, icons.get_qicon("FOLDER"))
        if parent_item != self.tree.invisibleRootItem():
            parent_item.setExpanded(True)
        self._save_templates()

    @Slot()
    def _on_add_template(self):
        text_to_save = self.get_current_text_func()
        if not text_to_save.strip():
            QMessageBox.information(self, "Инфо", "Текстовое поле пустое. Нечего сохранять.")
            return
        dialog = CustomInputDialog("Новый шаблон", "Имя шаблона:", self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.get_text().strip()
        if not name:
            return
        parent_item = self._get_target_parent_for_new_item()
        for i in range(parent_item.childCount()):
            if parent_item.child(i).text(0) == name:
                QMessageBox.warning(self, "Ошибка", "Элемент с таким именем уже существует!")
                return
        new_item = QTreeWidgetItem(parent_item)
        new_item.setText(0, name)
        new_item.setData(0, Qt.ItemDataRole.UserRole, "template")
        new_item.setData(0, Qt.ItemDataRole.UserRole + 1, text_to_save)
        new_item.setIcon(0, icons.get_qicon("FILE_TXT"))
        preview = text_to_save[:300] + "..." if len(text_to_save) > 300 else text_to_save
        new_item.setToolTip(0, preview)
        if parent_item != self.tree.invisibleRootItem():
            parent_item.setExpanded(True)
        self._save_templates()

    @Slot()
    def _on_rename_item(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        old_name = item.text(0)
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        title = "Переименование категории" if node_type == "folder" else "Переименование шаблона"
        
        dialog = CustomInputDialog(title, "Новое имя:", self, initial_text=old_name)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = dialog.get_text().strip()
        if not new_name or new_name == old_name:
            return
        parent_item = item.parent() or self.tree.invisibleRootItem()
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child != item and child.text(0) == new_name:
                QMessageBox.warning(self, "Ошибка", "Элемент с таким именем уже существует!")
                return
        item.setText(0, new_name)
        self._save_templates()

    def _get_target_parent_for_new_item(self) -> QTreeWidgetItem:
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return self.tree.invisibleRootItem()
        item = selected_items[0]
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type == "folder":
            return item
        else:
            return item.parent() or self.tree.invisibleRootItem()

    @Slot()
    def _on_delete_item(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        reply = QMessageBox.question(
            self, "Подтверждение", 
            f"Вы уверены, что хотите удалить '{item.text(0)}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            parent = item.parent() or self.tree.invisibleRootItem()
            parent.removeChild(item)
            self._save_templates()

    @Slot(QTreeWidgetItem, int)
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type == "template":
            text = item.data(0, Qt.ItemDataRole.UserRole + 1)
            # Передаем состояние чек-бокса вместе с текстом
            self.template_applied.emit(text, self.chk_replace.isChecked())