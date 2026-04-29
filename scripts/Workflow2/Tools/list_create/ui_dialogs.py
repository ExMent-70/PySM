"""
ui_dialogs.py
=============
Модуль содержит диалоговые окна приложения:
- Редактор списка услуг (ServicesEditorDialog)
- Редактор дополнительных услуг ученика (ExtraServicesDialog)
- Редактор словаря имен (NamesEditorDialog)
- Редактор схемы доп. информации (InfoSchemaEditorDialog)
- Редактор значений доп. информации (StudentInfoEditorDialog)
"""

from typing import Dict, List, Any, Tuple, Optional
import json
import pathlib

from PySide6.QtCore import Qt, QEvent  # Добавили QEvent
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QDialogButtonBox, QHeaderView,
    QMessageBox, QComboBox, QSpinBox, QLineEdit, QListWidget, QListWidgetItem,
    QInputDialog, QFormLayout, QLabel, QScrollArea, QPlainTextEdit, QFrame, QTabWidget, QTextEdit, QApplication 
)
from PySide6.QtGui import QKeySequence, QShortcut

from domain import ExtraService, Student
import io_services


class ServicesEditorDialog(QDialog):
    """
    Диалоговое окно для редактирования глобального списка услуг и их цен.
    """
    def __init__(self, services_data: Dict[str, int], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Редактор Услуг")
        self.setMinimumSize(450, 750)

        self.layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Название услуги", "Цена"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.layout.addWidget(self.table)
        
        self._populate_table(services_data)
        
        button_layout = QHBoxLayout()
        add_button = QPushButton("Добавить")
        remove_button = QPushButton("Удалить")
        add_button.clicked.connect(self._add_row)
        remove_button.clicked.connect(self._remove_row)
        
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        button_layout.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        button_layout.addWidget(self.button_box)
        self.layout.addLayout(button_layout)

    def _populate_table(self, data: Dict[str, int]) -> None:
        self.table.setRowCount(len(data))
        # ИЗМЕНЕНО: Сортировка ключей (названий услуг) по алфавиту
        for i, (name, cost) in enumerate(sorted(data.items())):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(str(cost)))

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("Новая услуга"))
        self.table.setItem(row, 1, QTableWidgetItem("0"))
        self.table.setCurrentCell(row, 0)

    def _remove_row(self) -> None:
        curr = self.table.currentRow()
        if curr >= 0:
            self.table.removeRow(curr)

    def get_services(self) -> Dict[str, int]:
        services = {}
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            cost_item = self.table.item(row, 1)
            
            if not name_item or not name_item.text().strip():
                raise ValueError(f"Название услуги в строке {row + 1} не может быть пустым.")
            
            name = name_item.text().strip()
            
            if not cost_item or not cost_item.text().strip():
                 raise ValueError(f"Цена для услуги '{name}' не может быть пустой.")
            
            try:
                cost = int(cost_item.text().strip())
                if cost < 0: raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(f"Цена для услуги '{name}' должна быть целым неотрицательным числом.")
            
            if name in services:
                raise ValueError(f"Название услуги '{name}' дублируется.")
            
            services[name] = cost
        return services


class ExtraServicesDialog(QDialog):
    """
    Диалоговое окно для редактирования списка дополнительных услуг.
    """
    def __init__(self, current_extras: List[ExtraService], services_dict: Dict[str, int], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Дополнительные услуги")
        self.setMinimumSize(600, 500)
        
        self.services_dict = services_dict or {}
        self.available_services = sorted(self.services_dict.keys())
        
        try:
            temp_dicts = [ex.to_dict() for ex in current_extras]
            self.extras_data = [ExtraService.from_dict(d) for d in temp_dicts]
        except Exception:
            self.extras_data = []

        self.layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Услуга", "Кол-во", "Цена (за шт.)", "Комментарий"])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 100)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.layout.addWidget(self.table)

        tb_layout = QHBoxLayout()
        add_btn = QPushButton("Добавить услугу")
        add_btn.clicked.connect(self._add_row)
        del_btn = QPushButton("Удалить выделенные")
        del_btn.clicked.connect(self._remove_row)
        tb_layout.addWidget(add_btn)
        tb_layout.addWidget(del_btn)
        tb_layout.addStretch()


        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        tb_layout.addWidget(self.button_box)
        self.layout.addLayout(tb_layout)

        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(0)
        for item in self.extras_data:
            self._add_row_ui(item)

    def _add_row(self):
        default_item = ExtraService(name="", qty=1, cost=0, comment="")
        self._add_row_ui(default_item)

    def _add_row_ui(self, item: ExtraService):
        row = self.table.rowCount()
        self.table.insertRow(row)

        combo = QComboBox()
        combo.setEditable(True) 
        combo.addItems(self.available_services)
        combo.setCurrentText(item.name)
        self.table.setCellWidget(row, 0, combo)

        sb_qty = QSpinBox()
        sb_qty.setRange(1, 999)
        sb_qty.setValue(item.qty)
        self.table.setCellWidget(row, 1, sb_qty)

        le_cost = QLineEdit()
        le_cost.setText(str(item.cost))
        le_cost.setPlaceholderText("0")
        self.table.setCellWidget(row, 2, le_cost)

        le_comment = QLineEdit(item.comment)
        self.table.setCellWidget(row, 3, le_comment)
        
        combo.currentTextChanged.connect(self._on_service_changed)

    def _on_service_changed(self, text: str):
        sender = self.sender()
        if not isinstance(sender, QComboBox): return

        target_row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 0) == sender:
                target_row = r
                break
        
        if target_row != -1:
            service_name = text.strip()
            if service_name in self.services_dict:
                price = self.services_dict[service_name]
                le_cost = self.table.cellWidget(target_row, 2)
                if isinstance(le_cost, QLineEdit):
                    le_cost.setText(str(price))

    def _remove_row(self):
        curr = self.table.currentRow()
        if curr >= 0: self.table.removeRow(curr)

    def get_data(self) -> List[ExtraService]:
        result = []
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 0)
            sb_qty = self.table.cellWidget(r, 1)
            le_cost = self.table.cellWidget(r, 2)
            le_comment = self.table.cellWidget(r, 3)
            
            if combo and sb_qty and le_cost:
                name = combo.currentText().strip()
                if not name: continue
                
                try:
                    cost_val = int(le_cost.text().strip())
                except ValueError:
                    cost_val = 0

                result.append(ExtraService(
                    name=name,
                    qty=sb_qty.value(),
                    cost=cost_val,
                    comment=le_comment.text() if le_comment else ""
                ))
        return result


class NamesEditorDialog(QDialog):
    """
    Диалоговое окно для редактирования словаря нормализации имен.
    """
    def __init__(self, names_data: Dict[str, str], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Редактор Словаря Имен")
        self.setMinimumSize(450, 750)

        self.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Сокращенное имя", "Полное имя"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.layout.addWidget(self.table)
        
        self._populate_table(names_data)
        
        button_layout = QHBoxLayout()
        add_button = QPushButton("Добавить")
        remove_button = QPushButton("Удалить")
        add_button.clicked.connect(self._add_row)
        remove_button.clicked.connect(self._remove_row)
        
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        button_layout.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        button_layout.addWidget(self.button_box)
        self.layout.addLayout(button_layout)

    def _populate_table(self, data: Dict[str, str]) -> None:
        self.table.setRowCount(len(data))
        # ИЗМЕНЕНО: Сортировка ключей (сокращенных имен) по алфавиту
        for i, (short, full) in enumerate(sorted(data.items())):
            self.table.setItem(i, 0, QTableWidgetItem(short))
            self.table.setItem(i, 1, QTableWidgetItem(full))

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("Сокращенное_Имя"))
        self.table.setItem(row, 1, QTableWidgetItem("Полное_Имя"))
        self.table.setCurrentCell(row, 0)

    def _remove_row(self) -> None:
        curr = self.table.currentRow()
        if curr >= 0: self.table.removeRow(curr)

    def get_names_dict(self) -> Dict[str, str]:
        names_dict = {}
        for row in range(self.table.rowCount()):
            short_item = self.table.item(row, 0)
            full_item = self.table.item(row, 1)
            
            if not short_item or not short_item.text().strip():
                raise ValueError(f"Сокращенное имя в строке {row + 1} не может быть пустым.")
            short = short_item.text().strip()
            
            if not full_item or not full_item.text().strip():
                 raise ValueError(f"Полное имя для '{short}' не может быть пустым.")
            full = full_item.text().strip()
            
            if short in names_dict:
                raise ValueError(f"Сокращенное имя '{short}' дублируется.")
            
            names_dict[short] = full
        return names_dict


# --- ИЗМЕНЕННЫЙ БЛОК 3: ui_dialogs.py (НОВЫЙ КЛАСС RanksEditorDialog) ---
class RanksEditorDialog(QDialog):
    """
    Диалоговое окно для редактирования списка рангов.
    """
    def __init__(self, ranks_data: List[str], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Редактор Рангов")
        self.setMinimumSize(400, 400)

        self.layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        
        # Сортируем и добавляем элементы, игнорируя возможные дубли
        for rank in sorted(set(ranks_data)):
            self._add_item_to_list(rank)
            
        self.layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        add_btn = QPushButton("Добавить")
        del_btn = QPushButton("Удалить")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._remove_row)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(del_btn)
        button_layout.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)

        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        #self.layout.addLayout(button_layout)
        #self.layout.addWidget(self.button_box)
        button_layout.addWidget(self.button_box)
        self.layout.addLayout(button_layout)        

    def _add_item_to_list(self, text: str):
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list_widget.addItem(item)

    def _add_row(self):
        self._add_item_to_list("Новый_ранг")
        item = self.list_widget.item(self.list_widget.count() - 1)
        self.list_widget.setCurrentItem(item)
        self.list_widget.editItem(item)

    def _remove_row(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def get_ranks(self) -> List[str]:
        ranks = set()
        for i in range(self.list_widget.count()):
            text = self.list_widget.item(i).text().strip()
            if text:
                ranks.add(text)
        return sorted(list(ranks))

# ==============================================================================
# НОВЫЕ КЛАССЫ ДЛЯ РАБОТЫ С ДОП. ИНФОРМАЦИЕЙ
# ==============================================================================

class InfoSchemaEditorDialog(QDialog):
    """
    Редактор схемы полей (заголовков) для дополнительной информации.
    """
# --- ИЗМЕНЕННЫЙ БЛОК: ui_dialogs.py (Весь метод __init__ в InfoSchemaEditorDialog) ---
    def __init__(self, current_columns: List[str], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Настройка полей информации")
        self.setMinimumSize(400, 300)

        self.rename_map: Dict[str, str] = {}
        
        self.layout = QVBoxLayout(self)
        main_layout = QHBoxLayout()
        self.layout.addLayout(main_layout)

        # 1. СНАЧАЛА создаем виджет
        self.list_widget = QListWidget()
        
        # 2. ЗАТЕМ добавляем в него отсортированные элементы
        for col in sorted(current_columns):
            self.list_widget.addItem(col)
            
        main_layout.addWidget(self.list_widget)

        btn_layout = QVBoxLayout()
        self.btn_add = QPushButton("Добавить")
        self.btn_rename = QPushButton("Переименовать")
        self.btn_remove = QPushButton("Удалить")
        self.btn_up = QPushButton("Вверх")
        self.btn_down = QPushButton("Вниз")

        self.btn_add.clicked.connect(self._add_field)
        self.btn_rename.clicked.connect(self._rename_field)
        self.btn_remove.clicked.connect(self._remove_field)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_rename)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_up)
        btn_layout.addWidget(self.btn_down)
        main_layout.addLayout(btn_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def _add_field(self):
        text, ok = QInputDialog.getText(self, "Новое поле", "Название поля:")
        if ok and text.strip():
            name = text.strip()
            if self.list_widget.findItems(name, Qt.MatchFlag.MatchExactly):
                QMessageBox.warning(self, "Ошибка", "Поле с таким названием уже существует.")
                return
            self.list_widget.addItem(name)

    def _rename_field(self):
        item = self.list_widget.currentItem()
        if not item: return
        
        old_name = item.text()
        text, ok = QInputDialog.getText(self, "Переименование", f"Новое название для '{old_name}':", text=old_name)
        if ok and text.strip():
            new_name = text.strip()
            if new_name == old_name: return
            
            if self.list_widget.findItems(new_name, Qt.MatchFlag.MatchExactly):
                QMessageBox.warning(self, "Ошибка", "Поле с таким названием уже существует.")
                return
            
            item.setText(new_name)
            
            original_key = None
            for key, val in self.rename_map.items():
                if val == old_name:
                    original_key = key
                    break
            
            if original_key:
                self.rename_map[original_key] = new_name
            else:
                self.rename_map[old_name] = new_name

    def _remove_field(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.takeItem(row)
            del item

    def _move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_widget.currentRow()
        if row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def get_result(self) -> Tuple[List[str], Dict[str, str]]:
        columns = []
        for i in range(self.list_widget.count()):
            columns.append(self.list_widget.item(i).text())
        return columns, self.rename_map


class StudentInfoEditorDialog(QDialog):
    """
    Редактор значений полей информации для конкретного студента.
    UI адаптирован под поддержку тем оформления.
    Навигация реализована через QShortcut (Ctrl+Left, Ctrl+Right).
    """
    def __init__(self, students: List[Student], current_index: int, info_columns: List[str], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Редактирование информации")
        self.resize(600, 450)
        
        self.students = students
        self.current_index = current_index
        self.info_columns = info_columns
        self.field_widgets: Dict[str, QPlainTextEdit] = {}
        # НОВОЕ: Хранилище изменений (Ключ - индекс строки, Значение - новый словарь info)
        self.staged_changes: Dict[int, Dict[str, str]] = {}

        # --- Основная верстка (без изменений) ---
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Шапка
        self.header_widget = QWidget()
        header_layout = QVBoxLayout(self.header_widget)
        header_layout.setContentsMargins(20, 15, 20, 0) 
        
        self.name_label = QLabel()
        font = self.name_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self.name_label.setFont(font)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.name_label)
        
        self.main_layout.addWidget(self.header_widget)

        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line1)

        # 2. Форма
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        form_widget = QWidget()
        form_container_layout = QVBoxLayout(form_widget)
        form_container_layout.setContentsMargins(20, 0, 20, 20)
        form_container_layout.setSpacing(15)

        self.form_layout = QFormLayout()
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.form_layout.setVerticalSpacing(15)
        self.form_layout.setHorizontalSpacing(15)
        
        for col in self.info_columns:
            text_edit = QPlainTextEdit()
            text_edit.setTabChangesFocus(True)
            text_edit.setMinimumHeight(70)
            text_edit.setMaximumHeight(120)
            
            label = QLabel(f"{col}:")
            l_font = label.font()
            l_font.setBold(True)
            label.setFont(l_font)
            
            self.field_widgets[col] = text_edit
            self.form_layout.addRow(label, text_edit)
            
        form_container_layout.addLayout(self.form_layout)
        form_container_layout.addStretch()
        scroll.setWidget(form_widget)
        self.main_layout.addWidget(scroll)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line2)

        # 3. Панель кнопок
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(15, 10, 15, 10)
        
        self.btn_prev = QPushButton("← Предыдущий")
        self.btn_next = QPushButton("Следующий →")
        self.btn_close = QPushButton("Закрыть")
        
        # Подсказки
        self.btn_prev.setToolTip("Ctrl + Стрелка Влево")
        self.btn_next.setToolTip("Ctrl + Стрелка Вправо")

        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next.clicked.connect(self._go_next)
        self.btn_close.clicked.connect(self.accept)

        bottom_layout.addWidget(self.btn_prev)
        bottom_layout.addWidget(self.btn_next)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_close)
        
        self.main_layout.addWidget(bottom_widget)

        # --- ИЗМЕНЕНИЕ: Глобальные шорткаты (работают даже если фокус в поле ввода) ---
        # Используем QShortcut вместо keyPressEvent, так как QPlainTextEdit перехватывает события
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Left), self)
        self.shortcut_prev.activated.connect(self._go_prev)

        self.shortcut_next = QShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Right), self)
        self.shortcut_next.activated.connect(self._go_next)

        self._load_student(self.current_index)

    def _load_student(self, index: int):
        if not (0 <= index < len(self.students)): return
        
        self.current_index = index
        student = self.students[index]
        self.name_label.setText(f"{student.surname} {student.name}")
        
        # Берем либо уже измененные данные, либо копию оригинальных
        current_info = self.staged_changes.get(index, student.info.copy())
        
        for col, widget in self.field_widgets.items():
            val = current_info.get(col, "")
            widget.setPlainText(val)
            
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < len(self.students) - 1)

    def _save_current(self):
        new_info = {}
        for col, widget in self.field_widgets.items():
            txt = widget.toPlainText().strip()
            if txt:
                new_info[col] = txt
        # Сохраняем изменения во временный словарь
        self.staged_changes[self.current_index] = new_info

    def get_changes(self) -> Dict[int, Dict[str, str]]:
        """НОВОЕ: Возвращает все накопленные изменения."""
        return self.staged_changes


    def _go_prev(self):
        self._save_current()
        self._load_student(self.current_index - 1)

    def _go_next(self):
        self._save_current()
        self._load_student(self.current_index + 1)

    def accept(self):
        self._save_current()
        super().accept()
        
class AIParsingDialog(QDialog):
    """
    Диалог для взаимодействия с AI (Gemini).
    Вкладка 1: Генерация промпта (вставка данных в шаблон).
    Вкладка 2: Импорт JSON-ответа от AI.
    """
    def __init__(self, students: List[Student], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("AI Обработка данных")
        self.resize(700, 500)
        self.students = students
        
        self.layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self._init_tab_generate()
        self._init_tab_import()
        
        # Кнопка закрытия общая для всех
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        self.layout.addWidget(self.btn_close)

    def _init_tab_generate(self):
        """Вкладка 1: Подготовка данных для отправки в чат."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("1. Вставьте неструктурированный текст (из чата/документа):"))
        self.raw_text_edit = QPlainTextEdit()
        self.raw_text_edit.setPlaceholderText("Пример:\nВася Пупкин: любит футбол, цитата 'Вперед!'\nМаша Иванова: любит танцы...")
        layout.addWidget(self.raw_text_edit)
        
        self.btn_copy_prompt = QPushButton("Сформировать промпт и скопировать в буфер")
        self.btn_copy_prompt.setStyleSheet("background-color: #e3f2fd; font-weight: bold; padding: 10px;")
        self.btn_copy_prompt.clicked.connect(self._generate_and_copy)
        layout.addWidget(self.btn_copy_prompt)
        
        layout.addWidget(QLabel("Предпросмотр того, что скопировано:"))
        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setStyleSheet("color: #666; background-color: #f5f5f5;")
        layout.addWidget(self.preview_edit)
        
        self.tabs.addTab(tab, "1. Генерация запроса")

    def _init_tab_import(self):
        """Вкладка 2: Обработка ответа."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("2. Вставьте JSON-ответ от Gemini:"))
        self.json_input_edit = QPlainTextEdit()
        self.json_input_edit.setPlaceholderText('[{"surname": "...", "name": "...", "info": {...}}, ...]')
        layout.addWidget(self.json_input_edit)
        
        self.btn_apply = QPushButton("Распознать и Обновить список")
        self.btn_apply.setStyleSheet("background-color: #e8f5e9; font-weight: bold; padding: 10px;")
        self.btn_apply.clicked.connect(self._apply_json)
        layout.addWidget(self.btn_apply)
        
        self.tabs.addTab(tab, "2. Импорт ответа")

    def _generate_and_copy(self):
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            QMessageBox.warning(self, "Внимание", "Введите текст с данными об учениках.")
            return

        # 1. Формируем мини-JSON (только идентификаторы и info)
        mini_list = []
        for s in self.students:
            mini_list.append({
                "surname": s.surname,
                "name": s.name,
                "info": s.info
            })
        
        json_str = json.dumps(mini_list, ensure_ascii=False, indent=2)
        
        # 2. Получаем шаблон
        template = io_services.get_ai_prompt_template(pathlib.Path.cwd()) # Или путь скрипта
        
        # 3. Заменяем плейсхолдеры
        final_prompt = template.replace("{{STUDENT_LIST_JSON}}", json_str)
        final_prompt = final_prompt.replace("{{RAW_TEXT}}", raw_text)
        
        # 4. Копируем
        clipboard = QApplication.clipboard()
        clipboard.setText(final_prompt)
        
        self.preview_edit.setPlainText(final_prompt)
        QMessageBox.information(self, "Готово", "Промпт скопирован в буфер обмена!\nВставьте его в чат Gemini.")

    def _apply_json(self):
        json_text = self.json_input_edit.toPlainText().strip()
        if not json_text: return

        # Очистка от markdown (```json ... ```) если нейросеть добавила их
        import re
        match = re.search(r'\[.*\]', json_text, re.DOTALL)
        if match:
            json_text = match.group(0)
        
        try:
            imported_data = json.loads(json_text)
            if not isinstance(imported_data, list):
                raise ValueError("Ожидался список объектов (list).")
            
            updated_count = 0
            not_found_list = []

            # Создаем карту для быстрого поиска: (Фамилия_lower, Имя_lower) -> Student
            student_map = {
                (s.surname.strip().lower(), s.name.strip().lower()): s 
                for s in self.students
            }

            for item in imported_data:
                s_surname = item.get("surname", "").strip()
                s_name = item.get("name", "").strip()
                new_info = item.get("info", {})
                
                key = (s_surname.lower(), s_name.lower())
                
                if key in student_map:
                    target_student = student_map[key]
                    # Обновляем info (merge)
                    if isinstance(new_info, dict):
                        target_student.info.update(new_info)
                        updated_count += 1
                else:
                    not_found_list.append(f"{s_surname} {s_name}")

            msg = f"Успешно обновлено учеников: {updated_count}"
            if not_found_list:
                msg += f"\n\nНе найдено в списке ({len(not_found_list)}):\n" + "\n".join(not_found_list[:10])
                if len(not_found_list) > 10: msg += "\n..."
            
            QMessageBox.information(self, "Результат", msg)
            
            # Если успешно, можно закрыть (или оставить для проверки)
            # self.accept() 

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Ошибка JSON", f"Некорректный формат JSON:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки:\n{e}")        