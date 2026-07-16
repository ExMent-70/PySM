"""
ui_models.py
============
Модуль содержит модели данных Qt (QAbstractTableModel) и делегаты.
"""

import sys
import re
from typing import List, Dict, Any, Optional

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QEvent, Signal, QTimer, QSortFilterProxyModel
)
from PySide6.QtGui import (
    QBrush, QColor
)
from PySide6.QtWidgets import (
    QStyledItemDelegate, QComboBox, QLineEdit, QAbstractItemDelegate, QMessageBox
)

from .domain import Student, ExtraService


class ValidationError(Exception):
    """Исключение, возникающее при ошибках валидации данных в модели."""
    pass


class StudentTableModel(QAbstractTableModel):
    """
    Модель таблицы для отображения списка учеников (Student).
    Поддерживает динамические колонки для доп. информации.
    """

    # Индексы фиксированных колонок
    COL_SHOOT_ORDER = 0
    COL_ALPHA_ORDER = 1
    COL_SURNAME = 2
    COL_NAME = 3
    COL_PATRONYMIC = 4   # НОВОЕ: Отчество
    COL_RANK = 5         # НОВОЕ: Ранг
    COL_SERVICE = 6
    COL_EXTRAS = 7
    COL_TOTAL = 8
    
    # Количество фиксированных колонок
    FIXED_COL_COUNT = 9

    # Колонки, доступные для редактирования (фиксированные)
    EDITABLE_FIXED_COLUMNS = {
        COL_SHOOT_ORDER, COL_SURNAME, COL_NAME, 
        COL_PATRONYMIC, COL_RANK, COL_SERVICE
    }
    


    # Паттерн валидации (обновленный)
    NAME_VALIDATION_PATTERN = re.compile(r"^[A-Za-zА-Яа-яЁё\s\-]+$")

    def __init__(self, 
                 data: List[Student] = None, 
                 services: Dict[str, int] = None,
                 ranks: List[str] = None,
                 surname_style: Dict[str, str] = None, 
                 name_style: Dict[str, str] = None,
                 base_bg_color: QColor = None, 
                 alternate_bg_color: QColor = None):
        super().__init__()
        self._data: List[Student] = data or []
        self.services = services or {}
        self.ranks = ranks or ["ученик"]
        self.info_columns: List[str] = [] # Список заголовков доп. полей

        self._base_headers =["№ съемки", "№ п/п", "Фамилия", "Имя", "Отчество", "Ранг", "Услуга", "Доп.", "Итого"]

        # Настройка стилей
        surname_style = surname_style or {}
        name_style = name_style or {}
        
        self.base_bg_brush = QBrush(base_bg_color) if base_bg_color else QBrush()
        self.alternate_bg_brush = QBrush(alternate_bg_color) if alternate_bg_color else QBrush()
        self._bg_brush_cache: Dict[str, QBrush] = {}

    def set_info_columns(self, columns: List[str]):
        """Обновляет список динамических колонок и перерисовывает таблицу."""
        self.beginResetModel()
        self.info_columns = columns
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        # Фиксированные + Динамические
        return self.FIXED_COL_COUNT + len(self.info_columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section < self.FIXED_COL_COUNT:
                return self._base_headers[section]
            else:
                # Заголовок динамической колонки
                idx = section - self.FIXED_COL_COUNT
                if 0 <= idx < len(self.info_columns):
                    return self.info_columns[idx]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
            
        row = index.row()
        col = index.column()
        student = self._data[row]
        
        if role in[Qt.DisplayRole, Qt.EditRole]:
            # Обработка фиксированных колонок
            if col == self.COL_SHOOT_ORDER:
                return str(student.shoot_order) if student.shoot_order is not None else ""
            elif col == self.COL_ALPHA_ORDER:
                return str(student.alpha_order)
            elif col == self.COL_SURNAME:
                return student.surname
            elif col == self.COL_NAME:
                return student.name
            elif col == self.COL_PATRONYMIC:   # НОВОЕ
                return student.patronymic
            elif col == self.COL_RANK:         # НОВОЕ
                return student.rank
            elif col == self.COL_SERVICE:
                return student.service_type
            elif col == self.COL_EXTRAS:
                return f"{len(student.extra_services)} шт." if student.extra_services else ""
            elif col == self.COL_TOTAL:
                return str(student.total_cost)
            
            # Обработка динамических колонок (Доп. информация)
            elif col >= self.FIXED_COL_COUNT:
                idx = col - self.FIXED_COL_COUNT
                if 0 <= idx < len(self.info_columns):
                    key = self.info_columns[idx]
                    return student.info.get(key, "")
            
            return ""
        
        if role == Qt.BackgroundRole:
            color_hex = None
            if col == self.COL_SURNAME:
                color_hex = student.color1
            elif col == self.COL_NAME:
                color_hex = student.color2
            
            if color_hex:
                if color_hex in self._bg_brush_cache:
                    return self._bg_brush_cache[color_hex]
                brush = QBrush(QColor(color_hex))
                self._bg_brush_cache[color_hex] = brush
                return brush
            else:
                return self.alternate_bg_brush if row % 2 else self.base_bg_brush
        
        elif role == Qt.ForegroundRole:
            color_hex_fg = None
            if col == self.COL_SURNAME:
                color_hex_fg = student.color1_fg
            elif col == self.COL_NAME:
                color_hex_fg = student.color2_fg
            
            if color_hex_fg:
                return QBrush(QColor(color_hex_fg))
            
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False

        row = index.row()
        col = index.column()
        student = self._data[row]
        
        try:
            # Фиксированные колонки
            if col == self.COL_SHOOT_ORDER:
                val_str = str(value).strip()
                if not val_str:
                    student.shoot_order = None
                else:
                    new_val = int(value)
                    if new_val <= 0: raise ValidationError("Номер съемки должен быть > 0.")
                    # ИЗМЕНЕНО: Оптимизированная проверка уникальности через set (хэш-таблицу)
                    used_orders = {s.shoot_order for i, s in enumerate(self._data) if i != row and s.shoot_order is not None}
                    if new_val in used_orders:
                        raise ValidationError(f"Номер '{new_val}' уже используется.")
                    student.shoot_order = new_val

            elif col == self.COL_SURNAME:
                val_str = str(value).strip()
                if not val_str or not self.NAME_VALIDATION_PATTERN.match(val_str):
                    raise ValidationError("Только буквы, пробелы и дефисы.")
                student.surname = val_str

            elif col == self.COL_NAME:
                val_str = str(value).strip()
                if not val_str or not self.NAME_VALIDATION_PATTERN.match(val_str):
                    raise ValidationError("Только буквы, пробелы и дефисы.")
                student.name = val_str
                
            elif col == self.COL_PATRONYMIC:  # НОВОЕ
                val_str = str(value).strip()
                # Отчество может быть пустым, проверяем только если ввели текст
                if val_str and not self.NAME_VALIDATION_PATTERN.match(val_str):
                    raise ValidationError("Только буквы, пробелы и дефисы.")
                student.patronymic = val_str
                
            elif col == self.COL_RANK:        # НОВОЕ
                student.rank = str(value).strip()
                
            elif col == self.COL_SERVICE:
                new_service_name = str(value)
                if (cost := self.services.get(new_service_name)) is not None:
                    student.service_type = new_service_name
                    student.service_cost = cost
                    self.dataChanged.emit(index, self.index(row, self.COL_TOTAL), [role])
                else:
                    return False

            # Динамические колонки
            elif col >= self.FIXED_COL_COUNT:
                idx = col - self.FIXED_COL_COUNT
                if 0 <= idx < len(self.info_columns):
                    key = self.info_columns[idx]
                    val_str = str(value).strip()
                    if val_str:
                        student.info[key] = val_str
                    else:
                        # Если строка пустая, удаляем ключ
                        if key in student.info:
                            del student.info[key]
            else:
                return False

        except (ValueError, TypeError, ValidationError) as e:
            raise ValidationError(str(e)) from e

        self.dataChanged.emit(index, index, [role])
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        col = index.column()
        # Разрешаем редактирование фиксированных (кроме Итого и Доп) и всех динамических
        if col in self.EDITABLE_FIXED_COLUMNS or col >= self.FIXED_COL_COUNT:
            flags |= Qt.ItemIsEditable
        return flags


        if column not in self.SORTABLE_COLUMNS and column < self.FIXED_COL_COUNT:
            return
        
        self.layoutAboutToBeChanged.emit()
        is_reverse = (order == Qt.SortOrder.DescendingOrder)
        
        if column == self.COL_SURNAME:
            self._data.sort(key=lambda x: (x.surname, x.name), reverse=is_reverse)
            self._renumber_alpha_order_internal()
        elif column == self.COL_SHOOT_ORDER:
            self._data.sort(key=lambda x: x.shoot_order if x.shoot_order is not None else float('inf'), reverse=is_reverse)
        elif column == self.COL_TOTAL:
            self._data.sort(key=lambda x: x.total_cost, reverse=is_reverse)
        
        # Сортировка по динамической колонке
        elif column >= self.FIXED_COL_COUNT:
            idx = column - self.FIXED_COL_COUNT
            if 0 <= idx < len(self.info_columns):
                key = self.info_columns[idx]
                self._data.sort(key=lambda x: x.info.get(key, ""), reverse=is_reverse)

        self.layoutChanged.emit()


    def sort_and_renumber(self):
        """
        Принудительно сортирует исходные данные по алфавиту 
        и пересчитывает порядковые номера (№ п/п).
        Вызывается после парсинга, загрузки или изменения количества строк.
        """
        self.layoutAboutToBeChanged.emit()
        self._data.sort(key=lambda x: (x.surname.lower(), x.name.lower()))
        for i, student in enumerate(self._data, 1):
            student.alpha_order = i
        self.layoutChanged.emit()

    def swap_name_surname(self, row: int) -> bool:
        if 0 <= row < len(self._data):
            s = self._data[row]
            s.name, s.surname = s.surname, s.name
            s.color1, s.color2 = s.color2, s.color1
            s.color1_fg, s.color2_fg = s.color2_fg, s.color1_fg
            self.dataChanged.emit(self.index(row, self.COL_SURNAME), self.index(row, self.COL_NAME))
            return True
        return False


    def update_data(self, data: List[Student]):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def get_all_data(self) -> List[Student]:
        return self._data

    def insert_row(self, r: int, student: Student):
        self.beginInsertRows(QModelIndex(), r, r)
        self._data.insert(r, student)
        self.endInsertRows()

    def remove_rows(self, rows: List[int]):
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self._data):
                self.beginRemoveRows(QModelIndex(), r, r)
                del self._data[r]
                self.endRemoveRows()

    def _renumber_alpha_order_internal(self):
        for i, student in enumerate(self._data, 1):
            student.alpha_order = i

    def update_all_services(self, s_type: str, s_cost: int):
        for student in self._data:
            student.service_type = s_type
            student.service_cost = s_cost
        if self.rowCount() > 0:
            self.dataChanged.emit(self.index(0, self.COL_SERVICE), self.index(self.rowCount() - 1, self.COL_TOTAL))

    def update_info_bulk(self, changes: Dict[int, Dict[str, str]]):
        """Безопасно применяет изменения доп. информации и уведомляет View."""
        for row, new_info in changes.items():
            if 0 <= row < len(self._data):
                self._data[row].info = new_info
                
        # Если были изменения, сообщаем таблице, что динамические колонки нужно перерисовать
        if changes and len(self.info_columns) > 0:
            start_row = min(changes.keys())
            end_row = max(changes.keys())
            start_idx = self.index(start_row, self.FIXED_COL_COUNT)
            end_idx = self.index(end_row, self.columnCount() - 1)
            self.dataChanged.emit(start_idx, end_idx, [Qt.DisplayRole])
            
    def update_extras(self, row: int, extras: List[ExtraService]):
        if 0 <= row < len(self._data):
            self._data[row].extra_services = extras
            idx_extras = self.index(row, self.COL_EXTRAS)
            idx_total = self.index(row, self.COL_TOTAL)
            self.dataChanged.emit(idx_extras, idx_total, [Qt.DisplayRole])


class EnterKeyDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, services: List[str] = None, ranks: List[str] = None):
        super().__init__(parent)
        # ИЗМЕНЕНО: Сортируем списки для комбобоксов
        self.services = sorted(services) if services else[]
        self.ranks = sorted(ranks) if ranks else[]

# --- ИЗМЕНЕННЫЙ БЛОК: ui_models.py (Метод createEditor в EnterKeyDelegate) ---
    def createEditor(self, parent, option, index):
        col = index.column()
        
        # Комбобокс для Услуг
        if col == StudentTableModel.COL_SERVICE:
            editor = QComboBox(parent)
            editor.addItems(self.services)
            
        # НОВОЕ: Комбобокс для Рангов
        elif col == StudentTableModel.COL_RANK:
            editor = QComboBox(parent)
            editor.addItems(self.ranks)
            
        # Обычное текстовое поле для остальных редактируемых колонок
        else:
            editor = QLineEdit(parent)
        
        editor.installEventFilter(self)
        return editor

    def setEditorData(self, editor, index):
        val = index.model().data(index, Qt.EditRole)
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(val))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(val))

    def setModelData(self, editor, model, index):
        view = self.parent()
        value = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
        
        try:
            model.setData(index, value, Qt.EditRole)
        except ValidationError as e:
            QMessageBox.warning(view, "Ошибка валидации", str(e))
            QTimer.singleShot(0, lambda: view.edit(index))

    def eventFilter(self, editor, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Up, Qt.Key_Down):
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, QAbstractItemDelegate.EndEditHint.NoHint)
                if key in (Qt.Key_Up, Qt.Key_Down):
                    view = self.parent()
                    idx = view.currentIndex()
                    next_row = idx.row() + (-1 if key == Qt.Key_Up else 1)
                    if 0 <= next_row < idx.model().rowCount():
                        view.setCurrentIndex(idx.model().index(next_row, idx.column()))
                return True
        return super().eventFilter(editor, event)
        
# --- ИЗМЕНЕННЫЙ БЛОК 2: ui_models.py (Новый класс StudentProxyModel) ---
class StudentProxyModel(QSortFilterProxyModel):
    """
    Прокси-модель для умной сортировки (понимает где числа, а где текст).
    """
    def lessThan(self, source_left: QModelIndex, source_right: QModelIndex) -> bool:
        col = source_left.column()
        model = self.sourceModel()
        
        # Получаем оригинальные объекты Student для честного сравнения (числа с числами)
        student_left = model._data[source_left.row()]
        student_right = model._data[source_right.row()]
        
        if col == model.COL_SHOOT_ORDER:
            val_l = student_left.shoot_order if student_left.shoot_order is not None else float('inf')
            val_r = student_right.shoot_order if student_right.shoot_order is not None else float('inf')
            return val_l < val_r
        elif col == model.COL_ALPHA_ORDER:
            return student_left.alpha_order < student_right.alpha_order
        elif col == model.COL_SURNAME or col == model.COL_NAME:
            return (student_left.surname, student_left.name) < (student_right.surname, student_right.name)
        elif col == model.COL_TOTAL:
            return student_left.total_cost < student_right.total_cost
            
        return super().lessThan(source_left, source_right)
