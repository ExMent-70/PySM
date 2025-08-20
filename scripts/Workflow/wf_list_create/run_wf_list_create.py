#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_wf_list_create.py
=====================
Модуль для создания и редактирования списков учеников с поддержкой интеллектуального
разбора имен и фамилий с использованием Natasha, а также простого парсинга по шаблону.
Предоставляет графический интерфейс на основе PySide6 для управления списками.
"""

# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import csv
import json
import os
import pathlib
import regex as re
import sys
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple

# Опциональные зависимости
try: 
    from natasha import NamesExtractor
except ImportError: 
    NamesExtractor = None

try: 
    from bs4 import BeautifulSoup
except ImportError: 
    BeautifulSoup = None

try: 
    import pymorphy3
except ImportError: 
    pymorphy3 = None

try: 
    from ipymarkup.palette import Palette, Color, material 
    from ipymarkup.span import format_span_box_markup
except ImportError: 
    format_span_box_markup, Palette, Color, material = None, None, None, None

try: 
    from pysm_lib import pysm_context 
    from pysm_lib import theme_api
    from pysm_lib.pysm_context import ConfigResolver 
    IS_MANAGED_RUN = True
except ImportError: 
    IS_MANAGED_RUN, pysm_context, ConfigResolver = False, None, None

try:
    from PySide6.QtCore import (Qt, QAbstractTableModel, QModelIndex, QEvent, Signal, QUrl, QPoint, QTimer)
    from PySide6.QtGui import (QAction, QKeySequence, QDesktopServices, QBrush, QColor)
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                   QSplitter, QLabel, QLineEdit, QTextEdit, QTableView, QPushButton, QHeaderView,
                                   QComboBox, QMenu, QStyle, QTabWidget, QTextBrowser, QStyledItemDelegate,
                                   QAbstractItemDelegate, QFileDialog, QMessageBox)
except ImportError: print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr); sys.exit(1)
try: import jinja2
except ImportError: jinja2 = None

# Константы
CHILDREN_LIST_FILENAME = "children.txt"

class ValidationError(Exception):
    """Исключение, возникающее при ошибках валидации данных в модели."""
    pass

def smart_capitalize(text: str) -> str:
    """Умно капитализирует строку, обрабатывая слова с дефисами или пробелами."""
    def capitalize_word(word: str) -> str:
        if not word: return ""
        return word[0].upper() + word[1:]
    by_hyphen = ['-'.join(capitalize_word(part) for part in part.split('-')) for part in text.split()]
    return ' '.join(by_hyphen)

def get_raw_config() -> Namespace:
    """Получает конфигурацию из аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Редактор списков классов для фотографа", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--wf_dest_dir", type=str, help="Директория назначения.")
    parser.add_argument("--wf_output_txt_file", type=str, help="Опциональный путь для сохранения файла children.txt.")
    parser.add_argument("--wf_autosave_formats", type=str, nargs='+', choices=["html", "txt", "csv"], default=["html", "txt"], help="Форматы для автосохранения.")
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def simple_parse_text(text: str) -> List[Dict[str, Any]]:
    """Извлекает имена и фамилии с помощью простого regex."""
    parsed_data: List[Dict[str, Any]] = []
    lines = text.splitlines()
    pattern = re.compile(r"^[ \d.\-)]*([\p{L}-]+)[ ,]+([\p{L}-]+)", flags=re.UNICODE)
    for line in lines:
        line = line.strip()
        if not line: continue
        match = pattern.search(line)
        if match:
            surname, name = match.groups()
            parsed_data.append({"surname": surname, "name": name})
    return parsed_data

# 3. БЛОК: "Умный" парсер
# ==============================================================================
class SmartParser:
    def __init__(self):
        self.morph, self.extractor, self.palette = None, None, None
        self.normalization_dict = {}
        # Явные HEX-коды для надежной работы с Qt
        self.SURNAME_COLOR_HEX = "#e3f2fd"
        self.NAME_COLOR_HEX = "#e8f5e9"
        
        if Palette and Color and material:
            try:
                # Используем material() только для ipymarkup, который это понимает
                surname_color = Color('Фамилия', background=material('Blue', '50'), border=material('Blue', '100'), text=material('Blue', '900'))
                name_color = Color('Имя', background=material('Green', '50'), border=material('Green', '100'), text=material('Green', '900'))
                fio_color = Color('ФИО', background=material('Orange', '50'), border=material('Orange', '100'), text=material('Orange', '900'))
                self.palette = Palette([surname_color, name_color, fio_color])
            except Exception as e: print(f"Ошибка палитры: {e}", file=sys.stderr)
        
        if pymorphy3 and NamesExtractor:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                self.extractor = NamesExtractor(self.morph)
            except Exception as e: print(f"Ошибка Natasha: {e}", file=sys.stderr)
        self._load_normalization_dict()

    def _load_normalization_dict(self):
        dict_path = pathlib.Path(__file__).parent / "_names_normalization.json"
        default_dict = {"Саша": "Александр", "Аня": "Анна", "Настя": "Анастасия"}
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f: self.normalization_dict = json.load(f)
            except Exception: self.normalization_dict = default_dict
        else:
            try:
                with open(dict_path, 'w', encoding='utf-8') as f: json.dump(default_dict, f, ensure_ascii=False, indent=4)
                self.normalization_dict = default_dict
            except Exception: self.normalization_dict = default_dict

    def _normalize_name(self, name: str) -> str:
        return self.normalization_dict.get(name.capitalize(), name.capitalize())

    def parse_text(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Извлекает имена и фамилии с помощью Natasha, генерирует HTML-разметку,
        а затем парсит эту разметку для извлечения цветов, чтобы гарантировать
        полное соответствие между предпросмотром и таблицей.
        """
        if not self.extractor or not self.morph:
            return [], "<p style='color: red;'>Парсер не инициализирован.</p>"

        # ШАГ 1: Основной парсинг для извлечения имен и фамилий
        parsed_data = []
        spans = []
        matches = self.extractor(text)
        
        for match in matches:
            fact = match.fact
            if not (fact.first and fact.last): continue
            if "доп" in fact.first.lower() or "фото" in fact.first.lower(): continue

            p_first = self.morph.parse(fact.first)[0]
            p_last = self.morph.parse(fact.last)[0]
            
            if ('Name' in p_last.tag and 'Surn' in p_first.tag) or ('Surn' in p_first.tag and 'Surn' not in p_last.tag):
                surname, name = fact.first, fact.last
                span1_label, span2_label = " (Фамилия)", " (Имя)"
            else:
                surname, name = fact.last, fact.first
                span1_label, span2_label = " (Имя)", " (Фамилия)"

            parsed_data.append({
                "surname": smart_capitalize(surname), 
                "name": smart_capitalize(self._normalize_name(name))
            })
            
            try:
                match_text = text[match.start:match.stop]
                s_match = re.search(re.escape(surname), match_text, re.IGNORECASE)
                n_match = re.search(re.escape(name), match_text, re.IGNORECASE)
                if s_match and n_match:
                    spans_to_add = ((s_match, span1_label), (n_match, span2_label)) if span1_label == " (Фамилия)" else ((n_match, span1_label), (s_match, span2_label))
                    spans.append((match.start + spans_to_add[0][0].start(), match.start + spans_to_add[0][0].end(), spans_to_add[0][1]))
                    spans.append((match.start + spans_to_add[1][0].start(), match.start + spans_to_add[1][0].end(), spans_to_add[1][1]))
                else:
                    spans.append((match.start, match.stop, "ФИО"))
            except Exception:
                spans.append((match.start, match.stop, "ФИО"))

        # ШАГ 2: Генерация HTML-разметки
        markup_html = ""
        if format_span_box_markup and self.palette:
            try: markup_html = "".join(format_span_box_markup(text, spans, palette=self.palette))
            except Exception as e: print(f"IPYMARKUP ERROR: {e}")
        
        # ШАГ 3: Парсинг HTML для извлечения цветов (надежный метод)
        if BeautifulSoup and markup_html:
            background_regex = re.compile(r"background:\s*([^;]+)")
            soup = BeautifulSoup(markup_html, 'html.parser')
            all_color_spans = soup.find_all('span', style=background_regex)
            
            # Проверяем, что количество найденных span'ов соответствует количеству людей (по 2 на каждого)
            if len(all_color_spans) == len(parsed_data) * 2:
                for i, person_data in enumerate(parsed_data):
                    span1 = all_color_spans[i * 2]
                    span2 = all_color_spans[i * 2 + 1]

                    match1 = background_regex.search(span1.get('style', ''))
                    color1 = match1.group(1).strip() if match1 else None
                    
                    match2 = background_regex.search(span2.get('style', ''))
                    color2 = match2.group(1).strip() if match2 else None
                    
                    # Присваиваем цвета в том порядке, в котором они идут в HTML
                    person_data['color1'] = color1
                    person_data['color2'] = color2
            else:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Рассинхронизация парсера. Найдено людей: {len(parsed_data)}, цветных блоков: {len(all_color_spans)}. Раскраска отменена.")

        return parsed_data, markup_html

# 4. БЛОК: Модель данных
# ==============================================================================
class StudentTableModel(QAbstractTableModel):
    row_focus_requested = Signal(int, int)
    COL_SHOOT_ORDER, COL_ALPHA_ORDER, COL_SURNAME, COL_NAME, COL_SERVICE, COL_COST = range(6)
    EDITABLE_COLUMNS = {COL_SHOOT_ORDER, COL_SURNAME, COL_NAME, COL_SERVICE, COL_COST}
    SORTABLE_COLUMNS = {COL_SHOOT_ORDER, COL_SURNAME}
    NAME_VALIDATION_PATTERN = re.compile(r"^[\p{L}\s-]+\Z", flags=re.UNICODE)

    def __init__(self, data: List[Dict[str, Any]] = None, services: Dict[str, int] = None):
        super().__init__()
        self._data = data or []
        self.services = services or {}
        self._headers = ["№ съемки", "№ п/п", "Фамилия", "Имя", "Услуга", "Стоимость"]
        self._brush_cache: Dict[str, QBrush] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self._data)
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self._headers)
    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal: return self._headers[section]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid(): return None
        row, col, d = index.row(), index.column(), self._data[index.row()]
        if role in [Qt.DisplayRole, Qt.EditRole]:
            return d.get(["shoot_order", "alpha_order", "surname", "name", "service_type", "service_cost"][col], "")
        elif role == Qt.BackgroundRole:
            color_hex = d.get("color1") if col == self.COL_SURNAME else d.get("color2") if col == self.COL_NAME else None
            if color_hex:
                if color_hex in self._brush_cache: return self._brush_cache[color_hex]
                brush = QBrush(QColor(color_hex))
                self._brush_cache[color_hex] = brush
                return brush

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if role != Qt.EditRole or not index.isValid(): return False
        row, col, row_data = index.row(), index.column(), self._data[index.row()]
        try:
            if col == self.COL_SHOOT_ORDER:
                if not str(value).strip(): row_data['shoot_order'] = ""
                else:
                    new_val = int(value)
                    if new_val <= 0: raise ValidationError("Номер съемки должен быть > 0.")
                    if new_val > self.rowCount(): raise ValidationError(f"Номер не может быть > {self.rowCount()}.")
                    if any(i != row and s.get("shoot_order") == new_val for i, s in enumerate(self._data)): raise ValidationError(f"Номер '{new_val}' уже используется.")
                    row_data['shoot_order'] = new_val
            elif col in [self.COL_SURNAME, self.COL_NAME]:
                val_str = str(value).strip()
                if not val_str or not self.NAME_VALIDATION_PATTERN.match(val_str): raise ValidationError("Поле должно содержать только буквы и дефис.")
                row_data['surname' if col == self.COL_SURNAME else 'name'] = val_str
                if col == self.COL_SURNAME: self.sort_and_refocus(row, col)
            elif col == self.COL_SERVICE:
                if (cost := self.services.get(str(value))) is not None:
                    row_data['service_type'], row_data['service_cost'] = str(value), cost
                    self.dataChanged.emit(index, self.index(row, self.COL_COST), [role])
                else: return False
            elif col == self.COL_COST:
                new_cost = int(value)
                if new_cost < 0: raise ValidationError("Стоимость не может быть < 0.")
                row_data['service_cost'] = new_cost
            else: return False
        except (ValueError, TypeError, ValidationError) as e:
            raise ValidationError(str(e)) from e
        self.dataChanged.emit(index, index, [role])
        return True
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        if index.column() in self.EDITABLE_COLUMNS: flags |= Qt.ItemIsEditable
        return flags

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        if column not in self.SORTABLE_COLUMNS: return
        self.layoutAboutToBeChanged.emit()
        is_reverse = order == Qt.SortOrder.DescendingOrder
        if column == self.COL_SURNAME:
            self._data.sort(key=lambda x: (x.get("surname", ""), x.get("name", "")), reverse=is_reverse)
            self._renumber_alpha_order_internal()
        elif column == self.COL_SHOOT_ORDER:
            self._data.sort(key=lambda x: int(v) if (v := x.get("shoot_order")) and str(v).strip().isdigit() else float('inf'), reverse=is_reverse)
        self.layoutChanged.emit()

    def swap_name_surname(self, row: int) -> bool:
        if 0 <= row < len(self._data):
            d = self._data[row]
            d['name'], d['surname'] = d['surname'], d['name']
            if 'color1' in d and 'color2' in d:
                d['color1'], d['color2'] = d['color2'], d['color1']
            self.dataChanged.emit(self.index(row, self.COL_SURNAME), self.index(row, self.COL_NAME))
            return True
        return False

    def sort_and_refocus(self, row: int, col: int):
        if 0 <= row < len(self._data):
            item = self._data[row]
            self.sort(self.COL_SURNAME)
            try: self.row_focus_requested.emit(self._data.index(item), col)
            except ValueError: pass

    def update_data(self, data: List[Dict[str, Any]]):
        self.beginResetModel(); self._data = data; self.endResetModel()
    def get_all_data(self) -> List[Dict[str, Any]]: return self._data
    def insert_row(self, r: int, data: Dict[str, Any]):
        self.beginInsertRows(QModelIndex(), r, r); self._data.insert(r, data); self.endInsertRows()
    def remove_rows(self, rows: List[int]):
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self._data): self.beginRemoveRows(QModelIndex(), r, r); del self._data[r]; self.endRemoveRows()
    def _renumber_alpha_order_internal(self):
        for i, row in enumerate(self._data, 1): row["alpha_order"] = i
    def update_all_services(self, s_type: str, s_cost: int):
        for row in self._data: row["service_type"], row["service_cost"] = s_type, s_cost
        self.dataChanged.emit(self.index(0, self.COL_SERVICE), self.index(self.rowCount() - 1, self.COL_COST))

# 5. БЛОК: Кастомный делегат
# ==============================================================================
class EnterKeyDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, services: List[str] = None):
        super().__init__(parent)
        self.services = services or []
    def createEditor(self, parent, option, index):
        editor = QComboBox(parent) if index.column() == StudentTableModel.COL_SERVICE else QLineEdit(parent)
        if isinstance(editor, QComboBox): editor.addItems(self.services)
        editor.installEventFilter(self)
        return editor
    def setEditorData(self, editor, index):
        val = index.model().data(index, Qt.EditRole)
        if isinstance(editor, QComboBox): editor.setCurrentText(str(val))
        elif isinstance(editor, QLineEdit): editor.setText(str(val))
    def setModelData(self, editor, model, index):
        view = self.parent()
        value = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
        try: model.setData(index, value, Qt.EditRole)
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
                    if 0 <= next_row < idx.model().rowCount(): view.setCurrentIndex(idx.model().index(next_row, idx.column()))
                return True
        return super().eventFilter(editor, event)

# 6. БЛОК: Основное окно
# ==============================================================================
# 6. БЛОК: Основное окно
# ==============================================================================
class ClassListEditor(QMainWindow):
    """Главное окно приложения для редактирования списков учеников."""

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.config = config
        self._is_dirty: bool = False
        self._save_children: bool = False
        self.SERVICES: Dict[str, int] = {}
        self._load_services()
        self.smart_parser = SmartParser()
        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализирует и компонует все виджеты интерфейса."""
        self.setWindowTitle("PySM - Редактор списка класса")
        self.resize(1200, 800)
        
        self._create_menu()

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self._create_top_panel(main_layout)
        self._create_main_panels(main_layout)
        self.left_tabs.setStyleSheet(
            """
            /* Стиль для неактивной вкладки */
            QTabBar::tab:!selected {
                background-color: #ffc300;
                color: #000000;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }

            /* Стиль для неактивной вкладки при наведении */
            QTabBar::tab:!selected:hover {
                color: #ffffff;
                font: bold;
            }

            /* Стиль для активной (выбранной) вкладки */
            QTabBar::tab:selected {
                background-color: #ffffff;
                font: bold;                
                border: 1px solid #c0c0c0;
                border-bottom: 1px solid white; /* "Сливается" с фоном */
                margin-bottom: -1px; /* Сдвиг для эффекта слияния */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
            }

            """
        )        

        self.statusBar().showMessage("Готово")
        if self.config.wf_dest_dir:
            self.class_name_input.setText(pathlib.Path(self.config.wf_dest_dir).name)
            self._load_current_session()
        self._update_cost_label()
        self._update_summary_info()
    
    def _create_menu(self) -> None:
        """Создает и настраивает меню приложения."""
        file_menu = self.menuBar().addMenu("&Файл")
        actions = {
            "Загрузить список текущей сессии": self._load_current_session,
            "Загрузить список...": self._load_any_session,
            "-1": None,
            "Сохранить список": (self._save_list, QKeySequence.StandardKey.Save),
            "Сохранить список как...": lambda: self._save_list(save_as=True),
            "-2": None,
            "Сохранить HTML как...": lambda: self._save_html(save_as=True),
            "Сохранить как CSV...": self._save_csv,
            "-3": None,
            "Экспортировать для обработки": self._save_for_processing,
            "-4": None,
            "Печать HTML": (self._print_html, QKeySequence.StandardKey.Print),
            "-5": None,
            "Выход": self.close,
        }
        for name, handler in actions.items():
            if name.startswith("-"):
                file_menu.addSeparator()
                continue
            action = QAction(name, self)
            if isinstance(handler, tuple):
                action.triggered.connect(handler[0])
                action.setShortcut(handler[1])
            else:
                action.triggered.connect(handler)
            file_menu.addAction(action)

    def _create_top_panel(self, parent_layout: QVBoxLayout) -> None:
        """Создает верхнюю панель с названием класса и выбором услуги."""
        top_panel = QGridLayout()
        parent_layout.addLayout(top_panel)
        top_panel.addWidget(QLabel("Название класса:"), 0, 0)

        class_name_layout = QHBoxLayout()
        class_name_layout.setContentsMargins(0, 0, 0, 0)
        self.class_name_input = QLineEdit()
        self.open_folder_button = QPushButton(icon=self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_folder_button.setFixedSize(self.class_name_input.sizeHint().height(), self.class_name_input.sizeHint().height())
        self.open_folder_button.clicked.connect(self._open_session_folder)
        class_name_layout.addWidget(self.class_name_input)
        class_name_layout.addWidget(self.open_folder_button)
        top_panel.addLayout(class_name_layout, 0, 1)

        top_panel.addWidget(QLabel("Вид фотоуслуги:"), 0, 2)
        self.service_type_combo = QComboBox()
        self.service_type_combo.addItems(self.SERVICES.keys())
        top_panel.addWidget(self.service_type_combo, 0, 3)
        top_panel.addWidget(QLabel("Стоимость:"), 0, 4)
        self.service_cost_label = QLabel()
        self.service_cost_label.setStyleSheet("font-weight: bold;")
        top_panel.addWidget(self.service_cost_label, 0, 5)
        self.service_type_combo.currentIndexChanged.connect(self._update_cost_label)

    def _create_main_panels(self, parent_layout: QVBoxLayout) -> None:
        """Создает основную область с левой (ввод) и правой (таблица) панелями."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        parent_layout.addWidget(splitter)

        # Левая панель
        self.left_tabs = QTabWidget()
        self._create_input_tab()
        self._create_markup_tab()
        splitter.addWidget(self.left_tabs)
        
        # Правая панель
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Обработанный список (ПКМ для действий):"))
        self._create_table_view(right_layout)
        self._create_summary_panel(right_layout)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 800])

    def _create_input_tab(self) -> None:
        """Создает вкладку для ввода и парсинга текста."""
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        parser_mode_layout = QHBoxLayout()
        self.parser_mode_combo = QComboBox()
        self.parser_mode_combo.addItems(["Интеллектуальный (Natasha)", "Простой (по шаблону)"])
        parser_mode_layout.addWidget(QLabel("Режим разбора:"))
        parser_mode_layout.addWidget(self.parser_mode_combo)
        input_layout.addLayout(parser_mode_layout)

        self.raw_list_input = QTextEdit()
        self.raw_list_input.setHtml(
            "<b>ИНСТРУКЦИЯ:</b><br><br>"
            "1. Скопируйте и вставьте сюда текст со списком учеников.<br><br>"
            "2. Нажмите кнопку <b>Обработать текст</b>.<br><br>"
            "3. В таблице справа появится отсортированный список."
        )
        input_layout.addWidget(self.raw_list_input)
        self.process_button = QPushButton("Обработать текст")
        self.process_button.clicked.connect(self._process_raw_list)
        input_layout.addWidget(self.process_button)
        self.left_tabs.addTab(input_tab, "Ввод")
        

    def _create_markup_tab(self) -> None:
        """Создает вкладку для отображения результата парсинга."""
        markup_tab = QWidget()
        markup_layout = QVBoxLayout(markup_tab)
        self.markup_browser = QTextBrowser()
        markup_layout.addWidget(self.markup_browser)
        self.left_tabs.addTab(markup_tab, "Результат разбора")
        
    def _create_table_view(self, parent_layout: QVBoxLayout) -> None:
        """Создает и настраивает таблицу для отображения списка."""
        self.table_model = StudentTableModel(services=self.SERVICES)
        self.table_model.dataChanged.connect(self._on_data_changed)
        self.table_model.rowsInserted.connect(self._update_summary_info)
        self.table_model.rowsRemoved.connect(self._update_summary_info)
        self.table_model.row_focus_requested.connect(self._handle_row_focus_request)
        
        self.processed_table = QTableView()
        self.processed_table.setModel(self.table_model)
        self.processed_table.setAlternatingRowColors(True)
        delegate = EnterKeyDelegate(self.processed_table, services=list(self.SERVICES.keys()))
        self.processed_table.setItemDelegate(delegate)
        self.processed_table.setSortingEnabled(True)
        self.processed_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.processed_table.customContextMenuRequested.connect(self._show_table_context_menu)
        
        header = self.processed_table.horizontalHeader()
        modes = {
            StudentTableModel.COL_SHOOT_ORDER: QHeaderView.ResizeMode.ResizeToContents,
            StudentTableModel.COL_ALPHA_ORDER: QHeaderView.ResizeMode.ResizeToContents,
            StudentTableModel.COL_COST: QHeaderView.ResizeMode.ResizeToContents,
            StudentTableModel.COL_SURNAME: QHeaderView.ResizeMode.Stretch,
            StudentTableModel.COL_NAME: QHeaderView.ResizeMode.Stretch,
            StudentTableModel.COL_SERVICE: QHeaderView.ResizeMode.Stretch,
        }
        for col, mode in modes.items():
            header.setSectionResizeMode(col, mode)
        parent_layout.addWidget(self.processed_table)

    def _create_summary_panel(self, parent_layout: QVBoxLayout) -> None:
        """Создает панель с итоговой информацией (количество, сумма)."""
        summary_layout = QHBoxLayout()
        self.summary_label_count = QLabel("Всего учеников: 0")
        self.summary_label_total_cost = QLabel("Итоговая сумма: 0 руб.")
        summary_layout.addWidget(self.summary_label_count)
        summary_layout.addStretch()
        summary_layout.addWidget(self.summary_label_total_cost)
        parent_layout.addLayout(summary_layout)

    def _load_services(self) -> None:
        services_path = pathlib.Path(__file__).parent / "_services.json"
        default_services = {"Стандарт": 1500}
        if services_path.exists():
            try:
                with open(services_path, 'r', encoding='utf-8') as f: self.SERVICES = json.load(f)
            except Exception: self.SERVICES = default_services
        else:
            try:
                with open(services_path, 'w', encoding='utf-8') as f: json.dump(default_services, f, ensure_ascii=False, indent=4)
                self.SERVICES = default_services
            except Exception: self.SERVICES = default_services

    def _update_cost_label(self) -> None:
        s_type = self.service_type_combo.currentText()
        cost = self.SERVICES.get(s_type, 0)
        self.service_cost_label.setText(f"{cost} руб.")
        if self.table_model.rowCount() > 0:
            self._is_dirty = True
            self.table_model.update_all_services(s_type, cost)

    def _process_raw_list(self) -> None:
        if self.table_model.rowCount() > 0 and QMessageBox.question(self, "Подтверждение", "Заменить список?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No: return
        self.statusBar().showMessage("Анализ...")
        raw_text = self.raw_list_input.toPlainText()
        if "Интеллектуальный" in self.parser_mode_combo.currentText():
            students, markup_html = self.smart_parser.parse_text(raw_text)
            self.markup_browser.setHtml(markup_html)
            self.left_tabs.setCurrentIndex(1)
        else:
            students = simple_parse_text(raw_text)
            self.markup_browser.clear()
        s_type, cost = self.service_type_combo.currentText(), self.SERVICES.get(self.service_type_combo.currentText(), 0)
        for s in students: s.update({"service_type": s_type, "service_cost": cost})
        self.table_model.update_data(students)
        self.table_model.sort(StudentTableModel.COL_SURNAME)
        self.statusBar().showMessage(f"Найдено: {len(students)}", 5000)
        self._is_dirty = True

    def _show_table_context_menu(self, pos: QPoint) -> None:
        menu = QMenu()
        menu.addAction(QAction("Добавить строку", self, triggered=self._add_new_row))
        if (idx := self.processed_table.indexAt(pos)).isValid():
            row = idx.row()
            menu.addSeparator()
            menu.addAction(QAction("Поменять Имя/Фамилию", self, triggered=lambda: self._on_swap_name_surname(row)))
            menu.addAction(QAction("Удалить выделенные строки", self, triggered=self._delete_selected_rows))
        menu.exec(self.processed_table.viewport().mapToGlobal(pos))

    def _add_new_row(self) -> None:
        s_color = self.smart_parser.SURNAME_COLOR_HEX
        n_color = self.smart_parser.NAME_COLOR_HEX
        data = {"surname": "Ученик", "name": "Новый", "color1": s_color, "color2": n_color, "service_type": self.service_type_combo.currentText(), "service_cost": self.SERVICES.get(self.service_type_combo.currentText(), 0)}
        self.table_model.insert_row(self.table_model.rowCount(), data)
        self.table_model.sort(StudentTableModel.COL_SURNAME)
        try:
            new_idx = self.table_model.get_all_data().index(data)
            self._handle_row_focus_request(new_idx, StudentTableModel.COL_SURNAME)
        except ValueError: pass
        self._is_dirty = True
        self._update_summary_info()
        
    def _delete_selected_rows(self) -> None:
        rows = sorted(list(set(i.row() for i in self.processed_table.selectionModel().selectedIndexes())))
        if rows and QMessageBox.question(self, "Подтверждение", f"Удалить {len(rows)} строк?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.table_model.remove_rows(rows)
            self.table_model.sort(StudentTableModel.COL_SURNAME)

    def _on_swap_name_surname(self, row: int) -> None:
        self.table_model.swap_name_surname(row)

    def _on_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, roles: List[int] = []) -> None:
        self._is_dirty = True
        self._update_summary_info()
        if top_left.isValid() and top_left.column() in [StudentTableModel.COL_SURNAME, StudentTableModel.COL_NAME]:
            self.table_model.sort_and_refocus(top_left.row(), top_left.column())

    def _handle_row_focus_request(self, row: int, col: int) -> None:
        index = self.table_model.index(row, col)
        self.processed_table.scrollTo(index, QTableView.ScrollHint.EnsureVisible)
        self.processed_table.setCurrentIndex(index)

    def _get_default_filepath(self, ext: str) -> Optional[pathlib.Path]:
        if self.config.wf_dest_dir:
            dir_path = pathlib.Path(self.config.wf_dest_dir)
            return dir_path / f"{dir_path.name}.{ext}"
        return None

    def _load_current_session(self) -> None:
        if (path := self._get_default_filepath('list')) and path.exists():
            self._load_from_file(path)

    def _load_any_session(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Загрузить...", "", "Списки (*.list)")
        if path_str:
            self._load_from_file(pathlib.Path(path_str))

    def _load_from_file(self, path: pathlib.Path) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            self.class_name_input.setText(data.get("class_name", path.stem))
            self.service_type_combo.setCurrentText(data.get("service_type", next(iter(self.SERVICES))))
            self.table_model.update_data(data.get("students", []))
            self.table_model.sort(StudentTableModel.COL_SURNAME)
            self.config.wf_dest_dir = str(path.parent)
            self.statusBar().showMessage(f"Загружено: {path.name}", 5000)
            self._is_dirty = False
        except Exception as e: QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{e}")

    def _save_list(self, save_as: bool = False) -> bool:
        path = self._get_default_filepath('list') if not save_as else None
        if not path:
            default_save_path = str(pathlib.Path(self.config.wf_dest_dir or os.getcwd()) / f"{self.class_name_input.text()}.list")
            path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить как...", default_save_path, "Списки (*.list)")
            if not path_str: return False
            path = pathlib.Path(path_str)
            self.config.wf_dest_dir = str(path.parent)
        self.class_name_input.setText(path.stem)
        data = {"class_name": path.stem, "service_type": self.service_type_combo.currentText(), "students": self.table_model.get_all_data()}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)
            self.statusBar().showMessage(f"Сохранено: {path.name}", 5000)
            self._is_dirty = False
            for fmt in self.config.wf_autosave_formats or []:
                if fmt == "html": self._save_html(save_as=False)
                elif fmt == "csv": self._save_csv(save_as=False)
                elif fmt == "txt": self._save_for_processing()
            return True
        except Exception as e: QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}"); return False

    def _save_csv(self, save_as: bool = True) -> None:
        if self.table_model.rowCount() == 0: return
        path_str: Optional[str] = None
        if save_as:
            path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить как CSV...", str(self._get_default_filepath('csv') or ''), "CSV (*.csv)")
        else:
            if path_obj := self._get_default_filepath('csv'): path_str = str(path_obj)
        if not path_str: return
        fieldnames = ["shoot_order", "alpha_order", "surname", "name", "service_type", "service_cost"]
        try:
            with open(path_str, 'w', newline='', encoding='utf-16') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader(); writer.writerows(self.table_model.get_all_data())
            self.statusBar().showMessage(f"CSV сохранен: {pathlib.Path(path_str).name}", 5000)
        except Exception as e: QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить CSV:\n{e}")

    def _save_html(self, save_as: bool = False) -> Optional[pathlib.Path]:
        if not jinja2: return None
        path = self._get_default_filepath('html') if not save_as else None
        if not path:
            path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить HTML как...", str(self._get_default_filepath('html') or ''), "HTML (*.html)")
            if not path_str: return None
            path = pathlib.Path(path_str)
        context = {"class_name": self.class_name_input.text(), "students": self.table_model.get_all_data(), "total_cost": sum(s.get('service_cost', 0) for s in self.table_model.get_all_data())}
        try:
            template_path = pathlib.Path(__file__).parent / "_list_template.html"
            template = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path.parent)).get_template(template_path.name)
            with open(path, 'w', encoding='utf-8') as f: f.write(template.render(context))
            self.statusBar().showMessage(f"HTML сохранен: {path.name}", 5000)
            return path
        except Exception as e: QMessageBox.critical(self, "Ошибка", f"Не удалось создать HTML:\n{e}"); return None

    def _save_for_processing(self) -> None:
        if self.table_model.rowCount() == 0: return
        if self.config.wf_output_txt_file: 
            output_path = pathlib.Path(self.config.wf_output_txt_file)
        elif self.config.wf_dest_dir: 
            output_path = pathlib.Path(self.config.wf_dest_dir) / CHILDREN_LIST_FILENAME
        else: 
            self.statusBar().showMessage("Путь не определен.", 4000); return

        self.table_model.sort(StudentTableModel.COL_SHOOT_ORDER, Qt.SortOrder.AscendingOrder)
        data, lines, skipped = self.table_model.get_all_data(), [], []
        for s in data: (lines if s.get("shoot_order") else skipped).append(f"{s.get('surname', '')} {s.get('name', '')}".strip())
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f: f.write("\n".join(lines))
            self.statusBar().showMessage(f"Файл '{output_path.name}' сохранен.", 5000)
            self._save_children = True
        except Exception as e: QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить '{output_path.name}':\n{e}")
        if skipped: QMessageBox.warning(self, "Внимание", "Пропущены ученики без '№ съемки':\n\n" + "\n".join(skipped))

    def _print_html(self) -> None:
        path = self._save_html(save_as=False)
        if path and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(self, "Ошибка", "Не удалось открыть HTML-файл в браузере.")

    def _update_summary_info(self, *args: Any) -> None:
        data = self.table_model.get_all_data()
        self.summary_label_count.setText(f"Всего учеников: {len(data)}")
        self.summary_label_total_cost.setText(f"Итоговая сумма: {sum(s.get('service_cost', 0) for s in data)} руб.")

    def add_link(self) -> None:
        if IS_MANAGED_RUN and pysm_context and self.config.wf_dest_dir:
            print(" ", file=sys.stderr)
            if self._save_children:
                output_path = pathlib.Path(self.config.wf_output_txt_file or pathlib.Path(self.config.wf_dest_dir) / CHILDREN_LIST_FILENAME)
                pysm_context.log_link(url_or_path=str(output_path), text=f"Открыть файл <i>{output_path.name}</i>")
                pysm_context.log_link(url_or_path=str(output_path.parent), text=f"Открыть папку с файлом <i>{output_path.name}</i>")
            pysm_context.log_link(url_or_path=str(self.config.wf_dest_dir), text="Открыть папку с файлами <i>(list, html, csv)</i>")
            print(" ", file=sys.stderr)

    def closeEvent(self, event: QEvent) -> None:
        if not self._is_dirty:
            self.add_link()
            event.accept()
            return
        reply = QMessageBox.question(self, "Несохраненные изменения", "Сохранить перед выходом?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Save:
            if self._save_list():
                self.add_link()
                event.accept()
            else:
                event.ignore()
        elif reply == QMessageBox.StandardButton.Discard:
            self.add_link()
            event.accept()
        else:
            event.ignore()

    def _open_session_folder(self) -> None:
        if self.config.wf_dest_dir and os.path.isdir(self.config.wf_dest_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config.wf_dest_dir))
        else:
            self.statusBar().showMessage("Папка сессии не найдена.", 4000)

# 7. БЛОК: Основная логика запуска
# ==============================================================================
def main() -> None:
    """Инициализирует и запускает Qt приложение."""
    config = get_raw_config()
    app = QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)
    
    window = ClassListEditor(config)
    window.show()
    sys.exit(app.exec())

# 8. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()