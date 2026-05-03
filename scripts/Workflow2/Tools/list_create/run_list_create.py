#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
Главный модуль приложения. Собирает воедино UI, логику парсинга
и управления данными.
"""
print("<b>РЕДАКТИРОВАНИЕ СПИСКА КЛАССА</b>")

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, Optional, Any, List

# Опциональная зависимость для интеграции с внутренней экосистемой
try:
    from pysm_lib import theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib import pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    theme_api = None
    ConfigResolver = None
    pysm_context = None

from PySide6.QtCore import Qt, QUrl, QPoint, QEvent, QTimer, QModelIndex
from PySide6.QtGui import QAction, QKeySequence, QDesktopServices, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QLabel, QLineEdit, QTextEdit, QTableView, QPushButton, QHeaderView,
    QComboBox, QMenu, QStyle, QTabWidget, QTextBrowser, QMessageBox, QFileDialog
)

from domain import AppConfig, Student, ExtraService, CHILDREN_LIST_FILENAME
from parser import SmartParser, simple_parse_text
from ui_models import StudentTableModel, EnterKeyDelegate, StudentProxyModel
from ui_dialogs import (
    ServicesEditorDialog, ExtraServicesDialog, NamesEditorDialog,
    InfoSchemaEditorDialog, StudentInfoEditorDialog, AIParsingDialog, RanksEditorDialog
)


import io_services


def get_raw_config() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Редактор списков классов для фотографа", 
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-d", "--wf_dest_dir", type=str, help="Директория назначения.")
    parser.add_argument("--wf_output_txt_file", type=str, help="Опциональный путь для сохранения children.txt.")
    parser.add_argument(
        "--wf_autosave_formats", type=str, nargs='+', 
        choices=["html", "txt", "csv"], default=["html", "txt"], 
        help="Форматы для автосохранения."
    )
    
    # --- ИЗМЕНЕНИЕ: Новый параметр CLI ---
    parser.add_argument(
        "--wf_default_info_fields", type=str, nargs='+', default=[],
        help="Список полей информации, которые добавляются автоматически (например: 'Цитата' 'ВК')."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        args = ConfigResolver(parser).resolve_all()
    else:
        args = parser.parse_args()
        
    return AppConfig.from_args(args)


class ClassListEditor(QMainWindow):
    """Главное окно приложения."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._is_dirty: bool = False
        self._save_children: bool = False
        self._is_loading: bool = False

        self.SERVICES: Dict[str, int] = {}
        self.INFO_COLUMNS: List[str] =[]  # Список заголовков доп. информации
        self.RANKS: List[str] =[]         # НОВОЕ: Хранение списка рангов

        self._load_services()
        self._load_ranks()
        self._load_theme_colors()

        self.smart_parser = SmartParser(
            surname_style=self.surname_style,
            name_style=self.name_style,
            fio_style=self.fio_style,
        )

        self._init_ui()
        self._apply_default_info_fields()

    # --- ИЗМЕНЕНИЕ: Новый метод логики ---
    def _apply_default_info_fields(self) -> None:
        """
        Проверяет наличие обязательных полей из конфига в текущей схеме.
        Если их нет — добавляет и обновляет таблицу.
        """
        if not self.config.wf_default_info_fields:
            return

        changed = False
        for field in self.config.wf_default_info_fields:
            if field not in self.INFO_COLUMNS:
                self.INFO_COLUMNS.append(field)
                changed = True
        
        if changed:
            self.table_model.set_info_columns(self.INFO_COLUMNS)
            self._setup_table_headers()
            # Если мы добавили поля, это изменение состояния, но при старте
            # приложения мы обычно не ставим _is_dirty=True, чтобы не надоедать вопросом
            # "Сохранить?" сразу после запуска.
            # Однако, если это произошло после загрузки файла — возможно стоит пометить.
            # Оставим на усмотрение пользователя, здесь не ставим _is_dirty, 
            # так как это "настройки по умолчанию".


    def _load_theme_colors(self):
        if IS_MANAGED_RUN and theme_api:
            self.surname_style = theme_api.get_parsed_style("markup_surname", "background-color: #e3f2fd; border-color: #bbdefb; color: #0d47a1;")
            self.name_style = theme_api.get_parsed_style("markup_name", "background-color: #e8f5e9; border-color: #c8e6c9; color: #1b5e20;")
            self.fio_style = theme_api.get_parsed_style("markup_fio", "background-color: #fff3e0; border-color: #ffe0b2; color: #e65100;")
            self.table_base_bg_color = theme_api.get_qcolor("table_background_base", "color", "#ffffff")
            self.table_alternate_bg_color = theme_api.get_qcolor("table_background_alternate", "color", "#f6f6f6")
        else:
            self.surname_style = {"background-color": "#e3f2fd", "border-color": "#bbdefb", "color": "#0d47a1"}
            self.name_style = {"background-color": "#e8f5e9", "border-color": "#c8e6c9", "color": "#1b5e20"}
            self.fio_style = {"background-color": "#fff3e0", "border-color": "#ffe0b2", "color": "#e65100"}
            self.table_base_bg_color = QColor("#ffffff")
            self.table_alternate_bg_color = QColor("#f6f6f6")

    def _init_ui(self) -> None:
        self.setWindowTitle("PySM - Редактор списка класса")
        self.resize(1400, 800)
        
        self._create_actions()
        self._create_menus()

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self._create_top_panel(main_layout)
        self._create_main_panels(main_layout)

        self.statusBar().showMessage("Готово")

        self._is_loading = True
        if self.config.wf_dest_dir:
            path = pathlib.Path(self.config.wf_dest_dir)
            self.class_name_input.setText(path.name)
            self._load_current_session()
        
        self._update_cost_label() 
        self._update_summary_info()
        self._is_loading = False

# --- ИЗМЕНЕННЫЙ БЛОК: run_list_create.py (Внутри класса ClassListEditor) ---
    def _create_actions(self) -> None:
        self.add_row_action = QAction("Добавить строку", self)
        self.add_row_action.triggered.connect(self._add_new_row)

        self.delete_rows_action = QAction("Удалить выделенные строки", self)
        self.delete_rows_action.triggered.connect(self._delete_selected_rows)

        self.swap_names_action = QAction("Поменять Имя/Фамилию", self)
        self.swap_names_action.triggered.connect(self._swap_current_row_names)
        
        self.edit_extras_action = QAction("Редактировать доп. услуги", self)
        # ИЗМЕНЕНО: Используем QTimer.singleShot(0, ...), чтобы отвязать запуск 
        # диалога от цикла закрытия контекстного меню и избежать мерцания (flicker).
        self.edit_extras_action.triggered.connect(
            lambda: QTimer.singleShot(0, self._open_extra_services_editor)
        )

        self.edit_info_action = QAction("Редактировать информацию", self)
        # ИЗМЕНЕНО: То же самое для редактора информации
        self.edit_info_action.triggered.connect(
            lambda: QTimer.singleShot(0, self._open_student_info_editor)
        )

    def _create_menus(self) -> None:
            file_menu = self.menuBar().addMenu("&Файл")
            
            actions = [
                ("Загрузить список текущей сессии", self._load_current_session, None),
                ("Загрузить список...", self._load_any_session, None),
                ("-", None, None),
                ("Сохранить список", self._save_list, QKeySequence.StandardKey.Save),
                ("Сохранить список как...", lambda: self._save_list(save_as=True), None),
                ("-", None, None),
                # --- ИЗМЕНЕНИЕ: Разделение на обычный и полный HTML ---
                ("Сохранить HTML как...", lambda: self._save_html(save_as=True, extended_mode=False), None),
                ("Сохранить полный HTML как...", lambda: self._save_html(save_as=True, extended_mode=True), None),
                # -----------------------------------------------------
                ("Сохранить CSV как...", self._save_csv, None),
                ("-", None, None),
                ("Сохранить как имена кластеров (TXT)", self._save_for_processing, None),
                ("-", None, None),
                ("Печать HTML", self._print_html, QKeySequence.StandardKey.Print),
                ("-", None, None),
                ("Выход", self.close, None),
            ]

            for name, handler, shortcut in actions:
                if name == "-":
                    file_menu.addSeparator()
                    continue
                action = QAction(name, self)
                action.triggered.connect(handler)
                if shortcut:
                    action.setShortcut(shortcut)
                file_menu.addAction(action)

            settings_menu = self.menuBar().addMenu("&Настройки")
            settings_menu.addAction("Редактировать услуги...", self._open_services_editor)
            settings_menu.addAction("Редактировать словарь имен...", self._open_names_editor)
            settings_menu.addAction("Редактировать ранги...", self._open_ranks_editor)
            settings_menu.addSeparator()
            settings_menu.addAction("Настроить поля информации...", self._open_info_schema_editor)

            ai_menu = self.menuBar().addMenu("&AI")
            ai_menu.addAction("Обработка данных (Gemini)...", self._open_ai_dialog)            

    def _create_top_panel(self, parent_layout: QVBoxLayout) -> None:
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
        self.service_type_combo.addItems(sorted(self.SERVICES.keys()))
        top_panel.addWidget(self.service_type_combo, 0, 3)
        top_panel.addWidget(QLabel("Стоимость:"), 0, 4)
        self.service_cost_label = QLabel()
        self.service_cost_label.setStyleSheet("font-weight: bold;")
        top_panel.addWidget(self.service_cost_label, 0, 5)
        self.service_type_combo.currentIndexChanged.connect(self._update_cost_label)

    def _create_main_panels(self, parent_layout: QVBoxLayout) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        parent_layout.addWidget(splitter)

        # Tabs
        self.left_tabs = QTabWidget()
        
        # Input Tab
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        parser_mode_layout = QHBoxLayout()
        self.parser_mode_combo = QComboBox()
        self.parser_mode_combo.addItems(["Интеллектуальный (Natasha)", "Простой (по шаблону)"])
        parser_mode_layout.addWidget(QLabel("Режим разбора:"))
        parser_mode_layout.addWidget(self.parser_mode_combo)
        input_layout.addLayout(parser_mode_layout)

        self.raw_list_input = QTextEdit()
        self.raw_list_input.setHtml("<b>ИНСТРУКЦИЯ:</b><br>1. Вставьте текст.<br>2. Нажмите <b>Обработать текст</b>.")
        input_layout.addWidget(self.raw_list_input)
        self.process_button = QPushButton("Обработать текст")
        self.process_button.clicked.connect(self._process_raw_list)
        input_layout.addWidget(self.process_button)
        self.left_tabs.addTab(input_tab, "Ввод")
        
        # Markup Tab
        markup_tab = QWidget()
        markup_layout = QVBoxLayout(markup_tab)
        self.markup_browser = QTextBrowser()
        markup_layout.addWidget(self.markup_browser)
        self.left_tabs.addTab(markup_tab, "Результат разбора")
        
        splitter.addWidget(self.left_tabs)
        
        # Right Panel (Table)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Обработанный список:"))
        self._create_table_view(right_layout)
        self._create_summary_panel(right_layout)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1100])

    def _create_table_view(self, parent_layout: QVBoxLayout) -> None:
        self.processed_table = QTableView()
        self.processed_table.setAlternatingRowColors(False)

        self.table_model = StudentTableModel(
            services=self.SERVICES,
            ranks=self.RANKS,
            surname_style=self.surname_style,
            name_style=self.name_style,
            base_bg_color=self.table_base_bg_color,
            alternate_bg_color=self.table_alternate_bg_color
        )
        self.table_model.set_info_columns(self.INFO_COLUMNS) # Установка начальных (пустых) колонок
        
        self.table_model.dataChanged.connect(self._on_data_changed)
        self.table_model.rowsInserted.connect(self._update_summary_info)
        self.table_model.rowsRemoved.connect(self._update_summary_info)

        
        # НОВОЕ: Настраиваем Прокси-модель
        self.proxy_model = StudentProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        self.proxy_model.setDynamicSortFilter(True) # Авто-сортировка при редактировании
        
        self.processed_table.setModel(self.proxy_model) # Таблица теперь смотрит через прокси!
        
        delegate = EnterKeyDelegate(
            self.processed_table, 
            services=list(self.SERVICES.keys()),
            ranks=self.RANKS
        )
        self.processed_table.setItemDelegate(delegate)
        self.processed_table.setSortingEnabled(True)
        self.processed_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.processed_table.customContextMenuRequested.connect(self._show_table_context_menu)
        
        self._setup_table_headers()
        parent_layout.addWidget(self.processed_table)

    def _setup_table_headers(self):
        header = self.processed_table.horizontalHeader()
        
        # 1. Делаем ВСЕ колонки интерактивными (разрешаем ручное изменение мышью)
        for i in range(self.table_model.columnCount()):
             header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)

        # 2. Автоматически подгоняем стартовую ширину под текст заголовков и содержимого
        self.processed_table.resizeColumnsToContents()

        # 3. Задаем дефолтную минимальную ширину для текстовых колонок, 
        # чтобы они не были слишком узкими, если таблица пустая
        self.processed_table.setColumnWidth(StudentTableModel.COL_SURNAME, 130)
        self.processed_table.setColumnWidth(StudentTableModel.COL_NAME, 100)
        self.processed_table.setColumnWidth(StudentTableModel.COL_PATRONYMIC, 80)
        self.processed_table.setColumnWidth(StudentTableModel.COL_RANK, 100)        
        self.processed_table.setColumnWidth(StudentTableModel.COL_SERVICE, 220)

        # 4. Последняя колонка будет тянуться до правого края окна
        header.setStretchLastSection(True)

    def _create_summary_panel(self, parent_layout: QVBoxLayout) -> None:
        summary_layout = QHBoxLayout()
        self.summary_label_count = QLabel("Всего учеников: 0")
        self.summary_label_total_cost = QLabel("Итоговая сумма: 0 руб.")
        summary_layout.addWidget(self.summary_label_count)
        summary_layout.addStretch()
        summary_layout.addWidget(self.summary_label_total_cost)
        parent_layout.addLayout(summary_layout)

    # --- Data Logic ---

    def _load_services(self) -> None:
        services_path = pathlib.Path(__file__).parent / "_services.json"
        default_services = {"Стандарт": 1500, "-": 0}
        if services_path.exists():
            try:
                with open(services_path, 'r', encoding='utf-8') as f:
                    self.SERVICES = json.load(f)
            except Exception:
                self.SERVICES = default_services
        else:
            self.SERVICES = default_services
            self._save_services()

    def _save_services(self) -> None:
        services_path = pathlib.Path(__file__).parent / "_services.json"
        try:
            with open(services_path, 'w', encoding='utf-8') as f:
                json.dump(self.SERVICES, f, ensure_ascii=False, indent=4)
        except IOError as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось записать файл услуг:\n{e}")

    def _load_ranks(self) -> None:
        """НОВОЕ: Загрузка списка рангов из файла."""
        ranks_path = pathlib.Path(__file__).parent / "_ranks.json"
        default_ranks =["ученик", "учитель", "директор", "завуч", "классный руководитель"]
        if ranks_path.exists():
            try:
                with open(ranks_path, 'r', encoding='utf-8') as f:
                    self.RANKS = json.load(f)
            except Exception:
                self.RANKS = default_ranks
        else:
            self.RANKS = default_ranks
            self._save_ranks()

    def _save_ranks(self) -> None:
        """НОВОЕ: Сохранение списка рангов в файл."""
        ranks_path = pathlib.Path(__file__).parent / "_ranks.json"
        try:
            with open(ranks_path, 'w', encoding='utf-8') as f:
                json.dump(self.RANKS, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Ошибка сохранения рангов: {e}", file=sys.stderr)

    def _process_raw_list(self) -> None:
        if self.table_model.rowCount() > 0:
             if QMessageBox.question(self, "Подтверждение", "Заменить текущий список?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
                 return

        self.statusBar().showMessage("Анализ...")
        raw_text = self.raw_list_input.toPlainText()
        
        if "Интеллектуальный" in self.parser_mode_combo.currentText():
            students, markup_html = self.smart_parser.parse_text(raw_text)
            self.markup_browser.setHtml(markup_html)
            self.left_tabs.setCurrentIndex(1)
        else:
            students = simple_parse_text(raw_text)
            self.markup_browser.clear()

        s_type = self.service_type_combo.currentText()
        cost = self.SERVICES.get(s_type, 0)
        
        for s in students:
            s.service_type = s_type
            s.service_cost = cost
        
        self.table_model.update_data(students)
        # НОВОЕ: Сортируем базу и проставляем номера п/п
        self.table_model.sort_and_renumber()
        self.processed_table.sortByColumn(StudentTableModel.COL_SURNAME, Qt.SortOrder.AscendingOrder)
        
        self.statusBar().showMessage(f"Найдено: {len(students)}", 5000)
        self._is_dirty = True

    def _update_cost_label(self) -> None:
        s_type = self.service_type_combo.currentText()
        cost = self.SERVICES.get(s_type, 0)
        self.service_cost_label.setText(f"{cost} руб.")
        if not self._is_loading and self.table_model.rowCount() > 0:
            self._is_dirty = True
            self.table_model.update_all_services(s_type, cost)

    def _update_summary_info(self, *args: Any) -> None:
        data = self.table_model.get_all_data()
        total_sum = sum(s.total_cost for s in data)
        self.summary_label_count.setText(f"Всего учеников: {len(data)}")
        self.summary_label_total_cost.setText(f"Итоговая сумма: {total_sum} руб.")

    # --- Actions ---
    def _add_new_row(self) -> None:
        new_student = Student(
            surname="Ученик", name="Новый",
            color1=self.smart_parser.SURNAME_COLOR_HEX, color2=self.smart_parser.NAME_COLOR_HEX,
            color1_fg=self.surname_style.get("color", "#000000"), color2_fg=self.name_style.get("color", "#000000"),
            service_type=self.service_type_combo.currentText(), service_cost=self.SERVICES.get(self.service_type_combo.currentText(), 0)
        )
        self.table_model.insert_row(self.table_model.rowCount(), new_student)
        
        # Пересчитываем номера
        self.table_model.sort_and_renumber()
        self.processed_table.sortByColumn(StudentTableModel.COL_SURNAME, Qt.SortOrder.AscendingOrder)
        
        # Находим строку, куда встал новый ученик после сортировки, и ставим фокус
        try:
            new_source_row = self.table_model.get_all_data().index(new_student)
            source_idx = self.table_model.index(new_source_row, StudentTableModel.COL_SURNAME)
            proxy_idx = self.proxy_model.mapFromSource(source_idx)
            self.processed_table.selectionModel().clear()
            self.processed_table.setCurrentIndex(proxy_idx)
            self.processed_table.scrollTo(proxy_idx)
        except ValueError: pass
        self._is_dirty = True

    def _delete_selected_rows(self) -> None:
        proxy_indexes = self.processed_table.selectionModel().selectedIndexes()
        rows = sorted(list(set(self.proxy_model.mapToSource(i).row() for i in proxy_indexes)))
        if rows and QMessageBox.question(self, "Подтверждение", f"Удалить {len(rows)} строк?", 
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.table_model.remove_rows(rows)
            # Пересчитываем номера после удаления
            self.table_model.sort_and_renumber()

    def _swap_current_row_names(self) -> None:
        proxy_idx = self.processed_table.currentIndex()
        if proxy_idx.isValid():
            source_idx = self.proxy_model.mapToSource(proxy_idx)
            student = self.table_model.get_all_data()[source_idx.row()]
            self.table_model.swap_name_surname(source_idx.row())
            
            # Меняем местами и пересчитываем номера
            self.table_model.sort_and_renumber()
            
            # Возвращаем фокус на этого же человека
            try:
                new_row = self.table_model.get_all_data().index(student)
                new_proxy_idx = self.proxy_model.mapFromSource(self.table_model.index(new_row, StudentTableModel.COL_SURNAME))
                self.processed_table.setCurrentIndex(new_proxy_idx)
            except ValueError: pass

    def _open_extra_services_editor(self) -> None:
        proxy_idx = self.processed_table.currentIndex()
        if not proxy_idx.isValid(): return
        source_row = self.proxy_model.mapToSource(proxy_idx).row()
        student = self.table_model.get_all_data()[source_row]
        
        dialog = ExtraServicesDialog(student.extra_services, self.SERVICES, self)
        if dialog.exec():
            new_extras = dialog.get_data()
            self.table_model.update_extras(source_row, new_extras)
            self._is_dirty = True
            self._update_summary_info()
            
    def _open_student_info_editor(self) -> None:
        proxy_idx = self.processed_table.currentIndex()
        if not proxy_idx.isValid(): 
            QMessageBox.information(self, "Информация", "Выберите ученика в таблице.")
            return
        
        if not self.INFO_COLUMNS:
             QMessageBox.information(self, "Информация", "Сначала настройте поля.")
             return

        source_row = self.proxy_model.mapToSource(proxy_idx).row()
        students = self.table_model.get_all_data()
        
        dialog = StudentInfoEditorDialog(students, source_row, self.INFO_COLUMNS, self)
        if dialog.exec():
            changes = dialog.get_changes()
            if changes:
                self.table_model.update_info_bulk(changes)
                self._is_dirty = True







    def _open_services_editor(self) -> None:
        dialog = ServicesEditorDialog(self.SERVICES, self)
        if dialog.exec():
            try:
                self.SERVICES = dialog.get_services()
                self._save_services()
                curr = self.service_type_combo.currentText()
                self.service_type_combo.blockSignals(True)
                self.service_type_combo.clear()
                self.service_type_combo.addItems(self.SERVICES.keys())
                if curr in self.SERVICES: self.service_type_combo.setCurrentText(curr)
                self.service_type_combo.blockSignals(False)
                self._update_cost_label()
                self.statusBar().showMessage("Услуги обновлены.", 3000)
            except ValueError as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")

    def _open_names_editor(self) -> None:
        dialog = NamesEditorDialog(self.smart_parser.normalization_dict.copy(), self)
        if dialog.exec():
            try:
                self.smart_parser.normalization_dict = dialog.get_names_dict()
                dict_path = pathlib.Path(__file__).parent / "_names_normalization.json"
                with open(dict_path, 'w', encoding='utf-8') as f:
                    json.dump(self.smart_parser.normalization_dict, f, ensure_ascii=False, indent=4)
                self.statusBar().showMessage("Словарь имен обновлен.", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")





           
    def _open_info_schema_editor(self) -> None:
        dialog = InfoSchemaEditorDialog(self.INFO_COLUMNS, self)
        if dialog.exec():
            new_cols, rename_map = dialog.get_result()
            
            # Обновляем данные студентов (переименование ключей)
            students = self.table_model.get_all_data()
            for s in students:
                # 1. Переименование
                for old_key, new_key in rename_map.items():
                    if old_key in s.info:
                        s.info[new_key] = s.info.pop(old_key)
                
                # 2. Удаление ключей, которых больше нет в схеме
                # (делаем копию ключей, чтобы удалять во время итерации)
                keys_to_del = [k for k in s.info if k not in new_cols]
                for k in keys_to_del:
                    del s.info[k]

            self.INFO_COLUMNS = new_cols
            self.table_model.set_info_columns(self.INFO_COLUMNS)
            self._setup_table_headers() # Обновляем заголовки таблицы (важно для Stretch)
            self._is_dirty = True

    def _open_ranks_editor(self) -> None:
        """НОВОЕ: Открывает редактор рангов."""
        dialog = RanksEditorDialog(self.RANKS, self)
        if dialog.exec():
            try:
                self.RANKS = dialog.get_ranks()
                self._save_ranks()
                # Обновляем делегат в таблице с новыми рангами
                delegate = EnterKeyDelegate(
                    self.processed_table, 
                    services=sorted(list(self.SERVICES.keys())),
                    ranks=self.RANKS
                )
                self.processed_table.setItemDelegate(delegate)
                self.statusBar().showMessage("Ранги обновлены.", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")


    def _open_ai_dialog(self) -> None:
        """Открывает диалог интеграции с AI."""
        if self.table_model.rowCount() == 0:
            QMessageBox.warning(self, "Пусто", "Сначала загрузите или создайте список учеников.")
            return
            
        students = self.table_model.get_all_data()
        dialog = AIParsingDialog(students, self)
        
        # Если диалог завершился (пользователь нажал Закрыть после импорта),
        # нужно обновить таблицу, так как данные могли измениться.
        dialog.exec()
        
        # Обновляем таблицу, так как поля info могли измениться
        # Также это перерисует колонки, если AI добавил новые ключи? 
        # Нет, AI обновляет только словарь. Если AI добавил ключ, которого нет в схеме INFO_COLUMNS,
        # он не отобразится в таблице, пока мы не добавим его в схему.
        
        # Поэтому полезно проверить ключи
        self._sync_info_columns_with_data()
        
        self.table_model.layoutChanged.emit()
        self._is_dirty = True

    def _sync_info_columns_with_data(self):
        """
        Проверяет, появились ли в данных студентов новые ключи info,
        которых нет в INFO_COLUMNS, и добавляет их.
        """
        students = self.table_model.get_all_data()
        changed = False
        for s in students:
            for key in s.info.keys():
                if key not in self.INFO_COLUMNS:
                    self.INFO_COLUMNS.append(key)
                    changed = True
        
        if changed:
            self.table_model.set_info_columns(self.INFO_COLUMNS)
            self._setup_table_headers()
            QMessageBox.information(self, "Схема обновлена", "AI добавил новые поля информации. Таблица обновлена.")




    def _on_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, roles: List[int] = None) -> None:
        roles = roles or[]  # ИЗМЕНЕНО: Защита от мутабельного аргумента по умолчанию
        self._is_dirty = True
        self._update_summary_info()

    # --- I/O ---

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
            metadata, students = io_services.load_session(path)
            
            self.class_name_input.setText(metadata["class_name"])
            self.service_type_combo.setCurrentText(metadata["service_type"])
            
            # Загружаем схему полей из файла
            self.INFO_COLUMNS = metadata.get("info_columns", [])
            
            # --- ИЗМЕНЕНИЕ: После загрузки файла тоже применяем дефолтные поля ---
            # Это полезно, если мы открываем старый файл, где этих полей еще не было.
            self._apply_default_info_fields()
            
            for s in students:
                if not s.color1:
                    s.color1 = self.smart_parser.SURNAME_COLOR_HEX
                    s.color1_fg = self.surname_style.get("color", "#000000")
                if not s.color2:
                    s.color2 = self.smart_parser.NAME_COLOR_HEX
                    s.color2_fg = self.name_style.get("color", "#000000")

            self.table_model.set_info_columns(self.INFO_COLUMNS)
            self.table_model.update_data(students)
                       
            self.table_model.sort_and_renumber()
            self.processed_table.sortByColumn(StudentTableModel.COL_SURNAME, Qt.SortOrder.AscendingOrder)
            self._setup_table_headers()

            self.config.wf_dest_dir = str(path.parent)
            self.statusBar().showMessage(f"Загружено: {path.name}", 5000)
            self._is_dirty = False
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{e}")


    def _save_list(self, save_as: bool = False) -> bool:
        path = self._get_default_filepath('list') if not save_as else None
        if not path:
            default_path = str(pathlib.Path(self.config.wf_dest_dir or os.getcwd()) / f"{self.class_name_input.text()}.list")
            path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить как...", default_path, "Списки (*.list)")
            if not path_str: return False
            path = pathlib.Path(path_str)
            self.config.wf_dest_dir = str(path.parent)

        self.class_name_input.setText(path.stem)
        try:
            io_services.save_session(
                path, 
                self.class_name_input.text(),
                self.service_type_combo.currentText(),
                self.table_model.get_all_data(),
                self.INFO_COLUMNS # Сохраняем схему
            )
            self.statusBar().showMessage(f"Сохранено: {path.name}", 5000)
            self._is_dirty = False
            
            for fmt in self.config.wf_autosave_formats:
                if fmt == "html": self._save_html(save_as=False)
                elif fmt == "csv": self._save_csv(save_as=False)
                elif fmt == "txt": self._save_for_processing()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
            return False

    def _save_csv(self, save_as: bool = True) -> None:
        if self.table_model.rowCount() == 0: return
        path_str = None
        if save_as:
            default = str(self._get_default_filepath('csv') or '')
            path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV...", default, "CSV (*.csv)")
        else:
            if path_obj := self._get_default_filepath('csv'): path_str = str(path_obj)
        if not path_str: return
        
        try:
            io_services.export_to_csv(
                pathlib.Path(path_str), 
                self.table_model.get_all_data(),
                self.INFO_COLUMNS
            )
            self.statusBar().showMessage(f"CSV сохранен: {pathlib.Path(path_str).name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить CSV:\n{e}")

    def _save_html(self, save_as: bool = False, extended_mode: bool = None) -> Optional[pathlib.Path]:
            try:
                path = self._get_default_filepath('html') if not save_as else None
                if not path:
                    default = str(self._get_default_filepath('html') or '')
                    path_str, _ = QFileDialog.getSaveFileName(self, "Сохранить HTML...", default, "HTML (*.html)")
                    if not path_str: return None
                    path = pathlib.Path(path_str)

                # Логика выбора режима:
                # 1. Если extended_mode передан явно (из меню) -> используем его.
                # 2. Если extended_mode is None (автосохранение) -> включаем, если есть колонки.
                use_extended = extended_mode if extended_mode is not None else (len(self.INFO_COLUMNS) > 0)

                io_services.export_to_html(
                    path, 
                    self.class_name_input.text(), 
                    self.table_model.get_all_data(),
                    pathlib.Path(__file__).parent,
                    extended_mode=use_extended
                )
                self.statusBar().showMessage(f"HTML сохранен: {path.name}", 5000)
                return path
            except ImportError:
                 QMessageBox.warning(self, "Ошибка", "Библиотека Jinja2 не установлена.")
            except Exception as e:
                 QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить HTML:\n{e}")
            return None

    def _save_for_processing(self) -> None:
        if self.table_model.rowCount() == 0: return
        if self.config.wf_output_txt_file: 
            output_dir = pathlib.Path(self.config.wf_output_txt_file).parent
            filename = pathlib.Path(self.config.wf_output_txt_file).name
        elif self.config.wf_dest_dir: 
            output_dir = pathlib.Path(self.config.wf_dest_dir)
            filename = CHILDREN_LIST_FILENAME
        else: return

        try:
            io_services.export_to_txt(output_dir, self.table_model.get_all_data(), filename)
            self.statusBar().showMessage("Файл для обработки сохранен.", 3000)
            self._save_children = True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить TXT:\n{e}")

    def _print_html(self) -> None:
        path = self._save_html(save_as=False)
        if path and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(self, "Ошибка", "Не удалось открыть браузер.")

    # --- Events ---

    def keyPressEvent(self, event: QEvent) -> None:
        if self.processed_table.hasFocus():
            if event.key() == Qt.Key.Key_N and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._add_new_row()
                event.accept(); return
            if event.key() == Qt.Key.Key_Delete:
                self._delete_selected_rows()
                event.accept(); return
            if event.key() == Qt.Key.Key_T and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._swap_current_row_names()
                event.accept(); return
            if event.key() == Qt.Key.Key_E and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._open_extra_services_editor()
                event.accept(); return
            if event.key() == Qt.Key.Key_I and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._open_student_info_editor()
                event.accept(); return
        super().keyPressEvent(event)

    def _show_table_context_menu(self, pos: QPoint) -> None:
        menu = QMenu()
        self.add_row_action.setText("Добавить строку\tCtrl+N")
        menu.addAction(self.add_row_action)
        
        if self.processed_table.selectionModel().hasSelection():
            menu.addSeparator()
            self.swap_names_action.setText("Поменять Имя/Фамилию\tCtrl+T")
            menu.addAction(self.swap_names_action)
            self.edit_extras_action.setText("Редактировать доп. услуги...\tCtrl+E")
            menu.addAction(self.edit_extras_action)
            self.edit_info_action.setText("Редактировать информацию...\tCtrl+I")
            menu.addAction(self.edit_info_action)
            self.delete_rows_action.setText("Удалить\tDelete")
            menu.addAction(self.delete_rows_action)
            
        menu.exec(self.processed_table.viewport().mapToGlobal(pos))

    def _handle_row_focus_request_async(self, row: int, col: int) -> None:
        QTimer.singleShot(0, lambda: self._handle_row_focus_request(row, col))

    def _handle_row_focus_request(self, row: int, col: int, clear_selection: bool = False) -> None:
        if clear_selection: self.processed_table.selectionModel().clear()
        index = self.table_model.index(row, col)
        self.processed_table.scrollTo(index, QTableView.ScrollHint.EnsureVisible)
        self.processed_table.setCurrentIndex(index)

    def _open_session_folder(self) -> None:
        if self.config.wf_dest_dir and os.path.isdir(self.config.wf_dest_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config.wf_dest_dir))

    def add_link(self) -> None:
        if IS_MANAGED_RUN and pysm_context and self.config.wf_dest_dir:
            print(" ", file=sys.stderr)
            if self._save_children:
                path_str = self.config.wf_output_txt_file or str(pathlib.Path(self.config.wf_dest_dir) / CHILDREN_LIST_FILENAME)
                path = pathlib.Path(path_str)
                pysm_context.log_link(url_or_path=str(path), text=f"Открыть файл <i>{path.name}</i>")
            pysm_context.log_link(url_or_path=str(self.config.wf_dest_dir), text="Открыть папку с файлами")
            print(" ", file=sys.stderr)

    def closeEvent(self, event: QEvent) -> None:
        out_txt = self.config.wf_output_txt_file
        if out_txt and not pathlib.Path(out_txt).exists() and self.table_model.rowCount() > 0:
             self._save_for_processing()

        if not self._is_dirty:
            #self.add_link()
            event.accept(); 
            return
            
        reply = QMessageBox.question(
            self, "Несохраненные изменения", "Сохранить перед выходом?", 
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Save:
            if self._save_list(): 
                #self.add_link(); 
                event.accept()
            else: 
                event.ignore()
        elif reply == QMessageBox.StandardButton.Discard:
            #self.add_link(); 
            event.accept()
        else: 
            event.ignore()


def main() -> None:
    config = get_raw_config()
    app = QApplication(sys.argv)
    if IS_MANAGED_RUN and theme_api: theme_api.apply_theme_to_app(app)
    window = ClassListEditor(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()