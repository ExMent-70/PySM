# pysm_lib/gui/dialogs/settings_dialog.py

# 1. БЛОК: Импорты
# ==============================================================================
import os
import pathlib
import re
import sys
import logging
from typing import List, Dict, Optional, Any, cast

import cssutils
import xml.dom

from PySide6.QtCore import Slot, Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMessageBox, QComboBox,
    QInputDialog, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QToolBox, QTextEdit
)

from ...config_manager import ValidLogLevels, AppConfigModel
from ...theme_manager import ThemeManager
from ...locale_manager import LocaleManager
from typing import get_args
from ..widgets.path_list_editor import PathListEditor
from ...app_constants import APPLICATION_ROOT_DIR

cssutils_log = logging.getLogger("cssutils")
cssutils_log.setLevel(logging.CRITICAL)

def is_valid_stylesheet(style_string: str) -> bool:
    """Проверяет, является ли строка валидным набором CSS-правил."""
    if not style_string.strip():
        return True
    try:
        sheet = cssutils.css.CSSStyleSheet()
        sheet.cssText = f"A{{ {style_string} }}"
        return sheet.valid
    except (xml.dom.SyntaxErr, Exception):
        return False

# 2. БЛОК: Основной класс SettingsDialog
# ==============================================================================
class SettingsDialog(QDialog):
    theme_changed = Signal(str)

    PROTECTED_STYLE_KEYS = {
        "api_image_description", "api_link", "set_header", "set_info",
        "script_header_block", "script_success_block", "script_error_block",
        "script_info", "script_stdout", "script_stderr", "runner_info",
        "script_arg_value", "tooltip_script_args_block", "tooltip_instance_args_block",
        "console_background", "collection_info", "status_running", "status_success",
        "status_error", "status_pending", "status_skipped", "script_description",
        "tooltip_arg_value",
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Новые ключи для делегатов
        "delegate_hover_border",
        "delegate_changed_indicator",
        "delegate_preview_background",
        "delegate_secondary_text",
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---        
    }
    
    INVALID_STYLE_COLOR = QColor("#fdecf0")

    def __init__(
        self,
        config_model: AppConfigModel,
        theme_manager: ThemeManager,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.config = config_model.model_copy(deep=True)
        self.theme_manager = theme_manager
        self._current_theme_styles_cache: Dict[str, Any] = {}

        self.setWindowTitle(self.locale_manager.get("dialogs.settings.title"))
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        # Создаем только самый базовый каркас
        self.main_layout = QVBoxLayout(self)
        
        # Временная метка загрузки
        self.loading_label = QLabel(self.locale_manager.get("dialogs.settings.loading_text", default="Загрузка настроек..."))
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.loading_label, 1)

        # Основной контейнер для виджетов (пока скрыт)
        self.main_content_widget = QWidget()
        self.main_content_widget.setVisible(False)
        self.main_layout.addWidget(self.main_content_widget, 1)
        
        # Кнопки создаем сразу
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        # Откладываем создание и заполнение всего "тяжелого" UI
        QTimer.singleShot(0, self._initialize_ui_content)

    def _initialize_ui_content(self):
        """Создает и заполняет все основные виджеты диалога."""
        
        # Создаем компоновку для основного контейнера
        content_layout = QVBoxLayout(self.main_content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        content_layout.addWidget(self.tabs)

        main_settings_widget = QWidget()
        main_settings_layout = QVBoxLayout(main_settings_widget)
        self.tabs.addTab(main_settings_widget, self.locale_manager.get("dialogs.settings.tabs.main"))
        
        styles_widget = QWidget()
        self.styles_layout = QVBoxLayout(styles_widget)
        self.tabs.addTab(styles_widget, self.locale_manager.get("dialogs.settings.tabs.appearance"))
        
        # Заполняем вкладки
        self._populate_main_tab(main_settings_layout)
        self._populate_styles_tab(self.styles_layout)

        # Производим "подмену": скрываем метку и показываем готовый контент
        self.loading_label.setVisible(False)
        self.main_content_widget.setVisible(True)

    def _finish_ui_setup(self):
        """
        Завершает настройку UI, заполняя его данными, которые могут
        требовать времени на загрузку (например, сканирование папки тем).
        """
        # Теперь сканирование тем и заполнение QComboBox произойдет
        # после того, как окно уже будет показано.
        self._update_theme_list()


    def _populate_main_tab(self, layout: QVBoxLayout):
        py_interpreter_group = QGroupBox(self.locale_manager.get("dialogs.settings.interpreter_group"))
        py_path_layout = QHBoxLayout(py_interpreter_group)
        self.py_interpreter_label = QLabel(self.config.paths.python_interpreter)
        self.py_interpreter_label.setWordWrap(True)
        self.py_interpreter_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        btn_browse_py_inline = QPushButton(self.locale_manager.get("dialogs.settings.buttons.change"))
        btn_browse_py_inline.clicked.connect(self._browse_python_interpreter)
        py_path_layout.addWidget(self.py_interpreter_label, 1)
        py_path_layout.addWidget(btn_browse_py_inline)
        layout.addWidget(py_interpreter_group)

        paths_tool_box = QToolBox()
        layout.addWidget(paths_tool_box, 1)

        additional_paths_widget = QWidget()
        additional_paths_layout = QVBoxLayout(additional_paths_widget)
        additional_paths_layout.setContentsMargins(0, 0, 0, 0)
        self.path_editor = PathListEditor(
            locale_manager=self.locale_manager,
            dialog_title=self.locale_manager.get("dialogs.settings.path_dialog_title"),
        )
        self.path_editor.set_paths(self.config.paths.additional_env_paths)
        additional_paths_layout.addWidget(self.path_editor)
        path_bottom_layout = self.path_editor.findChild(QHBoxLayout)
        self.btn_view_path = QPushButton(self.locale_manager.get("dialogs.settings.buttons.view"))
        path_bottom_layout.addWidget(self.btn_view_path)

        python_paths_widget = QWidget()
        python_paths_layout = QVBoxLayout(python_paths_widget)
        python_paths_layout.setContentsMargins(0, 0, 0, 0)
        self.python_path_editor = PathListEditor(
            locale_manager=self.locale_manager,
            dialog_title=self.locale_manager.get("dialogs.settings.path_dialog_title"),
        )
        self.python_path_editor.set_paths(self.config.paths.python_paths)
        python_paths_layout.addWidget(self.python_path_editor)
        python_path_bottom_layout = self.python_path_editor.findChild(QHBoxLayout)
        self.btn_view_python_path = QPushButton(self.locale_manager.get("dialogs.settings.buttons.view"))
        python_path_bottom_layout.addWidget(self.btn_view_python_path)

        env_vars_widget = QWidget()
        env_vars_layout = QVBoxLayout(env_vars_widget)
        env_vars_layout.setContentsMargins(0, 0, 0, 0)
        self.env_vars_table = QTableWidget()
        self.env_vars_table.setColumnCount(2)
        self.env_vars_table.setHorizontalHeaderLabels([
            self.locale_manager.get("dialogs.settings.env_vars.header_key"),
            self.locale_manager.get("dialogs.settings.env_vars.header_value"),
        ])
        self.env_vars_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.env_vars_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.env_vars_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for key, value in sorted(self.config.environment_variables.items()):
            row_count = self.env_vars_table.rowCount()
            self.env_vars_table.insertRow(row_count)
            self.env_vars_table.setItem(row_count, 0, QTableWidgetItem(key))
            self.env_vars_table.setItem(row_count, 1, QTableWidgetItem(value))
        env_vars_layout.addWidget(self.env_vars_table)
        env_buttons_layout = QHBoxLayout()
        btn_add_env_var = QPushButton(self.locale_manager.get("dialogs.settings.buttons.add"))
        btn_edit_env_var = QPushButton(self.locale_manager.get("dialogs.settings.buttons.edit"))
        btn_remove_env_var = QPushButton(self.locale_manager.get("dialogs.settings.buttons.remove"))
        env_buttons_layout.addWidget(btn_add_env_var)
        env_buttons_layout.addWidget(btn_edit_env_var)
        env_buttons_layout.addWidget(btn_remove_env_var)
        env_buttons_layout.addStretch()
        self.btn_view_env = QPushButton(self.locale_manager.get("dialogs.settings.buttons.view"))
        env_buttons_layout.addWidget(self.btn_view_env)
        env_vars_layout.addLayout(env_buttons_layout)

        paths_tool_box.addItem(additional_paths_widget, self.locale_manager.get("dialogs.settings.path_group"))
        paths_tool_box.addItem(python_paths_widget, self.locale_manager.get("dialogs.settings.pythonpath_group"))
        paths_tool_box.addItem(env_vars_widget, self.locale_manager.get("dialogs.settings.env_vars.group_title"))

        bottom_settings_layout = QHBoxLayout()
        logging_group = QGroupBox(self.locale_manager.get("dialogs.settings.logging_group"))
        logging_form_layout = QFormLayout(logging_group)
        self.log_level_combo = QComboBox()
        self.valid_log_levels: tuple[str, ...] = get_args(ValidLogLevels)
        self.log_level_combo.addItems(self.valid_log_levels)
        self.log_level_combo.setCurrentText(self.config.logging.log_level)
        logging_form_layout.addRow(self.locale_manager.get("dialogs.settings.logging_label"), self.log_level_combo)
        
        lang_group = QGroupBox(self.locale_manager.get("dialogs.settings.language_group"))
        lang_form_layout = QFormLayout(lang_group)
        self.language_combo = QComboBox()
        available_langs = self.locale_manager.get_available_languages()
        for code, name in sorted(available_langs.items()):
            self.language_combo.addItem(name, code)
        current_lang_index = self.language_combo.findData(self.config.general.language)
        if current_lang_index != -1:
            self.language_combo.setCurrentIndex(current_lang_index)
        lang_form_layout.addRow(self.locale_manager.get("dialogs.settings.language_label"), self.language_combo)

        bottom_settings_layout.addWidget(logging_group)
        bottom_settings_layout.addWidget(lang_group)
        layout.addLayout(bottom_settings_layout)
        layout.addWidget(QLabel(self.locale_manager.get("dialogs.settings.restart_note")))

        btn_add_env_var.clicked.connect(self._add_env_var)
        btn_edit_env_var.clicked.connect(self._edit_env_var)
        btn_remove_env_var.clicked.connect(self._remove_env_var)
        self.env_vars_table.itemSelectionChanged.connect(lambda: self._update_env_var_buttons_state(btn_edit_env_var, btn_remove_env_var))
        self.env_vars_table.doubleClicked.connect(self._edit_env_var)
        self._update_env_var_buttons_state(btn_edit_env_var, btn_remove_env_var)

        self.btn_view_path.clicked.connect(self._show_current_path)
        self.btn_view_python_path.clicked.connect(self._show_current_pythonpath)
        self.btn_view_env.clicked.connect(self._show_current_env)

    def _populate_styles_tab(self, layout: QVBoxLayout):
        themes_group = QGroupBox(self.locale_manager.get("dialogs.settings.themes.group_title"))
        themes_layout = QHBoxLayout(themes_group)
        themes_layout.addWidget(QLabel(self.locale_manager.get("dialogs.settings.themes.current_theme_label")))
        self.theme_combo = QComboBox()
        themes_layout.addWidget(self.theme_combo, 1)
        self.theme_create_btn = QPushButton(self.locale_manager.get("dialogs.settings.buttons.create"))
        self.theme_delete_btn = QPushButton(self.locale_manager.get("dialogs.settings.buttons.remove"))
        themes_layout.addWidget(self.theme_create_btn)
        themes_layout.addWidget(self.theme_delete_btn)
        layout.addWidget(themes_group)

        self.styles_table = QTableWidget()
        self.styles_table.setColumnCount(3)
        self.styles_table.setHorizontalHeaderLabels(["Параметр", "CSS-свойства", "Предпросмотр"])
        header = self.styles_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.styles_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.styles_table, 1)

        style_buttons_layout = QHBoxLayout()
        self.btn_add_style = QPushButton(self.locale_manager.get("dialogs.settings.buttons.add"))
        self.btn_remove_style = QPushButton(self.locale_manager.get("dialogs.settings.buttons.remove"))
        style_buttons_layout.addWidget(self.btn_add_style)
        style_buttons_layout.addWidget(self.btn_remove_style)
        style_buttons_layout.addStretch()
        layout.addLayout(style_buttons_layout)

        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        self.theme_create_btn.clicked.connect(self._on_create_theme)
        self.theme_delete_btn.clicked.connect(self._on_delete_theme)
        self.styles_table.itemChanged.connect(self._on_style_item_changed)
        self.styles_table.itemSelectionChanged.connect(self._update_style_buttons_state)
        self.btn_add_style.clicked.connect(self._on_add_style)
        self.btn_remove_style.clicked.connect(self._on_remove_style)

        self._update_theme_list()

    def _update_theme_list(self):
        with QSignalBlocker(self.theme_combo):
            self.theme_combo.clear()
            available_themes = self.theme_manager.get_available_themes()
            self.theme_combo.addItems(available_themes)
            
            current_idx = self.theme_combo.findText(self.config.general.active_theme_name)
            if current_idx != -1:
                self.theme_combo.setCurrentIndex(current_idx)
            elif "default" in available_themes:
                self.theme_combo.setCurrentIndex(self.theme_combo.findText("default"))
        
        QTimer.singleShot(0, lambda: self._on_theme_selected(self.theme_combo.currentIndex()))

    def _on_theme_selected(self, index: int):
        theme_name = self.theme_combo.itemText(index)
        if not theme_name: return

        self.theme_changed.emit(theme_name)
        
        is_default_theme = theme_name == "default"
        self.theme_delete_btn.setEnabled(not is_default_theme)
        self.btn_add_style.setEnabled(not is_default_theme)
        
        theme_data = self.theme_manager.load_theme(theme_name)
        style_dict = theme_data.dynamic_styles.get_styles_as_dict() if theme_data else {}
        self._current_theme_styles_cache = style_dict.copy()

        with QSignalBlocker(self.styles_table):
            self.styles_table.setRowCount(0)
            for name, value in sorted(style_dict.items()):
                 self._add_style_row(name, value)

        self.styles_table.resizeColumnsToContents()
        self._update_style_buttons_state()

    def _on_create_theme(self):
        source_theme = self.theme_combo.currentText() or "default"
        new_name, ok = QInputDialog.getText(self, "Создать новую тему", f"Имя новой темы (на основе '{source_theme}'):")
        if ok and new_name:
            if not re.match(r"^[a-zA-Z0-9_-]+$", new_name):
                QMessageBox.warning(self, "Ошибка", "Имя темы может содержать только латинские буквы, цифры, дефис и подчеркивание.")
                return
            if new_name in self.theme_manager.get_available_themes():
                QMessageBox.warning(self, "Ошибка", f"Тема '{new_name}' уже существует.")
                return
            
            if self.theme_manager.create_theme_from_existing(source_theme, new_name):
                self.config.general.active_theme_name = new_name
                self._update_theme_list()

    def _on_delete_theme(self):
        theme_to_delete = self.theme_combo.currentText()
        if not theme_to_delete or theme_to_delete == "default": return

        reply = QMessageBox.question(self, "Удаление темы", f"Вы уверены, что хотите удалить тему '{theme_to_delete}'? Это действие необратимо.")
        if reply == QMessageBox.StandardButton.Yes:
            if self.theme_manager.delete_theme(theme_to_delete):
                self.config.general.active_theme_name = "default"
                self._update_theme_list()

    def _add_style_row(self, name: str, value: str, select_row: bool = False):
        row_position = self.styles_table.rowCount()
        self.styles_table.insertRow(row_position)
        name_item = QTableWidgetItem(name)
        value_item = QTableWidgetItem(value)

        is_protected = name in self.PROTECTED_STYLE_KEYS
        is_default_theme = self.theme_combo.currentText() == "default"

        if is_protected or is_default_theme:
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        
        if is_default_theme:
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self.styles_table.setItem(row_position, 0, name_item)
        self.styles_table.setItem(row_position, 1, value_item)
        preview_widget = QLabel(self.locale_manager.get("dialogs.settings.styles_preview_text"))
        self.styles_table.setCellWidget(row_position, 2, preview_widget)
        self._validate_and_update_row_visuals(row_position)
        if select_row: self.styles_table.selectRow(row_position)

    def _validate_and_update_row_visuals(self, row: int):
        value_item = self.styles_table.item(row, 1)
        preview_widget = self.styles_table.cellWidget(row, 2)
        if not (value_item and isinstance(preview_widget, QLabel)): return
        
        value = value_item.text().strip()
        is_valid = is_valid_stylesheet(value)
        
        value_item.setBackground(self.INVALID_STYLE_COLOR if not is_valid else QColor(Qt.GlobalColor.white))
        try:
            preview_widget.setStyleSheet(value if is_valid else "")
        except Exception:
            preview_widget.setStyleSheet("")

    def _on_style_item_changed(self, item: QTableWidgetItem):
        self._validate_and_update_row_visuals(item.row())

    def _update_style_buttons_state(self):
        selected_items = self.styles_table.selectedItems()
        can_remove = False
        if selected_items and self.theme_combo.currentText() != 'default':
            selected_row = self.styles_table.row(selected_items[0])
            name_item = self.styles_table.item(selected_row, 0)
            if name_item and name_item.text() not in self.PROTECTED_STYLE_KEYS:
                can_remove = True
        self.btn_remove_style.setEnabled(can_remove)

    @Slot()
    def _on_add_style(self):
        if self.theme_combo.currentText() == 'default':
            QMessageBox.warning(self, "Операция запрещена", "Нельзя добавлять новые стили в тему 'default'.\nСоздайте новую тему, чтобы продолжить.")
            return

        param_name, ok = QInputDialog.getText(self, "Новый параметр", "Введите имя параметра:")
        if not (ok and param_name):
            return

        param_name = param_name.strip()

        if not param_name:
            QMessageBox.warning(self, "Ошибка", "Имя параметра не может быть пустым.")
            return
        
        if not re.match(r"^[a-zA-Z0-9_]+$", param_name):
            QMessageBox.warning(self, "Ошибка", "Имя параметра может содержать только латинские буквы, цифры и знак подчеркивания.")
            return

        for row in range(self.styles_table.rowCount()):
            if self.styles_table.item(row, 0).text() == param_name:
                QMessageBox.warning(self, "Дубликат", f"Параметр '{param_name}' уже существует.")
                return

        self._add_style_row(param_name, "color: black;", select_row=True)

    @Slot()
    def _on_remove_style(self):
        selected_rows = self.styles_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row_to_delete = selected_rows[0].row()
        name_item = self.styles_table.item(row_to_delete, 0)
        
        if self.theme_combo.currentText() == 'default':
            QMessageBox.warning(self, "Операция запрещена", "Нельзя удалять стили из темы 'default'.")
            return

        if name_item and name_item.text() in self.PROTECTED_STYLE_KEYS:
            QMessageBox.warning(self, "Операция запрещена", "Нельзя удалить базовый параметр стиля.")
            return

        self.styles_table.removeRow(row_to_delete)

    def _on_accept(self):
        for row in range(self.styles_table.rowCount()):
            if self.styles_table.item(row, 1).background() == self.INVALID_STYLE_COLOR:
                QMessageBox.warning(self, "Невалидный стиль", f"Стиль в строке {row + 1} некорректен.")
                return

        active_theme_name = self.theme_combo.currentText()
        if active_theme_name:
            current_ui_styles = self._collect_styles_from_ui()
            if current_ui_styles != self._current_theme_styles_cache:
                self.theme_manager.save_dynamic_styles(active_theme_name, current_ui_styles)
        
        self.config.general.active_theme_name = active_theme_name
        self.accept()

    def _collect_styles_from_ui(self) -> Dict[str, str]:
        style_data = {}
        for row in range(self.styles_table.rowCount()):
            name_item = self.styles_table.item(row, 0)
            value_item = self.styles_table.item(row, 1)
            if name_item and value_item and name_item.text():
                style_data[name_item.text()] = value_item.text()
        return style_data

    def get_settings_data(self) -> Optional[Dict[str, Any]]:
        if self.result() == QDialog.DialogCode.Accepted:
            self.config.paths.python_interpreter = self.py_interpreter_label.text().strip()
            self.config.paths.additional_env_paths = self.path_editor.get_paths()
            self.config.paths.python_paths = self.python_path_editor.get_paths()
            self.config.logging.log_level = cast(ValidLogLevels, self.log_level_combo.currentText())
            self.config.general.language = self.language_combo.currentData(Qt.ItemDataRole.UserRole)
            
            env_vars = {}
            for row in range(self.env_vars_table.rowCount()):
                key_item = self.env_vars_table.item(row, 0)
                value_item = self.env_vars_table.item(row, 1)
                if key_item and value_item and key_item.text():
                    env_vars[key_item.text()] = value_item.text()
            self.config.environment_variables = env_vars
            
            return self.config.model_dump(mode="python")
        return None

    def _update_env_var_buttons_state(self, edit_btn: QPushButton, remove_btn: QPushButton):
        has_selection = bool(self.env_vars_table.selectionModel().selectedRows())
        edit_btn.setEnabled(has_selection)
        remove_btn.setEnabled(has_selection)

    @Slot()
    def _browse_python_interpreter(self):
        current_path_str = self.py_interpreter_label.text()
        start_dir = str(APPLICATION_ROOT_DIR)
        if current_path_str:
            current_path = pathlib.Path(current_path_str)
            if current_path.is_file() and current_path.parent.is_dir():
                start_dir = str(current_path.parent)
        filter_key = "dialogs.settings.interpreter_filter_win" if sys.platform == "win32" else "dialogs.settings.interpreter_filter_unix"
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.locale_manager.get("dialogs.settings.interpreter_select_title"),
            start_dir, self.locale_manager.get(filter_key)
        )
        if file_path:
            self.py_interpreter_label.setText(file_path)

    @Slot()
    def _add_env_var(self):
        key, ok1 = QInputDialog.getText(self, self.locale_manager.get("dialogs.settings.env_vars.add_key_title"),
                                      self.locale_manager.get("dialogs.settings.env_vars.add_key_label"))
        if not (ok1 and key.strip()):
            return
        
        existing_keys = {self.env_vars_table.item(row, 0).text() for row in range(self.env_vars_table.rowCount())}
        if key.strip() in existing_keys:
            QMessageBox.warning(self, self.locale_manager.get("dialogs.settings.env_vars.duplicate_title"),
                                self.locale_manager.get("dialogs.settings.env_vars.duplicate_warning", key=key.strip()))
            return

        value, ok2 = QInputDialog.getText(self, self.locale_manager.get("dialogs.settings.env_vars.add_value_title"),
                                        self.locale_manager.get("dialogs.settings.env_vars.add_value_label", key=key.strip()))
        if ok2:
            row_count = self.env_vars_table.rowCount()
            self.env_vars_table.insertRow(row_count)
            self.env_vars_table.setItem(row_count, 0, QTableWidgetItem(key.strip()))
            self.env_vars_table.setItem(row_count, 1, QTableWidgetItem(value))

    @Slot()
    def _edit_env_var(self):
        selected_rows = self.env_vars_table.selectionModel().selectedRows()
        if not selected_rows: return
        
        row = selected_rows[0].row()
        key_item = self.env_vars_table.item(row, 0)
        value_item = self.env_vars_table.item(row, 1)
        key = key_item.text()
        new_value, ok = QInputDialog.getText(self, self.locale_manager.get("dialogs.settings.env_vars.edit_value_title"),
                                            self.locale_manager.get("dialogs.settings.env_vars.edit_value_label", key=key),
                                            text=value_item.text())
        if ok:
            value_item.setText(new_value)

    @Slot()
    def _remove_env_var(self):
        selected_rows = self.env_vars_table.selectionModel().selectedRows()
        if not selected_rows: return
        
        for index in sorted(selected_rows, key=lambda i: i.row(), reverse=True):
            self.env_vars_table.removeRow(index.row())

    def _show_scrollable_message(self, title: str, content: str):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        font = QFont("Courier New", 9)
        text_edit.setFont(font)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(text_edit)
        layout.addWidget(button_box)
        dialog.exec()

    @Slot()
    def _show_current_path(self):
        new_path_parts: List[str] = []
        interpreter_path_obj = pathlib.Path(self.py_interpreter_label.text())
        if interpreter_path_obj.is_file():
            new_path_parts.append(str(interpreter_path_obj.parent.resolve()))
        new_path_parts.extend(self.path_editor.get_paths())
        original_path = os.environ.get("PATH", "")
        if original_path:
            new_path_parts.extend(original_path.split(os.pathsep))
        unique_path_parts = list(dict.fromkeys(filter(None, new_path_parts)))
        content = "\n".join(unique_path_parts)
        self._show_scrollable_message(self.locale_manager.get("dialogs.settings.view_dialog.path_title"), content)

    @Slot()
    def _show_current_pythonpath(self):
        python_path_parts = self.python_path_editor.get_paths()
        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            python_path_parts.extend(existing_pythonpath.split(os.pathsep))
        unique_python_paths = list(dict.fromkeys(filter(None, python_path_parts)))
        content = "\n".join(unique_python_paths)
        if not content:
            content = self.locale_manager.get("dialogs.settings.view_dialog.empty_variable")
        self._show_scrollable_message(self.locale_manager.get("dialogs.settings.view_dialog.pythonpath_title"), content)

    @Slot()
    def _show_current_env(self):
        env = os.environ.copy()
        custom_vars = {}
        for row in range(self.env_vars_table.rowCount()):
            key_item = self.env_vars_table.item(row, 0)
            value_item = self.env_vars_table.item(row, 1)
            if key_item and value_item and key_item.text():
                custom_vars[key_item.text()] = value_item.text()
        env.update(custom_vars)
        content = "\n".join(f"{k}={v}" for k, v in sorted(env.items()))
        self._show_scrollable_message(self.locale_manager.get("dialogs.settings.view_dialog.env_title"), content)