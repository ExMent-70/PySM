# pysm_lib/gui/dialogs/settings_dialog.py

import os
import pathlib
import re
import sys
from typing import List, Dict, Optional, Any, cast

# 1. БЛОК: Импорты (ИЗМЕНЕН)
# ==============================================================================
import cssutils
import logging
import xml.dom  # cssutils может выбрасывать это исключение

from PySide6.QtCore import Slot, Qt, QSignalBlocker, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QInputDialog,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QToolBox,
    QTextEdit,
)

from ...config_manager import ValidLogLevels, ConsoleStylesConfig, AppConfigModel
from ...locale_manager import LocaleManager
from typing import get_args
from ..widgets.path_list_editor import PathListEditor
from ...app_constants import APPLICATION_ROOT_DIR

# 2. БЛОК: Новая функция-валидатор и настройка логгера
# ==============================================================================
# Понижаем уровень логирования для cssutils, чтобы не засорять консоль
cssutils_log = logging.getLogger("cssutils")
cssutils_log.setLevel(logging.CRITICAL)

def is_valid_stylesheet(style_string: str) -> bool:
    """Проверяет, является ли строка валидным набором CSS-правил."""
    if not style_string.strip():
        return True
    try:
        # Создаем пустой набор стилей
        sheet = cssutils.css.CSSStyleSheet()
        # Устанавливаем его CSS-текст. cssutils попытается распарсить его.
        # Этот метод не выбрасывает исключения на семантические ошибки по умолчанию.
        sheet.cssText = f"A{{ {style_string} }}" # Оборачиваем в фиктивный селектор
        # Свойство 'valid' будет False, если были ЛЮБЫЕ ошибки, включая семантические.
        return sheet.valid
    except Exception:
        # Ловим любые другие неожиданные ошибки
        return False


class SettingsDialog(QDialog):
    PROTECTED_STYLE_KEYS = {
        "api_image_description",
        "api_link",
        "set_header",
        "set_info",
        "script_header_block",
        "script_success_block",
        "script_error_block",
        "script_info",
        "script_stdout",
        "script_stderr",
        "runner_info",
        "script_arg_value",
        "tooltip_script_args_block",
        "tooltip_instance_args_block",                
        "console_background",
        "collection_info",
        "status_running",
        "status_success",
        "status_error",
        "status_pending",
        "status_skipped",
        "script_description",
        "tooltip_arg_value",
    }
    
    # Константа COLOR_ONLY_STYLE_KEYS УДАЛЕНА
    
    INVALID_STYLE_COLOR = QColor("#fdecf0")


    def __init__(
        self,
        config_model: AppConfigModel,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.config = config_model.model_copy(deep=True)
        self.styles_table: Optional[QTableWidget] = None
        self.btn_view_path: Optional[QPushButton] = None
        self.btn_view_python_path: Optional[QPushButton] = None
        self.btn_view_env: Optional[QPushButton] = None

        self.setWindowTitle(self.locale_manager.get("dialogs.settings.title"))
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()      
        main_layout.addWidget(self.tabs)
                # --- ВОТ ПРАВИЛЬНОЕ МЕСТО ДЛЯ СТИЛЕЙ ---
        self.tabs.setStyleSheet(
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

        main_settings_widget = QWidget()
        main_settings_layout = QVBoxLayout(main_settings_widget)
        self.tabs.addTab(
            main_settings_widget, self.locale_manager.get("dialogs.settings.tabs.main")
        )

        styles_widget = QWidget()
        self.styles_layout = QVBoxLayout(styles_widget)
        self.tabs.addTab(
            styles_widget, self.locale_manager.get("dialogs.settings.tabs.appearance")
        )

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(self.button_box)

        self._populate_main_tab(main_settings_layout)
        self._populate_styles_tab(self.styles_layout)

        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

    def _populate_main_tab(self, layout: QVBoxLayout):
        py_interpreter_group = QGroupBox(
            self.locale_manager.get("dialogs.settings.interpreter_group")
        )
        py_path_layout = QHBoxLayout(py_interpreter_group)
        self.py_interpreter_label = QLabel(self.config.paths.python_interpreter)
        self.py_interpreter_label.setWordWrap(True)
        self.py_interpreter_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.btn_browse_py_inline = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.change")
        )
        self.btn_browse_py_inline.clicked.connect(self._browse_python_interpreter)
        py_path_layout.addWidget(self.py_interpreter_label, 1)
        py_path_layout.addWidget(self.btn_browse_py_inline)
        layout.addWidget(py_interpreter_group)

        self.paths_tool_box = QToolBox()
        self.paths_tool_box.setStyleSheet(
            """
            QToolBox::tab {
                background-color: #ffc300;
                border-radius: 5px;
                color: black;
                padding: 4px;
            }
            QToolBox::tab:!selected:hover {
                color: #ffffff;
                font: bold;
                
            }            
            QToolBox::tab:selected {
                font: bold;
                color: black;
                background-color: white;
            }
        """
        )
        layout.addWidget(self.paths_tool_box, 1)

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
        self.btn_view_path = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.view")
        )
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
        self.btn_view_python_path = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.view")
        )
        python_path_bottom_layout.addWidget(self.btn_view_python_path)

        env_vars_widget = QWidget()
        env_vars_layout = QVBoxLayout(env_vars_widget)
        env_vars_layout.setContentsMargins(0, 0, 0, 0)
        self.env_vars_table = QTableWidget()
        self.env_vars_table.setColumnCount(2)
        self.env_vars_table.setHorizontalHeaderLabels(
            [
                self.locale_manager.get("dialogs.settings.env_vars.header_key"),
                self.locale_manager.get("dialogs.settings.env_vars.header_value"),
            ]
        )
        self.env_vars_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.env_vars_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.env_vars_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.env_vars_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        for key, value in sorted(self.config.environment_variables.items()):
            row_count = self.env_vars_table.rowCount()
            self.env_vars_table.insertRow(row_count)
            self.env_vars_table.setItem(row_count, 0, QTableWidgetItem(key))
            self.env_vars_table.setItem(row_count, 1, QTableWidgetItem(value))
        env_vars_layout.addWidget(self.env_vars_table)
        env_buttons_layout = QHBoxLayout()
        self.btn_add_env_var = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.add")
        )
        self.btn_edit_env_var = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.edit")
        )
        self.btn_remove_env_var = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.remove")
        )
        env_buttons_layout.addWidget(self.btn_add_env_var)
        env_buttons_layout.addWidget(self.btn_edit_env_var)
        env_buttons_layout.addWidget(self.btn_remove_env_var)
        env_buttons_layout.addStretch()
        self.btn_view_env = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.view")
        )
        env_buttons_layout.addWidget(self.btn_view_env)
        env_vars_layout.addLayout(env_buttons_layout)

        self.paths_tool_box.addItem(
            additional_paths_widget,
            self.locale_manager.get("dialogs.settings.path_group"),
        )
        self.paths_tool_box.addItem(
            python_paths_widget,
            self.locale_manager.get("dialogs.settings.pythonpath_group"),
        )
        self.paths_tool_box.addItem(
            env_vars_widget,
            self.locale_manager.get("dialogs.settings.env_vars.group_title"),
        )

        bottom_settings_layout = QHBoxLayout()
        logging_group = QGroupBox(
            self.locale_manager.get("dialogs.settings.logging_group")
        )
        logging_form_layout = QFormLayout(logging_group)
        self.log_level_combo = QComboBox()
        self.valid_log_levels: tuple[str, ...] = get_args(ValidLogLevels)
        self.log_level_combo.addItems(self.valid_log_levels)
        self.log_level_combo.setCurrentText(self.config.logging.log_level)
        logging_form_layout.addRow(
            self.locale_manager.get("dialogs.settings.logging_label"),
            self.log_level_combo,
        )
        lang_group = QGroupBox(
            self.locale_manager.get("dialogs.settings.language_group")
        )
        lang_form_layout = QFormLayout(lang_group)
        self.language_combo = QComboBox()
        available_langs = self.locale_manager.get_available_languages()
        for code, name in sorted(available_langs.items()):
            self.language_combo.addItem(name, code)
        current_lang_index = self.language_combo.findData(self.config.general.language)
        if current_lang_index != -1:
            self.language_combo.setCurrentIndex(current_lang_index)
        lang_form_layout.addRow(
            self.locale_manager.get("dialogs.settings.language_label"),
            self.language_combo,
        )

        bottom_settings_layout.addWidget(logging_group)
        bottom_settings_layout.addWidget(lang_group)
        layout.addLayout(bottom_settings_layout)
        layout.addWidget(
            QLabel(self.locale_manager.get("dialogs.settings.restart_note"))
        )

        self.btn_add_env_var.clicked.connect(self._add_env_var)
        self.btn_edit_env_var.clicked.connect(self._edit_env_var)
        self.btn_remove_env_var.clicked.connect(self._remove_env_var)
        self.env_vars_table.itemSelectionChanged.connect(
            self._update_env_var_buttons_state
        )
        self.env_vars_table.doubleClicked.connect(self._edit_env_var)
        self._update_env_var_buttons_state()

        self.btn_view_path.clicked.connect(self._show_current_path)
        self.btn_view_python_path.clicked.connect(self._show_current_pythonpath)
        self.btn_view_env.clicked.connect(self._show_current_env)


    def _populate_styles_tab(self, layout: QVBoxLayout):
        themes_group = QGroupBox(
            self.locale_manager.get("dialogs.settings.themes.group_title")
        )
        themes_layout = QHBoxLayout(themes_group)
        themes_layout.addWidget(
            QLabel(
                self.locale_manager.get("dialogs.settings.themes.current_theme_label")
            )
        )
        self.theme_combo = QComboBox()
        themes_layout.addWidget(self.theme_combo, 1)
        self.theme_save_as_btn = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.save_as")
        )
        self.theme_delete_btn = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.remove")
        )
        themes_layout.addWidget(self.theme_save_as_btn)
        themes_layout.addWidget(self.theme_delete_btn)
        layout.addWidget(themes_group)

        self.styles_table = QTableWidget()
        self.styles_table.setColumnCount(3)
        self.styles_table.setHorizontalHeaderLabels(
            ["Параметр", "CSS-свойства", "Предпросмотр"]
        )
        header = self.styles_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.styles_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.styles_table, 1)

        style_buttons_layout = QHBoxLayout()
        self.btn_add_style = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.add")
        )
        self.btn_remove_style = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.remove")
        )
        style_buttons_layout.addWidget(self.btn_add_style)
        style_buttons_layout.addWidget(self.btn_remove_style)
        style_buttons_layout.addStretch()
        layout.addLayout(style_buttons_layout)

        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        self.theme_save_as_btn.clicked.connect(self._on_save_theme_as)
        self.theme_delete_btn.clicked.connect(self._on_delete_theme)
        self.styles_table.itemChanged.connect(self._on_style_item_changed)
        self.styles_table.itemSelectionChanged.connect(self._update_style_buttons_state)
        self.btn_add_style.clicked.connect(self._on_add_style)
        self.btn_remove_style.clicked.connect(self._on_remove_style)

        self._update_theme_list()

    def _update_theme_list(self):
        with QSignalBlocker(self.theme_combo):
            self.theme_combo.clear()
            for theme_name in sorted(self.config.themes.keys()):
                self.theme_combo.addItem(theme_name)
            current_idx = self.theme_combo.findText(
                self.config.general.active_theme_name
            )
            if current_idx != -1:
                self.theme_combo.setCurrentIndex(current_idx)
            else:
                self.config.general.active_theme_name = "default"
                self.theme_combo.setCurrentIndex(self.theme_combo.findText("default"))
        QTimer.singleShot(0, lambda: self._on_theme_selected(self.theme_combo.currentIndex()))

    def _on_theme_selected(self, index: int):
        theme_name = self.theme_combo.itemText(index)
        if not theme_name:
            return

        self.config.general.active_theme_name = theme_name
        is_default_theme = theme_name == "default"
        self.theme_delete_btn.setEnabled(not is_default_theme)
        self.btn_add_style.setEnabled(not is_default_theme)
        
        theme_styles = self.config.themes.get(theme_name, ConsoleStylesConfig())
        style_dict = theme_styles.model_dump()

        with QSignalBlocker(self.styles_table):
            self.styles_table.setRowCount(0)
            
            all_keys = sorted(style_dict.keys())
            protected_keys = [k for k in all_keys if k in self.PROTECTED_STYLE_KEYS]
            custom_keys = [k for k in all_keys if k not in self.PROTECTED_STYLE_KEYS]

            for name in protected_keys:
                 self._add_style_row(name, style_dict[name])
            for name in custom_keys:
                 self._add_style_row(name, style_dict[name])

        self.styles_table.resizeColumnsToContents()
        self._update_style_buttons_state()

    def _add_style_row(self, name: str, value: str, select_row: bool = False):
        row_position = self.styles_table.rowCount()
        self.styles_table.insertRow(row_position)

        name_item = QTableWidgetItem(name)
        value_item = QTableWidgetItem(value)

        is_protected = name in self.PROTECTED_STYLE_KEYS
        is_default_theme = self.theme_combo.currentText() == "default"

        if is_protected or is_default_theme:
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setBackground(QColor("#f0f0f0"))
            if is_protected:
                name_item.setToolTip("Базовый параметр, имя изменить нельзя.")
            else:
                 name_item.setToolTip("Имена параметров в теме 'default' менять нельзя.")
        else:
            name_item.setBackground(QColor("#eaf5ea"))
            name_item.setToolTip("Пользовательский параметр.")

        self.styles_table.setItem(row_position, 0, name_item)
        self.styles_table.setItem(row_position, 1, value_item)

        preview_widget = QLabel(self.locale_manager.get("dialogs.settings.styles_preview_text"))
        self.styles_table.setCellWidget(row_position, 2, preview_widget)

        self._validate_and_update_row_visuals(row_position)

        if select_row:
            self.styles_table.selectRow(row_position)

    # 3. БЛОК: Новый вспомогательный метод для валидации и обновления строки
    # ==============================================================================
    def _validate_and_update_row_visuals(self, row: int):
        """Проверяет валидность стиля в строке и обновляет ее внешний вид."""
        value_item = self.styles_table.item(row, 1)
        preview_widget = self.styles_table.cellWidget(row, 2)

        if not (value_item and isinstance(preview_widget, QLabel)):
            return

        value = value_item.text().strip()
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Используем новую, более надежную функцию-валидатор
        is_valid = is_valid_stylesheet(value)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        value_item.setBackground(self.INVALID_STYLE_COLOR if not is_valid else QColor(Qt.GlobalColor.white))
        
        try:
            preview_widget.setStyleSheet(value if is_valid else "")
        except Exception:
            preview_widget.setStyleSheet("")

    def _on_style_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        col = item.column()
        
        # Если изменилось имя параметра (колонка 0)
        if col == 0:
            # ... (логика валидации имени, как в прошлой версии) ...
            new_name = item.text().strip()
            if not new_name:
                QMessageBox.warning(self, "Ошибка", "Имя параметра не может быть пустым.")
                # TODO: Implement rollback
                return

            for r in range(self.styles_table.rowCount()):
                if r != row and self.styles_table.item(r, 0).text() == new_name:
                    QMessageBox.warning(self, "Дубликат", f"Параметр '{new_name}' уже существует.")
                    # TODO: Implement rollback
                    return
            
            if not re.match(r"^[a-zA-Z0-9_]+$", new_name):
                QMessageBox.warning(self, "Ошибка", "Имя параметра может содержать только латинские буквы, цифры и знак подчеркивания.")
                # TODO: Implement rollback
                return
        
        # Если изменилось значение CSS (колонка 1)
        elif col == 1:
            self._validate_and_update_row_visuals(row)

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


    def _update_style_buttons_state(self):
        selected_items = self.styles_table.selectedItems()
        can_remove = False
        if selected_items:
            selected_row = self.styles_table.row(selected_items[0])
            name_item = self.styles_table.item(selected_row, 0)
            if name_item and name_item.text() not in self.PROTECTED_STYLE_KEYS and self.theme_combo.currentText() != 'default':
                can_remove = True
        
        self.btn_remove_style.setEnabled(can_remove)

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

    def _collect_styles_from_ui(self) -> Dict[str, str]:
        style_data = {}
        for row in range(self.styles_table.rowCount()):
            name_item = self.styles_table.item(row, 0)
            value_item = self.styles_table.item(row, 1)
            if name_item and value_item and name_item.text():
                style_data[name_item.text()] = value_item.text()
        return style_data
    
    # 4. БЛОК: Метод _on_accept (ИЗМЕНЕН)
    # ==============================================================================
    def _on_accept(self):
        # Проверяем все строки на валидность перед сохранением
        for row in range(self.styles_table.rowCount()):
            value_item = self.styles_table.item(row, 1)
            # Проверяем по цвету фона, который мы установили при валидации
            if value_item.background() == self.INVALID_STYLE_COLOR:
                param_name = self.styles_table.item(row, 0).text()
                QMessageBox.warning(self, "Невалидный стиль", f"Стиль для параметра '{param_name}' (строка {row + 1}) некорректен. Пожалуйста, исправьте его перед сохранением.")
                return

        active_theme_name = self.theme_combo.currentText()
        if active_theme_name:
            style_dict = self._collect_styles_from_ui()
            self.config.themes[active_theme_name] = ConsoleStylesConfig(**style_dict)
            
        self.accept()
        
    def _on_save_theme_as(self):
        current_name = self.theme_combo.currentText()
        new_name, ok = QInputDialog.getText(
            self,
            self.locale_manager.get("dialogs.settings.themes.save_as_title"),
            self.locale_manager.get("dialogs.settings.themes.save_as_label"),
            text=f"{current_name}_copy",
        )
        if ok and new_name and new_name != "default":
            style_dict = self._collect_styles_from_ui()
            new_theme = ConsoleStylesConfig(**style_dict)
            self.config.themes[new_name] = new_theme
            self.config.general.active_theme_name = new_name
            self._update_theme_list()

    def _on_delete_theme(self):
        theme_to_delete = self.theme_combo.currentText()
        if not theme_to_delete or theme_to_delete == "default":
            return
        reply = QMessageBox.question(
            self,
            self.locale_manager.get("dialogs.settings.themes.delete_confirm_title"),
            self.locale_manager.get(
                "dialogs.settings.themes.delete_confirm_text", name=theme_to_delete
            ),
        )
        if reply == QMessageBox.StandardButton.Yes:
            if theme_to_delete in self.config.themes:
                del self.config.themes[theme_to_delete]
            self.config.general.active_theme_name = "default"
            self._update_theme_list()
            
    def get_settings_data(self) -> Optional[Dict[str, Any]]:
        if self.result() == QDialog.DialogCode.Accepted:
            self.config.paths.python_interpreter = (
                self.py_interpreter_label.text().strip()
            )
            self.config.paths.additional_env_paths = self.path_editor.get_paths()
            self.config.paths.python_paths = self.python_path_editor.get_paths()

            self.config.logging.log_level = cast(
                ValidLogLevels, self.log_level_combo.currentText()
            )
            self.config.general.language = self.language_combo.currentData(
                Qt.ItemDataRole.UserRole
            )

            env_vars = {}
            for row in range(self.env_vars_table.rowCount()):
                key_item = self.env_vars_table.item(row, 0)
                value_item = self.env_vars_table.item(row, 1)
                if key_item and value_item and key_item.text():
                    env_vars[key_item.text()] = value_item.text()
            self.config.environment_variables = env_vars

            return self.config.model_dump(mode="python")
        return None

    def _update_env_var_buttons_state(self):
        has_selection = bool(self.env_vars_table.selectionModel().selectedRows())
        self.btn_edit_env_var.setEnabled(has_selection)
        self.btn_remove_env_var.setEnabled(has_selection)

    @Slot()
    def _add_path_to_list_editor(self, target_editor: PathListEditor):
        start_dir = self._last_browsed_path

        directory = QFileDialog.getExistingDirectory(
            self,
            target_editor.dialog_title,
            start_dir,
        )

        if directory:
            self._last_browsed_path = directory

            items = target_editor.get_paths()
            if directory not in items:
                target_editor.list_widget.addItem(directory)
            else:
                QMessageBox.information(
                    self,
                    self.locale_manager.get("path_list_editor.duplicate_dialog.title"),
                    self.locale_manager.get(
                        "path_list_editor.duplicate_dialog.text", path=directory
                    ),
                )

    def _browse_python_interpreter(self):
        current_path_str = self.py_interpreter_label.text()
        start_dir = str(APPLICATION_ROOT_DIR)

        if current_path_str:
            current_path = pathlib.Path(current_path_str)
            if current_path.is_file() and current_path.parent.is_dir():
                start_dir = str(current_path.parent)

        filter_key = (
            "dialogs.settings.interpreter_filter_win"
            if sys.platform == "win32"
            else "dialogs.settings.interpreter_filter_unix"
        )
        filter_str = self.locale_manager.get(filter_key)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.locale_manager.get("dialogs.settings.interpreter_select_title"),
            start_dir,
            filter_str,
        )
        if file_path:
            self.py_interpreter_label.setText(file_path)

    def _add_env_var(self):
        key, ok1 = QInputDialog.getText(
            self,
            self.locale_manager.get("dialogs.settings.env_vars.add_key_title"),
            self.locale_manager.get("dialogs.settings.env_vars.add_key_label"),
        )
        if not (ok1 and key.strip()):
            return

        existing_keys = {
            self.env_vars_table.item(row, 0).text()
            for row in range(self.env_vars_table.rowCount())
        }
        if key.strip() in existing_keys:
            QMessageBox.warning(
                self,
                self.locale_manager.get("dialogs.settings.env_vars.duplicate_title"),
                self.locale_manager.get(
                    "dialogs.settings.env_vars.duplicate_warning", key=key.strip()
                ),
            )
            return
        value, ok2 = QInputDialog.getText(
            self,
            self.locale_manager.get("dialogs.settings.env_vars.add_value_title"),
            self.locale_manager.get(
                "dialogs.settings.env_vars.add_value_label", key=key.strip()
            ),
        )
        if ok2:
            row_count = self.env_vars_table.rowCount()
            self.env_vars_table.insertRow(row_count)
            self.env_vars_table.setItem(row_count, 0, QTableWidgetItem(key.strip()))
            self.env_vars_table.setItem(row_count, 1, QTableWidgetItem(value))

    def _edit_env_var(self):
        selected_rows = self.env_vars_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        key_item = self.env_vars_table.item(row, 0)
        value_item = self.env_vars_table.item(row, 1)
        key = key_item.text()
        new_value, ok = QInputDialog.getText(
            self,
            self.locale_manager.get("dialogs.settings.env_vars.edit_value_title"),
            self.locale_manager.get(
                "dialogs.settings.env_vars.edit_value_label", key=key
            ),
            text=value_item.text(),
        )
        if ok:
            value_item.setText(new_value)

    def _remove_env_var(self):
        selected_rows = self.env_vars_table.selectionModel().selectedRows()
        if not selected_rows:
            return

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

        self._show_scrollable_message(
            self.locale_manager.get("dialogs.settings.view_dialog.path_title"), content
        )

    @Slot()
    def _show_current_pythonpath(self):
        python_path_parts = self.python_path_editor.get_paths()

        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            python_path_parts.extend(existing_pythonpath.split(os.pathsep))

        unique_python_paths = list(dict.fromkeys(filter(None, python_path_parts)))
        content = "\n".join(unique_python_paths)

        if not content:
            content = self.locale_manager.get(
                "dialogs.settings.view_dialog.empty_variable"
            )
        self._show_scrollable_message(
            self.locale_manager.get("dialogs.settings.view_dialog.pythonpath_title"),
            content,
        )

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

        self._show_scrollable_message(
            self.locale_manager.get("dialogs.settings.view_dialog.env_title"),
            content,
        )