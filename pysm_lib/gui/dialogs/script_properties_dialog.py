# pysm_lib/gui/dialogs/script_properties_dialog.py

import pathlib
import sys
from typing import Optional
from enum import Enum

from PySide6.QtCore import Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QStyle,
    QApplication,
    QToolBox,
    QInputDialog,
    QSplitter,
    QCheckBox,
)
from PySide6.QtGui import QAction

from ...models import (
    ScriptInfoModel,
    ScriptArgMetaDetailModel,
    ScriptSetEntryValueEnabled,
    ScriptSetEntryModel,
)
from ...locale_manager import LocaleManager
from ...argument_parser import scan_for_arguments
from ...app_constants import APPLICATION_ROOT_DIR
from ..widgets.parameter_editor_widget import ParameterEditorWidget, EditorMode
from ..tooltip_generator import _generate_base_script_html


class EditMode(Enum):
    PASSPORT = 1
    INSTANCE = 2


class ScriptPropertiesDialog(QDialog):
    def __init__(
        self,
        edit_mode: EditMode,
        script_info: ScriptInfoModel,
        locale_manager: LocaleManager,
        instance_entry: Optional[ScriptSetEntryModel] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.edit_mode = edit_mode
        self.locale_manager = locale_manager
        self.script_info_model = script_info
        if instance_entry:
            self.instance_entry_model = instance_entry.model_copy(deep=True)
            self._ensure_instance_args_match_passport()
        else:
            self.instance_entry_model = None

        self.instance_name_edit: Optional[QLineEdit] = None
        self.silent_mode_checkbox: Optional[QCheckBox] = None
        self.instance_name_label: Optional[QLabel] = None
        self.instance_id_label: Optional[QLabel] = None
        self.instance_id_edit: Optional[QLineEdit] = None
        self.instance_description_edit: Optional[QTextEdit] = None
        self.parameter_editor: Optional[ParameterEditorWidget] = None
        self.args_buttons_container: Optional[QWidget] = None
        self.copy_id_action = QAction(self)

        self._init_ui()
        self._connect_signals()
        self._apply_edit_mode()
        self._populate_arguments_tab()

        if self.edit_mode == EditMode.INSTANCE:
            if self.instance_name_edit:
                display_name = (
                    self.instance_entry_model.name or self.script_info_model.name
                )
                self.instance_name_edit.setText(display_name)
            if self.instance_id_edit:
                self.instance_id_edit.setText(self.instance_entry_model.instance_id)
            if self.instance_description_edit:
                self.instance_description_edit.setPlainText(
                    self.instance_entry_model.description or ""
                )
            if self.silent_mode_checkbox:
                self.silent_mode_checkbox.setChecked(
                    self.instance_entry_model.silent_mode
                )

    def _ensure_instance_args_match_passport(self):
        if not self.instance_entry_model:
            return
        new_args = {}
        meta_args = self.script_info_model.command_line_args_meta or {}
        current_args = self.instance_entry_model.command_line_args
        for name, meta in meta_args.items():
            if name in current_args:
                new_args[name] = current_args[name]
            else:
                is_enabled = meta.required or meta.default is not None
                new_args[name] = ScriptSetEntryValueEnabled(
                    value=meta.default, enabled=is_enabled
                )
        self.instance_entry_model.command_line_args = new_args

    def _init_ui(self):
        title_key = (
            "dialogs.script_properties.title_passport"
            if self.edit_mode == EditMode.PASSPORT
            else "dialogs.script_properties.title_instance"
        )
        self.setWindowTitle(
            self.locale_manager.get(title_key, name=self.script_info_model.name)
        )
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        main_layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        self.name_edit = QLineEdit(self.script_info_model.name)
        self.description_edit = QPlainTextEdit(self.script_info_model.description or "")
        self.author_edit = QLineEdit(self.script_info_model.author or "")
        self.version_edit = QLineEdit(self.script_info_model.version or "")
        general_layout.addRow(
            self.locale_manager.get("dialogs.script_properties.general_tab.name_label"),
            self.name_edit,
        )
        general_layout.addRow(
            self.locale_manager.get(
                "dialogs.script_properties.general_tab.description_label"
            ),
            self.description_edit,
        )
        general_layout.addRow(
            self.locale_manager.get(
                "dialogs.script_properties.general_tab.author_label"
            ),
            self.author_edit,
        )
        general_layout.addRow(
            self.locale_manager.get(
                "dialogs.script_properties.general_tab.version_label"
            ),
            self.version_edit,
        )
        self.tabs.addTab(
            general_tab,
            self.locale_manager.get("dialogs.script_properties.tabs.general"),
        )

        args_tab = QWidget()
        args_layout = QVBoxLayout(args_tab)
       
        self.tabs.addTab(
            args_tab,
            self.locale_manager.get("dialogs.script_properties.tabs.arguments"),
        )

        top_form_layout = QFormLayout()
        top_form_layout.setContentsMargins(0, 5, 0, 5)
        self.instance_name_label = QLabel(
            self.locale_manager.get("dialogs.script_properties.instance_name_label")
        )
        self.instance_name_edit = QLineEdit()
        top_form_layout.addRow(self.instance_name_label, self.instance_name_edit)
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Добавляем чекбокс "Тихий режим" в форму
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        self.instance_id_label = QLabel(
            self.locale_manager.get("tooltips.instance.label_instance_id")
        )
        self.instance_id_edit = QLineEdit()
        self.instance_id_edit.setReadOnly(True)
        self.copy_id_action.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
        )
        self.copy_id_action.setToolTip("Копировать ID в буфер обмена")
        self.instance_id_edit.addAction(
            self.copy_id_action, QLineEdit.ActionPosition.TrailingPosition
        )
        top_form_layout.addRow(self.instance_id_label, self.instance_id_edit)
        args_layout.addLayout(top_form_layout)

        # --- 1. БЛОК: Создаем сплиттер ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(0)
        args_layout.addWidget(
            splitter, 1
        )  # Растягиваем сплиттер на все доступное место

        self.silent_mode_checkbox = QCheckBox(
            self.locale_manager.get("dialogs.script_properties.silent_mode_label")
        )
        args_layout.addWidget(self.silent_mode_checkbox)




        self.details_toolbox = QToolBox()
        # --- 2. БЛОК: Добавляем стили для QToolBox ---
        #        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E1E1E1, stop: 0.4 #DDDDDD, stop: 1.0 #D3D3D3);



        script_description_widget = QTextEdit()
        script_description_widget.setReadOnly(True)
        script_description_widget.setHtml(
            _generate_base_script_html(self.script_info_model, self.locale_manager)
        )
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Убираем рамку у QTextEdit, чтобы избежать двойного обрамления
        # script_description_widget.setStyleSheet("border: none;")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        self.details_toolbox.addItem(
            script_description_widget,
            self.locale_manager.get(
                "dialogs.script_properties.base_script_description_label"
            ),
        )

        self.instance_description_edit = QTextEdit()

        self.details_toolbox.addItem(
            self.instance_description_edit,
            self.locale_manager.get(
                "dialogs.script_properties.instance_description_label"
            ),
        )
        self.details_toolbox.setCurrentIndex(0)

        splitter.addWidget(self.details_toolbox)



        # --- 4. БЛОК: Создаем контейнер для таблицы и кнопок ---
        params_container_widget = QWidget()
        params_container_layout = QVBoxLayout(params_container_widget)
        params_container_layout.setContentsMargins(
            0, 5, 0, 0
        )  # Небольшой отступ сверху

        self.parameter_editor = ParameterEditorWidget(
            mode=(
                EditorMode.INSTANCE_ARGS
                if self.edit_mode == EditMode.INSTANCE
                else EditorMode.PASSPORT_ARGS
            ),
            locale_manager=self.locale_manager,
        )
        params_container_layout.addWidget(
            self.parameter_editor
        )  # Добавляем таблицу в контейнер




        self.args_buttons_container = QWidget()
        args_button_layout = QHBoxLayout(self.args_buttons_container)
        args_button_layout.setContentsMargins(0, 0, 0, 0)
        self.add_arg_btn = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.add")
        )
        self.remove_arg_btn = QPushButton(
            self.locale_manager.get("dialogs.settings.buttons.remove")
        )
        self.scan_args_btn = QPushButton(
            self.locale_manager.get("dialogs.script_properties.scan_button")
        )
        self.remove_arg_btn.setEnabled(False)
        args_button_layout.addWidget(self.add_arg_btn)
        args_button_layout.addWidget(self.remove_arg_btn)
        args_button_layout.addStretch()
        args_button_layout.addWidget(self.scan_args_btn)

        params_container_layout.addWidget(
            self.args_buttons_container
        )  # Добавляем кнопки в контейнер

        # --- 5. БЛОК: Добавляем контейнер в сплиттер ---
        splitter.addWidget(params_container_widget)
        splitter.setSizes([350, 350])  # Устанавливаем начальные размеры

        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)
        py_path_layout = QHBoxLayout()
        self.py_interpreter_edit = QLineEdit(
            self.script_info_model.specific_python_interpreter or ""
        )
        btn_browse_py = QPushButton("...")
        btn_browse_py.setFixedWidth(30)
        btn_browse_py.clicked.connect(self._on_browse_for_interpreter)
        py_path_layout.addWidget(self.py_interpreter_edit)
        py_path_layout.addWidget(btn_browse_py)
        advanced_layout.addRow(
            self.locale_manager.get(
                "dialogs.script_properties.advanced_tab.interpreter_label"
            ),
            py_path_layout,
        )
        self.tabs.addTab(
            advanced_tab,
            self.locale_manager.get("dialogs.script_properties.tabs.advanced"),
        )

        self.button_box = QDialogButtonBox()
        self.help_button = self.button_box.addButton(
            "Справка", QDialogButtonBox.ButtonRole.HelpRole
        )
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

    # ... (методы с _connect_signals по _on_copy_instance_id без изменений) ...
    def _connect_signals(self):
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        if self.edit_mode == EditMode.INSTANCE:
            self.copy_id_action.triggered.connect(self._on_copy_instance_id)
        self.help_button.clicked.connect(self._on_help_clicked)
        self.add_arg_btn.clicked.connect(self._on_add_arg)
        self.remove_arg_btn.clicked.connect(self._on_remove_arg)
        self.scan_args_btn.clicked.connect(self._on_scan_args)
        self.parameter_editor.table.itemSelectionChanged.connect(
            self._on_arg_selection_changed
        )

    def _populate_arguments_tab(self):
        args_meta = self.script_info_model.command_line_args_meta or {}
        if self.edit_mode == EditMode.PASSPORT:
            self.parameter_editor.set_data(data=args_meta)
        else:
            self.parameter_editor.set_data(
                data=args_meta,
                instance_values=self.instance_entry_model.command_line_args,
            )

    def _on_accept(self):
        if self.edit_mode == EditMode.PASSPORT:
            self.script_info_model.name = self.name_edit.text().strip()
            self.script_info_model.description = (
                self.description_edit.toPlainText().strip()
            )
            self.script_info_model.author = self.author_edit.text().strip()
            self.script_info_model.version = self.version_edit.text().strip()
            self.script_info_model.specific_python_interpreter = (
                self.py_interpreter_edit.text().strip() or None
            )
            self.script_info_model.command_line_args_meta = (
                self.parameter_editor.get_updated_meta()
            )
        elif self.edit_mode == EditMode.INSTANCE:
            if self.instance_name_edit.text().strip():
                self.instance_entry_model.name = self.instance_name_edit.text().strip()
            else:
                self.instance_entry_model.name = None

            if self.instance_description_edit.toPlainText().strip():
                self.instance_entry_model.description = (
                    self.instance_description_edit.toPlainText()
                )
            else:
                self.instance_entry_model.description = None
            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # КОММЕНТАРИЙ: Сохраняем состояние чекбокса в модель
            self.instance_entry_model.silent_mode = (
                self.silent_mode_checkbox.isChecked()
            )
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            updated_values = self.parameter_editor.get_updated_values()
            self.instance_entry_model.command_line_args = updated_values
        self.accept()

    @Slot()
    def _on_help_clicked(self):
        script_folder = pathlib.Path(self.script_info_model.folder_abs_path)
        manual_file = script_folder / "manual.md"
        if not manual_file.is_file():
            QMessageBox.information(
                self,
                "Справка",
                f"Файл справки (manual.md) отсутствует в папке скрипта:\n{script_folder}",
            )
            return
        try:
            with open(manual_file, "r", encoding="utf-8") as f:
                content = f.read()
            help_dialog = QDialog(self)
            help_dialog.setWindowTitle(
                f"Справка по скрипту: {self.script_info_model.name}"
            )
            help_dialog.setMinimumSize(700, 500)
            layout = QVBoxLayout(help_dialog)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setMarkdown(content)
            layout.addWidget(text_edit)
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            button_box.accepted.connect(help_dialog.accept)
            button_box.rejected.connect(help_dialog.reject)
            layout.addWidget(button_box)
            help_dialog.exec()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка чтения справки",
                f"Не удалось прочитать файл manual.md:\n{e}",
            )

    @Slot()
    def _on_arg_selection_changed(self):
        self.remove_arg_btn.setEnabled(
            bool(self.parameter_editor.table.selectionModel().selectedRows())
        )

    @Slot()
    def _on_add_arg(self):
        text, ok = QInputDialog.getText(
            self,
            self.locale_manager.get("dialogs.script_properties.add_arg_dialog.title"),
            self.locale_manager.get("dialogs.script_properties.add_arg_dialog.label"),
        )
        if ok and text:
            name = text.strip()
            if not name:
                return
            if self.script_info_model.command_line_args_meta is None:
                self.script_info_model.command_line_args_meta = {}
            if name in self.script_info_model.command_line_args_meta:
                QMessageBox.warning(
                    self,
                    self.locale_manager.get(
                        "dialogs.script_properties.duplicate_arg_warning.title"
                    ),
                    self.locale_manager.get(
                        "dialogs.script_properties.duplicate_arg_warning.text",
                        name=name,
                    ),
                )
                return
            self.script_info_model.command_line_args_meta[name] = (
                ScriptArgMetaDetailModel(type="string")
            )
            self._populate_arguments_tab()

    @Slot()
    def _on_remove_arg(self):
        selected_rows = self.parameter_editor.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row_index = selected_rows[0].row()
        name_item = self.parameter_editor.table.item(row_index, 1)  # Column 1 is Name
        if not name_item:
            return
        name = name_item.text()
        reply = QMessageBox.question(
            self,
            self.locale_manager.get(
                "dialogs.script_properties.delete_arg_dialog.title"
            ),
            self.locale_manager.get(
                "dialogs.script_properties.delete_arg_dialog.text", name=name
            ),
        )
        if (
            reply == QMessageBox.StandardButton.Yes
            and self.script_info_model.command_line_args_meta
            and name in self.script_info_model.command_line_args_meta
        ):
            del self.script_info_model.command_line_args_meta[name]
            self._populate_arguments_tab()

    @Slot()
    def _on_scan_args(self):
        run_file = self.script_info_model.run_file_abs_path
        if not run_file or not pathlib.Path(run_file).is_file():
            QMessageBox.warning(
                self,
                self.locale_manager.get("general.error_title"),
                self.locale_manager.get(
                    "dialogs.script_properties.scan_error.not_found", file=run_file
                ),
            )
            return
        reply = QMessageBox.question(
            self,
            self.locale_manager.get("dialogs.script_properties.scan_confirm.title"),
            self.locale_manager.get("dialogs.script_properties.scan_confirm.text"),
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        scanned_args = scan_for_arguments(run_file)
        if scanned_args:
            self.script_info_model.command_line_args_meta = scanned_args
            self._populate_arguments_tab()
            QMessageBox.information(
                self,
                self.locale_manager.get(
                    "dialogs.script_properties.scan_result.success_title"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.scan_result.success_text",
                    count=len(scanned_args),
                ),
            )
        else:
            QMessageBox.information(
                self,
                self.locale_manager.get(
                    "dialogs.script_properties.scan_result.info_title"
                ),
                self.locale_manager.get(
                    "dialogs.script_properties.scan_result.info_text"
                ),
            )

    @Slot()
    def _on_copy_instance_id(self):
        clipboard = QApplication.clipboard()
        if self.instance_id_edit and self.instance_id_edit.text():
            clipboard.setText(self.instance_id_edit.text())
            original_icon = self.copy_id_action.icon()
            original_tooltip = self.copy_id_action.toolTip()
            self.copy_id_action.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            )
            self.copy_id_action.setToolTip("Скопировано!")
            QTimer.singleShot(
                1500,
                lambda: (
                    self.copy_id_action.setIcon(original_icon),
                    self.copy_id_action.setToolTip(original_tooltip),
                ),
            )

    def _apply_edit_mode(self):
        is_passport_mode = self.edit_mode == EditMode.PASSPORT
        self.tabs.setTabVisible(0, is_passport_mode)
        self.tabs.setTabVisible(2, is_passport_mode)

        # --- 6. БЛОК: Управляем видимостью родительского виджета кнопок ---
        self.args_buttons_container.setVisible(is_passport_mode)

        self.instance_name_label.setVisible(not is_passport_mode)
        self.instance_name_edit.setVisible(not is_passport_mode)
        self.silent_mode_checkbox.setVisible(not is_passport_mode)

        self.instance_id_label.setVisible(not is_passport_mode)
        self.instance_id_edit.setVisible(not is_passport_mode)
        self.details_toolbox.setVisible(not is_passport_mode)

        if not is_passport_mode:
            self.tabs.setCurrentIndex(1)
        else:
            # В режиме паспорта сплиттер не нужен
            self.parameter_editor.parentWidget().parentWidget().setVisible(False)
            # Отображаем только редактор параметров
            self.tabs.widget(1).layout().addWidget(self.parameter_editor)
            self.tabs.widget(1).layout().addWidget(self.args_buttons_container)

    def _on_browse_for_interpreter(self):
        current_path_str = self.py_interpreter_edit.text()
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
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.locale_manager.get("dialogs.settings.interpreter_select_title"),
            start_dir,
            self.locale_manager.get(filter_key),
        )
        if file_path:
            self.py_interpreter_edit.setText(file_path)

    def get_updated_script_info_model(self) -> Optional[ScriptInfoModel]:
        return (
            self.script_info_model
            if self.result() and self.edit_mode == EditMode.PASSPORT
            else None
        )

    def get_updated_instance_entry_model(self) -> Optional[ScriptSetEntryModel]:
        return (
            self.instance_entry_model
            if self.result() and self.edit_mode == EditMode.INSTANCE
            else None
        )
