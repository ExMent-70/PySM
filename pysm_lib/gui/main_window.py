# pysm_lib/gui/main_window.py

import pathlib
import logging
from typing import Optional

from PySide6.QtCore import Slot, QSize, Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QStatusBar,
    QMessageBox,
    QSplitter,
    QSizePolicy,
    QStyle,
    QFileDialog,
    QDialog,
    QToolButton,
    QMenu,
)
from PySide6.QtGui import QAction

from ..app_controller import AppController
from ..app_constants import (
    COLLECTION_EXTENSION,
    COLLECTION_FILE_TYPE_NAME,
)
from ..locale_manager import LocaleManager
from .console_widget import ConsoleWidget
from .dialogs import SettingsDialog, CollectionPassportDialog
from .available_scripts_widget import AvailableScriptsWidget
from .script_collection_widget import ScriptCollectionWidget

mw_logger = logging.getLogger(f"PyScriptManager.{__name__}")


class MainWindow(QMainWindow):
    def __init__(
        self,
        app_controller: AppController,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.controller = app_controller
        self.locale_manager = locale_manager
        self.base_window_title = self.locale_manager.get("main_window.base_title")
        self.initial_width = 950
        self.initial_height = 900
        self.setGeometry(100, 100, self.initial_width, self.initial_height)
        self.console_visible_state: bool = True

        self.scripts_widget: Optional[AvailableScriptsWidget] = None
        self.collection_widget: Optional[ScriptCollectionWidget] = None
        self.console_widget: Optional[ConsoleWidget] = None

        self._init_ui()
        self._connect_signals()
        self._update_window_title()

    def _init_ui(self):
        self._create_toolbar()
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        self._create_left_panel()
        self._create_right_panel()
        self._setup_splitter_and_statusbar()
        self.setMenuBar(None)

    def _create_toolbar(self):
        self.toolbar = QToolBar(
            self.locale_manager.get("main_window.toolbar.main_toolbar_title")
        )
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setFixedHeight(50)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.action_new_collection = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon),
            self.locale_manager.get("main_window.toolbar.new_collection"),
            self,
        )
        self.toolbar.addAction(self.action_new_collection)

        self.action_open_collection = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            self.locale_manager.get("main_window.toolbar.open_collection"),
            self,
        )
        self.toolbar.addAction(self.action_open_collection)

        self.save_button = QToolButton()
        self.save_button.setToolTip(
            self.locale_manager.get("main_window.toolbar.save_collection_tooltip")
        )
        self.save_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.save_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.save_button.clicked.connect(
            self._on_save_collection
        )

        save_menu = QMenu(self.save_button)
        self.action_save_collection = save_menu.addAction(
            self.locale_manager.get("main_window.toolbar.save_collection")
        )
        self.action_save_collection_as = save_menu.addAction(
            self.locale_manager.get("main_window.toolbar.save_collection_as")
        )
        self.save_button.setMenu(save_menu)
        self.toolbar.addWidget(self.save_button)

        self.toolbar.addSeparator()

        self.action_collection_passport = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView),
            self.locale_manager.get("main_window.toolbar.collection_passport"),
            self,
        )
        self.toolbar.addAction(self.action_collection_passport)

        self.action_refresh_scripts_toolbar = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            self.locale_manager.get("main_window.toolbar.refresh_scripts"),
            self,
        )
        self.toolbar.addAction(self.action_refresh_scripts_toolbar)

        self.toolbar.addSeparator()

        self.action_settings_toolbar = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton),
            self.locale_manager.get("main_window.toolbar.settings"),
            self,
        )
        self.toolbar.addAction(self.action_settings_toolbar)

        self.action_toggle_console_toolbar = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon),
            self.locale_manager.get("main_window.toolbar.toggle_console"),
            self,
        )
        self.action_toggle_console_toolbar.setCheckable(True)
        self.action_toggle_console_toolbar.setChecked(self.console_visible_state)
        self.toolbar.addAction(self.action_toggle_console_toolbar)

        self.toolbar.addSeparator()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        self.action_exit_toolbar = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TabCloseButton),
            self.locale_manager.get("main_window.toolbar.exit"),
            self,
        )
        self.action_exit_toolbar.triggered.connect(self.close)
        self.toolbar.addAction(self.action_exit_toolbar)

    def _create_left_panel(self):
        self.left_panel_splitter = QSplitter(Qt.Orientation.Vertical)
        self.scripts_widget = AvailableScriptsWidget(
            controller=self.controller, locale_manager=self.locale_manager, parent=self
        )
        self.left_panel_splitter.addWidget(self.scripts_widget)
        self.collection_widget = ScriptCollectionWidget(
            controller=self.controller, locale_manager=self.locale_manager, parent=self
        )
        self.left_panel_splitter.addWidget(self.collection_widget)
        self.main_splitter.addWidget(self.left_panel_splitter)

    def _create_right_panel(self):
        self.console_widget = ConsoleWidget(
            config_manager=self.controller.config_manager,
            locale_manager=self.locale_manager,
            parent=self
        )
        self.main_splitter.addWidget(self.console_widget)

    def _setup_splitter_and_statusbar(self):
        self.main_splitter.setSizes([450, 500])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, True)
        self.left_panel_splitter.setSizes([300, 600])
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(self.locale_manager.get("main_window.status.ready"))

    def _connect_signals(self):
        self.controller.log_message_to_console.connect(
            self.console_widget.append_to_console
        )
        self.controller.clear_console_request.connect(self.console_widget.clear_console)
        self.controller.script_progress_updated.connect(
            self.console_widget.update_progress_bar
        )
        self.controller.status_message_updated.connect(self.statusBar.showMessage)
        self.controller.collection_dirty_state_changed.connect(
            self._update_window_title
        )
        self.controller.app_busy_state_changed.connect(self._on_app_busy_state_changed)
        self.controller.config_updated.connect(self._on_config_updated)

        self.action_refresh_scripts_toolbar.triggered.connect(
            self.controller.refresh_available_scripts_list
        )
        self.action_toggle_console_toolbar.triggered.connect(
            self._toggle_console_panel_action
        )
        self.action_new_collection.triggered.connect(self._on_new_collection)
        self.action_open_collection.triggered.connect(self._on_open_collection)

        self.action_save_collection.triggered.connect(self._on_save_collection)
        self.action_save_collection_as.triggered.connect(self._on_save_collection_as)

        self.action_collection_passport.triggered.connect(
            self._on_collection_passport_clicked
        )
        self.action_settings_toolbar.triggered.connect(self._on_settings_clicked)

        self.controller.available_scripts_updated.connect(
            self.scripts_widget.update_scripts_tree
        )
        self.controller.script_info_updated.connect(
            self.scripts_widget.on_script_info_updated
        )
        self.scripts_widget.add_script_to_collection_requested.connect(
            self.controller.add_script_to_active_set_node
        )
        self.scripts_widget.focus_requested_on_collection_widget.connect(
            self._on_focus_requested_on_collection_widget
        )
        self.scripts_widget.save_passport_requested.connect(
            self.controller.save_script_passport
        )

        self.controller.current_collection_updated.connect(
            self.collection_widget.on_collection_updated
        )
        self.controller.script_instance_updated.connect(
            self.collection_widget.on_script_instance_updated
        )
        self.controller.collection_dirty_state_changed.connect(
            self.collection_widget.on_collection_dirty_state_changed
        )
        self.controller.script_instance_status_changed.connect(
            self.collection_widget._on_script_status_changed
        )
        self.controller.active_set_node_changed.connect(
            self.collection_widget.on_active_set_node_changed
        )

        self.controller.active_set_node_changed.connect(
            lambda model: self.scripts_widget.on_collection_target_selection_changed(
                model is not None
            )
        )

    @Slot()
    def _on_focus_requested_on_collection_widget(self):
        if self.collection_widget and self.collection_widget.collection_tree_view:
            self.collection_widget.collection_tree_view.setFocus()

    @Slot()
    def _on_config_updated(self):
        if self.console_widget:
            self.console_widget.apply_theme()
        if self.collection_widget:
            self.collection_widget.on_collection_dirty_state_changed(
                self.controller.set_manager.is_dirty
            )

    @Slot()
    def _on_settings_clicked(self):
        cfg = self.controller.config_manager.config
        dialog = SettingsDialog(
            config_model=cfg, locale_manager=self.locale_manager, parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings_data = dialog.get_settings_data()
            if settings_data:
                self.controller.apply_new_config(settings_data)

    @Slot()
    def _on_collection_passport_clicked(self):
        collection_model = self.controller.set_manager.current_collection_model
        dialog = CollectionPassportDialog(
            controller=self.controller,
            collection_model=collection_model,
            locale_manager=self.locale_manager,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if data:
                self.controller.update_collection_properties(
                    data["name"], data["description"]
                )
                if "context_data" in data:
                    self.controller.update_collection_context(data["context_data"])

                new_roots = data["script_roots"]
                old_roots = collection_model.script_roots
                new_paths = {r.path for r in new_roots}
                old_paths_map = {r.path: r.id for r in old_roots}
                for old_path, old_id in old_paths_map.items():
                    if old_path not in new_paths:
                        self.controller.remove_script_root_from_collection(old_id)
                for new_root in new_roots:
                    if new_root.path not in old_paths_map:
                        self.controller.add_script_root_to_collection(new_root.path)

                self._update_window_title()

    @Slot(bool)
    def _update_window_title(self, is_dirty: bool = False):
        title = (
            self.base_window_title
            + f" - [{self.controller.set_manager.current_collection_model.collection_name}]"
        )
        if is_dirty or self.controller.set_manager.is_dirty:
            title += self.locale_manager.get("main_window.dirty_indicator")
        self.setWindowTitle(title)

    @Slot(bool)
    def _on_app_busy_state_changed(self, is_busy: bool):
        is_enabled = not is_busy

        if self.scripts_widget:
            self.scripts_widget.setEnabled(is_enabled)
        if self.collection_widget:
            self.collection_widget.collection_tree_view.setEnabled(is_enabled)
            self.collection_widget.combo_set_run_mode.setEnabled(is_enabled)
            self.collection_widget.chk_continue_on_error.setEnabled(is_enabled)

        self.action_new_collection.setEnabled(is_enabled)
        self.action_open_collection.setEnabled(is_enabled)
        self.save_button.setEnabled(is_enabled)
        self.action_collection_passport.setEnabled(is_enabled)
        self.action_refresh_scripts_toolbar.setEnabled(is_enabled)
        self.action_settings_toolbar.setEnabled(is_enabled)

    def _check_unsaved_changes(self) -> bool:
        if not self.controller.set_manager.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            self.locale_manager.get("main_window.unsaved_dialog.title"),
            self.locale_manager.get("main_window.unsaved_dialog.text"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._on_save_collection()
        elif reply == QMessageBox.StandardButton.Cancel:
            return False
        return True

    @Slot()
    def _on_new_collection(self):
        if self._check_unsaved_changes():
            self.controller.new_collection_requested_by_gui()

    @Slot()
    def _on_open_collection(self):
        if not self._check_unsaved_changes():
            return

        start_dir = self.controller.set_manager.default_sets_root_dir
        if self.controller.current_collection_file_path:
            start_dir = self.controller.current_collection_file_path.parent

        file_filter = self.locale_manager.get(
            "main_window.file_dialog.filter",
            extension=COLLECTION_EXTENSION,
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.locale_manager.get("main_window.file_dialog.open_title"),
            str(start_dir),
            file_filter,
        )
        if file_path:
            self.controller.open_collection_requested_by_gui(pathlib.Path(file_path))

    @Slot()
    def _on_save_collection(self) -> bool:
        if self.controller.current_collection_file_path:
            return self.controller.save_current_collection_requested_by_gui(None)
        else:
            return self._on_save_collection_as()

    @Slot()
    def _on_save_collection_as(self) -> bool:
        start_dir = self.controller.set_manager.default_sets_root_dir
        if self.controller.current_collection_file_path:
            start_dir = self.controller.current_collection_file_path.parent

        suggested_name = (
            self.controller.set_manager.current_collection_model.collection_name
            + COLLECTION_EXTENSION
        )
        file_filter = self.locale_manager.get(
            "main_window.file_dialog.filter",
            extension=COLLECTION_EXTENSION,
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.locale_manager.get("main_window.file_dialog.save_as_title"),
            str(start_dir / suggested_name),
            file_filter,
        )
        if file_path:
            return self.controller.save_current_collection_requested_by_gui(
                pathlib.Path(file_path)
            )
        return False

    def closeEvent(self, event):
        if self._check_unsaved_changes():
            self.controller.config_manager.save_config()
            event.accept()
        else:
            event.ignore()

    @Slot(bool)
    def _toggle_console_panel_action(self, checked: bool):
        self.console_visible_state = checked
        if self.console_widget:
            self.console_widget.setVisible(checked)
        self.action_toggle_console_toolbar.setChecked(self.console_visible_state)