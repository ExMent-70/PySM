# pysm_lib/gui/available_scripts_widget.py

import logging
from typing import List, Optional, Dict

from PySide6.QtCore import Qt, Slot, QSize, Signal, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTreeView, QPushButton,
    QAbstractItemView, QHeaderView, QStyle, QMessageBox, QMenu
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QAction, QPalette

from ..models import ScriptInfoModel, CategoryNodeModel, ScanTreeNodeType
from ..locale_manager import LocaleManager
from ..theme_manager import ThemeManager
from .dialogs import ScriptPropertiesDialog, EditMode
from .tooltip_generator import generate_script_tooltip_html


class AvailableScriptsWidget(QWidget):
    add_script_to_collection_requested = Signal(str)
    focus_requested_on_collection_widget = Signal()
    save_passport_requested = Signal(str, ScriptInfoModel)

    def __init__(
        self,
        controller: "AppController",
        theme_manager: ThemeManager,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(f"PyScriptManager.{self.__class__.__name__}")
        self.controller = controller
        self.theme_manager = theme_manager
        self.locale_manager = locale_manager
        self._is_add_target_selected_in_collection: bool = False
        self._items_by_id: Dict[str, QStandardItem] = {}
        self._init_ui()
        self._connect_signals()
        self.logger.debug(self.locale_manager.get("available_scripts_widget.log_debug.init"))

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scripts_groupbox = QGroupBox(self.locale_manager.get("available_scripts_widget.group_title"))
        scripts_layout = QVBoxLayout(scripts_groupbox)
        main_layout.addWidget(scripts_groupbox)
        self.tree_available_scripts = QTreeView()
        self.tree_available_scripts.setAlternatingRowColors(True)
        self.tree_available_scripts.setIconSize(QSize(24, 24))
        self.tree_available_scripts.setHeaderHidden(False)
        self.tree_available_scripts.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree_available_scripts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_available_scripts.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_available_scripts.setUniformRowHeights(True)

        scripts_layout.addWidget(self.tree_available_scripts)
        self.scripts_model = QStandardItemModel()
        self.scripts_model.setColumnCount(2)
        self.scripts_model.setHorizontalHeaderLabels([
            self.locale_manager.get("available_scripts_widget.header.script_name"),
            self.locale_manager.get("available_scripts_widget.header.description"),
        ])
        self.tree_available_scripts.setModel(self.scripts_model)
        if header := self.tree_available_scripts.header():
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.resizeSection(0, 200)
            header.setMinimumSectionSize(200)

        script_actions_layout = QHBoxLayout()
        self.btn_add_to_set = QPushButton(self.locale_manager.get("available_scripts_widget.add_to_set_button"))
        self.btn_add_to_set.setEnabled(False)
        script_actions_layout.addWidget(self.btn_add_to_set)
        scripts_layout.addLayout(script_actions_layout)

    def _connect_signals(self):
        self.tree_available_scripts.selectionModel().selectionChanged.connect(self._on_tree_selection_changed)
        self.tree_available_scripts.doubleClicked.connect(self._on_double_clicked)
        self.tree_available_scripts.customContextMenuRequested.connect(self._show_context_menu)
        self.btn_add_to_set.clicked.connect(self._on_add_to_collection_clicked)
        self.controller.config_updated.connect(self._update_all_tooltips)
        
    @Slot()
    def _update_all_tooltips(self):
        root = self.scripts_model.invisibleRootItem()
        self._recursive_tooltip_update(root)

    def _recursive_tooltip_update(self, parent_item: QStandardItem):
        for row in range(parent_item.rowCount()):
            if not (item_col0 := parent_item.child(row, 0)): continue
            
            item_data = item_col0.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_data, ScriptInfoModel):
                tooltip_html = generate_script_tooltip_html(item_data, self.locale_manager, self.theme_manager)
                item_col0.setToolTip(tooltip_html)
                if item_col1 := parent_item.child(row, 1):
                    item_col1.setToolTip(tooltip_html)

            if item_col0.hasChildren(): self._recursive_tooltip_update(item_col0)

    @Slot(list)
    def update_scripts_tree(self, root_nodes: List[ScanTreeNodeType]):
        self.logger.debug(self.locale_manager.get("available_scripts_widget.log_debug.update_tree", count=len(root_nodes)))
        self.scripts_model.clear()
        self._items_by_id.clear()
        self.scripts_model.setColumnCount(2)
        self.scripts_model.setHorizontalHeaderLabels([
            self.locale_manager.get("available_scripts_widget.header.script_name"),
            self.locale_manager.get("available_scripts_widget.header.description"),
        ])
        if not root_nodes:
            placeholder_item = QStandardItem(self.locale_manager.get("available_scripts_widget.no_scripts_placeholder"))
            placeholder_item.setEditable(False)
            placeholder_item.setSelectable(False)
            self.scripts_model.appendRow([placeholder_item, QStandardItem("")])
        else:
            self._populate_tree_recursive(self.scripts_model.invisibleRootItem(), root_nodes)
            self.tree_available_scripts.expandToDepth(0)
        self._update_add_button_state()

    @Slot(ScriptInfoModel)
    def on_script_info_updated(self, updated_model: ScriptInfoModel):
        item_col0 = self._items_by_id.get(updated_model.id)
        if not item_col0: return

        item_col0.setData(updated_model, Qt.ItemDataRole.UserRole)
        parent = item_col0.parent() or self.scripts_model.invisibleRootItem()
        if not (item_col1 := parent.child(item_col0.row(), 1)): return

        item_col0.setText(updated_model.name)
        description_text = updated_model.description or ""
        tooltip_html = generate_script_tooltip_html(updated_model, self.locale_manager, self.theme_manager)

        palette = self.palette()
        item_col0.setForeground(palette.color(QPalette.ColorRole.Text))
        item_col1.setForeground(palette.color(QPalette.ColorRole.Text))

        if not updated_model.passport_valid:
            item_col0.setForeground(QColor("red"))
            item_col1.setForeground(QColor("red"))
            item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
            description_text = f"[{updated_model.passport_error}]" if not updated_model.description else self.locale_manager.get("general.passport_error_format", description=updated_model.description)
        elif updated_model.is_raw:
            item_col0.setForeground(QColor("#808000"))
            item_col1.setForeground(QColor("#808000"))
            item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton))
            description_text = self.locale_manager.get("general.setup_required_format")
        else:
            item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))

        item_col1.setText(description_text)
        item_col0.setToolTip(tooltip_html)
        item_col1.setToolTip(tooltip_html)
        self.tree_available_scripts.setCurrentIndex(item_col0.index())

    @Slot(bool)
    def on_collection_target_selection_changed(self, is_target_selected: bool):
        self._is_add_target_selected_in_collection = is_target_selected
        self._update_add_button_state()

    def _populate_tree_recursive(self, parent_qt_item: QStandardItem, nodes_data: List[ScanTreeNodeType]):
        for node_data in nodes_data:
            item_col0 = QStandardItem()
            item_col1 = QStandardItem()
            item_col0.setData(node_data, Qt.ItemDataRole.UserRole)
            self._items_by_id[node_data.id] = item_col0

            if isinstance(node_data, CategoryNodeModel):
                item_col0.setText(node_data.name)
                item_col1.setText(node_data.description or "")
                item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirClosedIcon))
                parent_qt_item.appendRow([item_col0, item_col1])
                if node_data.children: self._populate_tree_recursive(item_col0, node_data.children)
            elif isinstance(node_data, ScriptInfoModel):
                item_col0.setText(node_data.name)
                description_text = node_data.description or ""
                tooltip_html = generate_script_tooltip_html(node_data, self.locale_manager, self.theme_manager)

                if not node_data.passport_valid:
                    item_col0.setForeground(QColor("red"))
                    item_col1.setForeground(QColor("red"))
                    item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
                    description_text = f"[{node_data.passport_error}]" if not node_data.description else self.locale_manager.get("general.passport_error_format", description=node_data.description)
                elif node_data.is_raw:
                    item_col0.setForeground(QColor("#808000"))
                    item_col1.setForeground(QColor("#808000"))
                    item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton))
                    description_text = self.locale_manager.get("general.setup_required_format")
                else:
                    item_col0.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))

                item_col1.setText(description_text)
                item_col0.setToolTip(tooltip_html)
                item_col1.setToolTip(tooltip_html)
                parent_qt_item.appendRow([item_col0, item_col1])

    @Slot()
    def _on_tree_selection_changed(self):
        self._update_add_button_state()



    def _update_add_button_state(self):
        selected_model = self.get_selected_script_model()
        can_add = (
            selected_model is not None
            and selected_model.passport_valid
            and not selected_model.is_raw
            and self._is_add_target_selected_in_collection
        )
        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        # Явно преобразуем результат в bool. bool(None) вернет False,
        # что предотвратит TypeError и является безопасным поведением.
        self.btn_add_to_set.setEnabled(bool(can_add))
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---



    @Slot()
    def _on_add_to_collection_clicked(self):
        if selected_model := self.get_selected_script_model():
            self.add_script_to_collection_requested.emit(selected_model.id)
            self.focus_requested_on_collection_widget.emit()
        else:
            QMessageBox.warning(self, self.locale_manager.get("available_scripts_widget.add_warning.title"), self.locale_manager.get("available_scripts_widget.add_warning.text"))

    def get_selected_script_model(self) -> Optional[ScriptInfoModel]:
        current_index = self.tree_available_scripts.currentIndex()
        if not current_index.isValid(): return None
        if item := self.scripts_model.itemFromIndex(current_index.siblingAtColumn(0)):
            if isinstance(item_data := item.data(Qt.ItemDataRole.UserRole), ScriptInfoModel):
                return item_data
        return None

    @Slot(QModelIndex)
    def _on_double_clicked(self, index: QModelIndex):
        if not index.isValid(): return
        if not (item := self.scripts_model.itemFromIndex(index.siblingAtColumn(0))): return
        if not isinstance(item_data := item.data(Qt.ItemDataRole.UserRole), ScriptInfoModel): return
        self._show_script_properties_dialog(item_data.model_copy(deep=True))

    @Slot("QPoint")
    def _show_context_menu(self, position):
        index = self.tree_available_scripts.indexAt(position)
        if not index.isValid(): return
        if not (item := self.scripts_model.itemFromIndex(index.siblingAtColumn(0))): return
        
        if isinstance(item_data := item.data(Qt.ItemDataRole.UserRole), ScriptInfoModel):
            menu = QMenu(self)
            action_configure = QAction(self.locale_manager.get("available_scripts_widget.context_menu.configure"), self)
            action_configure.triggered.connect(lambda: self._show_script_properties_dialog(item_data.model_copy(deep=True)))
            menu.addAction(action_configure)
            menu.exec(self.tree_available_scripts.viewport().mapToGlobal(position))

    def _show_script_properties_dialog(self, script_info: ScriptInfoModel):
        dialog = ScriptPropertiesDialog(
            edit_mode=EditMode.PASSPORT, script_info=script_info,
            locale_manager=self.locale_manager, parent=self
        )
        if dialog.exec():
            original_id = script_info.id
            if updated_model := dialog.get_updated_script_info_model():
                self.save_passport_requested.emit(original_id, updated_model)