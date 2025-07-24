# pysm_lib/gui/script_collection_widget.py

import logging
import re
from typing import List, Optional, Dict, Union, TYPE_CHECKING, Set, Tuple

from PySide6.QtCore import Qt, QSize, Slot, QModelIndex, QMimeData
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QTreeView,
    QPushButton,
    QAbstractItemView,
    QStyle,
    QMessageBox,
    QComboBox,
    QLabel,
    QMenu,
    QInputDialog,
    QCheckBox,
)
from PySide6.QtGui import (
    QStandardItemModel,
    QStandardItem,
    QColor,
    QAction,
    QBrush,
    QPalette,
)

from ..models import (
    SetHierarchyNodeType,
    SetFolderNodeModel,
    ScriptSetNodeModel,
    ScriptSetEntryModel,
)
from ..config_manager import ConsoleStylesConfig
from ..config_manager import ConfigManager # <--- НОВЫЙ ИМПОРТ
from .gui_utils import resolve_themed_text # <--- ИЗМЕНЕН ИМПОРТ
from ..app_enums import SetRunMode, AppState, ScriptRunStatus
from ..locale_manager import LocaleManager
from .dialogs import ScriptPropertiesDialog, EditMode
from .tooltip_generator import generate_instance_tooltip_html

if TYPE_CHECKING:
    from ..app_controller import AppController


class CollectionModel(QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dragged_item: Optional[QStandardItem] = None

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        default_flags = super().flags(index)
        if not index.isValid():
            return default_flags | Qt.ItemFlag.ItemIsDropEnabled
        item = self.itemFromIndex(index)
        if not item:
            return default_flags
        item_data = item.data(Qt.ItemDataRole.UserRole)

        if isinstance(
            item_data, (SetFolderNodeModel, ScriptSetNodeModel, ScriptSetEntryModel)
        ):
            if item.parent() is not None or isinstance(
                item_data, (ScriptSetNodeModel, ScriptSetEntryModel)
            ):
                default_flags |= Qt.ItemFlag.ItemIsDragEnabled
        if isinstance(
            item_data, (SetFolderNodeModel, ScriptSetNodeModel, ScriptSetEntryModel)
        ):
            default_flags |= Qt.ItemFlag.ItemIsDropEnabled
        return default_flags

    def canDropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent_index: QModelIndex,
    ) -> bool:
        if not self.dragged_item:
            return False
        source_data = self.dragged_item.data(Qt.ItemDataRole.UserRole)
        target_item = (
            self.itemFromIndex(parent_index) if parent_index.isValid() else None
        )
        target_data = (
            target_item.data(Qt.ItemDataRole.UserRole) if target_item else None
        )

        if isinstance(source_data, (SetFolderNodeModel, ScriptSetNodeModel)):
            if target_data is None or isinstance(target_data, SetFolderNodeModel):
                if isinstance(source_data, SetFolderNodeModel):
                    temp_parent = target_item
                    while temp_parent:
                        if temp_parent == self.dragged_item:
                            return False
                        temp_parent = temp_parent.parent()
                return True
        elif isinstance(source_data, ScriptSetEntryModel):
            source_parent = self.dragged_item.parent()
            target_parent = None
            if target_data and isinstance(target_data, ScriptSetEntryModel):
                target_parent = target_item.parent()
            elif target_data and isinstance(target_data, ScriptSetNodeModel):
                target_parent = target_item
            if source_parent and target_parent and source_parent == target_parent:
                return True
        return False


class ScriptCollectionWidget(QWidget):
    def __init__(
        self,
        controller: "AppController",
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(f"PyScriptManager.{self.__class__.__name__}")
        self.controller = controller
        self.locale_manager = locale_manager
        self.expanded_ids: Set[str] = set()
        self._items_by_instance_id: Dict[str, QStandardItem] = {}

        self._init_ui()
        self.default_palette = self.collection_groupbox.palette()
        self.base_groupbox_title = self.locale_manager.get(
            "collection_widget.group_title"
        )
        self._connect_signals()
        self._update_buttons_state()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.collection_groupbox = QGroupBox(
            self.locale_manager.get("collection_widget.group_title")
        )
        collection_layout = QVBoxLayout(self.collection_groupbox)
        main_layout.addWidget(self.collection_groupbox)

        self.collection_tree_view = QTreeView()
        self.collection_tree_view.setAlternatingRowColors(True)
        self.collection_tree_view.setHeaderHidden(True)
        self.collection_tree_view.setIconSize(QSize(20, 20))
        self.collection_tree_view.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.collection_tree_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.collection_tree_view.setDropIndicatorShown(True)
        self.collection_tree_view.setDragEnabled(True)
        self.collection_tree_view.setAcceptDrops(True)
        self.collection_tree_view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.collection_tree_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.collection_tree_view.setUniformRowHeights(True)

        collection_layout.addWidget(self.collection_tree_view)

        self.collection_model = CollectionModel()
        self.collection_tree_view.setModel(self.collection_model)

        run_controls_layout = QVBoxLayout()
        run_controls_layout.setSpacing(5)

        top_line_layout = QHBoxLayout()

        self.combo_set_run_mode = QComboBox()
        self.combo_set_run_mode.addItem(
            self.locale_manager.get("collection_widget.run_mode_full"),
            SetRunMode.SEQUENTIAL_FULL,
        )
        self.combo_set_run_mode.addItem(
            self.locale_manager.get("collection_widget.run_mode_step"),
            SetRunMode.SEQUENTIAL_STEP,
        )

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # 1. БЛОК: Добавление новых режимов в выпадающий список
        # Добавляем разделитель для визуального отделения новых режимов
        self.combo_set_run_mode.insertSeparator(2)

        # !! ВАЖНО: нужно будет добавить новые строки в файлы локализации
        self.combo_set_run_mode.addItem(
            "Условный (авто)",  # Временно, пока нет ключа локализации
            SetRunMode.CONDITIONAL_FULL,
        )
        self.combo_set_run_mode.addItem(
            "Условный (пошагово)",  # Временно, пока нет ключа локализации
            SetRunMode.CONDITIONAL_STEP,
        )
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        self.combo_set_run_mode.addItem(
            self.locale_manager.get("collection_widget.run_mode_single"),
            SetRunMode.SINGLE_FROM_SET,
        )

        self.btn_run_action = QPushButton(
            self.locale_manager.get("collection_widget.run_button_run")
        )
        self.btn_stop_action = QPushButton(
            self.locale_manager.get("collection_widget.run_button_stop")
        )
        self.btn_stop_action.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )

        top_line_layout.addWidget(
            QLabel(self.locale_manager.get("collection_widget.run_mode_label"))
        )
        top_line_layout.addWidget(self.combo_set_run_mode, 1)
        top_line_layout.addWidget(self.btn_run_action)
        top_line_layout.addWidget(self.btn_stop_action)

        run_controls_layout.addLayout(top_line_layout)

        self.chk_continue_on_error = QCheckBox(
            self.locale_manager.get("collection_widget.continue_on_error_checkbox")
        )
        run_controls_layout.addWidget(self.chk_continue_on_error)

        collection_layout.addLayout(run_controls_layout)

    def _connect_signals(self):
        self.collection_tree_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.collection_tree_view.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self.collection_tree_view.doubleClicked.connect(self._on_double_clicked)

        self.btn_run_action.clicked.connect(self._on_run_action_clicked)
        self.btn_stop_action.clicked.connect(self.controller.stop_current_set_run)

        self.combo_set_run_mode.currentIndexChanged.connect(self._on_run_mode_changed)

        self.controller.controller_state_updated.connect(self._update_buttons_state)
        # Сигнал app_busy_state_changed больше не нужен этому виджету напрямую
        # self.controller.app_busy_state_changed.connect(self.on_app_busy_state_changed)

        self.controller.script_instance_status_changed.connect(
            self._on_script_status_changed
        )
        self.controller.active_set_node_changed.connect(self.on_active_set_node_changed)
        self.controller.collection_dirty_state_changed.connect(
            self.on_collection_dirty_state_changed
        )
        self.controller.run_mode_restored.connect(self.on_run_mode_restored)

        self.collection_tree_view.expanded.connect(self._on_item_expanded)
        self.collection_tree_view.collapsed.connect(self._on_item_collapsed)

        self.collection_model.dropMimeData = self._custom_drop_mime_data
        self.controller.config_updated.connect(self._update_all_tooltips)        


    @Slot()
    def _update_all_tooltips(self):
        """Рекурсивно обновляет все подсказки в дереве после смены темы."""
        root = self.collection_model.invisibleRootItem()
        self._recursive_tooltip_update(root)

    def _recursive_tooltip_update(self, parent_item: QStandardItem):
        """Рекурсивный помощник для _update_all_tooltips."""
        for row in range(parent_item.rowCount()):
            item = parent_item.child(row)
            if not item:
                continue
            
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_data, ScriptSetEntryModel):
                script_info = self.controller.get_script_info_by_id(item_data.id)
                # --- ИЗМЕНЕНИЕ ---
                tooltip_html = generate_instance_tooltip_html(
                    script_info, item_data, self.locale_manager, self.controller.config_manager
                )
                item.setToolTip(tooltip_html)
            
            if item.hasChildren():
                self._recursive_tooltip_update(item)


    # --- НОВЫЙ СЛОТ: Обработка сигнала от контроллера ---
    @Slot(str)
    def on_run_mode_restored(self, mode_id: str):
        """Устанавливает режим запуска в комбо-боксе при загрузке коллекции."""
        index = self.combo_set_run_mode.findData(mode_id)
        if index != -1:
            self.combo_set_run_mode.setCurrentIndex(index)

    @Slot(list, object)
    def on_collection_updated(
        self,
        root_nodes: List[SetHierarchyNodeType],
        node_id_to_select: Optional[str] = None,
    ):
        id_to_reselect = node_id_to_select
        if not id_to_reselect:
            item_data = self._get_data_from_selected_item()
            if item_data:
                item_id = getattr(item_data, "instance_id", None) or getattr(
                    item_data, "id", None
                )
                if item_id:
                    id_to_reselect = item_id

        self._items_by_instance_id.clear()
        self.collection_model.clear()
        self.collection_model.setHorizontalHeaderLabels(
            [self.locale_manager.get("collection_widget.header_collection")]
        )
        self._populate_collection_recursive(
            self.collection_model.invisibleRootItem(), root_nodes
        )
        self._restore_expanded_state(self.collection_model.invisibleRootItem())

        if id_to_reselect:
            self._select_item_by_id(id_to_reselect)

        self._update_buttons_state()


    @Slot(str, ScriptSetEntryModel)
    def on_script_instance_updated(
        self, set_id: str, updated_entry: ScriptSetEntryModel
    ):
        item = self._items_by_instance_id.get(updated_entry.instance_id)
        if not item:
            return

        item.setData(updated_entry, Qt.ItemDataRole.UserRole)
        script_info = self.controller.get_script_info_by_id(updated_entry.id)
        display_name = updated_entry.name or (
            script_info.name
            if script_info
            else self.locale_manager.get(
                "collection_widget.script_not_found_format", id=updated_entry.id
            )
        )
        item.setText(display_name)
        
        # --- 1. ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        tooltip_html = generate_instance_tooltip_html(
            script_info, updated_entry, self.locale_manager, self.controller.config_manager
        )


        item.setToolTip(tooltip_html)
        self.collection_tree_view.setCurrentIndex(item.index())

    # --- 1. БЛОК: Метод on_collection_dirty_state_changed (ИЗМЕНЕН) ---
    @Slot(bool)
    def on_collection_dirty_state_changed(self, is_dirty: bool):
        if is_dirty:
            active_theme = self.controller.config_manager.get_active_theme()
            # --- ИЗМЕНЕНИЕ: Используем парсер, а не _get_color_from_css ---
            _fg, bg_color = self._parse_status_style(
                active_theme.status_error,
                self.palette().color(QPalette.ColorRole.Text),
                QColor("orange")
            )

            dirty_palette = self.collection_groupbox.palette()
            dirty_palette.setColor(QPalette.ColorRole.Window, bg_color)
            self.collection_groupbox.setPalette(dirty_palette)
            self.collection_groupbox.setAutoFillBackground(True)
            dirty_title = f"{self.base_groupbox_title} {self.locale_manager.get('collection_widget.unsaved_suffix')}"
            self.collection_groupbox.setTitle(dirty_title)
        else:
            self.collection_groupbox.setPalette(self.default_palette)
            self.collection_groupbox.setAutoFillBackground(False)
            self.collection_groupbox.setTitle(self.base_groupbox_title)
        self._update_buttons_state()

    @Slot(object)
    def on_active_set_node_changed(
        self, active_set_node_model: Optional[ScriptSetNodeModel]
    ):
        log_name = "None" if not active_set_node_model else active_set_node_model.name
        self.logger.debug(
            self.locale_manager.get(
                "collection_widget.log_debug.active_set_changed", name=log_name
            )
        )
        self._update_buttons_state()

    @Slot(QModelIndex)
    def _on_item_expanded(self, index: QModelIndex):
        item = self.collection_model.itemFromIndex(index)
        if item:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data and hasattr(item_data, "id"):
                self.expanded_ids.add(item_data.id)

    @Slot(QModelIndex)
    def _on_item_collapsed(self, index: QModelIndex):
        item = self.collection_model.itemFromIndex(index)
        if item:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if (
                item_data
                and hasattr(item_data, "id")
                and item_data.id in self.expanded_ids
            ):
                self.expanded_ids.remove(item_data.id)

    @Slot(str, object)
    def _on_script_status_changed(
        self, instance_id: str, status: Optional[ScriptRunStatus]
    ):
        item = self._items_by_instance_id.get(instance_id)
        if item:
            theme_styles = self.controller.config_manager.get_active_theme()
            self._update_item_visuals(item, status, theme_styles)

    # --- ИЗМЕНЕНИЕ: Метод теперь вызывает контроллер ---
    @Slot(int)
    def _on_run_mode_changed(self, index: int):
        """Обрабатывает смену режима запуска, сообщает контроллеру и обновляет кнопки."""
        mode_id = self.combo_set_run_mode.itemData(index)
        if mode_id:
            self.controller.set_collection_run_mode(mode_id)
        self._update_buttons_state()

    # --- 2. БЛОК: Новый вспомогательный метод-парсер ---
    def _parse_status_style(self, css_str: str, default_fg: QColor, default_bg: QColor) -> Tuple[QColor, QColor]:
        """Извлекает color и background-color из CSS строки."""
        fg_color = default_fg
        bg_color = default_bg

        # Ищем цвет текста
        fg_match = re.search(r"color:\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", css_str)
        if fg_match:
            color_val = fg_match.group(1).strip()
            if QColor.isValidColor(color_val):
                fg_color = QColor(color_val)
        
        # Ищем цвет фона
        bg_match = re.search(r"background-color:\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", css_str)
        if bg_match:
            color_val = bg_match.group(1).strip()
            if QColor.isValidColor(color_val):
                bg_color = QColor(color_val)
        
        return fg_color, bg_color

    # --- 3. БЛОК: Метод _update_item_visuals (ИЗМЕНЕН) ---
    def _update_item_visuals(
        self,
        item: QStandardItem,
        status: Optional[ScriptRunStatus],
        theme: ConsoleStylesConfig,
    ):
        # Сбрасываем стили к дефолтным
        default_fg = self.palette().color(QPalette.ColorRole.Text)
        item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon))
        item.setBackground(QBrush(Qt.GlobalColor.transparent))
        item.setForeground(QBrush(default_fg))

        # Карта соответствия статуса и CSS-строки из темы
        status_css_map = {
            ScriptRunStatus.RUNNING: theme.status_running,
            ScriptRunStatus.SUCCESS: theme.status_success,
            ScriptRunStatus.ERROR: theme.status_error,
            ScriptRunStatus.PENDING: theme.status_pending,
            ScriptRunStatus.SKIPPED: theme.status_skipped,
        }
        
        css_str = status_css_map.get(status)
        if css_str:
            # Парсим fg и bg цвета из CSS-строки
            fg_color, bg_color = self._parse_status_style(
                css_str,
                default_fg,
                QColor(Qt.GlobalColor.transparent)
            )
            item.setForeground(QBrush(fg_color))
            item.setBackground(QBrush(bg_color))

        # Устанавливаем иконки
        if status == ScriptRunStatus.RUNNING:
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        elif status == ScriptRunStatus.SUCCESS:
            item.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            )
        elif status == ScriptRunStatus.ERROR:
            item.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
            )
        elif status == ScriptRunStatus.PENDING:
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        elif status == ScriptRunStatus.SKIPPED:
            item.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton)
            )
            # Для пропущенных всегда серый цвет текста, переопределяя тему
            item.setForeground(QColor("gray"))


    def _restore_expanded_state(self, parent_item: QStandardItem):
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row)
            if not child_item:
                continue
            item_data = child_item.data(Qt.ItemDataRole.UserRole)
            if (
                item_data
                and hasattr(item_data, "id")
                and item_data.id in self.expanded_ids
            ):
                self.collection_tree_view.expand(child_item.index())
                if child_item.hasChildren():
                    self._restore_expanded_state(child_item)

    def _populate_collection_recursive(
        self, parent_qt_item: QStandardItem, nodes_data: List[SetHierarchyNodeType]
    ):
        for node_data in nodes_data:
            node_item = QStandardItem(node_data.name)
            node_item.setEditable(False)
            node_item.setData(node_data, Qt.ItemDataRole.UserRole)
            node_item.setToolTip(node_data.description or node_data.name)
            if isinstance(node_data, SetFolderNodeModel):
                node_item.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_DirClosedIcon)
                )
                parent_qt_item.appendRow(node_item)
                if node_data.children:
                    self._populate_collection_recursive(node_item, node_data.children)
            elif isinstance(node_data, ScriptSetNodeModel):
                node_item.setIcon(
                    self.style().standardIcon(
                        QStyle.StandardPixmap.SP_FileDialogNewFolder
                    )
                )
                parent_qt_item.appendRow(node_item)
                theme_styles = self.controller.config_manager.get_active_theme()
                for entry in node_data.script_entries:
                    script_info = self.controller.get_script_info_by_id(entry.id)
                    display_name = entry.name or (
                        script_info.name
                        if script_info
                        else self.locale_manager.get(
                            "collection_widget.script_not_found_format", id=entry.id
                        )
                    )
                    entry_item = QStandardItem(display_name)
                    entry_item.setEditable(False)
                    entry_item.setData(entry, Qt.ItemDataRole.UserRole)
                    self._items_by_instance_id[entry.instance_id] = entry_item
                    status = self.controller.script_run_statuses.get(entry.instance_id)
                    self._update_item_visuals(entry_item, status, theme_styles)

                    # --- 2. ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                    tooltip_html = generate_instance_tooltip_html(
                        script_info, entry, self.locale_manager, self.controller.config_manager
                    )

                    entry_item.setToolTip(tooltip_html)
                    if not (script_info and script_info.passport_valid):
                        entry_item.setIcon(
                            self.style().standardIcon(
                                QStyle.StandardPixmap.SP_MessageBoxWarning
                            )
                        )
                        entry_item.setForeground(QColor("red"))

                    node_item.appendRow(entry_item)

    # ... (остальные методы без изменений) ...

    def _get_selected_qstandarditem(self) -> Optional[QStandardItem]:
        selected_indexes = self.collection_tree_view.selectedIndexes()
        return (
            self.collection_model.itemFromIndex(selected_indexes[0])
            if selected_indexes
            else None
        )

    def _get_data_from_selected_item(
        self,
    ) -> Optional[Union[SetHierarchyNodeType, ScriptSetEntryModel]]:
        item = self._get_selected_qstandarditem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _select_item_by_id(self, node_id: str):
        root = self.collection_model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row)
            if not item:
                continue
            found_index = self._find_item_recursive(item, node_id)
            if found_index.isValid():
                self.collection_tree_view.setCurrentIndex(found_index)
                return

    def _find_item_recursive(
        self, parent_item: QStandardItem, node_id_or_instance_id: str
    ) -> QModelIndex:
        item_data = parent_item.data(Qt.ItemDataRole.UserRole)
        item_id = getattr(item_data, "instance_id", getattr(item_data, "id", None))
        if item_data and item_id == node_id_or_instance_id:
            return self.collection_model.indexFromItem(parent_item)
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row)
            if not child_item:
                continue
            found_index = self._find_item_recursive(child_item, node_id_or_instance_id)
            if found_index.isValid():
                return found_index
        return QModelIndex()

    @Slot()
    def _on_selection_changed(self):
        selected_item = self._get_selected_qstandarditem()
        self.collection_model.dragged_item = selected_item
        selected_data = (
            selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None
        )

        node_id_to_activate = None
        if isinstance(selected_data, ScriptSetNodeModel):
            node_id_to_activate = selected_data.id
        elif isinstance(selected_data, ScriptSetEntryModel):
            item = self._get_selected_qstandarditem()
            if item and item.parent():
                parent_data = item.parent().data(Qt.ItemDataRole.UserRole)
                if isinstance(parent_data, ScriptSetNodeModel):
                    node_id_to_activate = parent_data.id

        self.controller.set_active_script_set_node(node_id_to_activate)

        self._update_buttons_state()

    @Slot()
    def _on_run_action_clicked(self):
        if self.controller.is_waiting_for_next_step():
            self.controller.proceed_to_next_script_in_set_step()
            return

        selected_data = self._get_data_from_selected_item()
        run_mode = self.combo_set_run_mode.currentData(Qt.ItemDataRole.UserRole)
        # --- НОВАЯ ЛОГИКА: Считываем состояние чекбокса ---
        continue_on_error = self.chk_continue_on_error.isChecked()

        set_node_id = self.controller.selected_set_node_id
        instance_id = (
            selected_data.instance_id
            if isinstance(selected_data, ScriptSetEntryModel)
            else None
        )
        if not set_node_id:
            QMessageBox.warning(
                self,
                self.locale_manager.get("collection_widget.run_error.title"),
                self.locale_manager.get("collection_widget.run_error.no_set_selected"),
            )
            return
        if run_mode == SetRunMode.SINGLE_FROM_SET and not instance_id:
            QMessageBox.warning(
                self,
                self.locale_manager.get("collection_widget.run_error.title"),
                self.locale_manager.get("collection_widget.run_error.no_script_in_set"),
            )
            return

        # --- ИЗМЕНЕНИЕ: Передаем новый параметр в контроллер ---
        self.controller.run_script_set(
            set_node_id, run_mode, instance_id, continue_on_error
        )

    def _update_buttons_state(self):
        current_state = self.controller._app_state
        # --- СТРОКА ИЗМЕНЕНА ---
        # is_dirty больше не используется для блокировки, только для подсказки
        collection_was_never_saved = (
            self.controller.current_collection_file_path is None
        )

        # 1. Кнопка "Остановить"
        self.btn_stop_action.setEnabled(current_state != AppState.IDLE)

        # 2. Кнопка "Выполнить" / "Следующий шаг"
        if current_state == AppState.SET_RUNNING_STEP_WAIT:
            self.btn_run_action.setText(
                self.locale_manager.get("collection_widget.run_button_next_step")
            )
            self.btn_run_action.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
            )
            self.btn_run_action.setEnabled(True)
            self.btn_run_action.setToolTip("") # Очищаем подсказку
        else:
            self.btn_run_action.setText(
                self.locale_manager.get("collection_widget.run_button_run")
            )
            self.btn_run_action.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

            can_run = (
                current_state == AppState.IDLE
            ) and self._can_run_based_on_selection()
            self.btn_run_action.setEnabled(can_run)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            # Показываем подсказку о необходимости сохранения, только если
            # коллекция ни разу не сохранялась.
            tooltip = ""
            if collection_was_never_saved:
                tooltip = self.locale_manager.get(
                    "collection_widget.run_tooltip_save_first"
                )
            self.btn_run_action.setToolTip(tooltip)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

    def _can_run_based_on_selection(self) -> bool:
        """Вспомогательный метод для проверки, можно ли запустить выбранное."""
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Блокируем запуск, только если коллекция ни разу не была сохранена,
        # так как для запуска нужен путь к файлу контекста.
        if self.controller.current_collection_file_path is None:
            return False
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        run_mode = self.combo_set_run_mode.currentData(Qt.ItemDataRole.UserRole)
        selected_data = self._get_data_from_selected_item()

        if run_mode == SetRunMode.SINGLE_FROM_SET:
            script_info = None
            if isinstance(selected_data, ScriptSetEntryModel):
                script_info = self.controller.get_script_info_by_id(selected_data.id)
            return script_info is not None and script_info.passport_valid
        else:
            return (
                self.controller.selected_set_node_id is not None
                and self.controller.selected_set_node_model is not None
                and bool(self.controller.selected_set_node_model.script_entries)
            )

    @Slot("QPoint")
    def _show_context_menu(self, position):
        if self.controller.is_busy():
            return
        index = self.collection_tree_view.indexAt(position)
        item = self.collection_model.itemFromIndex(index) if index.isValid() else None
        item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
        menu = QMenu(self)
        parent_id_for_new_node = None

        if isinstance(item_data, SetFolderNodeModel):
            parent_id_for_new_node = item_data.id
        elif isinstance(item_data, ScriptSetNodeModel):
            parent_id_for_new_node = (
                item_data.id
            )  # Можно создавать папки внутри наборов, но они не будут показаны
        elif item and item.parent():
            parent_data = item.parent().data(Qt.ItemDataRole.UserRole)
            if isinstance(parent_data, SetFolderNodeModel):
                parent_id_for_new_node = parent_data.id

        action_create_folder = QAction(
            self.locale_manager.get("collection_widget.context_menu.create_folder"),
            self,
        )
        action_create_set = QAction(
            self.locale_manager.get("collection_widget.context_menu.create_set"), self
        )
        action_create_folder.triggered.connect(
            lambda: self._create_node_action(True, parent_id_for_new_node)
        )
        action_create_set.triggered.connect(
            lambda: self._create_node_action(False, parent_id_for_new_node)
        )
        menu.addAction(action_create_folder)
        menu.addAction(action_create_set)

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Логика добавления пункта "Вставить"
        if isinstance(item_data, ScriptSetNodeModel):
            menu.addSeparator()
            action_paste = QAction(
                self.locale_manager.get(
                    "collection_widget.context_menu.paste_instance"
                ),
                self,
            )
            # Пункт меню активен, только если в буфере контроллера что-то есть
            action_paste.setEnabled(self.controller._copied_script_entry is not None)
            action_paste.triggered.connect(
                lambda: self.controller.paste_script_instance_from_buffer(item_data.id)
            )
            menu.addAction(action_paste)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        if item_data and isinstance(
            item_data, (SetFolderNodeModel, ScriptSetNodeModel)
        ):
            menu.addSeparator()
            action_rename = QAction(
                self.locale_manager.get("collection_widget.context_menu.rename"), self
            )
            action_delete = QAction(
                self.locale_manager.get("collection_widget.context_menu.delete"), self
            )
            action_rename.triggered.connect(lambda: self._rename_node_action(item_data))
            action_delete.triggered.connect(lambda: self._delete_node_action(item_data))
            menu.addAction(action_rename)
            menu.addAction(action_delete)

        if isinstance(item_data, ScriptSetEntryModel):
            menu.addSeparator()
            action_params = QAction(
                self.locale_manager.get(
                    "collection_widget.context_menu.configure_params"
                ),
                self,
            )
            entry_data_copy = item_data.model_copy(deep=True)
            action_params.triggered.connect(
                lambda: self._show_script_instance_properties_dialog(entry_data_copy)
            )
            menu.addAction(action_params)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # КОММЕНТАРИЙ: Добавляем новые действия для экземпляра
            menu.addSeparator()
            action_copy = QAction(
                self.locale_manager.get("collection_widget.context_menu.copy_instance"),
                self,
            )
            action_duplicate = QAction(
                self.locale_manager.get(
                    "collection_widget.context_menu.duplicate_instance"
                ),
                self,
            )
            action_delete_entry = QAction(
                self.locale_manager.get(
                    "collection_widget.context_menu.delete_from_set"
                ),
                self,
            )

            parent_set_id = self.controller.selected_set_node_id
            instance_id = item_data.instance_id

            action_copy.triggered.connect(
                lambda: self.controller.copy_script_instance_to_buffer(
                    parent_set_id, instance_id
                )
            )
            action_duplicate.triggered.connect(
                lambda: self.controller.duplicate_script_instance(
                    parent_set_id, instance_id
                )
            )
            action_delete_entry.triggered.connect(
                lambda: self._delete_entry_action(item_data)
            )

            menu.addAction(action_copy)
            menu.addAction(action_duplicate)
            menu.addAction(action_delete_entry)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        menu.exec(self.collection_tree_view.viewport().mapToGlobal(position))

    def _create_node_action(self, is_folder: bool, parent_id: Optional[str]):
        title = (
            self.locale_manager.get("collection_widget.create_folder.title")
            if is_folder
            else self.locale_manager.get("collection_widget.create_set.title")
        )
        label = (
            self.locale_manager.get("collection_widget.create_folder.label")
            if is_folder
            else self.locale_manager.get("collection_widget.create_set.label")
        )
        text, ok = QInputDialog.getText(self, title, label)
        if ok and text:
            if is_folder:
                self.controller.create_folder_in_collection(text, parent_id)
            else:
                self.controller.create_set_in_collection(text, parent_id)

    def _rename_node_action(self, node_data: SetHierarchyNodeType):
        text, ok = QInputDialog.getText(
            self,
            self.locale_manager.get("collection_widget.rename.title"),
            self.locale_manager.get("collection_widget.rename.label"),
            text=node_data.name,
        )
        if ok and text and text != node_data.name:
            self.controller.rename_node_in_collection(node_data.id, text)

    def _delete_node_action(self, node_data: SetHierarchyNodeType):
        reply = QMessageBox.question(
            self,
            self.locale_manager.get("collection_widget.delete_node.confirm_title"),
            self.locale_manager.get(
                "collection_widget.delete_node.confirm_text", name=node_data.name
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.delete_node_from_collection(node_data.id)

    def _delete_entry_action(self, entry_data: ScriptSetEntryModel):
        if not self.controller.selected_set_node_id:
            return
        script_info = self.controller.get_script_info_by_id(entry_data.id)
        script_name = script_info.name if script_info else entry_data.id
        set_name = (
            self.controller.selected_set_node_model.name
            if self.controller.selected_set_node_model
            else ""
        )
        reply = QMessageBox.question(
            self,
            self.locale_manager.get("collection_widget.delete_entry.title"),
            self.locale_manager.get(
                "collection_widget.delete_entry.text",
                script_name=script_name,
                set_name=set_name,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.remove_script_from_active_set_node(entry_data.instance_id)

    # pysm_lib/gui/script_collection_widget.py

    # 2. БЛОК: Метод _show_script_instance_properties_dialog (ИЗМЕНЕН)
    def _show_script_instance_properties_dialog(self, entry_data: ScriptSetEntryModel):
        script_info = self.controller.get_script_info_by_id(entry_data.id)
        if not script_info:
            QMessageBox.critical(
                self,
                self.locale_manager.get("general.error_title"),
                self.locale_manager.get(
                    "collection_widget.error.script_info_not_found",
                    id=entry_data.id,
                ),
            )
            return

        script_info_copy = script_info.model_copy(deep=True)

        dialog = ScriptPropertiesDialog(
            edit_mode=EditMode.INSTANCE,
            script_info=script_info_copy,
            instance_entry=entry_data,
            locale_manager=self.locale_manager,
            parent=self,
        )
        if dialog.exec():
            # --- ИЗМЕНЕНИЕ: Получаем всю обновленную модель и вызываем новый метод контроллера ---
            updated_instance_model = dialog.get_updated_instance_entry_model()
            if updated_instance_model:
                self.controller.update_script_instance_in_active_set_node(
                    updated_instance_model
                )

    @Slot(QModelIndex)
    def _on_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return

        item = self.collection_model.itemFromIndex(index)
        if not item:
            return

        item_data = item.data(Qt.ItemDataRole.UserRole)

        if not isinstance(item_data, ScriptSetEntryModel):
            return

        entry_data_copy = item_data.model_copy(deep=True)
        self._show_script_instance_properties_dialog(entry_data_copy)

    def _custom_drop_mime_data(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent_index: QModelIndex,
    ) -> bool:
        if not self.collection_model.canDropMimeData(
            data, action, row, column, parent_index
        ):
            return False
        source_item = self.collection_model.dragged_item
        if not source_item:
            return False
        source_data = source_item.data(Qt.ItemDataRole.UserRole)

        if isinstance(source_data, (SetFolderNodeModel, ScriptSetNodeModel)):
            target_item = (
                self.collection_model.itemFromIndex(parent_index)
                if parent_index.isValid()
                else None
            )
            new_parent_id = None
            if target_item:
                new_parent_id = target_item.data(Qt.ItemDataRole.UserRole).id
            self.controller.move_node_in_collection(source_data.id, new_parent_id, row)
            return True

        elif isinstance(source_data, ScriptSetEntryModel):
            source_parent_item = source_item.parent()
            set_node_data = source_parent_item.data(Qt.ItemDataRole.UserRole)
            current_ids = [entry.instance_id for entry in set_node_data.script_entries]
            dragged_id = source_data.instance_id
            current_ids.remove(dragged_id)
            if row == -1:
                current_ids.append(dragged_id)
            else:
                current_ids.insert(row, dragged_id)
            self.controller.reorder_scripts_in_active_set_node(current_ids)
            return True
        return False
