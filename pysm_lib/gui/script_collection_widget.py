# pysm_lib/gui/script_collection_widget.py

import logging
import re
from typing import List, Optional, Dict, Union, TYPE_CHECKING, Set, Tuple

from PySide6.QtCore import Qt, QSize, Slot, QModelIndex, QMimeData
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTreeView, QPushButton,
    QAbstractItemView, QStyle, QMessageBox, QComboBox, QLabel, QMenu, QDialog,
    QInputDialog, QCheckBox, QToolButton, QSizePolicy, QScrollArea, QFrame
)
from PySide6.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QAction, QBrush, QPalette, QWheelEvent
)

from ..models import (
    SetHierarchyNodeType, SetFolderNodeModel, ScriptSetNodeModel, ScriptSetEntryModel
)
# --- 1. БЛОК: ИЗМЕНЕННЫЕ ИМПОРТЫ ---
from ..theme_manager import ThemeManager
from ..pysm_icons import icons
from .gui_utils import resolve_themed_text
from ..app_enums import SetRunMode, AppState, ScriptRunStatus
from ..locale_manager import LocaleManager
from .dialogs import ScriptPropertiesDialog, EditMode
from .tooltip_generator import generate_instance_tooltip_html, generate_favorite_tooltip_html
from .widgets.icon_selection_dialog import IconSelectionDialog

if TYPE_CHECKING:
    from ..app_controller import AppController

# --- НАЧАЛО ИЗМЕНЕНИЙ ---
class HorizontalScrollArea(QScrollArea):
    """Кастомная область прокрутки, которая переводит вертикальное колесико мыши в горизонтальный скролл."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def wheelEvent(self, event: QWheelEvent):
        # Получаем значение поворота колесика
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.angleDelta().x()
            
        # Двигаем скрытый горизонтальный ползунок
        scroll_bar = self.horizontalScrollBar()
        scroll_bar.setValue(scroll_bar.value() - delta)
# --- КОНЕЦ ИЗМЕНЕНИЙ ---


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
        if not item: return default_flags
        item_data = item.data(Qt.ItemDataRole.UserRole)

        if isinstance(item_data, (SetFolderNodeModel, ScriptSetNodeModel, ScriptSetEntryModel)):
            if item.parent() is not None or isinstance(item_data, (ScriptSetNodeModel, ScriptSetEntryModel)):
                default_flags |= Qt.ItemFlag.ItemIsDragEnabled
        if isinstance(item_data, (SetFolderNodeModel, ScriptSetNodeModel, ScriptSetEntryModel)):
            default_flags |= Qt.ItemFlag.ItemIsDropEnabled
        return default_flags

    def canDropMimeData(self, data: QMimeData, action: Qt.DropAction, row: int, column: int, parent_index: QModelIndex) -> bool:
        if not self.dragged_item: return False
        source_data = self.dragged_item.data(Qt.ItemDataRole.UserRole)
        target_item = self.itemFromIndex(parent_index) if parent_index.isValid() else None
        target_data = target_item.data(Qt.ItemDataRole.UserRole) if target_item else None

        if isinstance(source_data, (SetFolderNodeModel, ScriptSetNodeModel)):
            if target_data is None or isinstance(target_data, SetFolderNodeModel):
                if isinstance(source_data, SetFolderNodeModel):
                    temp_parent = target_item
                    while temp_parent:
                        if temp_parent == self.dragged_item: return False
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
        theme_manager: ThemeManager,
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(f"PyScriptManager.{self.__class__.__name__}")
        self.controller = controller
        self.theme_manager = theme_manager
        self.locale_manager = locale_manager
        self.expanded_ids: Set[str] = set()
        self._items_by_instance_id: Dict[str, QStandardItem] = {}

        self._init_ui()
        self.default_palette = self.collection_groupbox.palette()
        self.base_groupbox_title = self.locale_manager.get("collection_widget.group_title")
        self._connect_signals()
        self._update_buttons_state()


    def _init_ui(self):
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            self.collection_groupbox = QGroupBox(self.locale_manager.get("collection_widget.group_title"))
            collection_layout = QVBoxLayout(self.collection_groupbox)
            main_layout.addWidget(self.collection_groupbox)

            self.collection_tree_view = QTreeView()
            self.collection_tree_view.setAlternatingRowColors(True)
            self.collection_tree_view.setHeaderHidden(True)
            self.collection_tree_view.setIconSize(QSize(24, 24))
            self.collection_tree_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.collection_tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.collection_tree_view.setDropIndicatorShown(True)
            self.collection_tree_view.setDragEnabled(True)
            self.collection_tree_view.setAcceptDrops(True)
            self.collection_tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.collection_tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.collection_tree_view.setUniformRowHeights(True)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # Панель избранных скриптов (Quick Access)
            # Используем наш кастомный класс для работы колесика мыши
            self.favorites_scroll = HorizontalScrollArea()
            self.favorites_scroll.setFixedHeight(44) # Высота для кнопок 36x36 + отступы
            
            self.favorites_container = QWidget()
            self.favorites_layout = QHBoxLayout(self.favorites_container)
            self.favorites_layout.setContentsMargins(0, 0, 0, 0)
            self.favorites_layout.setSpacing(5)
            self.favorites_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            
            self.favorites_scroll.setWidget(self.favorites_container)
            collection_layout.addWidget(self.favorites_scroll)
            self.favorites_scroll.setVisible(False)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            collection_layout.addWidget(self.collection_tree_view)
            self.collection_model = CollectionModel()
            self.collection_tree_view.setModel(self.collection_model)

            run_controls_layout = QVBoxLayout()
            run_controls_layout.setSpacing(5)
            top_line_layout = QHBoxLayout()

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # 1. Используем QPushButton вместо QToolButton для стандартного внешнего вида
            self.btn_run_action = QPushButton(self.locale_manager.get("collection_widget.run_button_run"))
            #self.btn_run_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            #self.btn_run_action.setIcon(icons.get_qicon("PLAY"))

            # 1. Генерируем иконку в хорошем качестве (например, 32px)
            self.btn_run_action.setIcon(icons.get_qicon("PLAY", size=28))
            # 2. Явно задаем размер иконки для кнопки
            self.btn_run_action.setIconSize(QSize(28, 28)) 
            
            
            
            # 2. Устанавливаем политику размера Preferred, чтобы кнопка НЕ растягивалась
            self.btn_run_action.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            
            # 3. Применяем класс акцента (синий цвет), если он поддерживается темой для QPushButton
            self.btn_run_action.setProperty("class", "accent")

            # 4. Создаем меню
            self.run_menu = QMenu(self.btn_run_action)
            
            self.action_run_full = QAction(self.locale_manager.get("collection_widget.run_mode_conditional_full"), self)
            self.action_run_step = QAction(self.locale_manager.get("collection_widget.run_mode_conditional_step"), self)
            self.action_run_single = QAction(self.locale_manager.get("collection_widget.run_mode_single"), self)

            # --- ДОБАВЛЕНИЕ ИКОНОК В МЕНЮ ---
            self.action_run_full.setIcon(icons.get_qicon("PLAY"))
            self.action_run_step.setIcon(icons.get_qicon("NEXT")) # Шаг вперед
            self.action_run_single.setIcon(icons.get_qicon("FILE_PY")) # Один файл
            # --------------------------------            


            self.run_menu.addAction(self.action_run_full)
            self.run_menu.addAction(self.action_run_step)
            self.run_menu.addSeparator()
            self.run_menu.addAction(self.action_run_single)
            
            # Назначаем меню кнопке. В QPushButton это добавляет стрелочку выпадающего списка.
            self.btn_run_action.setMenu(self.run_menu)

            self.btn_stop_action = QPushButton(self.locale_manager.get("collection_widget.run_button_stop"))
            #self.btn_stop_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.btn_stop_action.setIcon(icons.get_qicon("STOP", size=28))
            # 2. Явно задаем размер иконки для кнопки
            self.btn_stop_action.setIconSize(QSize(28, 28)) 



            # Добавляем кнопки в лейаут.
            # Добавляем stretch в конце, чтобы прижать кнопки влево, если они не занимают всю ширину
            top_line_layout.addWidget(self.btn_run_action)
            top_line_layout.addWidget(self.btn_stop_action)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            run_controls_layout.addLayout(top_line_layout)
            self.chk_continue_on_error = QCheckBox(self.locale_manager.get("collection_widget.continue_on_error_checkbox"))
            run_controls_layout.addWidget(self.chk_continue_on_error)
            collection_layout.addLayout(run_controls_layout)


    def _connect_signals(self):
            self.collection_tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
            self.collection_tree_view.customContextMenuRequested.connect(self._show_context_menu)
            self.collection_tree_view.doubleClicked.connect(self._on_double_clicked)

            # Новые подключения меню
            self.action_run_full.triggered.connect(lambda: self._start_run(SetRunMode.CONDITIONAL_FULL))
            self.action_run_step.triggered.connect(lambda: self._start_run(SetRunMode.CONDITIONAL_STEP))
            self.action_run_single.triggered.connect(lambda: self._start_run(SetRunMode.SINGLE_FROM_SET))
            
            # Клик по кнопке (только для шагов)
            self.btn_run_action.clicked.connect(self._on_run_button_clicked)

            self.btn_stop_action.clicked.connect(self.controller.stop_current_set_run)
            
            # --- УДАЛИТЬ ЭТИ СТРОКИ (они вызывают ошибку) ---
            # self.combo_set_run_mode.currentIndexChanged.connect(self._on_run_mode_changed)
            # ------------------------------------------------
            
            self.controller.controller_state_updated.connect(self._update_buttons_state)
            self.controller.script_instance_status_changed.connect(self._on_script_status_changed)
            self.controller.active_set_node_changed.connect(self.on_active_set_node_changed)
            self.controller.collection_dirty_state_changed.connect(self.on_collection_dirty_state_changed)
            
            # --- УДАЛИТЬ ЭТУ СТРОКУ (сигнал удален из контроллера) ---
            # self.controller.run_mode_restored.connect(self.on_run_mode_restored)
            # ---------------------------------------------------------
            self.controller.favorites_updated.connect(self._update_favorites_panel)
            
            self.collection_tree_view.expanded.connect(self._on_item_expanded)
            self.collection_tree_view.collapsed.connect(self._on_item_collapsed)
            self.collection_model.dropMimeData = self._custom_drop_mime_data
            self.controller.config_updated.connect(self._update_all_tooltips)   

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    def _start_run(self, mode: str):
        """Единая точка входа для запуска из меню."""
        selected_data = self._get_data_from_selected_item()
        continue_on_error = self.chk_continue_on_error.isChecked()
        set_node_id = self.controller.selected_set_node_id
        
        # Определение instance_id, если выбран скрипт
        instance_id = None
        if isinstance(selected_data, ScriptSetEntryModel):
            instance_id = selected_data.instance_id
        
        # Валидация: выбран ли вообще какой-то набор?
        if not set_node_id:
            QMessageBox.warning(self, self.locale_manager.get("collection_widget.run_error.title"), 
                                self.locale_manager.get("collection_widget.run_error.no_set_selected"))
            return

        # Валидация: если режим SINGLE, обязательно должен быть выбран скрипт (instance_id)
        if mode == SetRunMode.SINGLE_FROM_SET and not instance_id:
            # Текст ошибки: "Для запуска одиночного скрипта выберите конкретный скрипт в списке."
            # (Используем существующий ключ локализации или fallback)
            msg = self.locale_manager.get("collection_widget.run_error.no_script_in_set")
            if "collection_widget" in msg: # Fallback если ключа нет
                msg = "Для этого режима необходимо выбрать конкретный скрипт, а не папку или набор."
            
            QMessageBox.warning(self, self.locale_manager.get("collection_widget.run_error.title"), msg)
            return

        self.controller.run_script_set(set_node_id, mode, instance_id, continue_on_error)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    @Slot()
    def _on_run_button_clicked(self):
        """
        Обрабатывает клик по кнопке. 
        В состоянии IDLE кнопка работает как меню (InstantPopup), этот слот не вызывается.
        В состоянии STEP_WAIT кнопка работает как обычная кнопка (Next Step).
        """
        if self.controller.is_waiting_for_next_step():
            self.controller.proceed_to_next_script_in_set_step()
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


    @Slot()
    def _update_all_tooltips(self):
        root = self.collection_model.invisibleRootItem()
        self._recursive_tooltip_update(root)

    def _recursive_tooltip_update(self, parent_item: QStandardItem):
        for row in range(parent_item.rowCount()):
            item = parent_item.child(row)
            if not item: continue
            
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_data, ScriptSetEntryModel):
                script_info = self.controller.get_script_info_by_id(item_data.id)
                tooltip_html = generate_instance_tooltip_html(
                    script_info, item_data, self.locale_manager, self.theme_manager
                )
                item.setToolTip(tooltip_html)
            
            if item.hasChildren():
                self._recursive_tooltip_update(item)


    @Slot(list, object)
    def on_collection_updated(self, root_nodes: List[SetHierarchyNodeType], node_id_to_select: Optional[str] = None):
        id_to_reselect = node_id_to_select
        if not id_to_reselect:
            item_data = self._get_data_from_selected_item()
            if item_data:
                item_id = getattr(item_data, "instance_id", None) or getattr(item_data, "id", None)
                if item_id: id_to_reselect = item_id

        self._items_by_instance_id.clear()
        self.collection_model.clear()
        self.collection_model.setHorizontalHeaderLabels([self.locale_manager.get("collection_widget.header_collection")])
        self._populate_collection_recursive(self.collection_model.invisibleRootItem(), root_nodes)
        self._restore_expanded_state(self.collection_model.invisibleRootItem())

        if id_to_reselect: self._select_item_by_id(id_to_reselect)
        self._update_buttons_state()

    @Slot(str, ScriptSetEntryModel)
    def on_script_instance_updated(self, set_id: str, updated_entry: ScriptSetEntryModel):
        item = self._items_by_instance_id.get(updated_entry.instance_id)
        if not item: return

        item.setData(updated_entry, Qt.ItemDataRole.UserRole)
        script_info = self.controller.get_script_info_by_id(updated_entry.id)
        display_name = updated_entry.name or (script_info.name if script_info else self.locale_manager.get("collection_widget.script_not_found_format", id=updated_entry.id))
        item.setText(display_name)
        
        tooltip_html = generate_instance_tooltip_html(script_info, updated_entry, self.locale_manager, self.theme_manager)
        item.setToolTip(tooltip_html)

        # 4. ОБНОВЛЯЕМ ВИЗУАЛ (Иконку и цвет статуса)
        current_status = self.controller.script_run_statuses.get(updated_entry.instance_id)
        self._update_item_visuals(item, current_status)        
        
        self.collection_tree_view.setCurrentIndex(item.index())

    @Slot(bool)
    def on_collection_dirty_state_changed(self, is_dirty: bool):
        # Устанавливаем динамическое свойство
        self.collection_groupbox.setProperty("isDirty", is_dirty)

        # Формируем заголовок
        if is_dirty:
            dirty_title = f"{self.base_groupbox_title} {self.locale_manager.get('collection_widget.unsaved_suffix')}"
            self.collection_groupbox.setTitle(dirty_title)
        else:
            self.collection_groupbox.setTitle(self.base_groupbox_title)

        # Принудительно обновляем стиль виджета, чтобы QSS применил новые правила
        self.style().unpolish(self.collection_groupbox)
        self.style().polish(self.collection_groupbox)
        self.collection_groupbox.update()
        
        # Этот вызов остается, так как он влияет на кнопки
        self._update_buttons_state()
      

    @Slot(object)
    def on_active_set_node_changed(self, active_set_node_model: Optional[ScriptSetNodeModel]):
        log_name = "None" if not active_set_node_model else active_set_node_model.name
        self.logger.debug(self.locale_manager.get("collection_widget.log_debug.active_set_changed", name=log_name))
        self._update_buttons_state()

    @Slot(QModelIndex)
    def _on_item_expanded(self, index: QModelIndex):
        item = self.collection_model.itemFromIndex(index)
        if item and (item_data := item.data(Qt.ItemDataRole.UserRole)) and hasattr(item_data, "id"):
            self.expanded_ids.add(item_data.id)

    @Slot(QModelIndex)
    def _on_item_collapsed(self, index: QModelIndex):
        item = self.collection_model.itemFromIndex(index)
        if item and (item_data := item.data(Qt.ItemDataRole.UserRole)) and hasattr(item_data, "id") and item_data.id in self.expanded_ids:
            self.expanded_ids.remove(item_data.id)

    @Slot(str, object)
    def _on_script_status_changed(self, instance_id: str, status: Optional[ScriptRunStatus]):
        item = self._items_by_instance_id.get(instance_id)
        if item:
            self._update_item_visuals(item, status)

    def _parse_status_style(self, css_str: str, default_fg: QColor, default_bg: QColor) -> Tuple[QColor, QColor]:
        fg_color, bg_color = default_fg, default_bg
        if fg_match := re.search(r"color:\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", css_str):
            if QColor.isValidColor(color_val := fg_match.group(1).strip()): fg_color = QColor(color_val)
        if bg_match := re.search(r"background-color:\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", css_str):
            if QColor.isValidColor(color_val := bg_match.group(1).strip()): bg_color = QColor(color_val)
        return fg_color, bg_color

    def _update_item_visuals(self, item: QStandardItem, status: Optional[ScriptRunStatus]):
        dynamic_styles = self.theme_manager.get_active_theme_dynamic_styles()
        
        # Сбрасываем стили к значениям по умолчанию для текущей палитры
        default_fg = self.palette().color(QPalette.ColorRole.Text)
        
        # --- ИСПРАВЛЕНИЕ: Получаем данные из элемента перед их использованием ---
        item_data = item.data(Qt.ItemDataRole.UserRole)
        base_icon = getattr(item_data, "icon_name", "INSTANCE_ITEM") if item_data else "INSTANCE_ITEM"
        icon_color = getattr(item_data, "icon_color", None) if item_data else None
        
        # Устанавливаем пользовательскую иконку
        item.setIcon(icons.get_qicon(base_icon, color=icon_color))
        # ------------------------------------------------------------------------
        
        item.setBackground(QBrush(Qt.GlobalColor.transparent))
        item.setForeground(QBrush(default_fg))

        status_css_map = {
            ScriptRunStatus.RUNNING: dynamic_styles.get("status_running"),
            ScriptRunStatus.SUCCESS: dynamic_styles.get("status_success"),
            ScriptRunStatus.ERROR: dynamic_styles.get("status_error"),
            ScriptRunStatus.PENDING: dynamic_styles.get("status_pending"),
            ScriptRunStatus.SKIPPED: dynamic_styles.get("status_skipped"),
        }

        css_str = status_css_map.get(status)
        if css_str:
            # Парсим fg и bg цвета из CSS-строки
            fg_color, bg_color = self._parse_status_style(
                css_str,
                default_fg,
                QColor(Qt.GlobalColor.transparent)
            )
            # Устанавливаем ТОЛЬКО фон. Цвет текста оставляем по умолчанию.
            item.setBackground(QBrush(bg_color))

        # Переопределение иконок в зависимости от статуса выполнения
        if status == ScriptRunStatus.RUNNING:
            item.setIcon(icons.get_qicon("PLAY"))
        elif status == ScriptRunStatus.SUCCESS:
            item.setIcon(icons.get_qicon("OK"))
        elif status == ScriptRunStatus.ERROR:
            item.setIcon(icons.get_qicon("ERROR"))
        elif status == ScriptRunStatus.PENDING:
            item.setIcon(icons.get_qicon("ARROW_SUB"))
        elif status == ScriptRunStatus.SKIPPED:
            item.setIcon(icons.get_qicon("WARNING", color="gray"))
            item.setForeground(QBrush(QColor("gray")))

    def _restore_expanded_state(self, parent_item: QStandardItem):
        for row in range(parent_item.rowCount()):
            if child_item := parent_item.child(row):
                item_data = child_item.data(Qt.ItemDataRole.UserRole)
                if item_data and hasattr(item_data, "id") and item_data.id in self.expanded_ids:
                    self.collection_tree_view.expand(child_item.index())
                    if child_item.hasChildren(): self._restore_expanded_state(child_item)

    def _populate_collection_recursive(self, parent_qt_item: QStandardItem, nodes_data: List[SetHierarchyNodeType]):
        for node_data in nodes_data:
            node_item = QStandardItem(node_data.name)
            node_item.setEditable(False)
            node_item.setData(node_data, Qt.ItemDataRole.UserRole)
            node_item.setToolTip(node_data.description or node_data.name)
            
            if isinstance(node_data, SetFolderNodeModel):
                node_item.setIcon(icons.get_qicon("FOLDER_VIRTUAL"))
                parent_qt_item.appendRow(node_item)
                if node_data.children: self._populate_collection_recursive(node_item, node_data.children)
            elif isinstance(node_data, ScriptSetNodeModel):
                node_item.setIcon(icons.get_qicon("INSTANCE_SET"))
                parent_qt_item.appendRow(node_item)
                for entry in node_data.script_entries:
                    script_info = self.controller.get_script_info_by_id(entry.id)
                    display_name = entry.name or (script_info.name if script_info else self.locale_manager.get("collection_widget.script_not_found_format", id=entry.id))
                    entry_item = QStandardItem(display_name)
                    entry_item.setEditable(False)
                    entry_item.setData(entry, Qt.ItemDataRole.UserRole)
                    self._items_by_instance_id[entry.instance_id] = entry_item
                    status = self.controller.script_run_statuses.get(entry.instance_id)
                    self._update_item_visuals(entry_item, status)

                    tooltip_html = generate_instance_tooltip_html(script_info, entry, self.locale_manager, self.theme_manager)
                    entry_item.setToolTip(tooltip_html)
                    
                    if not (script_info and script_info.passport_valid):
                        entry_item.setIcon(icons.get_qicon("ERROR"))
                        entry_item.setForeground(QColor("red"))

                    node_item.appendRow(entry_item)

    def _get_selected_qstandarditem(self) -> Optional[QStandardItem]:
        selected_indexes = self.collection_tree_view.selectedIndexes()
        return self.collection_model.itemFromIndex(selected_indexes[0]) if selected_indexes else None

    def _get_data_from_selected_item(self) -> Optional[Union[SetHierarchyNodeType, ScriptSetEntryModel]]:
        item = self._get_selected_qstandarditem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _select_item_by_id(self, node_id: str):
        root = self.collection_model.invisibleRootItem()
        for row in range(root.rowCount()):
            if item := root.child(row):
                if (found_index := self._find_item_recursive(item, node_id)).isValid():
                    self.collection_tree_view.setCurrentIndex(found_index)
                    return

    def _find_item_recursive(self, parent_item: QStandardItem, node_id_or_instance_id: str) -> QModelIndex:
        item_data = parent_item.data(Qt.ItemDataRole.UserRole)
        item_id = getattr(item_data, "instance_id", getattr(item_data, "id", None))
        if item_data and item_id == node_id_or_instance_id:
            return self.collection_model.indexFromItem(parent_item)
        for row in range(parent_item.rowCount()):
            if child_item := parent_item.child(row):
                if (found_index := self._find_item_recursive(child_item, node_id_or_instance_id)).isValid():
                    return found_index
        return QModelIndex()

    @Slot()
    def _on_selection_changed(self):
        selected_item = self._get_selected_qstandarditem()
        self.collection_model.dragged_item = selected_item
        selected_data = selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None
        node_id_to_activate = None
        if isinstance(selected_data, ScriptSetNodeModel):
            node_id_to_activate = selected_data.id
        elif isinstance(selected_data, ScriptSetEntryModel):
            if (item := self._get_selected_qstandarditem()) and item.parent():
                if isinstance(parent_data := item.parent().data(Qt.ItemDataRole.UserRole), ScriptSetNodeModel):
                    node_id_to_activate = parent_data.id
        self.controller.set_active_script_set_node(node_id_to_activate)
        self._update_buttons_state()


    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    def _update_buttons_state(self):
        """
        Обновляет состояние кнопок. Логика блокировки отдельных пунктов меню удалена.
        """
        current_state = self.controller._app_state
        collection_loaded = self.controller.current_collection_file_path is not None
        
        # 1. Кнопка Стоп
        self.btn_stop_action.setEnabled(current_state != AppState.IDLE)

        # 2. Кнопка Запуск
        if current_state == AppState.SET_RUNNING_STEP_WAIT:
            # Режим ожидания шага
            self.btn_run_action.setText(self.locale_manager.get("collection_widget.run_button_next_step"))
            self.btn_run_action.setIcon(icons.get_qicon("NEXT"))
            
            # Убираем меню, чтобы кнопка работала как обычный клик
            self.btn_run_action.setMenu(None)
            self.btn_run_action.setEnabled(True)
            self.btn_run_action.setProperty("class", "accent")
            self.style().polish(self.btn_run_action)

        elif current_state == AppState.IDLE:
            # Режим простоя
            self.btn_run_action.setText(self.locale_manager.get("collection_widget.run_button_run"))
            self.btn_run_action.setIcon(icons.get_qicon("PLAY"))
            
            # Возвращаем меню
            self.btn_run_action.setMenu(self.run_menu)
            self.btn_run_action.setProperty("class", "accent")
            self.style().polish(self.btn_run_action)
            
            # Логика доступности самой кнопки:
            # Кнопка активна, если загружена коллекция и выбрано хоть что-то (папка, набор, скрипт)
            has_selection = self.controller.selected_set_node_id is not None
            # (selected_set_node_id устанавливается контроллером даже если выбрана папка, 
            # но для надежности можно проверить наличие данных)
            selected_data = self._get_data_from_selected_item()
            is_something_selected = selected_data is not None
            
            can_run = collection_loaded and is_something_selected
            
            self.btn_run_action.setEnabled(can_run)
            
            # Сбрасываем блокировку пунктов меню (делаем все активными),
            # так как проверка теперь внутри _start_run
            self.action_run_full.setEnabled(True)
            self.action_run_step.setEnabled(True)
            self.action_run_single.setEnabled(True)
            
            tooltip = self.locale_manager.get("collection_widget.run_tooltip_save_first") if not collection_loaded else ""
            self.btn_run_action.setToolTip(tooltip)
            
        else:
            # Скрипты выполняются (RUNNING) - блокируем кнопку целиком
            self.btn_run_action.setEnabled(False)
            self.btn_run_action.setProperty("class", "")
            self.style().polish(self.btn_run_action)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


    @Slot("QPoint")
    def _show_context_menu(self, position):
        if self.controller.is_busy(): return
        index = self.collection_tree_view.indexAt(position)
        item = self.collection_model.itemFromIndex(index) if index.isValid() else None
        item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
        menu = QMenu(self)
        
        parent_id_for_new_node = None
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Логика определения родителя для новых узлов и импорта
        parent_folder_for_import = None
        if isinstance(item_data, SetFolderNodeModel):
            parent_id_for_new_node = item_data.id
            parent_folder_for_import = item_data
        elif isinstance(item_data, ScriptSetNodeModel):
            # Для набора тоже можно импортировать "рядом" с ним, т.е. в ту же папку
            parent_id_for_new_node = item_data.id # Для создания узлов "после"
            if item and item.parent():
                parent_folder_for_import = item.parent().data(Qt.ItemDataRole.UserRole)
        elif item and item.parent(): # Если выбран экземпляр скрипта
            parent_data = item.parent().data(Qt.ItemDataRole.UserRole)
            if isinstance(parent_data, SetFolderNodeModel):
                parent_id_for_new_node = parent_data.id
                parent_folder_for_import = parent_data
            elif isinstance(parent_data, ScriptSetNodeModel): # Родитель - набор
                 if item.parent().parent(): # Ищем "дедушку" - папку
                     parent_folder_for_import = item.parent().parent().data(Qt.ItemDataRole.UserRole)
        
        # Если клик был не на элементе, но есть корневая папка
        if not parent_folder_for_import:
            main_root = self.controller.set_manager.get_main_root_folder()
            if main_root:
                parent_folder_for_import = main_root
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        action_create_folder = QAction(self.locale_manager.get("collection_widget.context_menu.create_folder"), self)
        action_create_set = QAction(self.locale_manager.get("collection_widget.context_menu.create_set"), self)
        action_create_folder.triggered.connect(lambda: self._create_node_action(True, parent_id_for_new_node))
        action_create_set.triggered.connect(lambda: self._create_node_action(False, parent_id_for_new_node))
        menu.addAction(action_create_folder)
        menu.addAction(action_create_set)

        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Логика добавления пунктов импорта/экспорта
        menu.addSeparator()
        
        # Действие "Импортировать" доступно всегда, если определена родительская папка
        if parent_folder_for_import:
            action_import_set = QAction(self.locale_manager.get("collection_widget.context_menu.import_set"), self)
            action_import_set.triggered.connect(
                lambda: self.controller.import_set_requested(parent_folder_for_import.id)
            )
            menu.addAction(action_import_set)
        
        # Действие "Экспортировать" доступно, только если выбран набор
        if isinstance(item_data, ScriptSetNodeModel):
            action_export_set = QAction(self.locale_manager.get("collection_widget.context_menu.export_set"), self)
            action_export_set.triggered.connect(
                lambda: self.controller.export_set_requested(item_data.id)
            )
            menu.addAction(action_export_set)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        if isinstance(item_data, ScriptSetNodeModel):
            menu.addSeparator()
            action_paste = QAction(self.locale_manager.get("collection_widget.context_menu.paste_instance"), self)
            action_paste.setEnabled(self.controller._copied_script_entry is not None)
            action_paste.triggered.connect(lambda: self.controller.paste_script_instance_from_buffer(item_data.id))
            menu.addAction(action_paste)

        if item_data and isinstance(item_data, (SetFolderNodeModel, ScriptSetNodeModel)):
            menu.addSeparator()
            action_rename = QAction(self.locale_manager.get("collection_widget.context_menu.rename"), self)
            action_delete = QAction(self.locale_manager.get("collection_widget.context_menu.delete"), self)
            action_rename.triggered.connect(lambda: self._rename_node_action(item_data))
            action_delete.triggered.connect(lambda: self._delete_node_action(item_data))
            menu.addAction(action_rename)
            menu.addAction(action_delete)

        if isinstance(item_data, ScriptSetEntryModel):
            menu.addSeparator()

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # Управление избранным
            is_fav = self.controller.set_manager.is_favorite(item_data.instance_id)
            fav_text_key = "collection_widget.context_menu.remove_favorite" if is_fav else "collection_widget.context_menu.add_favorite"
            action_toggle_fav = QAction(self.locale_manager.get(fav_text_key), self)
            action_toggle_fav.setIcon(icons.get_qicon("STAR"))

            # Извлекаем иконку экземпляра (по умолчанию INSTANCE_ITEM, если пользователь ее не менял)
            icon_to_use = getattr(item_data, "icon_name", "INSTANCE_ITEM")
            color_to_use = getattr(item_data, "icon_color", None)
            
            # Передаем icon_to_use в слот
            action_toggle_fav.triggered.connect(
                lambda: self._toggle_favorite_action(item_data.instance_id, icon_to_use, color_to_use)
            )            
           
            menu.addAction(action_toggle_fav)
            menu.addSeparator()
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---            
            
            # Смена иконки для самого экземпляра
            action_change_tree_icon = QAction(self.locale_manager.get("collection_widget.context_menu.change_icon"), self)
            action_change_tree_icon.setIcon(icons.get_qicon("SETTINGS"))
            
            # Находим ID родительского набора, он нужен для обновления
            parent_set_id = None
            if item and item.parent():
                parent_data = item.parent().data(Qt.ItemDataRole.UserRole)
                if isinstance(parent_data, ScriptSetNodeModel):
                    parent_set_id = parent_data.id
            
            if parent_set_id:
                action_change_tree_icon.triggered.connect(
                    lambda: self._change_instance_icon_in_tree(parent_set_id, item_data)
                )
                menu.addAction(action_change_tree_icon)
                
            menu.addSeparator()
            
            action_params = QAction(self.locale_manager.get("collection_widget.context_menu.configure_params"), self)
            entry_data_copy = item_data.model_copy(deep=True)
            action_params.triggered.connect(lambda: self._show_script_instance_properties_dialog(entry_data_copy))
            menu.addAction(action_params)
            menu.addSeparator()
            action_copy = QAction(self.locale_manager.get("collection_widget.context_menu.copy_instance"), self)
            action_duplicate = QAction(self.locale_manager.get("collection_widget.context_menu.duplicate_instance"), self)
            action_delete_entry = QAction(self.locale_manager.get("collection_widget.context_menu.delete_from_set"), self)
            parent_set_id = self.controller.selected_set_node_id
            instance_id = item_data.instance_id
            action_copy.triggered.connect(lambda: self.controller.copy_script_instance_to_buffer(parent_set_id, instance_id))
            action_duplicate.triggered.connect(lambda: self.controller.duplicate_script_instance(parent_set_id, instance_id))
            action_delete_entry.triggered.connect(lambda: self._delete_entry_action(item_data))
            menu.addAction(action_copy)
            menu.addAction(action_duplicate)
            menu.addAction(action_delete_entry)

        menu.exec(self.collection_tree_view.viewport().mapToGlobal(position))

    def _create_node_action(self, is_folder: bool, parent_id: Optional[str]):
        title = self.locale_manager.get("collection_widget.create_folder.title") if is_folder else self.locale_manager.get("collection_widget.create_set.title")
        label = self.locale_manager.get("collection_widget.create_folder.label") if is_folder else self.locale_manager.get("collection_widget.create_set.label")
        text, ok = QInputDialog.getText(self, title, label)
        if ok and text:
            if is_folder: self.controller.create_folder_in_collection(text, parent_id)
            else: self.controller.create_set_in_collection(text, parent_id)

    def _rename_node_action(self, node_data: SetHierarchyNodeType):
        text, ok = QInputDialog.getText(self, self.locale_manager.get("collection_widget.rename.title"), self.locale_manager.get("collection_widget.rename.label"), text=node_data.name)
        if ok and text and text != node_data.name:
            self.controller.rename_node_in_collection(node_data.id, text)

    def _delete_node_action(self, node_data: SetHierarchyNodeType):
        reply = QMessageBox.question(self, self.locale_manager.get("collection_widget.delete_node.confirm_title"),
                                     self.locale_manager.get("collection_widget.delete_node.confirm_text", name=node_data.name),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.delete_node_from_collection(node_data.id)

    def _delete_entry_action(self, entry_data: ScriptSetEntryModel):
        if not self.controller.selected_set_node_id: return
        script_info = self.controller.get_script_info_by_id(entry_data.id)
        script_name = script_info.name if script_info else entry_data.id
        set_name = self.controller.selected_set_node_model.name if self.controller.selected_set_node_model else ""
        reply = QMessageBox.question(self, self.locale_manager.get("collection_widget.delete_entry.title"),
                                     self.locale_manager.get("collection_widget.delete_entry.text", script_name=script_name, set_name=set_name),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.remove_script_from_active_set_node(entry_data.instance_id)

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    @Slot(str, ScriptSetEntryModel)
    def _change_instance_icon_in_tree(self, set_id: str, entry_data: ScriptSetEntryModel):
        """Открывает диалог выбора иконки для экземпляра скрипта в основном дереве."""
        dialog = IconSelectionDialog(self.locale_manager, current_color=entry_data.icon_color, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            icon_name, icon_color = dialog.get_selected()
            if icon_name:
                updated_entry = entry_data.model_copy(deep=True)
                updated_entry.icon_name = icon_name
                updated_entry.icon_color = icon_color
                self.controller.update_script_instance_in_active_set_node(updated_entry)

    def _show_script_instance_properties_dialog(self, entry_data: ScriptSetEntryModel):
        script_info = self.controller.get_script_info_by_id(entry_data.id)
        if not script_info:
            QMessageBox.critical(self, self.locale_manager.get("general.error_title"),
                                 self.locale_manager.get("collection_widget.error.script_info_not_found", id=entry_data.id))
            return
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Собираем ВСЕ экземпляры из всей коллекции для выпадающих списков
        all_entries_in_collection: List[Tuple[str, ScriptSetEntryModel]] =[]
        def find_entries(nodes):
            for n in nodes:
                if isinstance(n, ScriptSetNodeModel):
                    for e in n.script_entries:
                        all_entries_in_collection.append((n.name, e))
                elif isinstance(n, SetFolderNodeModel) and n.children:
                    find_entries(n.children)
                    
        root_nodes = self.controller.set_manager.current_collection_model.root_nodes
        find_entries(root_nodes)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        
        dialog = ScriptPropertiesDialog(
            edit_mode=EditMode.INSTANCE, 
            script_info=script_info.model_copy(deep=True),
            instance_entry=entry_data, 
            locale_manager=self.locale_manager,
            theme_manager=self.theme_manager,
            available_script_entries=all_entries_in_collection, # <--- Передаем полный список
            get_script_name_func=self.controller.get_script_info_by_id,
            parent=self
        )
        
        if dialog.exec():
            if updated_instance_model := dialog.get_updated_instance_entry_model():
                self.controller.update_script_instance_in_active_set_node(updated_instance_model)


    @Slot(QModelIndex)
    def _on_double_clicked(self, index: QModelIndex):
        if not index.isValid(): return
        item = self.collection_model.itemFromIndex(index)
        if not item: return
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(item_data, ScriptSetEntryModel): return
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Теперь просто вызываем наш обновленный метод, передав ему копию данных
        self._show_script_instance_properties_dialog(item_data.model_copy(deep=True))
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---        
        

    def _custom_drop_mime_data(self, data: QMimeData, action: Qt.DropAction, row: int, column: int, parent_index: QModelIndex) -> bool:
        if not self.collection_model.canDropMimeData(data, action, row, column, parent_index): return False
        source_item = self.collection_model.dragged_item
        if not source_item: return False
        source_data = source_item.data(Qt.ItemDataRole.UserRole)

        if isinstance(source_data, (SetFolderNodeModel, ScriptSetNodeModel)):
            target_item = self.collection_model.itemFromIndex(parent_index) if parent_index.isValid() else None
            new_parent_id = target_item.data(Qt.ItemDataRole.UserRole).id if target_item else None
            self.controller.move_node_in_collection(source_data.id, new_parent_id, row)
            return True
        elif isinstance(source_data, ScriptSetEntryModel):
            source_parent_item = source_item.parent()
            set_node_data = source_parent_item.data(Qt.ItemDataRole.UserRole)
            current_ids = [entry.instance_id for entry in set_node_data.script_entries]
            dragged_id = source_data.instance_id
            current_ids.remove(dragged_id)
            if row == -1: current_ids.append(dragged_id)
            else: current_ids.insert(row, dragged_id)
            self.controller.reorder_scripts_in_active_set_node(current_ids)
            return True
        return False
        
    # ==============================================================================
    # БЛОК: Обработка избранного (Quick Access)
    # ==============================================================================
    @Slot()
    def _update_favorites_panel(self):
        """Очищает и перерисовывает панель избранных скриптов."""
        # Очистка старых кнопок
        while self.favorites_layout.count():
            item = self.favorites_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        favorites = self.controller.set_manager.get_all_favorites()
        self.favorites_scroll.setVisible(len(favorites) > 0)
        
        for fav in favorites:
            entry = self.controller.set_manager.find_script_entry_by_instance_id(fav.instance_id)
            if not entry:
                continue
                
            script_info = self.controller.get_script_info_by_id(entry.id)
            
            btn = QToolButton()
            btn.setIcon(icons.get_qicon(fav.icon_name, size=32, color=fav.icon_color))
            btn.setIconSize(QSize(32, 32))
            # Строго фиксируем размер, чтобы они не сжимались, а уходили за край экрана
            btn.setFixedSize(42, 42)            
            btn.setAutoRaise(True)
            # Меняем курсор на "руку" при наведении
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("class", "favorite-btn")

            
            # Применяем тот же тултип, что и в основном дереве
            tooltip_html = generate_favorite_tooltip_html(
                script_info=script_info,
                instance_entry=entry,
                locale_manager=self.locale_manager,
                theme_manager=self.theme_manager
            )
            btn.setToolTip(tooltip_html)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # Разрешаем кастомное контекстное меню для кнопки
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            # Привязываем сигнал меню. Используем lambda для захвата текущей кнопки и instance_id
            btn.customContextMenuRequested.connect(
                lambda pos, b=btn, i_id=fav.instance_id: self._show_favorite_context_menu(b, pos, i_id)
            )
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---            
            
            
            # Подключаем запуск
            btn.clicked.connect(lambda checked=False, i_id=fav.instance_id: self._run_favorite_script(i_id))
            self.favorites_layout.addWidget(btn)

    @Slot(str)
    def _run_favorite_script(self, instance_id: str):
        """Запускает скрипт из панели избранного."""
        if self.controller.is_busy():
            return
            
        set_id = self.controller.set_manager.find_parent_set_id_for_instance(instance_id)
        if not set_id:
            QMessageBox.warning(
                self, 
                self.locale_manager.get("collection_widget.run_error.title"), 
                self.locale_manager.get("collection_widget.run_error.parent_set_not_found")
            )
            return
            
        continue_on_error = self.chk_continue_on_error.isChecked()
        self.controller.run_script_set(set_id, SetRunMode.SINGLE_FROM_SET, instance_id, continue_on_error)

    @Slot(str, str, str) # Типизацию слота можно оставить так, или убрать @Slot для надежности с None
    def _toggle_favorite_action(self, instance_id: str, icon_name: str = "STAR", icon_color: Optional[str] = None):
        self.controller.toggle_script_favorite(instance_id, icon_name, icon_color)

    @Slot(str)
    def _change_favorite_icon_action(self, instance_id: str):
        # Находим текущий цвет, чтобы передать в диалог
        current_color = None
        for fav in self.controller.set_manager.get_all_favorites():
            if fav.instance_id == instance_id:
                current_color = fav.icon_color
                break
                
        dialog = IconSelectionDialog(self.locale_manager, current_color=current_color, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            icon_name, icon_color = dialog.get_selected()
            if icon_name:
                self.controller.change_favorite_icon(instance_id, icon_name, icon_color)

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    @Slot(object, "QPoint", str)
    def _show_favorite_context_menu(self, button: QToolButton, pos, instance_id: str):
        """Отображает контекстное меню при клике ПКМ по кнопке в панели избранного."""
        menu = QMenu(self)
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Добавляем действие "Показать в дереве"
        action_locate = QAction(self.locale_manager.get("collection_widget.context_menu.locate_in_tree"), self)
        action_locate.setIcon(icons.get_qicon("TARGET"))
        action_locate.triggered.connect(lambda: self._select_item_by_id(instance_id))
        menu.addAction(action_locate)
        
        menu.addSeparator()
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        
        action_change_icon = QAction(self.locale_manager.get("collection_widget.context_menu.change_icon"), self)
        action_change_icon.setIcon(icons.get_qicon("SETTINGS"))
        action_change_icon.triggered.connect(lambda: self._change_favorite_icon_action(instance_id))
        menu.addAction(action_change_icon)
        
        menu.addSeparator()
        
        action_remove = QAction(self.locale_manager.get("collection_widget.context_menu.remove_favorite"), self)
        action_remove.setIcon(icons.get_qicon("DELETE"))
        action_remove.triggered.connect(lambda: self._toggle_favorite_action(instance_id))
        menu.addAction(action_remove)
        
        menu.exec(button.mapToGlobal(pos))