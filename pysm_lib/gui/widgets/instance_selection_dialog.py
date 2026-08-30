# pysm_lib/gui/widgets/instance_selection_dialog.py

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QWidget,
)

from ...models import ScriptSetEntryModel, ScriptInfoModel
from ...locale_manager import LocaleManager
from ...theme_manager import ThemeManager
from ..tooltip_generator import generate_instance_tooltip_html
from ...pysm_icons import icons


class _InstanceTreeItemDelegate(QStyledItemDelegate):
    """Дорисовывает ясный минус для частично отмеченных групп."""

    def paint(self, painter: QPainter, option, index) -> None:
        paint_option = QStyleOptionViewItem(option)
        paint_option.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, paint_option, index)

        if (
            index.data(Qt.ItemDataRole.CheckStateRole)
            != Qt.CheckState.PartiallyChecked.value
        ):
            return

        prepared_option = QStyleOptionViewItem(paint_option)
        self.initStyleOption(prepared_option, index)
        indicator_rect = paint_option.widget.style().subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator,
            prepared_option,
            paint_option.widget,
        )
        if not indicator_rect.isValid():
            return

        pen = QPen(paint_option.palette.highlightedText().color(), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.save()
        painter.setPen(pen)
        center_y = indicator_rect.center().y()
        painter.drawLine(
            indicator_rect.left() + 4,
            center_y,
            indicator_rect.right() - 4,
            center_y,
        )
        painter.restore()


class InstanceSelectionDialog(QDialog):
    """
    Диалог для мульти-выбора экземпляров скриптов из коллекции (макрос).
    Группирует скрипты по родительским наборам в QTreeWidget с чекбоксами.
    """

    def __init__(
        self,
        title: str,
        script_entries: List[Tuple[str, ScriptSetEntryModel]],
        get_script_name_func: callable,
        locale_manager: LocaleManager,
        theme_manager: ThemeManager,
        current_value: Optional[str] = None,
        forbidden_instance_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.theme_manager = theme_manager

        self.setWindowTitle(title)
        self.setMinimumSize(550, 650)
        self.setObjectName("InstanceSelectionDialog")

        main_layout = QVBoxLayout(self)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(
            self.locale_manager.get("dialogs.context_editor.search_placeholder")
        )
        main_layout.addWidget(self.search_bar)

        buttons_layout = QHBoxLayout()

        self.btn_expand_all = QPushButton(self.locale_manager.get("dialogs.instance_editor.expand_all"))
        self.btn_collapse_all = QPushButton(self.locale_manager.get("dialogs.instance_editor.collapse_all"))
        self.btn_clear_selection = QPushButton(self.locale_manager.get("dialogs.instance_editor.clear_selection"))

        self.btn_expand_all.setFixedHeight(26)
        self.btn_collapse_all.setFixedHeight(26)
        self.btn_clear_selection.setFixedHeight(26)

        buttons_layout.addWidget(self.btn_expand_all)
        buttons_layout.addWidget(self.btn_collapse_all)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_clear_selection)

        main_layout.addLayout(buttons_layout)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setObjectName("InstanceSelectionTree")
        self.tree_widget.setHeaderHidden(True)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setItemDelegate(_InstanceTreeItemDelegate(self.tree_widget))
        main_layout.addWidget(self.tree_widget, 1)

        # Парсим текущие выбранные ID
        current_ids = set([x.strip() for x in (current_value or "").split(",") if x.strip()])

        def resolve_instance_name(instance_id: str) -> Optional[str]:
            for target_set_name, target_entry in script_entries:
                if target_entry.instance_id != instance_id:
                    continue
                try:
                    target_script_info = get_script_name_func(target_entry.id)
                except Exception:
                    target_script_info = None
                target_name = target_entry.name or (
                    target_script_info.name if target_script_info else None
                )
                return (
                    f"[{target_set_name}] {target_name}" if target_name else None
                )
            return None

        sets_map = {}
        first_checked_item: Optional[QTreeWidgetItem] = None
        for set_name, entry in script_entries:
            if set_name not in sets_map:
                set_item = QTreeWidgetItem([set_name])
                set_item.setIcon(0, icons.get_qicon("INSTANCE_SET"))
                # Включаем чекбокс и авто-управление детьми (Tristate)
                set_item.setFlags(set_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                set_item.setCheckState(0, Qt.CheckState.Unchecked)
                self.tree_widget.addTopLevelItem(set_item)
                sets_map[set_name] = set_item

            parent_item = sets_map[set_name]

            script_name = "Unknown Script"
            script_info: Optional[ScriptInfoModel] = None
            try:
                script_info = get_script_name_func(entry.id)
                if script_info:
                    script_name = script_info.name
            except Exception:
                pass

            display_name = entry.name or script_name

            item = QTreeWidgetItem(parent_item, [display_name])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.instance_id)
            # Включаем чекбокс
            if forbidden_instance_id and entry.instance_id == forbidden_instance_id:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip(0, "Нельзя ссылаться на самого себя")
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if entry.instance_id in current_ids:
                    item.setCheckState(0, Qt.CheckState.Checked)
                    if first_checked_item is None:
                        first_checked_item = item
                else:
                    item.setCheckState(0, Qt.CheckState.Unchecked)

                tooltip_html = generate_instance_tooltip_html(
                    script_info=script_info,
                    instance_entry=entry,
                    locale_manager=self.locale_manager,
                    theme_manager=self.theme_manager,
                    instance_name_resolver=resolve_instance_name,
                )
                item.setToolTip(0, tooltip_html)

            icon_to_use = getattr(entry, "icon_name", "INSTANCE_ITEM")
            icon_color = getattr(entry, "icon_color", None)
            item.setIcon(0, icons.get_qicon(icon_to_use, color=icon_color))

            tooltip_html = generate_instance_tooltip_html(
                script_info=script_info,
                instance_entry=entry,
                locale_manager=self.locale_manager,
                theme_manager=self.theme_manager,
                instance_name_resolver=resolve_instance_name,
            )
            item.setToolTip(0, tooltip_html)

        self._set_initial_tree_state(first_checked_item)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.search_bar.textChanged.connect(self._filter_list)

        # --- Сигналы для кнопок ---
        self.btn_expand_all.clicked.connect(self.tree_widget.expandAll)
        self.btn_collapse_all.clicked.connect(self.tree_widget.collapseAll)
        self.btn_clear_selection.clicked.connect(self._clear_checkboxes)

    def _set_initial_tree_state(
        self,
        first_checked_item: Optional[QTreeWidgetItem],
    ) -> None:
        """Сворачивает дерево и показывает первый отмеченный экземпляр."""
        self.tree_widget.collapseAll()
        if first_checked_item is None:
            return

        parent_item = first_checked_item.parent()
        if parent_item is not None:
            parent_item.setExpanded(True)

        self.tree_widget.setCurrentItem(first_checked_item)
        QTimer.singleShot(0, self._scroll_to_current_item)

    @Slot()
    def _scroll_to_current_item(self) -> None:
        """Прокручивает дерево после расчёта геометрии диалога."""
        current_item = self.tree_widget.currentItem()
        if current_item is not None:
            self.tree_widget.scrollToItem(
                current_item,
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    @Slot(str)
    def _filter_list(self, text: str):
        search_text = text.lower()
        for i in range(self.tree_widget.topLevelItemCount()):
            top_item = self.tree_widget.topLevelItem(i)
            any_visible = False
            for j in range(top_item.childCount()):
                child = top_item.child(j)
                if search_text in child.text(0).lower():
                    child.setHidden(False)
                    any_visible = True
                else:
                    child.setHidden(True)
            top_item.setHidden(not any_visible)

    @Slot()
    def _clear_checkboxes(self):
        """Снимает галочки со всех элементов дерева (надежный обход)."""
        for i in range(self.tree_widget.topLevelItemCount()):
            top_item = self.tree_widget.topLevelItem(i)
            top_item.setCheckState(0, Qt.CheckState.Unchecked)
            # Явно снимаем галочки с дочерних элементов для надежности
            for j in range(top_item.childCount()):
                child = top_item.child(j)
                if child.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    child.setCheckState(0, Qt.CheckState.Unchecked)

    def get_selected_instance_id(self) -> Optional[str]:
        """Возвращает строку с ID выбранных экземпляров через запятую. Если пусто - возвращает ''."""
        if self.result() == QDialog.DialogCode.Accepted:
            selected_ids = list()  # Избегаем пустых скобок для парсера

            for i in range(self.tree_widget.topLevelItemCount()):
                top_item = self.tree_widget.topLevelItem(i)
                for j in range(top_item.childCount()):
                    child = top_item.child(j)
                    if child.checkState(0) == Qt.CheckState.Checked:
                        if child.flags() & Qt.ItemFlag.ItemIsEnabled:
                            selected_ids.append(child.data(0, Qt.ItemDataRole.UserRole))
            
            # Возвращаем склеенную строку или пустую строку, если список пуст
            return ",".join(selected_ids)

        # Возвращаем None ТОЛЬКО если пользователь нажал Отмена (Cancel)
        return None
