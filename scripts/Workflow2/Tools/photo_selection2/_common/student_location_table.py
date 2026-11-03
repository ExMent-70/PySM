"""Flat student/location matrix with individually selectable photo lines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QStyledItemDelegate, QStyle,
    QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QToolTip,
)

from pysm_lib.pysm_icons import icons
from .constants import ITEM_STUDENT_ROLE, ITEM_LOCATION_ROLE

PHOTOS_ROLE = ITEM_LOCATION_ROLE + 1


@dataclass(frozen=True)
class PhotoLine:
    """Display-only photo information; assignment and filesystem state stay untouched."""

    number: str
    icon_names: tuple[str, ...]
    paths: tuple[Path, ...]
    tooltip: str


class PhotoCellDelegate(QStyledItemDelegate):
    """Paint compact photo lines using the existing theme-aware PySM icons."""

    PADDING = 5
    ICON_SIZE = 18
    ICON_STEP = 21

    def line_height(self):
        return max(24, self.parent().fontMetrics().height() + 6)

    def line_at(self, rect, position, count):
        if not count:
            return -1
        offset = (position.y() - rect.top() - self.PADDING) // self.line_height()
        return int(offset) if 0 <= offset < count else -1

    def paint(self, painter, option, index):
        if index.column() == 0:
            super().paint(painter, option, index)
            return
        entries = index.data(PHOTOS_ROLE) or ()
        background = QStyleOptionViewItem(option)
        self.initStyleOption(background, index)
        background.text = ""
        widget = option.widget
        widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, background, painter, widget)
        painter.save()
        painter.setClipRect(option.rect)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        color = option.palette.color(QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text)
        painter.setPen(color)
        if not entries:
            if not selected:
                tint = QColor("#d69b22")
                tint.setAlpha(18)
                painter.fillRect(option.rect, tint)
                painter.setPen(option.palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(option.rect.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignCenter, "—")
        for offset, entry in enumerate(entries):
            rect = QRect(option.rect.left() + self.PADDING,
                         option.rect.top() + self.PADDING + offset * self.line_height(),
                         option.rect.width() - 2 * self.PADDING, self.line_height())
            if selected and entry.number == self.parent().selected_number():
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
            x = rect.left() + 3
            for name in entry.icon_names:
                icon_rect = QRect(x, rect.center().y() - self.ICON_SIZE // 2,
                                  self.ICON_SIZE, self.ICON_SIZE)
                icons.get_qicon(name, self.ICON_SIZE).paint(painter, icon_rect)
                x += self.ICON_STEP
            painter.drawText(QRect(x + 4, rect.top(), max(0, rect.right() - x - 4), rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, entry.number)
        painter.restore()

    def helpEvent(self, event, view, option, index):
        entries = index.data(PHOTOS_ROLE) or ()
        offset = self.line_at(option.rect, event.pos(), len(entries))
        if offset >= 0:
            QToolTip.showText(event.globalPos(), entries[offset].tooltip, view)
            return True
        return super().helpEvent(event, view, option, index)

    def sizeHint(self, option, index):
        entries = index.data(PHOTOS_ROLE) or ()
        if not entries:
            return super().sizeHint(option, index)
        width = max(len(entry.icon_names) * self.ICON_STEP
                    + option.fontMetrics.horizontalAdvance(entry.number) + 22 for entry in entries)
        return QSize(width, len(entries) * self.line_height() + 2 * self.PADDING)


class StudentLocationTable(QTableWidget):
    """One student per row, one location per column, with mouse/keyboard photo selection."""

    photoActivated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_number = ""
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setSortingEnabled(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setResizeContentsPrecision(-1)
        self.setItemDelegate(PhotoCellDelegate(self))
        self.currentCellChanged.connect(self._cell_changed)
        self.horizontalHeader().sortIndicatorChanged.connect(
            lambda *_: QTimer.singleShot(0, self.resizeRowsToContents)
        )

    def entries(self, item=None):
        item = self.currentItem() if item is None else item
        return tuple(item.data(PHOTOS_ROLE) or ()) if item is not None else ()

    def selected_number(self):
        entries = self.entries()
        if any(entry.number == self._active_number for entry in entries):
            return self._active_number
        return entries[0].number if entries else ""

    def selected_photo(self):
        return next((entry for entry in self.entries() if entry.number == self.selected_number()), None)

    def _cell_changed(self, *_):
        self._active_number = ""
        self.photoActivated.emit()

    def select_photo_at(self, position):
        """Resolve a click to a specific line, including right-click context menus."""
        item = self.itemAt(position)
        if item is None:
            return
        self.setCurrentItem(item)
        offset = self.itemDelegate().line_at(self.visualItemRect(item), position, len(self.entries(item)))
        self._active_number = self.entries(item)[offset].number if offset >= 0 else ""
        self.viewport().update()
        self.photoActivated.emit()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.select_photo_at(event.position().toPoint())

    def keyPressEvent(self, event):
        entries = self.entries()
        if entries and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            offset = next(i for i, entry in enumerate(entries) if entry.number == self.selected_number())
            following = offset + (1 if event.key() == Qt.Key.Key_Down else -1)
            if 0 <= following < len(entries):
                self._active_number = entries[following].number
                self.viewport().update()
                self.photoActivated.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def selection_key(self):
        item = self.currentItem()
        if item is None:
            return None
        return item.data(ITEM_STUDENT_ROLE), item.data(ITEM_LOCATION_ROLE), self.selected_number()

    def restore_selection(self, key):
        """Restore semantic selection after refreshing or sorting the matrix."""
        if key is None:
            return
        student_id, location, number = key
        for row in range(self.rowCount()):
            if self.item(row, 0).data(ITEM_STUDENT_ROLE) != student_id:
                continue
            for column in range(self.columnCount()):
                item = self.item(row, column)
                if item.data(ITEM_LOCATION_ROLE) == location:
                    previous = self.blockSignals(True)
                    try:
                        self.setCurrentItem(item)
                        self._active_number = number
                    finally:
                        self.blockSignals(previous)
                    self.viewport().update()
                    return

    def populate(self, locations, rows):
        """Fit location columns to all contents and give the remaining width to names."""
        section = self.horizontalHeader().sortIndicatorSection()
        sort_name = self.horizontalHeaderItem(section).text() if 0 <= section < self.columnCount() else "ФИО ученика"
        order = self.horizontalHeader().sortIndicatorOrder() if self.columnCount() else Qt.SortOrder.AscendingOrder
        headers = ["ФИО ученика", *locations]
        blocked = self.blockSignals(True)
        self.setUpdatesEnabled(False)
        self.setSortingEnabled(False)
        try:
            self.clear()
            self.setColumnCount(len(headers))
            self.setHorizontalHeaderLabels(headers)
            self.setRowCount(len(rows))
            for row, (student_id, name, by_location) in enumerate(rows):
                name_item = QTableWidgetItem(name)
                name_item.setData(ITEM_STUDENT_ROLE, student_id)
                count = sum(bool(entries) for entries in by_location.values())
                name_item.setToolTip(f"{name}\nВыбрано локаций: {count}/{len(locations)}")
                self.setItem(row, 0, name_item)
                for column, location in enumerate(locations, 1):
                    entries = tuple(by_location.get(location, ()))
                    item = QTableWidgetItem("\n".join(entry.number for entry in entries) or "—")
                    item.setData(ITEM_STUDENT_ROLE, student_id)
                    item.setData(ITEM_LOCATION_ROLE, location)
                    item.setData(PHOTOS_ROLE, entries)
                    item.setToolTip("\n\n".join(entry.tooltip for entry in entries)
                                    or f"{name}\n{location}\nВыбор отсутствует")
                    self.setItem(row, column, item)
            self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for column in range(1, len(headers)):
                self.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            self._active_number = ""
            self.setSortingEnabled(True)
            self.sortItems(headers.index(sort_name) if sort_name in headers else 0, order)
            self.resizeRowsToContents()
        finally:
            self.setUpdatesEnabled(True)
            self.blockSignals(blocked)

    def populate_records(self, locations, students, record_index, file_entries):
        """Adapt the existing assignment/file indexes without recalculating selection."""
        rows = []
        for student in sorted(students, key=lambda value: value.display_name.casefold()):
            by_location = {}
            for location in locations:
                entries = []
                for record in record_index.get((student.student_id, location), []):
                    icon_names, hints = [], []
                    if record.selected_student_ids:
                        icon_names.append("PHOTO_PORTRAIT")
                        hints.append("Фотография выбрана клиентом")
                    if record.photographer_selected:
                        icon_names.append("CAMERA")
                        hints.append("Фотография выбрана фотографом")
                    files = file_entries.get(record.number, [])
                    origins = {origin for _, origin, _ in files}
                    if any("исходник" in origin for origin in origins):
                        icon_names.append("FOLDER")
                        hints.append("Файл расположен в исходной папке")
                    if any("целевая папка" in origin for origin in origins):
                        icon_names.append("FOLDER_VIRTUAL")
                        hints.append("Файл расположен в целевой папке")
                    paths = tuple(dict.fromkeys(path for _, _, physical in files for path in physical))
                    if not paths:
                        icon_names.append("WARNING")
                        hints.append("Физические файлы не найдены")
                    tooltip = "\n".join([student.display_name, location, record.number, *hints,
                                          *(str(path) for path in paths)])
                    entries.append(PhotoLine(record.number, tuple(icon_names), paths, tooltip))
                by_location[location] = entries
            rows.append((student.student_id, student.display_name, by_location))
        self.populate(locations, rows)
