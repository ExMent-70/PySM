# analize/cluster_editor/_lib/editor_delegates.py
"""
Модуль, содержащий делегаты для отрисовки кастомных элементов
в стандартных виджетах Qt (например, QListWidget).
"""
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, QRect, QSize

from .editor_styles import THUMBNAIL_SIZE, PREVIEW_SIZE

class ClusterItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки кластера в QListWidget."""

    def paint(self, painter: QPainter, option, index):
        painter.save()
        
        # Получаем данные из элемента списка
        item_data = index.data(Qt.ItemDataRole.UserRole)
        cluster_name = item_data.get("name", "N/A")
        count = item_data.get("count", 0)
        pixmap: QPixmap = item_data.get("pixmap")
        is_changed = item_data.get("is_changed", False)
        
        # Состояния элемента
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_under_mouse = option.state & QStyle.StateFlag.State_MouseOver

        # --- Отрисовка фона и рамки ---
        bg_rect = option.rect
        if is_selected:
            painter.setPen(QColor("#007bff")) # dodgerblue
            painter.setBrush(QColor("#2e2e2e"))
        elif is_under_mouse:
            painter.setPen(QColor("orange"))
            painter.setBrush(QColor("#2e2e2e"))
        else:
            painter.setPen(QColor("#444"))
            painter.setBrush(QColor("#2e2e2e"))
        
        painter.drawRoundedRect(bg_rect.adjusted(1, 1, -1, -1), 5, 5)

        if is_changed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("orange"))
            painter.drawRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), 3)
            #painter.drawRect(bg_rect.x(), bg_rect.y()+bg_rect.height()-2, bg_rect.width(), 4)

        # --- Отрисовка превью ---
        preview_bg_rect = QRect(bg_rect.x() + 6, bg_rect.y() + 6, PREVIEW_SIZE, PREVIEW_SIZE)
        painter.setBrush(QColor("#222"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, 3, 3)

        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(QColor("#777"))
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "No Preview")

        # --- Отрисовка текста ---
        painter.setPen(QColor("white"))
        font = painter.font()
        font.setBold(False)
        font.setPointSize(11)
        painter.setFont(font)
        name_rect = QRect(bg_rect.x() + 6, preview_bg_rect.bottom() + 4, PREVIEW_SIZE, 40)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, cluster_name)

        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor("#aaa"))
        count_rect = QRect(name_rect.bottomLeft(), QSize(PREVIEW_SIZE, 20))
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignCenter, f"Фото: {count}")
        
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        """Возвращает размер элемента."""
        return QSize(PREVIEW_SIZE + 12, PREVIEW_SIZE + 12 + 40 + 20)

class ImageItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки изображения в QListWidget."""

    def paint(self, painter: QPainter, option, index):
        painter.save()
        
        pixmap: QPixmap = index.data(Qt.ItemDataRole.DecorationRole)
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_under_mouse = option.state & QStyle.StateFlag.State_MouseOver

        bg_rect = option.rect
        if is_selected:
            painter.setPen(QColor("dodgerblue"))
        elif is_under_mouse:
            painter.setPen(QColor("orange"))
        else:
            painter.setPen(QColor("#444"))
        
        painter.setBrush(QColor("#2e2e2e"))
        painter.drawRoundedRect(bg_rect.adjusted(1, 1, -1, -1), 5, 5)

        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(QColor("#777"))
            painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, "Image\nNot Found")
        
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(THUMBNAIL_SIZE + 12, THUMBNAIL_SIZE + 12)