# analize/cluster_editor/_lib/editor_delegates.py

from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtGui import QPixmap, QPainter, QColor, QPalette
from PySide6.QtCore import Qt, QRect, QSize

from .editor_styles import THUMBNAIL_SIZE, PREVIEW_SIZE

# 1. Блок: Новый базовый класс для общей логики отрисовки
# ==============================================================================
class BaseItemDelegate(QStyledItemDelegate):
    """
    Базовый делегат, отвечающий за отрисовку фона и рамок
    в зависимости от состояния и переданных цветов.
    """
    def __init__(self, hover_border_color: QColor, parent=None):
        super().__init__(parent)
        self.hover_border_color = hover_border_color

    def paint_background(self, painter: QPainter, option, index):
        """Отрисовывает фон и рамку элемента."""
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_under_mouse = option.state & QStyle.StateFlag.State_MouseOver
        
        # Используем цвета из палитры виджета, которая настраивается через QSS
        palette = option.palette
        bg_color = palette.color(QPalette.ColorRole.Base)
        border_color = palette.color(QPalette.ColorRole.Mid)

        if is_selected:
            bg_color = palette.color(QPalette.ColorRole.Highlight)
            border_color = palette.color(QPalette.ColorRole.Highlight)
        elif is_under_mouse:
            # Для наведения используем кастомный цвет, переданный в конструкторе
            border_color = self.hover_border_color

        painter.setBrush(bg_color)
        painter.setPen(border_color)
        painter.drawRoundedRect(option.rect.adjusted(1, 1, -1, -1), 5, 5)

# 2. Блок: Обновленный ClusterItemDelegate
# ==============================================================================
class ClusterItemDelegate(BaseItemDelegate):
    """Делегат для отрисовки карточки кластера."""
    def __init__(self, hover_border_color: QColor, changed_indicator_color: QColor,
                 preview_bg_color: QColor, secondary_text_color: QColor, parent=None):
        super().__init__(hover_border_color, parent)
        # Сохраняем остальные кастомные цвета
        self.changed_indicator_color = changed_indicator_color
        self.preview_bg_color = preview_bg_color
        self.secondary_text_color = secondary_text_color

    def paint(self, painter: QPainter, option, index):
        painter.save()
        
        # 1. Отрисовываем фон и рамку с помощью базового класса
        self.paint_background(painter, option, index)
        
        # Получаем данные и палитру
        item_data = index.data(Qt.ItemDataRole.UserRole)
        palette = option.palette
        is_selected = option.state & QStyle.StateFlag.State_Selected
        
        cluster_name = item_data.get("name", "N/A")
        count = item_data.get("count", 0)
        pixmap: QPixmap = item_data.get("pixmap")
        is_changed = item_data.get("is_changed", False)
        
        bg_rect = option.rect

        # 2. Отрисовка индикатора изменений (если нужно)
        if is_changed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.changed_indicator_color)
            painter.drawRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), 3)

        # 3. Отрисовка превью
        preview_bg_rect = QRect(bg_rect.x() + 6, bg_rect.y() + 6, PREVIEW_SIZE, PREVIEW_SIZE)
        painter.setBrush(self.preview_bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, 3, 3)

        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(self.secondary_text_color)
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "No Preview")

        # 4. Отрисовка текста (цвета из палитры)
        text_color = palette.color(QPalette.ColorRole.HighlightedText if is_selected else QPalette.ColorRole.WindowText)
        painter.setPen(text_color)
        
        font = painter.font()
        font.setBold(False)
        font.setPointSize(11)
        painter.setFont(font)
        name_rect = QRect(bg_rect.x() + 6, preview_bg_rect.bottom() + 4, PREVIEW_SIZE, 40)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, cluster_name)

        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(self.secondary_text_color)
        count_rect = QRect(name_rect.bottomLeft(), QSize(PREVIEW_SIZE, 20))
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignCenter, f"Фото: {count}")
        
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(PREVIEW_SIZE + 12, PREVIEW_SIZE + 12 + 40 + 20)

# 3. Блок: Обновленный ImageItemDelegate
# ==============================================================================
class ImageItemDelegate(BaseItemDelegate):
    """Делегат для отрисовки карточки изображения."""
    # Конструктор теперь принимает только цвет рамки при наведении
    def __init__(self, hover_border_color: QColor, parent=None):
        super().__init__(hover_border_color, parent)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        
        # 1. Отрисовываем фон и рамку с помощью базового класса
        self.paint_background(painter, option, index)
        
        pixmap: QPixmap = index.data(Qt.ItemDataRole.DecorationRole)
        bg_rect = option.rect

        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
            painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, "Image\nNot Found")
        
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(THUMBNAIL_SIZE + 12, THUMBNAIL_SIZE + 12)