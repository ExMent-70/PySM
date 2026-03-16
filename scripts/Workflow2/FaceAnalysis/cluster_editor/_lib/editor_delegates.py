# analize/cluster_editor/_lib/editor_delegates.py
# -*- coding: utf-8 -*-

"""
Модуль делегатов (Delegates) для кастомной отрисовки элементов списков.
"""

from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QStyledItemDelegate, 
    QStyle, 
    QStyleOptionViewItem, 
    QListWidget, 
    QWidget
)
from PySide6.QtGui import (
    QPixmap, 
    QPainter, 
    QColor, 
    QPalette, 
    QPen,
    QBrush # <--- Добавлено
)
from PySide6.QtCore import Qt, QRect, QSize, QRectF

# ==============================================================================
# VISUAL CONFIG (НАСТРОЙКИ ВНЕШНЕГО ВИДА)
# ==============================================================================

# --- Размеры элементов (в пикселях) ---
THUMBNAIL_SIZE = 180  # Размер квадратной миниатюры в галерее
PREVIEW_SIZE = 180    # Размер миниатюры в списке кластеров
FACE_SIZE = 130       # Базовый размер лица (используется в других модулях)
FACE_SIZE_PORTRAIT = 290       # Базовый размер лица (используется в других модулях)
FACE_MIN = 100        # Мин. размер слайдера лиц
FACE_MAX = 400        # Макс. размер слайдера лиц

# --- Отступы и рамки ---
ITEM_PADDING = 6           # Внутренний отступ от края ячейки до контента
BORDER_RADIUS = 3          # Радиус скругления серой подложки
BORDER_RADIUS_BG = 8       # <--- Радиус скругления фона выделения (Сделали побольше)
CHANGED_INDICATOR_H = 3    # Высота цветной полоски "Изменено"
DROP_BORDER_WIDTH = 2      # Толщина рамки при Drag&Drop

# --- Текст ---
TEXT_NAME_HEIGHT = 40
TEXT_COUNT_HEIGHT = 20
FONT_SIZE_TITLE = 11
FONT_SIZE_COUNT = 9

# ==============================================================================
# THEME API INTEGRATION
# ==============================================================================

try:
    from pysm_lib import theme_api
except ImportError:
    class ThemeAPIMock:
        def get_parsed_style(self, key: str, *args, **kwargs) -> Dict[str, str]:
            defaults = {
                "delegate_changed_indicator": {"color": "#f0ad4e"},
                "delegate_preview_background": {"color": "#e8e8e8"},
                "delegate_secondary_text": {"color": "#555555"},
                "delegate_hover_border": {"color": "#0078d7"}
            }
            return defaults.get(key, {})
    theme_api = ThemeAPIMock()


class ClusterItemDelegate(QStyledItemDelegate):
    """
    Делегат для отрисовки карточки кластера в левой панели.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        changed_styles = theme_api.get_parsed_style("delegate_changed_indicator")
        preview_styles = theme_api.get_parsed_style("delegate_preview_background")
        secondary_styles = theme_api.get_parsed_style("delegate_secondary_text")
        hover_styles = theme_api.get_parsed_style("delegate_hover_border")

        self.changed_indicator_color = QColor(changed_styles.get("color", "#f0ad4e"))
        self.preview_bg_color = QColor(preview_styles.get("color", "#e8e8e8"))
        self.secondary_text_color = QColor(secondary_styles.get("color", "#555555"))
        self.drop_border_color = QColor(hover_styles.get("color", "#0078d7"))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        """
        Основной метод отрисовки.
        """
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Шаг 2: Извлечение данных (Подняли наверх для проверки флагов) ---
        item_data = index.data(Qt.ItemDataRole.UserRole) or {}
        cluster_name = item_data.get("name", "N/A")
        count = item_data.get("count", 0)
        pixmap: Optional[QPixmap] = item_data.get("pixmap")
        is_changed = item_data.get("is_changed", False)
        
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_hovered = option.state & QStyle.StateFlag.State_MouseOver
        
        bg_rect = option.rect

        # --- Шаг 1: Отрисовка скругленного фона (Texture Buffer Method) ---
        # Мы рисуем фон только если элемент выделен или под курсором (оптимизация)
        if is_selected or is_hovered:
            # 1.1. Создаем временный буфер (QPixmap) размером с ячейку
            # Это позволяет нам отрисовать стиль темы (QSS) в картинку
            buffer = QPixmap(option.rect.size())
            buffer.fill(Qt.transparent)
            
            p = QPainter(buffer)
            # Создаем копию опций, смещенную в 0,0 (локальные координаты буфера)
            temp_option = QStyleOptionViewItem(option)
            temp_option.rect = QRect(0, 0, option.rect.width(), option.rect.height())
            # Убираем текст и иконки, нам нужен только фон от стиля
            temp_option.text = "" 
            temp_option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
            
            # Просим стиль нарисовать фон (учитывая цвета темы hover/select)
            option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, temp_option, p, option.widget)
            p.end()
            
            # 1.2. Рисуем этот буфер как текстуру (Brush) внутри скругленного прямоугольника
            # drawRoundedRect поддерживает качественный антиалиасинг, в отличие от setClipPath
            painter.setBrush(QBrush(buffer))
            painter.setBrushOrigin(option.rect.topLeft()) # Важно: совмещаем текстуру с ячейкой
            painter.setPen(Qt.NoPen)
            
            # Небольшой отступ, чтобы выделения не слипались
            draw_rect = QRectF(option.rect).adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(draw_rect, BORDER_RADIUS_BG, BORDER_RADIUS_BG)

        # --- Шаг 3: Индикатор изменений ---
        if is_changed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.changed_indicator_color)
            painter.drawRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), CHANGED_INDICATOR_H)

        # --- Шаг 4: Подложка под миниатюру ---
        preview_bg_rect = QRect(
            bg_rect.x() + ITEM_PADDING, 
            bg_rect.y() + ITEM_PADDING, 
            PREVIEW_SIZE, 
            PREVIEW_SIZE
        )
        painter.setBrush(self.preview_bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, BORDER_RADIUS, BORDER_RADIUS)

        # --- Шаг 5: Отрисовка миниатюры ---
        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(self.secondary_text_color)
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "No Preview")

        # --- Шаг 6: Отрисовка Названия кластера ---
        if is_selected:
            text_color = option.palette.color(QPalette.ColorRole.HighlightedText)
        elif is_hovered:
            text_color = QColor(0, 0, 0) # Черный при наведении
        else:
            text_color = option.palette.color(QPalette.ColorRole.WindowText)

        painter.setPen(text_color)
        font = painter.font()
        font.setBold(False)
        font.setPointSize(FONT_SIZE_TITLE)
        painter.setFont(font)
        
        name_rect = QRect(
            bg_rect.x() + ITEM_PADDING, 
            preview_bg_rect.bottom() + 4, 
            PREVIEW_SIZE, 
            TEXT_NAME_HEIGHT
        )
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, cluster_name)

        # --- Шаг 7: Отрисовка Счетчика фото ---
        font.setBold(False)
        font.setPointSize(FONT_SIZE_COUNT)
        painter.setFont(font)
        
        if not is_selected and not is_hovered:
            painter.setPen(self.secondary_text_color)
        else:
            painter.setPen(text_color)
            
        count_rect = QRect(name_rect.bottomLeft(), QSize(PREVIEW_SIZE, TEXT_COUNT_HEIGHT))
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignCenter, f"Фото: {count}")

        # --- Шаг 8: Рамка Drag&Drop ---
        widget = option.widget
        if hasattr(widget, 'drop_target_item') and isinstance(widget, QListWidget):
            current_item = widget.item(index.row())
            if widget.drop_target_item == current_item:
                pen = painter.pen()
                pen.setColor(self.drop_border_color)
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(DROP_BORDER_WIDTH)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(bg_rect.adjusted(2, 2, -2, -2), 5, 5)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        width = PREVIEW_SIZE + (ITEM_PADDING * 2)
        height = PREVIEW_SIZE + (ITEM_PADDING * 2) + TEXT_NAME_HEIGHT + TEXT_COUNT_HEIGHT
        return QSize(width, height)


class ImageItemDelegate(QStyledItemDelegate):
    """
    Делегат для отрисовки карточки изображения в галерее.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        preview_styles = theme_api.get_parsed_style("delegate_preview_background")
        self.preview_bg_color = QColor(preview_styles.get("color", "#e8e8e8"))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Шаг 1: Отрисовка скругленного фона (Texture Buffer Method) ---
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_hovered = option.state & QStyle.StateFlag.State_MouseOver

        if is_selected or is_hovered:
            buffer = QPixmap(option.rect.size())
            buffer.fill(Qt.transparent)
            
            p = QPainter(buffer)
            temp_option = QStyleOptionViewItem(option)
            temp_option.rect = QRect(0, 0, option.rect.width(), option.rect.height())
            temp_option.text = "" 
            temp_option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
            
            option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, temp_option, p, option.widget)
            p.end()
            
            painter.setBrush(QBrush(buffer))
            painter.setBrushOrigin(option.rect.topLeft())
            painter.setPen(Qt.NoPen)
            
            draw_rect = QRectF(option.rect).adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(draw_rect, BORDER_RADIUS_BG, BORDER_RADIUS_BG)

        # --- Шаг 2: Данные ---
        pixmap: QPixmap = index.data(Qt.ItemDataRole.DecorationRole)
        bg_rect = option.rect

        # --- Шаг 3: Серая подложка ---
        preview_bg_rect = bg_rect.adjusted(ITEM_PADDING, ITEM_PADDING, -ITEM_PADDING, -ITEM_PADDING)
        
        painter.setBrush(self.preview_bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, BORDER_RADIUS, BORDER_RADIUS)

        # --- Шаг 4: Изображение ---
        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "Image\nNot Found")

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        size = THUMBNAIL_SIZE + (ITEM_PADDING * 2)
        return QSize(size, size)