# -*- coding: utf-8 -*-

"""
Модуль делегатов (Delegates) для кастомной отрисовки элементов списков.
Выделения рисуются делегатами без дополнительных буферов QPixmap.
"""

from typing import Dict, Optional

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
    QBrush
)
from PySide6.QtCore import Qt, QRect, QSize, QRectF

# ==============================================================================
# VISUAL CONFIG (НАСТРОЙКИ ВНЕШНЕГО ВИДА)
# ==============================================================================

# --- Размеры элементов (в пикселях) ---
THUMBNAIL_SIZE = 180  # Размер квадратной миниатюры в галерее
PREVIEW_SIZE = 180    # Размер миниатюры в списке кластеров
FACE_SIZE = 130       # Базовый размер лица (используется в других модулях)
FACE_SIZE_PORTRAIT = 290 # Базовый размер лица
FACE_MIN = 100        # Мин. размер слайдера лиц
FACE_MAX = 400        # Макс. размер слайдера лиц

# Custom roles used by FaceItemDelegate. Face thumbnails are deliberately not
# stored as QIcon: QIcon changes the native item size when async data arrives.
FACE_PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 1
FACE_STATUS_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 2

# --- Отступы и рамки ---
ITEM_PADDING = 6           # Внутренний отступ от края ячейки до контента
BORDER_RADIUS = 3          # Радиус скругления серой подложки
BORDER_RADIUS_BG = 8       # Радиус скругления фона выделения
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
    from pysm_lib.pysm_icons import icons as pysm_icons
except ImportError:
    class ThemeAPIMock:
        def get_parsed_style(self, key: str, *args, **kwargs) -> Dict[str, str]:
            defaults = {
                "delegate_changed_indicator": {"color": "#f0ad4e"},
                "delegate_face_hover": {"color": "#f0ad4e"},
                "delegate_preview_background": {"color": "#e8e8e8"},
                "delegate_secondary_text": {"color": "#555555"},
                "delegate_hover_border": {"color": "#0078d7"}
            }
            return defaults.get(key, dict())
    theme_api = ThemeAPIMock()
    pysm_icons = None


class ClusterItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки кластера в левой панели."""

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        item_data = index.data(Qt.ItemDataRole.UserRole)
        if item_data is None:
            item_data = dict()
            
        cluster_name = item_data.get("name", "N/A")
        count = item_data.get("count", 0)
        pixmap: Optional[QPixmap] = item_data.get("pixmap")
        is_changed = item_data.get("is_changed", False)
        
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_hovered = option.state & QStyle.StateFlag.State_MouseOver
        
        bg_rect = option.rect

        # Скругленный фон состояния карточки.
        if is_selected or is_hovered:
            if is_selected:
                bg_color = option.palette.color(QPalette.ColorRole.Highlight)
            else:
                base_color = option.palette.color(QPalette.ColorRole.WindowText)
                bg_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 15)
                
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.PenStyle.NoPen)
            draw_rect = QRectF(option.rect).adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(draw_rect, BORDER_RADIUS_BG, BORDER_RADIUS_BG)

        # --- Шаг 2: Индикатор изменений ---
        if is_changed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.changed_indicator_color)
            painter.drawRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), CHANGED_INDICATOR_H)

        # --- Шаг 3: Подложка под миниатюру ---
        preview_bg_rect = QRect(
            bg_rect.x() + ITEM_PADDING, 
            bg_rect.y() + ITEM_PADDING, 
            PREVIEW_SIZE, 
            PREVIEW_SIZE
        )
        painter.setBrush(self.preview_bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, BORDER_RADIUS, BORDER_RADIUS)

        # --- Шаг 4: Отрисовка миниатюры ---
        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(self.secondary_text_color)
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "No Preview")

        # --- Шаг 5: Отрисовка Названия кластера ---
        if is_selected:
            text_color = option.palette.color(QPalette.ColorRole.HighlightedText)
        elif is_hovered:
            text_color = option.palette.color(QPalette.ColorRole.WindowText)
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

        # --- Шаг 6: Отрисовка Счетчика фото ---
        font.setBold(False)
        font.setPointSize(FONT_SIZE_COUNT)
        painter.setFont(font)
        
        if not is_selected and not is_hovered:
            painter.setPen(self.secondary_text_color)
        else:
            painter.setPen(text_color)
            
        count_rect = QRect(name_rect.bottomLeft(), QSize(PREVIEW_SIZE, TEXT_COUNT_HEIGHT))
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignCenter, f"Фото: {count}")

        # --- Шаг 7: Рамка Drag&Drop ---
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


class FaceItemDelegate(QStyledItemDelegate):
    """Draw a face cell with geometry independent from async image delivery."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        hover_style = theme_api.get_parsed_style("delegate_face_hover")
        preview_style = theme_api.get_parsed_style("delegate_preview_background")
        self.hover_color = QColor(hover_style.get("color", "#f0ad4e"))
        self.preview_color = QColor(preview_style.get("color", "#e8e8e8"))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        cell_rect = QRectF(option.rect).adjusted(4, 4, -4, -4)

        if hovered and not selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.hover_color)
            painter.drawRoundedRect(cell_rect, 5, 5)
        elif selected:
            pen = painter.pen()
            pen.setColor(option.palette.color(QPalette.ColorRole.Highlight))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(cell_rect, 5, 5)

        widget = option.widget
        icon_size = widget.iconSize() if isinstance(widget, QListWidget) else QSize(FACE_SIZE, FACE_SIZE)
        image_rect = QRect(
            option.rect.x() + (option.rect.width() - icon_size.width()) // 2,
            option.rect.y() + ITEM_PADDING,
            icon_size.width(),
            icon_size.height(),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.preview_color)
        painter.drawRoundedRect(image_rect, BORDER_RADIUS, BORDER_RADIUS)

        pixmap = index.data(FACE_PIXMAP_ROLE)
        drawn_rect = QRect()
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            drawn_size = pixmap.size().scaled(
                image_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            drawn_rect = QRect(0, 0, drawn_size.width(), drawn_size.height())
            drawn_rect.moveCenter(image_rect.center())
            painter.drawPixmap(drawn_rect, pixmap, pixmap.rect())

            status_color = str(index.data(FACE_STATUS_COLOR_ROLE) or "")
            if status_color:
                pen_width = max(
                    3,
                    int(min(drawn_rect.width(), drawn_rect.height()) * 0.04),
                )
                status_pen = painter.pen()
                status_pen.setColor(QColor(status_color))
                status_pen.setWidth(pen_width)
                status_pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(status_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                inset = max(1, pen_width // 2)
                painter.drawRect(drawn_rect.adjusted(inset, inset, -inset, -inset))

        text_rect = QRect(
            option.rect.x() + ITEM_PADDING,
            image_rect.bottom() + 4,
            option.rect.width() - 2 * ITEM_PADDING,
            max(1, option.rect.bottom() - image_rect.bottom() - 8),
        )
        painter.setPen(option.palette.color(QPalette.ColorRole.WindowText))
        lines = str(index.data(Qt.ItemDataRole.DisplayRole) or "").splitlines()
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        y = text_rect.top()
        for line in lines[:2]:
            line_rect = QRect(text_rect.x(), y, text_rect.width(), line_height)
            painter.drawText(
                line_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(line, Qt.TextElideMode.ElideRight, line_rect.width()),
            )
            y += line_height

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        widget = option.widget
        icon_size = widget.iconSize() if isinstance(widget, QListWidget) else QSize(FACE_SIZE, FACE_SIZE)
        return QSize(icon_size.width() + 20, icon_size.height() + 60)


class ImageItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки изображения в галерее."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        preview_styles = theme_api.get_parsed_style("delegate_preview_background")
        self.preview_bg_color = QColor(preview_styles.get("color", "#e8e8e8"))
        self._icon_cache = dict() # Кэш иконок для максимальной производительности

    def _get_cached_icon(self, icon_name: str, size: int):
        if not pysm_icons: return None
        cache_key = f"{icon_name}_{size}"
        if cache_key not in self._icon_cache:
            self._icon_cache[cache_key] = pysm_icons.get_qicon(icon_name, size=size)
        return self._icon_cache[cache_key]

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_hovered = option.state & QStyle.StateFlag.State_MouseOver

        bg_rect = option.rect
        # Создаем квадратную область специально для картинки (верхняя часть)
        img_size = bg_rect.width()
        img_rect = QRect(bg_rect.x(), bg_rect.y(), img_size, img_size)

        # --- Шаг 1: Отрисовка скругленного фона (на всю высоту, включая текст) ---
        if is_selected or is_hovered:
            if is_selected:
                bg_color = option.palette.color(QPalette.ColorRole.Highlight)
            else:
                base_color = option.palette.color(QPalette.ColorRole.WindowText)
                bg_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 15)
                
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.PenStyle.NoPen)
            draw_rect = QRectF(option.rect).adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(draw_rect, BORDER_RADIUS_BG, BORDER_RADIUS_BG)

        # --- Шаг 2: Данные ---
        pixmap: QPixmap = index.data(Qt.ItemDataRole.DecorationRole)

        # --- Шаг 3: Серая подложка (только для квадратной области картинки) ---
        preview_bg_rect = img_rect.adjusted(ITEM_PADDING, ITEM_PADDING, -ITEM_PADDING, -ITEM_PADDING)
        
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

        # --- Шаг 5: Отрисовка информационных значков (overlays) ---
        user_data = index.data(Qt.ItemDataRole.UserRole)
        if user_data and "overlays" in user_data and pysm_icons:
            overlays = user_data["overlays"]
            icon_size = 20
            padding = 6
            
            current_x = preview_bg_rect.right() - icon_size - padding
            current_y = preview_bg_rect.top() + padding
            
            for icon_name in overlays:
                qicon = self._get_cached_icon(icon_name, icon_size)
                if qicon and not qicon.isNull():
                    bg_badge = QRectF(current_x - 3, current_y - 3, icon_size + 6, icon_size + 6)
                    painter.setBrush(QColor(0, 0, 0, 140))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(bg_badge, (icon_size+6)/2, (icon_size+6)/2)
                    
                    icon_rect = QRect(int(current_x), int(current_y), icon_size, icon_size)
                    qicon.paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter)
                    
                    current_y += (icon_size + padding + 6)

        # --- Шаг 6: Отрисовка Beauty Score ---
        if user_data and "beauty_score" in user_data:
            b_score = user_data["beauty_score"]
            badge_size = 22
            padding = 6
            bx = preview_bg_rect.left() + padding
            by = preview_bg_rect.top() + padding

            bg_badge = QRectF(bx, by, badge_size, badge_size)
            painter.setBrush(QColor(230, 80, 150, 210))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_badge, badge_size/2, badge_size/2)

            painter.setPen(QColor("white"))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(bg_badge, Qt.AlignmentFlag.AlignCenter, str(b_score))

        # --- Шаг 7: Отрисовка имени файла ---
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text:
            # Место под текст — нижняя часть после img_rect
            text_rect = QRect(bg_rect.x() + ITEM_PADDING, img_rect.bottom(), bg_rect.width() - (ITEM_PADDING * 2), bg_rect.height() - img_size)
            
            if is_selected:
                painter.setPen(option.palette.color(QPalette.ColorRole.HighlightedText))
            else:
                painter.setPen(option.palette.color(QPalette.ColorRole.WindowText))
                
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            
            # Экранируем слишком длинные имена файлов точками (...)
            metrics = painter.fontMetrics()
            elided_text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, elided_text)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        size = THUMBNAIL_SIZE + (ITEM_PADDING * 2)
        text_height = 24 # Выделяем 24 пикселя по высоте для имени файла
        return QSize(size, size + text_height)
