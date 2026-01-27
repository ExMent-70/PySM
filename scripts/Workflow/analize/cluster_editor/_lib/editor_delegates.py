# analize/cluster_editor/_lib/editor_delegates.py

# --- ИСПРАВЛЕНИЕ: Добавляем QListWidget в импорт ---
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QStyleOptionViewItem, QListWidget
from PySide6.QtGui import QPixmap, QPainter, QColor, QPalette
from PySide6.QtCore import Qt, QRect, QSize

# --- КОНСТАНТЫ РАЗМЕРОВ ---
THUMBNAIL_SIZE = 180
PREVIEW_SIZE = 180
FACE_SIZE = 130
FACE_MIN = 100
FACE_MAX = 400


# --- Импорт API тем ---
try:
    from pysm_lib import theme_api
except ImportError:
    class ThemeAPIMock:
        def get_parsed_style(self, *args, **kwargs):
            return {}
    theme_api = ThemeAPIMock()


class ClusterItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки кластера."""

    def __init__(self, parent=None):
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
        painter.save()

        # 1. Рисуем фон через QSS
        background_option = QStyleOptionViewItem(option)
        if background_option.features & QStyleOptionViewItem.ViewItemFeature.HasDecoration:
            background_option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        background_option.text = ""
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, background_option, painter, option.widget)

        # 2. Получаем данные
        item_data = index.data(Qt.ItemDataRole.UserRole)
        is_selected = option.state & QStyle.StateFlag.State_Selected
        cluster_name = item_data.get("name", "N/A")
        count = item_data.get("count", 0)
        pixmap: QPixmap = item_data.get("pixmap")
        is_changed = item_data.get("is_changed", False)
        bg_rect = option.rect

        # 3. Рисуем индикатор изменений
        if is_changed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.changed_indicator_color)
            painter.drawRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), 3)

        # 4. Рисуем превью
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

        # 5. Рисуем текст
        text_color = option.palette.color(
            QPalette.ColorRole.HighlightedText if is_selected else QPalette.ColorRole.WindowText
        )
        painter.setPen(text_color)
        font = painter.font()
        font.setBold(False); font.setPointSize(11)
        painter.setFont(font)
        name_rect = QRect(bg_rect.x() + 6, preview_bg_rect.bottom() + 4, PREVIEW_SIZE, 40)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, cluster_name)
        font.setBold(False); font.setPointSize(9)
        painter.setFont(font)
        if not is_selected:
            painter.setPen(self.secondary_text_color)
        count_rect = QRect(name_rect.bottomLeft(), QSize(PREVIEW_SIZE, 20))
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignCenter, f"Фото: {count}")

        # 6. Рисуем рамку-индикатор
        widget = option.widget
        if hasattr(widget, 'drop_target_item') and isinstance(widget, QListWidget):
            current_item = widget.item(index.row())
            if widget.drop_target_item == current_item:
                pen = painter.pen()
                pen.setColor(self.drop_border_color)
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(bg_rect.adjusted(2, 2, -2, -2), 5, 5)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(PREVIEW_SIZE + 12, PREVIEW_SIZE + 12 + 40 + 20)


class ImageItemDelegate(QStyledItemDelegate):
    """Делегат для отрисовки карточки изображения."""

    def __init__(self, parent=None):
        super().__init__(parent)
        preview_styles = theme_api.get_parsed_style("delegate_preview_background")
        self.preview_bg_color = QColor(preview_styles.get("color", "#e8e8e8"))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        painter.save()

        # 1. Рисуем фон через QSS
        background_option = QStyleOptionViewItem(option)
        if background_option.features & QStyleOptionViewItem.ViewItemFeature.HasDecoration:
            background_option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        background_option.text = ""
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, background_option, painter, option.widget)

        # 2. Рисуем содержимое
        pixmap: QPixmap = index.data(Qt.ItemDataRole.DecorationRole)
        bg_rect = option.rect

        # Рисуем серую подложку
        preview_bg_rect = bg_rect.adjusted(6, 6, -6, -6)
        painter.setBrush(self.preview_bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(preview_bg_rect, 3, 3)

        if pixmap and not pixmap.isNull():
            target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
            target_rect.moveCenter(preview_bg_rect.center())
            painter.drawPixmap(target_rect, pixmap)
        else:
            painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
            painter.drawText(preview_bg_rect, Qt.AlignmentFlag.AlignCenter, "Image\nNot Found")

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(THUMBNAIL_SIZE + 12, THUMBNAIL_SIZE + 12)