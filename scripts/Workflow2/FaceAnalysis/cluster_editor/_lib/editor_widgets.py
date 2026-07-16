"""
Модуль, содержащий кастомные подклассы стандартных виджетов Qt,
такие как списки с поддержкой Drag & Drop.
"""
import logging
from PySide6.QtWidgets import QListWidget, QAbstractItemView
from PySide6.QtGui import QDropEvent, QDrag, QPainter, QColor, QPixmap
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint, QSize

from .editor_delegates import (
    FACE_SIZE,
    FACE_SIZE_PORTRAIT,
    THUMBNAIL_SIZE,
    FaceItemDelegate,
)

logger = logging.getLogger(__name__)


class ClusterDropListWidget(QListWidget):
    itemsDropped = Signal(str, str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.main_window = parent
        self.drop_target_item = None

    def dragEnterEvent(self, event: QDropEvent):
        # Просто сообщаем, что виджет в принципе может принимать данные этого типа.
        # Конкретное решение будет приниматься в dragMoveEvent.
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDropEvent):
        """
        Отслеживает элемент под курсором, обновляет подсветку и явно разрешает/запрещает сброс.
        """
        current_item = self.itemAt(event.position().toPoint())

        # Логика ручной подсветки (без изменений)
        if self.drop_target_item != current_item:
            old_target = self.drop_target_item
            self.drop_target_item = current_item
            if old_target:
                self.viewport().update(self.visualItemRect(old_target))
            if self.drop_target_item:
                self.viewport().update(self.visualItemRect(self.drop_target_item))
        
        # Явно сообщаем Qt, можно ли выполнить сброс в *данной конкретной точке*.
        # Это изменит курсор и позволит сработать dropEvent.
        if current_item:
            event.acceptProposedAction() # Разрешаем сброс, курсор изменится на "можно"
        else:
            event.ignore() # Запрещаем сброс, курсор изменится на "нельзя"
        
        # Вызов super() убран, так как он мешал и переопределял наше решение.

    def dropEvent(self, event: QDropEvent):
        """Обрабатывает сброс данных и сбрасывает подсветку."""
        target_item = self.itemAt(event.position().toPoint())
        
        # Сначала сбрасываем подсветку
        if self.drop_target_item:
            old_target = self.drop_target_item
            self.drop_target_item = None
            self.viewport().update(self.visualItemRect(old_target))
        
        if not target_item:
            event.ignore()
            return

        target_id = target_item.data(Qt.ItemDataRole.UserRole)["id"]
        mime_data = event.mimeData().text()

        try:
            source_id, filenames_str = mime_data.split("::", 1)
            if source_id != target_id:
                filenames = filenames_str.split('|')
                self.itemsDropped.emit(source_id, target_id, filenames)
                event.acceptProposedAction()
            else:
                event.ignore()
        except ValueError as e:
            logger.error(f"Error parsing MIME data: {e}")
            event.ignore()

    def dragLeaveEvent(self, event):
        """Срабатывает, когда курсор покидает виджет, сбрасывая подсветку."""
        if self.drop_target_item:
            old_target = self.drop_target_item
            self.drop_target_item = None
            self.viewport().update(self.visualItemRect(old_target))
        super().dragLeaveEvent(event)

class ImageDragListWidget(QListWidget):
    """Список, который правильно инициирует перетаскивание своих элементов."""
    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return

        main_window = self.window()
        if not hasattr(main_window, 'active_cluster_id'):
            logger.error("Главное окно или active_cluster_id не найдены")
            return

        mime_data = QMimeData()
        
        # Cleaning передаёт индекс лица вместе с именем файла.
        filenames =[]
        for item in items:
            user_data = item.data(Qt.ItemDataRole.UserRole)
            fname = user_data["filename"]
            # Для режима cleaning у нас сохранен face_index
            f_idx = user_data.get("face_index")
            if f_idx is not None:
                filenames.append(f"{fname}::{f_idx}")
            else:
                filenames.append(fname)
                
        mime_text = f"{main_window.active_cluster_id}::{'|'.join(filenames)}"
        mime_data.setText(mime_text)

        drag = QDrag(self)
        drag.setMimeData(mime_data)

        base_pixmap: QPixmap = items[0].data(Qt.ItemDataRole.DecorationRole)
        if base_pixmap.isNull():
            base_pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            base_pixmap.fill(Qt.GlobalColor.darkGray)

        drag_pixmap = QPixmap(base_pixmap.size())
        drag_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(drag_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)

        if len(items) > 1:
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRect(drag_pixmap.rect())
            font = painter.font()
            font.setPointSize(24); font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("orange"))
            painter.drawText(drag_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"+{len(items)}")

        pen = painter.pen()
        pen.setColor(QColor("orange")); pen.setWidth(5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(drag_pixmap.rect())
        painter.end()

        drag.setPixmap(drag_pixmap)
        drag.setHotSpot(QPoint(drag_pixmap.width() // 2, drag_pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)
       
class FaceDetailsWidget(QListWidget):
    """
    Виджет для отображения крупных планов лиц.
    Адаптирует размер в зависимости от режима.
    """
    def __init__(self, parent=None, mode: str = "face"): # <--- Добавили аргумент mode
        super().__init__(parent)
        self.mode = mode
        
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setItemDelegate(FaceItemDelegate(self))
        self.setUniformItemSizes(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust) 
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(10)
        # QAbstractScrollArea receives pointer events through its viewport.
        # Explicit tracking keeps ``QListWidget::item:hover`` current while the
        # cursor moves between faces without a pressed mouse button.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        
        # Настройка размеров в зависимости от режима
        if self.mode == 'face':
            base_size = FACE_SIZE_PORTRAIT
        else:
            base_size = FACE_SIZE
            
        self.setIconSize(QSize(base_size, base_size))
        # Высота сетки = иконка + место под текст
        self.setGridSize(QSize(base_size + 20, base_size + 60))
        
        self.setWordWrap(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
