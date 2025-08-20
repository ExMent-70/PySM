# analize/cluster_editor/_lib/editor_widgets.py
"""
Модуль, содержащий кастомные подклассы стандартных виджетов Qt,
такие как списки с поддержкой Drag & Drop.
"""
import logging
from PySide6.QtWidgets import QListWidget
from PySide6.QtGui import QDropEvent, QDrag, QPainter, QColor, QPixmap
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint



# Импортируем стили для получения размеров
from . import editor_styles as styles

logger = logging.getLogger(__name__)


class ClusterDropListWidget(QListWidget):
    itemsDropped = Signal(str, str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        # --- НОВЫЙ АТРИБУТ: ссылка на главное окно для доступа к его методам ---
        self.main_window = parent
        logger.debug(f"ClusterDropListWidget initialized: acceptDrops={self.acceptDrops()}, dragDropMode={self.dragDropMode()}, geometry={self.geometry()}")
        logger.debug(f"Viewport acceptDrops={self.viewport().acceptDrops()}")

    def dragEnterEvent(self, event: QDropEvent):
        logger.debug(f"dragEnterEvent: hasText={event.mimeData().hasText()}, pos={event.position().toPoint()}")
        if event.mimeData().hasText():
            event.acceptProposedAction()
            logger.debug("Drag enter event accepted")
        else:
            event.ignore()
            logger.debug("Drag enter event ignored: no text data")

    def dragMoveEvent(self, event: QDropEvent):
        logging.debug(f"dragMoveEvent: pos={event.position().toPoint()}")
        if event.mimeData().hasText():
            event.acceptProposedAction()
            logging.debug("Drag move event accepted")
        else:
            event.ignore()
            logger.debug("Drag move event ignored")

# --- ИЗМЕНЕННЫЙ МЕТОД: dropEvent ---
    def dropEvent(self, event: QDropEvent):
        logger.debug(f"Drop event received at position: {event.position().toPoint()}, widget geometry={self.geometry()}")
        
        if not event.mimeData().hasText():
            logger.warning("No text data in MIME")
            event.ignore()
            return
            
        target_item = self.itemAt(event.position().toPoint())
        if not target_item:
            logger.warning(f"No target item found at drop position: {event.position().toPoint()}, drop ignored.")
            event.ignore()
            return

        target_id = target_item.data(Qt.ItemDataRole.UserRole)["id"]
        logger.debug(f"Target item found: id={target_id}")
        
        mime_data = event.mimeData().text()
        logger.debug(f"MIME data: {mime_data}")

        try:
            source_id, filenames_str = mime_data.split("::", 1)
            if source_id != target_id:
                filenames = filenames_str.split('|')
                logger.debug(f"Dropping {len(filenames)} files from cluster {source_id} to {target_id}")

                # 1. Сначала обновляем модель данных через сигнал
                self.itemsDropped.emit(source_id, target_id, filenames)
                
                # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Предотвращаем "прыжок" выбора ---
                # 2. Находим QListWidgetItem, который был активен до drop
                # Мы просим главное окно найти его, так как только оно знает о всех виджетах
                if self.main_window and hasattr(self.main_window, '_get_list_item_by_id'):
                    # Используем active_cluster_id из главного окна, так как это источник правды
                    active_id_before_drop = self.main_window.active_cluster_id
                    source_item = self.main_window._get_list_item_by_id(active_id_before_drop)
                    if source_item:
                        # 3. Принудительно устанавливаем его как текущий.
                        # Это "перебивает" стандартное поведение Qt, которое пытается выбрать target_item.
                        logger.debug(f"Forcing current item back to source: {active_id_before_drop}")
                        self.setCurrentItem(source_item)
                
                event.acceptProposedAction()
            else:
                logger.debug("Source and target clusters are the same, ignoring drop")
                event.ignore()

        except ValueError as e:
            logger.error(f"Error parsing MIME data: {e}", file=sys.stderr)
            event.ignore()
         
           
class ImageDragListWidget(QListWidget):
    """Список, который правильно инициирует перетаскивание своих элементов."""

# --- ИЗМЕНЯЕМЫЙ БЛОК: метод startDrag в классе ImageDragListWidget ---
    def startDrag(self, supportedActions):
        """Переопределенный метод для начала перетаскивания."""
        items = self.selectedItems()
        if not items:
            return
        
        main_window = self.window() # Более надежный способ получить главное окно
        if not hasattr(main_window, 'active_cluster_id'):
            logger.error("Главное окно или active_cluster_id не найдены", file=sys.stderr)
            return

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # 1. Формируем MIME-данные, как и раньше
        mime_data = QMimeData()
        filenames = [item.data(Qt.ItemDataRole.UserRole)["filename"] for item in items]
        mime_text = f"{main_window.active_cluster_id}::{'|'.join(filenames)}"
        mime_data.setText(mime_text)
        
        # 2. Создаем "призрачное" изображение для перетаскивания
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # Получаем исходный pixmap
        base_pixmap: QPixmap = items[0].data(Qt.ItemDataRole.DecorationRole)
        if base_pixmap.isNull(): # Fallback
            base_pixmap = QPixmap(styles.THUMBNAIL_SIZE, styles.THUMBNAIL_SIZE)
            base_pixmap.fill(Qt.GlobalColor.darkGray)

        # Создаем новый холст для "призрака"
        drag_pixmap = QPixmap(base_pixmap.size())
        drag_pixmap.fill(Qt.GlobalColor.transparent) # Прозрачный фон
        
        painter = QPainter(drag_pixmap)
        
        # Рисуем исходное изображение
        painter.drawPixmap(0, 0, base_pixmap)
        
        # Если перетаскивается несколько изображений, добавляем счетчик
        if len(items) > 1:
            painter.setBrush(QColor(0, 0, 0, 150)) # Полупрозрачный оверлей
            painter.drawRect(drag_pixmap.rect())
            font = painter.font()
            font.setPointSize(24); font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("orange"))
            painter.drawText(drag_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"+{len(items)}")

        # Рисуем яркую рамку поверх всего
        pen = painter.pen()
        pen.setColor(QColor("orange"))
        pen.setWidth(15) # Ширина рамки
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 150)) # Полупрозрачный оверлей
        #painter.setBrush(Qt.BrushStyle.NoBrush) # Не заливаем
        painter.drawRect(drag_pixmap.rect())

        painter.end() # Завершаем рисование
        
        drag.setPixmap(drag_pixmap)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
        drag.setHotSpot(QPoint(drag_pixmap.width() // 2, drag_pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)