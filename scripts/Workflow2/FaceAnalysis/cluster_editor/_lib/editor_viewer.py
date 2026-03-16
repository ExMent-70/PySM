# analize/cluster_editor/_lib/editor_viewer.py
"""
Модуль, содержащий виджет для просмотра изображений (ImageViewer).
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, 
    QGraphicsRectItem
)
from PySide6.QtGui import (
    QPixmap, QWheelEvent, QAction, QKeySequence, QPainter, QTransform,
    QPen, QColor
)
from PySide6.QtCore import Qt, QTimer, QEvent

logger = logging.getLogger(__name__)


class ImageViewer(QDialog):
    """
    Модальное диалоговое окно для просмотра изображений с зумом и навигацией.
    Поддерживает выделение лиц (красная рамка).
    """

    def __init__(self, image_paths: List[Path], filenames: List[str],
                 current_index: int, parent=None, 
                 highlights_map: Optional[Dict[str, List[float]]] = None,
                 highlight_bbox: Optional[List[float]] = None): # <--- Вернули аргумент для совместимости
        """
        :param highlights_map: Словарь { "filename.jpg": [x1, y1, x2, y2] } для сквозной подсветки.
        :param highlight_bbox: [x1, y1, x2, y2] для подсветки только на стартовом фото.
        """
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.image_paths = image_paths
        self.filenames = filenames
        self.current_index = current_index
        
        # 1. Инициализируем карту подсветок
        self.highlights_map = highlights_map or {}
        
        # 2. Если передан одиночный bbox (режим Cleaning), добавляем его в карту
        if highlight_bbox and 0 <= current_index < len(filenames):
            current_fname = filenames[current_index]
            self.highlights_map[current_fname] = highlight_bbox
        
        self.is_fitted_in_view = False

        self.setWindowTitle("Просмотр изображений")
        self.setMinimumSize(800, 600)

        # --- UI ---
        main_layout = QVBoxLayout(self)

        # Сцена и View
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.installEventFilter(self)

        # Нижняя панель
        self.nav_label = QLabel()
        self.filename_label = QLabel()
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.prev_button = QPushButton("<< Предыдущее")
        self.next_button = QPushButton("Следующее >>")

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.nav_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.prev_button)
        bottom_layout.addWidget(self.next_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.filename_label)

        main_layout.addWidget(self.view, 1)
        main_layout.addLayout(bottom_layout)

        # Шорткаты
        prev_action = QAction(self); prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left)); prev_action.triggered.connect(self.show_previous_image)
        next_action = QAction(self); next_action.setShortcut(QKeySequence(Qt.Key.Key_Right)); next_action.triggered.connect(self.show_next_image)
        close_action = QAction(self); close_action.setShortcut(QKeySequence(Qt.Key.Key_Escape)); close_action.triggered.connect(self.reject)
        self.addActions([prev_action, next_action, close_action])

        self._load_image()
        QTimer.singleShot(0, self.fit_in_view)

    def eventFilter(self, source, event):
        if source is self.view and event.type() == QEvent.Type.MouseButtonDblClick:
            if self.is_fitted_in_view:
                self._zoom_to_100_percent()
            else:
                self.fit_in_view()
            return True 
        return super().eventFilter(source, event)

    def _load_image(self):
        """Загружает и отображает текущее изображение."""
        current_fname = self.filenames[self.current_index]
        path = self.image_paths[self.current_index]
        
        pixmap = QPixmap(str(path))
        
        # Очищаем сцену
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        if pixmap.isNull():
            logger.warning(f"Не удалось загрузить изображение: {path}")
            self.pixmap_item.setPixmap(QPixmap())
        else:
            self.pixmap_item.setPixmap(pixmap)
            
            # --- ПРОВЕРКА КАРТЫ ПОДСВЕТОК ---
            if current_fname in self.highlights_map:
                bbox = self.highlights_map[current_fname]
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    
                    rect_item = QGraphicsRectItem(x1, y1, w, h)
                    pen = QPen(QColor(255, 0, 0)) # Красная рамка
                    pen.setWidth(5) # Толстая линия
                    rect_item.setPen(pen)
                    self.scene.addItem(rect_item)

        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.nav_label.setText(
            f"Фото {self.current_index + 1} из {len(self.image_paths)}")
        self.filename_label.setText(current_fname)

        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)

    def fit_in_view(self):
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_fitted_in_view = True

    def _zoom_to_100_percent(self):
        self.view.setTransform(QTransform())
        self.is_fitted_in_view = False

    def wheelEvent(self, event: QWheelEvent):
        self.is_fitted_in_view = False
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(factor, factor)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_image()
            self.fit_in_view()

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_image()
            self.fit_in_view()