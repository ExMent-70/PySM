# analize/cluster_editor/editor_viewer.py
"""
Модуль, содержащий виджет для просмотра изображений (ImageViewer).
"""
import logging
from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton
)
from PySide6.QtGui import (
    QPixmap, QWheelEvent, QAction, QKeySequence, QPainter, QTransform
)
from PySide6.QtCore import Qt, QTimer, QEvent

from .editor_styles import SCROLLBAR_STYLE

logger = logging.getLogger(__name__)


class ImageViewer(QDialog):
    """Модальное диалоговое окно для просмотра изображений с зумом и навигацией."""

    def __init__(self, image_paths: List[Path], filenames: List[str],
                 current_index: int, scrollbar_style: str, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.filenames = filenames
        self.current_index = current_index
        self.is_fitted_in_view = False

        self.setWindowTitle("Просмотр изображений")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QDialog { background-color: #252525; color: #eee; }")

        # --- UI ---
        main_layout = QVBoxLayout(self)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setStyleSheet("QGraphicsView { border: none; }")
        self.view.verticalScrollBar().setStyleSheet(scrollbar_style)
        self.view.horizontalScrollBar().setStyleSheet(scrollbar_style)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.installEventFilter(self)

        info_layout = QHBoxLayout()
        self.nav_label = QLabel()
        self.filename_label = QLabel()
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(self.nav_label)
        info_layout.addWidget(self.filename_label)

        self.prev_button = QPushButton("<< Предыдущее")
        self.next_button = QPushButton("Следующее >>")

        button_style = """
            QPushButton { 
                background-color: #007bff; color: white; font-weight: bold; 
                border: none; padding: 10px; border-radius: 5px; min-width: 120px;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #555; color: #999; }
        """
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addStretch(1)

        main_layout.addLayout(info_layout)
        main_layout.addWidget(self.view)
        main_layout.addLayout(button_layout)

        # Создаем QAction для шорткатов
        prev_action = QAction(self)
        prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        prev_action.triggered.connect(self.show_previous_image)

        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        next_action.triggered.connect(self.show_next_image)

        close_action = QAction(self)
        close_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        close_action.triggered.connect(self.reject)

        self.addActions([prev_action, next_action, close_action])

        self._load_image()
        QTimer.singleShot(0, self.fit_in_view)

    def eventFilter(self, source, event):
        if source is self.view and event.type() == QEvent.Type.MouseButtonDblClick:
            if self.is_fitted_in_view:
                self._zoom_to_100_percent()
            else:
                self.fit_in_view()
            return True  # Событие обработано
        return super().eventFilter(source, event)

    def _load_image(self):
        """Загружает и отображает текущее изображение."""
        path = self.image_paths[self.current_index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            logger.warning(f"Не удалось загрузить изображение: {path}")
            self.pixmap_item.setPixmap(QPixmap())  # Устанавливаем пустой pixmap
        else:
            self.pixmap_item.setPixmap(pixmap)

        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.nav_label.setText(
            f"Фото {self.current_index + 1} из {len(self.image_paths)}")
        self.filename_label.setText(self.filenames[self.current_index])

        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)

    def fit_in_view(self):
        """Вписывает изображение в размер окна просмотра."""
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_fitted_in_view = True

    def _zoom_to_100_percent(self):
        """Сбрасывает трансформацию для показа изображения 1 к 1."""
        self.view.setTransform(QTransform())
        self.is_fitted_in_view = False

    def wheelEvent(self, event: QWheelEvent):
        """Обрабатывает колесо мыши для масштабирования."""
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