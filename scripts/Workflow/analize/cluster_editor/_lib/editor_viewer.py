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

logger = logging.getLogger(__name__)


class ImageViewer(QDialog):
    """Модальное диалоговое окно для просмотра изображений с зумом и навигацией."""

    def __init__(self, image_paths: List[Path], filenames: List[str],
                 current_index: int, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.filenames = filenames
        self.current_index = current_index
        self.is_fitted_in_view = False

        self.setWindowTitle("Просмотр изображений")
        self.setMinimumSize(800, 600)

        # --- UI ---
        main_layout = QVBoxLayout(self)

        # 1. Элементы для отображения изображения (без изменений)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.installEventFilter(self)

        # 2. Создаем все виджеты для нижней панели
        self.nav_label = QLabel()  # "Фото X из Y"
        self.filename_label = QLabel()
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.prev_button = QPushButton("<< Предыдущее")
        self.next_button = QPushButton("Следующее >>")

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        # 3. --- ИЗМЕНЕНИЕ: Создаем единую нижнюю панель ---
        #    В один QHBoxLayout помещаем и информацию, и кнопки
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.nav_label)        # Инфо слева
        bottom_layout.addStretch()                       # Растяжитель, толкает кнопки к центру
        bottom_layout.addWidget(self.prev_button)
        bottom_layout.addWidget(self.next_button)
        bottom_layout.addStretch()                       # Растяжитель, толкает имя файла вправо
        bottom_layout.addWidget(self.filename_label)   # Имя файла справа

        # 4. --- ИЗМЕНЕНИЕ: Собираем основной layout ---
        #    Сначала виджет просмотра, потом нижняя панель. Верхняя панель удалена.
        main_layout.addWidget(self.view, 1)  # Добавляем 1, чтобы виджет занимал все доступное место
        main_layout.addLayout(bottom_layout)

        # 5. Шорткаты (без изменений)
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