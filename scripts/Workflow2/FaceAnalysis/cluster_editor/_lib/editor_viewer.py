# analize/cluster_editor/_lib/editor_viewer.py
"""
Модуль, содержащий виджет для просмотра изображений (ImageViewer).
Реализует навигацию по контексту локации и умную подсветку лиц.
"""
import logging
import re
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, 
    QGraphicsRectItem, QGraphicsTextItem
)
from PySide6.QtGui import (
    QPixmap, QWheelEvent, QAction, QKeySequence, QPainter, QTransform,
    QPen, QColor, QBrush, QFont
)
from PySide6.QtCore import Qt, QTimer, QEvent

logger = logging.getLogger(__name__)

def natural_keys(text):
    """Сортировка строк с числами в человеческом порядке."""
    return[int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

class ImageViewer(QDialog):
    """
    Модальное окно для просмотра изображений.
    Умеет отрисовывать цветные рамки лиц и осуществляет навигацию по локациям.
    """

    def __init__(self, data_manager, start_filename: str, parent=None, target_face_index: Optional[int] = None, draw_boxes: bool = True):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        
        self.data_manager = data_manager
        self.target_face_index = target_face_index
        self.draw_boxes = draw_boxes # Флаг отрисовки рамок
        
        self.is_fitted_in_view = False
        self.filenames = list()
        self.current_index = 0

        self.setWindowTitle("Просмотр изображений (Контекст локации)")
        self.setMinimumSize(900, 700)

        self._build_navigation_list(start_filename)
        self._init_ui()
        
        self._load_image()
        QTimer.singleShot(0, self.fit_in_view)

    def _build_navigation_list(self, start_filename: str):
        """Формирует список файлов для навигации Вперед/Назад на основе текущей локации."""
        record = self.data_manager.records.get(start_filename)
        location_cluster = record.location_cluster if record else None

        files_in_context = list()
        if location_cluster is not None:
            # Собираем все фото из этой же локации
            for f_name, r in self.data_manager.records.items():
                if r.location_cluster == location_cluster:
                    files_in_context.append(f_name)
        else:
            # Если локации нет, листаем по всем фото
            files_in_context = list(self.data_manager.records.keys())

        # Сортируем естественно (как в проводнике)
        self.filenames = sorted(files_in_context, key=natural_keys)

        try:
            self.current_index = self.filenames.index(start_filename)
        except ValueError:
            self.filenames.insert(0, start_filename)
            self.current_index = 0

    def _init_ui(self):
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

    def eventFilter(self, source, event):
        if source is self.view and event.type() == QEvent.Type.MouseButtonDblClick:
            if self.is_fitted_in_view:
                self._zoom_to_100_percent()
            else:
                self.fit_in_view()
            return True 
        return super().eventFilter(source, event)

    def _draw_bounding_box(self, bbox: List[float], is_recognized: bool, text: Optional[str] = None, tooltip: str = ""):
        """Отрисовывает рамку на сцене поверх изображения с поддержкой хинта (tooltip)."""
        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = map(int, bbox)
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        w = x2 - x1
        h = y2 - y1

        color = QColor(0, 255, 0) if is_recognized else QColor(255, 0, 0)
        
        rect_item = QGraphicsRectItem(x1, y1, w, h)
        pen = QPen(color)
        pen.setWidth(4)
        rect_item.setPen(pen)
        
        # Добавляем всплывающий хинт к рамке
        if tooltip:
            rect_item.setToolTip(tooltip)
            
        self.scene.addItem(rect_item)

        if text:
            text_item = QGraphicsTextItem(text)
            text_item.setDefaultTextColor(Qt.GlobalColor.white)
            
            font = text_item.font()
            font.setBold(True)
            font.setPointSize(14)
            text_item.setFont(font)
            
            text_bg = QGraphicsRectItem(text_item.boundingRect())
            text_bg.setBrush(QBrush(QColor(0, 0, 0, 180))) 
            text_bg.setPen(Qt.PenStyle.NoPen)
            
            pos_y = y1 - text_item.boundingRect().height() if y1 > 30 else y1
            text_item.setPos(x1, pos_y)
            text_bg.setPos(x1, pos_y)
            
            # Добавляем хинт к тексту и фону текста, чтобы он работал по всей площади
            if tooltip:
                text_item.setToolTip(tooltip)
                text_bg.setToolTip(tooltip)
            
            self.scene.addItem(text_bg)
            self.scene.addItem(text_item)

    def _load_image(self):
        """Загружает текущее изображение и рисует рамки (если включено)."""
        current_fname = self.filenames[self.current_index]
        
        if hasattr(self.parent(), '_get_image_path'):
            path = self.parent()._get_image_path(current_fname)
        else:
            path = self.data_manager.working_dir / "JPG" / current_fname
        
        pixmap = QPixmap(str(path))
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        if pixmap.isNull():
            logger.warning(f"Не удалось загрузить изображение: {path}")
            self.pixmap_item.setPixmap(QPixmap())
        else:
            self.pixmap_item.setPixmap(pixmap)
            
            # Отрисовываем рамки ТОЛЬКО если флаг draw_boxes = True
            if self.draw_boxes:
                record = self.data_manager.records.get(current_fname)
                if record:
                    for i, face in enumerate(record.faces):
                        # Собираем имя для тултипа
                        tooltip_text = face.extra_data.get('matched_child_name') or face.child_name or face.temp_child_name or ""
                        
                        # СЦЕНАРИЙ 1: Клик по ОПОЗНАННОМУ лицу
                        if self.target_face_index is not None:
                            if i == self.target_face_index:
                                self._draw_bounding_box(face.bbox, is_recognized=True, text=None, tooltip=tooltip_text)
                            continue 
                        
                        # СЦЕНАРИЙ 2: Листание или клик по НЕОПОЗНАННОМУ
                        rec_id = face.extra_data.get('matched_portrait_cluster_label')
                        if rec_id is None:
                            rec_id = face.cluster_label
                        
                        is_rec = (rec_id is not None and str(rec_id) not in ("-1", "trash", "None"))
                        label_text = f"ID: {rec_id}" if is_rec else None
                        
                        self._draw_bounding_box(face.bbox, is_recognized=is_rec, text=label_text, tooltip=tooltip_text)

        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.nav_label.setText(f"Фото {self.current_index + 1} из {len(self.filenames)}")
        self.filename_label.setText(current_fname)

        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.filenames) - 1)

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
            self.target_face_index = None 
            self._load_image()
            self.fit_in_view()

    def show_next_image(self):
        if self.current_index < len(self.filenames) - 1:
            self.current_index += 1
            self.target_face_index = None 
            self._load_image()
            self.fit_in_view()