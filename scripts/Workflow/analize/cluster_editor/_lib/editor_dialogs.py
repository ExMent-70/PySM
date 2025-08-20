# analize/cluster_editor/_lib/editor_dialogs.py
"""
Модуль, содержащий кастомные диалоговые окна для редактора кластеров.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, QSlider, QFrame
)
from PySide6.QtGui import QPixmap, QPainter, QTransform, QWheelEvent
from PySide6.QtCore import Qt, Slot, QEvent

from . import editor_styles as styles

try:
    from pysm_lib import pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    IS_MANAGED_RUN = False

# --- ИЗМЕНЕНИЕ: Исправляем импорт Pillow ---
try:
    from PIL import Image, ImageEnhance
    # Импортируем МОДУЛЬ ImageQt
    from PIL import ImageQt
    IS_PILLOW_AVAILABLE = True
except ImportError:
    IS_PILLOW_AVAILABLE = False
    Image = None
    ImageEnhance = None
    ImageQt = None
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

logger = logging.getLogger(__name__)


class EnhanceSettingsDialog(QDialog):
    """
    Диалоговое окно для интерактивной настройки параметров улучшения изображений.
    """
    RECOMMENDED_DEFAULTS = {
        "brightness": 1.0, "contrast": 1.1, "color": 1.1, "sharpness": 1.2
    }

    def __init__(self, preview_image_path: Path, parent=None):
        super().__init__(parent)
        if not IS_PILLOW_AVAILABLE:
            raise ImportError("Для работы этого диалога необходима библиотека Pillow.")

        self.preview_image_path = preview_image_path
        self.original_pil_image: Optional[Image.Image] = None
        self.original_qt_pixmap: Optional[QPixmap] = None
        self.enhancement_factors: Dict[str, float] = {}
        self.is_fitted_in_view = False

        self.setWindowTitle("Настройка улучшения изображений")
        self.setMinimumSize(1000, 700)
        #self.setStyleSheet(styles.MAIN_WINDOW_STYLE)

        self._load_original_image()
        self._init_ui()
        self._load_settings()
        self._update_preview()

    def _load_original_image(self):
        try:
            self.original_pil_image = Image.open(self.preview_image_path).convert("RGB")
            # --- ИЗМЕНЕНИЕ: Используем правильный вызов Модуль.Класс() ---
            if ImageQt:
                qimage = ImageQt.ImageQt(self.original_pil_image)
                self.original_qt_pixmap = QPixmap.fromImage(qimage)
        except Exception as e:
            logger.error(f"Не удалось загрузить изображение для предпросмотра: {e}")
            self.original_pil_image = Image.new("RGB", (500, 500), "black") if Image else None
            self.original_qt_pixmap = QPixmap(500, 500)
            if self.original_qt_pixmap: self.original_qt_pixmap.fill(Qt.GlobalColor.black)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        preview_container = QFrame()
        preview_layout = QVBoxLayout(preview_container)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        #self.view.setStyleSheet("QGraphicsView { border: 1px solid #444; }")
        #self.view.verticalScrollBar().setStyleSheet(styles.SCROLLBAR_STYLE)
        #self.view.horizontalScrollBar().setStyleSheet(styles.SCROLLBAR_STYLE)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.installEventFilter(self)
        #preview_title = QLabel("<b>Предпросмотр (Колесо - зум, Двойной клик - вписать/100%)</b>")
        #preview_title.setStyleSheet(styles.TITLE_LABEL_STYLE)
        #preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.view)
        settings_container = QFrame()
        settings_container.setFixedWidth(280)
        settings_layout = QVBoxLayout(settings_container)
        toggle_preview_button = QPushButton("До/После")
        #toggle_preview_button.setStyleSheet(styles.BUTTON_STYLE)
        toggle_preview_button.pressed.connect(self._show_original_preview)
        toggle_preview_button.released.connect(self._update_preview)
        settings_layout.addWidget(toggle_preview_button)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        #line.setStyleSheet("QFrame { border: 1px solid #444; }")
        settings_layout.addWidget(line)
        self.brightness_slider = self._create_slider("Яркость", "brightness")
        self.contrast_slider = self._create_slider("Контраст", "contrast")
        self.color_slider = self._create_slider("Насыщенность", "color")
        self.sharpness_slider = self._create_slider("Резкость", "sharpness")
        settings_layout.addWidget(self.brightness_slider["group"])
        settings_layout.addWidget(self.contrast_slider["group"])
        settings_layout.addWidget(self.color_slider["group"])
        settings_layout.addWidget(self.sharpness_slider["group"])

        apply_button = QPushButton("Применить и экспортировать")
        #apply_button.setStyleSheet(styles.BUTTON_STYLE + "background-color: #28a745;")
        apply_button.clicked.connect(self.accept)
        settings_layout.addWidget(apply_button)

        settings_layout.addStretch()


        reset_button = QPushButton("Сбросить")
        cancel_button = QPushButton("Отмена")
        #reset_button.setStyleSheet(styles.BUTTON_STYLE_ORANGE_COMPACT)
        #cancel_button.setStyleSheet(styles.BUTTON_STYLE_ORANGE_COMPACT)
        reset_button.clicked.connect(self._reset_sliders)
        cancel_button.clicked.connect(self.reject)
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.addWidget(reset_button)
        bottom_buttons_layout.addWidget(cancel_button)
        settings_layout.addLayout(bottom_buttons_layout)


        main_layout.addWidget(preview_container, 1)
        main_layout.addWidget(settings_container)

    def eventFilter(self, source, event: QEvent) -> bool:
        if source is self.view and event.type() == QEvent.Type.MouseButtonDblClick:
            if self.is_fitted_in_view:
                self._zoom_to_100_percent()
            else:
                self.fit_in_view()
            return True
        return super().eventFilter(source, event)

    def wheelEvent(self, event: QWheelEvent):
        if self.view.underMouse():
            self.is_fitted_in_view = False
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.view.scale(factor, factor)

    def _update_preview(self):
        if not self.original_pil_image or not ImageQt:
            return

        fit_in_view_on_first_load = not self.pixmap_item.pixmap()
        enhanced_image = self.original_pil_image
        if self.enhancement_factors.get("brightness", 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(self.enhancement_factors["brightness"])
        if self.enhancement_factors.get("contrast", 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(self.enhancement_factors["contrast"])
        if self.enhancement_factors.get("color", 1.0) != 1.0:
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(self.enhancement_factors["color"])
        if self.enhancement_factors.get("sharpness", 1.0) != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(self.enhancement_factors["sharpness"])
        
        # --- ИЗМЕНЕНИЕ: Используем правильный вызов Модуль.Класс() ---
        qimage = ImageQt.ImageQt(enhanced_image)
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)
        
        if fit_in_view_on_first_load:
            self.fit_in_view()
        elif self.is_fitted_in_view:
            self.fit_in_view()
            
    def fit_in_view(self):
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_fitted_in_view = True

    def _zoom_to_100_percent(self):
        self.view.setTransform(QTransform())
        self.is_fitted_in_view = False

    def _show_original_preview(self):
        if self.original_qt_pixmap:
            self.pixmap_item.setPixmap(self.original_qt_pixmap)
            
    def _load_settings(self):
        settings = self.RECOMMENDED_DEFAULTS
        if IS_MANAGED_RUN and pysm_context:
            settings = pysm_context.get("enhancer_settings", self.RECOMMENDED_DEFAULTS)
        else:
            logger.warning("pysm_context не доступен, используются рекомендуемые настройки.")
        for key, widget_dict in self._get_all_sliders().items():
            value = float(settings.get(key, self.RECOMMENDED_DEFAULTS.get(key, 1.0)))
            widget_dict["slider"].setValue(int(value * 100))

    def accept(self):
        if IS_MANAGED_RUN and pysm_context:
            pysm_context.set("enhancer_settings", self.enhancement_factors)
        super().accept()

    def _create_slider(self, name: str, key: str) -> Dict:
        group = QFrame(); layout = QVBoxLayout(group); layout.setSpacing(5)
        label_layout = QHBoxLayout(); label_name = QLabel(name); label_value = QLabel("1.00")
        label_layout.addWidget(label_name); label_layout.addStretch(); label_layout.addWidget(label_value)
        slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, 200); slider.setValue(100)
        slider.setTickInterval(10); slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        #slider.setStyleSheet(styles.SLIDER_STYLE)
        slider.valueChanged.connect(lambda value, k=key, lbl=label_value: self._on_slider_change(k, value, lbl))
        layout.addLayout(label_layout); layout.addWidget(slider)
        return {"group": group, "slider": slider, "label": label_value}

    @Slot(str, int, QLabel)
    def _on_slider_change(self, key: str, value: int, label: QLabel):
        factor = value / 100.0
        self.enhancement_factors[key] = factor
        label.setText(f"{factor:.2f}")
        self._update_preview()

    def _reset_sliders(self):
        for key, widget_dict in self._get_all_sliders().items():
            default_value = self.RECOMMENDED_DEFAULTS.get(key, 1.0)
            widget_dict["slider"].setValue(int(default_value * 100))

    def get_enhancement_factors(self) -> Dict[str, float]:
        return self.enhancement_factors

    def _get_all_sliders(self) -> Dict[str, Dict]:
        return {"brightness": self.brightness_slider, "contrast": self.contrast_slider,
                "color": self.color_slider, "sharpness": self.sharpness_slider}