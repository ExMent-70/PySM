# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, QSlider, QFrame, QComboBox, QDialogButtonBox,
    QListWidget, QListWidgetItem, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QToolBox, QLineEdit, QWidget, QScrollArea
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QTransform, QWheelEvent, QMouseEvent
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    Qt,
    Signal,
    Slot,
    QEvent,
    QSize,
    QTimer,
)

from pysm_lib.pysm_image_cache import (
    AsyncImageLoader,
    AsyncImageResult,
    ImageRequest,
    QtImageCache,
)

from .image_requests import face_thumbnail_request
from .editor_delegates import FACE_PIXMAP_ROLE, FaceItemDelegate


try:
    from pysm_lib import pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    IS_MANAGED_RUN = False

try:
    from PIL import Image, ImageQt
    from .image_processing import apply_color_corrections, create_watermark_layer
    IS_PILLOW_AVAILABLE = True
except ImportError:
    IS_PILLOW_AVAILABLE = False
    Image = None
    ImageQt = None
    apply_color_corrections, create_watermark_layer = None, None

logger = logging.getLogger(__name__)


# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ---

class DoubleClickSlider(QSlider):
    """Слайдер, который сбрасывается в дефолтное значение при двойном клике."""
    def __init__(self, orientation=Qt.Orientation.Horizontal, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self.default_value = default_value

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_value)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class _PreviewRenderSignals(QObject):
    finished = Signal(int, object, str)


class _PreviewRenderTask(QRunnable):
    """Render one export-preview generation outside the GUI thread."""

    def __init__(
        self,
        generation: int,
        source_image,
        factors: Dict[str, Any],
        bboxes: List[List[float]],
        show_watermark: bool,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.source_image = source_image
        self.factors = factors
        self.bboxes = bboxes
        self.show_watermark = show_watermark
        self.signals = _PreviewRenderSignals()

    @Slot()
    def run(self) -> None:
        try:
            image = apply_color_corrections(self.source_image, self.factors)
            if self.show_watermark:
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                layer = create_watermark_layer(
                    image.size,
                    self.bboxes,
                    self.factors,
                    "Имя Фамилия",
                )
                if layer is not None:
                    image = Image.alpha_composite(image, layer)
            qimage = QImage(ImageQt.ImageQt(image)).copy()
            self.signals.finished.emit(self.generation, qimage, "")
        except Exception as exc:
            self.signals.finished.emit(self.generation, QImage(), str(exc))


class EnhanceSettingsDialog(QDialog):
    RECOMMENDED_DEFAULTS = {
        # Base
        "brightness": 1.0, "contrast": 1.1, "color": 1.1, "sharpness": 1.2,
        # Advanced
        "temperature": 0.0, "tint": 0.0,
        "black_point": 0, "white_point": 0,
        # Watermark Settings
        "wm_stripe_alpha": 45, "wm_mask_fill": 10,
        "wm_pad_w": 0.1, "wm_pad_h": 0.2,
        "wm_text": "ВЫБОР ФОТОГРАФИИ", "wm_text_alpha": 150
    }

    def __init__(
        self,
        preview_image_path: Path,
        faces_bboxes: List[List[float]],
        parent=None,
        *,
        image_cache: QtImageCache,
        image_loader: AsyncImageLoader,
    ):
        super().__init__(parent)
        if not IS_PILLOW_AVAILABLE:
            raise ImportError("Pillow required.")

        self.preview_image_path = preview_image_path
        self.faces_bboxes = faces_bboxes 
        self.image_cache = image_cache
        self.image_loader = image_loader
        self._image_channel = ("enhance-preview", id(self))
        
        self.original_pil_image: Optional[Image.Image] = None
        self.original_qt_pixmap: Optional[QPixmap] = None
        
        self.enhancement_factors: Dict[str, Any] = self.RECOMMENDED_DEFAULTS.copy()
        self.slider_controls: Dict[str, Dict] = {} 
        
        self.original_size = self.image_cache.source_size(self.preview_image_path)
        if self.original_size[0] <= 0 or self.original_size[1] <= 0:
            self.original_size = (500, 500)
        self.original_dpi = 300
        self.is_fitted_in_view = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(80)
        self.preview_timer.timeout.connect(self._update_preview)
        self._preview_generation = 0
        self._preview_pool = QThreadPool(self)
        self._preview_pool.setMaxThreadCount(1)

        self.setWindowTitle("Настройка экспорта")
        self.setMinimumSize(1200, 850)

        self._init_ui()
        self._load_settings()
        self.image_loader.imageReady.connect(self._on_original_image_ready)
        self._load_original_image()

    def _load_original_image(self):
        request = ImageRequest(
            self.preview_image_path,
            (1600, 1200),
            mode="fit",
            variant="cluster_editor.enhance_preview.v2",
        )
        self.image_loader.request(
            request,
            channel=self._image_channel,
            persist=True,
            disk_format="PNG",
        )

    @Slot(object)
    def _on_original_image_ready(self, result: AsyncImageResult) -> None:
        if result.channel != self._image_channel:
            return
        if result.image.isNull():
            logger.error(f"Preview load error: {result.error}")
            self.original_pil_image = Image.new("RGB", (500, 500), "black")
            self.original_qt_pixmap = QPixmap(500, 500)
            self.original_qt_pixmap.fill(Qt.GlobalColor.black)
        else:
            self.original_qt_pixmap = QPixmap.fromImage(result.image)
            self.original_pil_image = ImageQt.fromqimage(result.image).convert("RGB")
            if result.image.dotsPerMeterX() > 0:
                self.original_dpi = round(result.image.dotsPerMeterX() * 0.0254)
        self.is_fitted_in_view = False
        self._update_preview()

    def showEvent(self, event):
        """
        Вызывается при отображении окна.
        Используем таймер, чтобы выполнить fit_in_view после того,
        как менеджер компоновки (layout) завершит расчет размеров.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_in_view)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # 1. Left: Preview
        preview_container = QFrame()
        preview_layout = QVBoxLayout(preview_container)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.installEventFilter(self)
        preview_layout.addWidget(self.view)

        # 2. Right: Toolbox
        settings_container = QFrame()
        settings_container.setFixedWidth(400) 
        settings_layout = QVBoxLayout(settings_container)
        
        self.toolbox = QToolBox()
        
        # --- TAB 1: Коррекция ---
        enhancement_page = QWidget()
        enhancement_layout = QVBoxLayout(enhancement_page)
        enhancement_layout.setSpacing(10)
        
        # Базовые
        gb_basic = QGroupBox("Базовая коррекция")
        gb_basic_layout = QVBoxLayout(gb_basic)
        self.slider_controls["brightness"] = self._create_slider("Яркость", "brightness", 0, 200, 100, scale=100.0)
        self.slider_controls["contrast"] = self._create_slider("Контраст", "contrast", 0, 200, 100, scale=100.0)
        self.slider_controls["color"] = self._create_slider("Насыщенность", "color", 0, 200, 100, scale=100.0)
        self.slider_controls["sharpness"] = self._create_slider("Резкость", "sharpness", 0, 300, 100, scale=100.0)
        
        gb_basic_layout.addWidget(self.slider_controls["brightness"]["group"])
        gb_basic_layout.addWidget(self.slider_controls["contrast"]["group"])
        gb_basic_layout.addWidget(self.slider_controls["color"]["group"])
        gb_basic_layout.addWidget(self.slider_controls["sharpness"]["group"])
        enhancement_layout.addWidget(gb_basic)
        
        # HDR
        gb_adv = QGroupBox("Цветовой баланс и уровни")
        gb_adv_layout = QVBoxLayout(gb_adv)
        
        self.slider_controls["temperature"] = self._create_slider("Температура (Cold <-> Warm)", "temperature", -100, 100, 0, scale=1.0)
        self.slider_controls["tint"] = self._create_slider("Оттенок (Green <-> Magenta)", "tint", -50, 50, 0, scale=1.0)
        self.slider_controls["black_point"] = self._create_slider("Точка черного (Blacks)", "black_point", -100, 100, 0, scale=1.0)
        self.slider_controls["white_point"] = self._create_slider("Точка белого (Whites)", "white_point", -100, 100, 0, scale=1.0)
        
        gb_adv_layout.addWidget(self.slider_controls["temperature"]["group"])
        gb_adv_layout.addWidget(self.slider_controls["tint"]["group"])
        gb_adv_layout.addWidget(self.slider_controls["black_point"]["group"])
        gb_adv_layout.addWidget(self.slider_controls["white_point"]["group"])
        enhancement_layout.addWidget(gb_adv)
        
        enhancement_layout.addStretch()
        self.toolbox.addItem(enhancement_page, "Коррекция изображения")

        # --- TAB 2: Экспорт ---
        export_page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        export_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        page_layout = QVBoxLayout(export_page)
        page_layout.addWidget(scroll)

        # Группа 1: Формат
        gb_fmt = QGroupBox("Формат и Размер")
        form_fmt = QFormLayout(gb_fmt)
        
        self.quality_spin = QSpinBox(); self.quality_spin.setRange(1, 100); self.quality_spin.setSuffix("%")
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(72, 1200); self.dpi_spin.setSuffix(" dpi")
        self.width_spin = QSpinBox(); self.width_spin.setRange(100, 20000); self.width_spin.setSuffix(" px")
        self.height_spin = QSpinBox(); self.height_spin.setRange(100, 20000); self.height_spin.setSuffix(" px")
        
        self.ratio_check = QCheckBox("Сохранять пропорции")
        
        form_fmt.addRow("Качество:", self.quality_spin)
        form_fmt.addRow("DPI:", self.dpi_spin)
        form_fmt.addRow("Ширина:", self.width_spin)
        form_fmt.addRow("Высота:", self.height_spin)
        # Добавляем виджет напрямую для выравнивания по левому краю
        form_fmt.addRow(self.ratio_check) 
        
        export_layout.addWidget(gb_fmt)

        # Группа 2: Водяные знаки
        gb_wm = QGroupBox("Водяные знаки (Watermarks)")
        wm_main_layout = QVBoxLayout(gb_wm)
        wm_main_layout.setSpacing(10)
        
       
        # Настройки водяных знаков (всегда видны, чтобы можно было настраивать)
        # Слайдеры
        self.slider_controls["wm_stripe_alpha"] = self._create_slider("Прозрачность полос (0-255)", "wm_stripe_alpha", 0, 255, 45, scale=1.0)
        wm_main_layout.addWidget(self.slider_controls["wm_stripe_alpha"]["group"])
        
        self.slider_controls["wm_mask_fill"] = self._create_slider("Заливка маски (0-255)", "wm_mask_fill", 0, 255, 10, scale=1.0)
        wm_main_layout.addWidget(self.slider_controls["wm_mask_fill"]["group"])
        
        # Текст
        wm_text_layout = QHBoxLayout()
        wm_text_layout.addWidget(QLabel("Текст:"))
        self.wm_text = QLineEdit()
        self.wm_text.textChanged.connect(lambda v: self._update_factor("wm_text", v))
        # Обновляем превью при завершении ввода текста
        self.wm_text.editingFinished.connect(self._update_preview)
        wm_text_layout.addWidget(self.wm_text)
        wm_main_layout.addLayout(wm_text_layout)
        
        self.slider_controls["wm_text_alpha"] = self._create_slider("Прозрачность текста (0-255)", "wm_text_alpha", 0, 255, 150, scale=1.0)
        wm_main_layout.addWidget(self.slider_controls["wm_text_alpha"]["group"])

        # Отступы
        pad_layout = QFormLayout()
        self.wm_pad_w = QDoubleSpinBox(); self.wm_pad_w.setRange(0.0, 2.0); self.wm_pad_w.setSingleStep(0.05)
        self.wm_pad_w.valueChanged.connect(lambda v: self._update_factor("wm_pad_w", v, update_preview=True))
        
        self.wm_pad_h = QDoubleSpinBox(); self.wm_pad_h.setRange(0.0, 2.0); self.wm_pad_h.setSingleStep(0.05)
        self.wm_pad_h.valueChanged.connect(lambda v: self._update_factor("wm_pad_h", v, update_preview=True))
        
        pad_layout.addRow("Отступ маски W:", self.wm_pad_w)
        pad_layout.addRow("Отступ маски H:", self.wm_pad_h)
        wm_main_layout.addLayout(pad_layout)

        # Заголовок с чекбоксом (для экспорта) и кнопкой превью (для глаз)
        wm_header_layout = QHBoxLayout()
        
        # Чекбокс для экспорта
        self.wm_export_check = QCheckBox("Включить при экспорте")
        self.wm_export_check.setToolTip("Если отмечено, водяные знаки будут наложены на сохраненные файлы.")
        
        # Кнопка для превью
        self.btn_preview_wm = QPushButton("Показать")
        self.btn_preview_wm.setCheckable(True)
        self.btn_preview_wm.setChecked(False) # По умолчанию выкл
        self.btn_preview_wm.setFixedWidth(100)
        self.btn_preview_wm.toggled.connect(self._update_preview)

        wm_header_layout.addWidget(self.wm_export_check)
        wm_header_layout.addStretch()
        wm_header_layout.addWidget(self.btn_preview_wm)
        wm_main_layout.addLayout(wm_header_layout)

        
        export_layout.addWidget(gb_wm)
        export_layout.addStretch()
        
        self.toolbox.addItem(export_page, "Параметры экспорта")
        settings_layout.addWidget(self.toolbox)

        # Buttons
        btn_layout = QHBoxLayout()
        toggle_btn = QPushButton("До/После")
        toggle_btn.pressed.connect(self._show_original_preview)
        toggle_btn.released.connect(self._update_preview)
        
        reset_btn = QPushButton("Сбросить")
        reset_btn.clicked.connect(self._reset_sliders)
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(toggle_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Применить и экспортировать")
        apply_btn.setProperty("class", "primary")
        apply_btn.clicked.connect(self.accept)

        settings_layout.addLayout(btn_layout)
        settings_layout.addWidget(apply_btn)

        main_layout.addWidget(preview_container, 1)
        main_layout.addWidget(settings_container)

        self.aspect_ratio = self.original_size[0] / self.original_size[1] if self.original_size[1] > 0 else 1.0
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)

    def _create_slider(self, name: str, key: str, min_val: int, max_val: int, default: int, scale: float = 1.0) -> Dict:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 2, 0, 5)
        layout.setSpacing(2)
        
        lbl_layout = QHBoxLayout()
        lbl_name = QLabel(name)
        lbl_val = QLabel(str(default / scale))
        lbl_val.setStyleSheet("font-weight: bold; color: #0078d7;")
        
        lbl_layout.addWidget(lbl_name)
        lbl_layout.addStretch()
        lbl_layout.addWidget(lbl_val)
        
        slider = DoubleClickSlider(Qt.Orientation.Horizontal, default_value=default)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        
        slider.valueChanged.connect(
            lambda v, k=key, l=lbl_val, s=scale: self._on_slider_change(k, v, l, s)
        )
        
        layout.addLayout(lbl_layout)
        layout.addWidget(slider)
        return {"group": group, "slider": slider, "label": lbl_val, "scale": scale, "default": default}

    def _on_slider_change(self, key: str, value: int, label: QLabel, scale: float):
        real_val = value / scale
        if scale == 1.0: 
            label.setText(f"{int(real_val)}")
        else:
            label.setText(f"{real_val:.2f}")
            
        self.enhancement_factors[key] = real_val
        self.preview_timer.start()

    def _update_factor(self, key: str, value: Any, update_preview: bool = False):
        self.enhancement_factors[key] = value
        # Если включен режим превью WM, то обновляем картинку
        if update_preview and self.btn_preview_wm.isChecked():
            self.preview_timer.start()

    def _load_settings(self):
        settings = self.RECOMMENDED_DEFAULTS.copy()
        if IS_MANAGED_RUN and pysm_context:
            saved = pysm_context.get("enhancer_settings", {})
            if isinstance(saved, dict):
                settings.update(saved)
        
        for key, ctrl in self.slider_controls.items():
            val = settings.get(key, ctrl["default"] / ctrl["scale"])
            ctrl["slider"].setValue(int(val * ctrl["scale"]))

        self.width_spin.setValue(self.original_size[0])
        self.height_spin.setValue(self.original_size[1])
        
        self.dpi_spin.setValue(int(settings.get("export_dpi", self.original_dpi)))
        self.quality_spin.setValue(int(settings.get("export_quality", 95)))
        self.wm_export_check.setChecked(bool(settings.get("export_watermarks", False)))
        self.ratio_check.setChecked(bool(settings.get("export_keep_ratio", True)))
        
        self.wm_pad_w.setValue(float(settings.get("wm_pad_w", 0.1)))
        self.wm_pad_h.setValue(float(settings.get("wm_pad_h", 0.2)))
        self.wm_text.setText(str(settings.get("wm_text", "ВЫБОР ФОТОГРАФИИ")))
        
        self.enhancement_factors.update(settings)

    def _reset_sliders(self):
        for key, ctrl in self.slider_controls.items():
            ctrl["slider"].setValue(ctrl["default"])
        
        self.width_spin.setValue(self.original_size[0])
        self.height_spin.setValue(self.original_size[1])
        self.dpi_spin.setValue(self.original_dpi)
        self.quality_spin.setValue(95)
        self.wm_export_check.setChecked(False)
        self.btn_preview_wm.setChecked(False)
        self.ratio_check.setChecked(True)
        
        d = self.RECOMMENDED_DEFAULTS
        self.wm_pad_w.setValue(d["wm_pad_w"])
        self.wm_pad_h.setValue(d["wm_pad_h"])
        self.wm_text.setText(d["wm_text"])
        
        self._update_preview()

    def _update_preview(self):
        if not self.original_pil_image:
            return

        self._preview_generation += 1
        generation = self._preview_generation
        self._preview_pool.clear()
        source = self.original_pil_image.copy()
        task = _PreviewRenderTask(
            generation,
            source,
            self.enhancement_factors.copy(),
            self._preview_bboxes(source.size),
            self.btn_preview_wm.isChecked(),
        )
        task.signals.finished.connect(self._on_preview_rendered)
        self._preview_pool.start(task)

    @Slot(int, object, str)
    def _on_preview_rendered(
        self,
        generation: int,
        image: QImage,
        error: str,
    ) -> None:
        if generation != self._preview_generation:
            return
        if image.isNull():
            logger.error(f"Preview render error: {error}")
            return
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        if not self.is_fitted_in_view:
            self.fit_in_view()

    def _preview_bboxes(self, preview_size: tuple[int, int]) -> List[List[float]]:
        source_width, source_height = self.original_size
        if source_width <= 0 or source_height <= 0:
            return []
        scale_x = preview_size[0] / source_width
        scale_y = preview_size[1] / source_height
        return [
            [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
            for bbox in self.faces_bboxes
            if len(bbox) == 4
        ]

    def fit_in_view(self):
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_fitted_in_view = True

    def _zoom_to_100_percent(self):
        self.view.setTransform(QTransform())
        self.is_fitted_in_view = False

    def _show_original_preview(self):
        if self.original_qt_pixmap:
            self.pixmap_item.setPixmap(self.original_qt_pixmap)

    def _on_width_changed(self, new_width):
        if self.ratio_check.isChecked():
            self.height_spin.blockSignals(True)
            self.height_spin.setValue(int(new_width / self.aspect_ratio))
            self.height_spin.blockSignals(False)

    def _on_height_changed(self, new_height):
        if self.ratio_check.isChecked():
            self.width_spin.blockSignals(True)
            self.width_spin.setValue(int(new_height * self.aspect_ratio))
            self.width_spin.blockSignals(False)

    def eventFilter(self, source, event: QEvent) -> bool:
        if source is self.view and event.type() == QEvent.Type.MouseButtonDblClick:
            if self.is_fitted_in_view: self._zoom_to_100_percent()
            else: self.fit_in_view()
            return True
        return super().eventFilter(source, event)

    def wheelEvent(self, event: QWheelEvent):
        if self.view.underMouse():
            self.is_fitted_in_view = False
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.view.scale(factor, factor)

    def accept(self):
        if IS_MANAGED_RUN and pysm_context:
            settings = self.enhancement_factors.copy()
            settings.update({
                "export_dpi": self.dpi_spin.value(),
                "export_quality": self.quality_spin.value(),
                "export_watermarks": self.wm_export_check.isChecked(),
                "export_keep_ratio": self.ratio_check.isChecked(),
                "wm_pad_w": self.wm_pad_w.value(),
                "wm_pad_h": self.wm_pad_h.value(),
                "wm_text": self.wm_text.text()
            })
            pysm_context.set("enhancer_settings", settings)
            self.enhancement_factors.update(settings)
        super().accept()

    def done(self, result: int) -> None:
        self._preview_generation += 1
        self._preview_pool.clear()
        self.image_loader.cancel(self._image_channel)
        try:
            self.image_loader.imageReady.disconnect(self._on_original_image_ready)
        except (RuntimeError, TypeError):
            pass
        super().done(result)

    def get_export_settings(self) -> Dict[str, Any]:
        non_slider_wm = {
            "wm_pad_w": self.wm_pad_w.value(),
            "wm_pad_h": self.wm_pad_h.value(),
            "wm_text": self.wm_text.text()
        }
        factors = self.enhancement_factors.copy()
        factors.update(non_slider_wm)

        return {
            "factors": factors, 
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "dpi": self.dpi_spin.value(),
            "quality": self.quality_spin.value(),
            "watermarks": self.wm_export_check.isChecked()
        }

# Остальные классы без изменений
class StudentSelectionDialog(QDialog):
    """Поиск и выбор записи ученика без свободного ввода личности."""

    def __init__(self, students, current_student_id: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор ученика")
        self.resize(520, 480)
        self._students = list(students)
        self._selected_student_id: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Поиск по фамилии, имени или student_id:"))
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Начните вводить фамилию, имя или ID")
        layout.addWidget(self.search)

        self.student_list = QListWidget(self)
        layout.addWidget(self.student_list, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.search.textChanged.connect(self._populate)
        self.student_list.itemDoubleClicked.connect(lambda _item: self._accept_selection())
        self._populate("")

        if current_student_id:
            for row in range(self.student_list.count()):
                item = self.student_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == current_student_id:
                    self.student_list.setCurrentItem(item)
                    break

    def _populate(self, query: str) -> None:
        query = query.strip().casefold()
        self.student_list.clear()
        for student in self._students:
            searchable = f"{student.display_name} {student.student_id}".casefold()
            if query and query not in searchable:
                continue
            item = QListWidgetItem(student.display_label)
            item.setData(Qt.ItemDataRole.UserRole, student.student_id)
            self.student_list.addItem(item)
        if self.student_list.count():
            self.student_list.setCurrentRow(0)

    def _accept_selection(self) -> None:
        item = self.student_list.currentItem()
        if not item:
            return
        self._selected_student_id = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def selected_student_id(self) -> Optional[str]:
        return self._selected_student_id


class RenameDialog(QDialog):
    def __init__(self, predefined_names: List[str], current_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Переименование кластера")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)
        info_label = QLabel("Выберите имя из списка или введите новое:")
        layout.addWidget(info_label)
        self.combo_box = QComboBox(self)
        self.combo_box.setEditable(True)
        if predefined_names: self.combo_box.addItems(predefined_names)
        self.combo_box.setCurrentText(current_name)
        self.combo_box.lineEdit().selectAll()
        layout.addWidget(self.combo_box)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    def get_selected_name(self) -> str: return self.combo_box.currentText().strip()

class FaceSelectorDialog(QDialog):
    def __init__(
        self,
        image_path: Path,
        faces: List,
        parent=None,
        instruction_text: Optional[str] = None,
        *,
        image_cache: QtImageCache,
        image_loader: AsyncImageLoader,
    ):
        super().__init__(parent)
        self.image_path = image_path
        self.faces = faces
        self.image_cache = image_cache
        self.image_loader = image_loader
        self._image_channels = set()
        self.selected_index = -1
        self.setWindowTitle("Выбор лица")
        self.setMinimumSize(600, 400)
        self.layout = QVBoxLayout(self)
        if instruction_text: final_text = f"Файл: <b>{image_path.name}</b><br><br>{instruction_text}"
        else: final_text = (f"На изображении <b>{image_path.name}</b> обнаружено несколько лиц.<br>"
                          "Выберите, какое из них соответствует целевому кластеру:")
        info_lbl = QLabel(final_text)
        info_lbl.setWordWrap(True)
        info_lbl.setTextFormat(Qt.TextFormat.RichText) 
        self.layout.addWidget(info_lbl)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setGridSize(QSize(170, 210))
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setItemDelegate(FaceItemDelegate(self.list_widget))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.layout.addWidget(self.list_widget)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        self.layout.addWidget(btn_box)
        self.image_loader.imageReady.connect(self._on_image_ready)
        self._load_faces()

    def _load_faces(self):
        source_size = self.image_cache.source_size(self.image_path)
        for i, face in enumerate(self.faces):
            item = QListWidgetItem(f"Лицо #{i + 1}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_widget.addItem(item)
            request = face_thumbnail_request(
                self.image_cache,
                self.image_path,
                face.bbox,
                (150, 150),
                padding=0.2,
                variant="cluster_editor.face_selector.v3",
                source_size=source_size,
            )
            if request is None:
                item.setText(f"Лицо #{i + 1} (ошибка координат)")
                continue
            channel = ("face-selector", id(self), i)
            self._image_channels.add(channel)
            self.image_loader.request(
                request,
                channel=channel,
                persist=True,
                disk_format="PNG",
            )

    @Slot(object)
    def _on_image_ready(self, result: AsyncImageResult) -> None:
        channel = result.channel
        if channel not in self._image_channels:
            return
        self._image_channels.discard(channel)
        face_index = channel[2]
        if result.image.isNull() or not 0 <= face_index < self.list_widget.count():
            return
        item = self.list_widget.item(face_index)
        item.setData(FACE_PIXMAP_ROLE, QPixmap.fromImage(result.image))
        self.list_widget.viewport().update(self.list_widget.visualItemRect(item))

    def done(self, result: int) -> None:
        for channel in list(self._image_channels):
            self.image_loader.cancel(channel)
        self._image_channels.clear()
        try:
            self.image_loader.imageReady.disconnect(self._on_image_ready)
        except (RuntimeError, TypeError):
            pass
        super().done(result)

    def _on_item_double_clicked(self, item):
        self.selected_index = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_accept(self):
        if len(self.list_widget.selectedItems()) > 0:
            item = self.list_widget.selectedItems()[0]
            self.selected_index = item.data(Qt.ItemDataRole.UserRole)
            self.accept()
        else:
            if self.list_widget.count() > 0:
                self.selected_index = 0
                self.accept()
            else:
                self.reject()

    def get_selected_index(self) -> int:
        return self.selected_index
