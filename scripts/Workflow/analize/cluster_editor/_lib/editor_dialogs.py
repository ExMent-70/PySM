# analize/cluster_editor/_lib/editor_dialogs.py
"""
Модуль, содержащий кастомные диалоговые окна для редактора кластеров.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, QSlider, QFrame, QComboBox, QDialogButtonBox,
    QListWidget, QListWidgetItem,
    QSpinBox, QCheckBox, QGroupBox, QFormLayout # <--- Добавлены новые виджеты
)
from PySide6.QtGui import QPixmap, QPainter, QTransform, QWheelEvent, QIcon 
from PySide6.QtCore import Qt, Slot, QEvent, QSize


from . import editor_styles as styles

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_theme_api import set_widget_class    
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    IS_MANAGED_RUN = False

# --- ИЗМЕНЕНИЕ: Исправляем импорт Pillow ---
try:
    from PIL import Image, ImageEnhance, ImageOps # <--- Добавили ImageOps
    from PIL import ImageQt
    IS_PILLOW_AVAILABLE = True
except ImportError:
    IS_PILLOW_AVAILABLE = False
    Image = None
    ImageEnhance = None
    ImageQt = None
    ImageOps = None
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
        
        # Параметры экспорта
        self.original_size = (0, 0)
        self.original_dpi = 300
        self.is_fitted_in_view = False

        self.setWindowTitle("Настройка экспорта")
        self.setMinimumSize(1100, 750) # Немного увеличили окно

        self._load_original_image()
        self._init_ui()
        self._load_settings()
        self._update_preview()


    def _load_original_image(self):
        try:
            # Загружаем оригинал
            self.original_pil_image = Image.open(self.preview_image_path).convert("RGB")
            
            # Сохраняем исходные параметры
            self.original_size = self.original_pil_image.size
            info = self.original_pil_image.info
            self.original_dpi = int(info.get('dpi', (300, 300))[0]) # Берем X DPI
            
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
        
        # 1. Левая часть - Превью (без изменений)
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

        # 2. Правая часть - Настройки
        settings_container = QFrame()
        settings_container.setFixedWidth(320)
        settings_layout = QVBoxLayout(settings_container)

        # --- Кнопка До/После ---
        #toggle_preview_button = QPushButton("До/После (Зажать)")
        #toggle_preview_button.pressed.connect(self._show_original_preview)
        #toggle_preview_button.released.connect(self._update_preview)
        #settings_layout.addWidget(toggle_preview_button)
        
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setFrameShadow(QFrame.Shadow.Sunken)
        settings_layout.addWidget(line)

        # --- Слайдеры улучшения ---
        self.brightness_slider = self._create_slider("Яркость", "brightness")
        self.contrast_slider = self._create_slider("Контраст", "contrast")
        self.color_slider = self._create_slider("Насыщенность", "color")
        self.sharpness_slider = self._create_slider("Резкость", "sharpness")
        settings_layout.addWidget(self.brightness_slider["group"])
        settings_layout.addWidget(self.contrast_slider["group"])
        settings_layout.addWidget(self.color_slider["group"])
        settings_layout.addWidget(self.sharpness_slider["group"])

        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine); line2.setFrameShadow(QFrame.Shadow.Sunken)
        settings_layout.addWidget(line2)

# --- НАЧАЛО ИСПРАВЛЕННОГО БЛОКА (Правильный порядок) ---
        # Сначала создаем Группу и Лайаут
        export_group = QGroupBox("Параметры вывода")
        export_layout = QFormLayout(export_group)

        # 1. Качество JPEG (теперь добавляем в существующий export_layout)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSuffix("%")
        export_layout.addRow("Качество JPG:", self.quality_spin)

        # 2. DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(self.original_dpi)
        self.dpi_spin.setSuffix(" dpi")
        export_layout.addRow("Разрешение:", self.dpi_spin)

        # 3. Размер
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 20000)
        self.width_spin.setValue(self.original_size[0])
        self.width_spin.setSuffix(" px")
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 20000)
        self.height_spin.setValue(self.original_size[1])
        self.height_spin.setSuffix(" px")

        self.ratio_check = QCheckBox("Сохранять пропорции")
        self.ratio_check.setChecked(True)

        export_layout.addRow("Ширина:", self.width_spin)
        export_layout.addRow("Высота:", self.height_spin)
        export_layout.addRow("", self.ratio_check)
        
        # Логика пропорций
        self.aspect_ratio = self.original_size[0] / self.original_size[1] if self.original_size[1] > 0 else 1.0
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)

        # 4. Водяные знаки
        self.watermark_check = QCheckBox("Наложить водяные знаки")
        self.watermark_check.setChecked(True)
        export_layout.addRow("", self.watermark_check)
        
        # Добавляем готовую группу в основной лайаут
        settings_layout.addWidget(export_group)
# --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---
        apply_button = QPushButton("Применить и экспортировать")
        apply_button.setProperty("class", "primary") 
        #set_widget_class(apply_button, "primary")        
        apply_button.clicked.connect(self.accept)
        settings_layout.addWidget(apply_button)


        settings_layout.addStretch()

        # Кнопки
        # --- Кнопка До/После ---
        toggle_preview_button = QPushButton("До/После")
        toggle_preview_button.pressed.connect(self._show_original_preview)
        toggle_preview_button.released.connect(self._update_preview)
        toggle_preview_button.setProperty("class", "primary") 
        

        reset_button = QPushButton("Сбросить")
        cancel_button = QPushButton("Отмена")
        reset_button.clicked.connect(self._reset_sliders)
        cancel_button.clicked.connect(self.reject)
        
        
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.addWidget(toggle_preview_button)
        bottom_buttons_layout.addWidget(reset_button)
        bottom_buttons_layout.addWidget(cancel_button)

        settings_layout.addLayout(bottom_buttons_layout)

        main_layout.addWidget(preview_container, 1)
        main_layout.addWidget(settings_container)


# --- НАЧАЛО ИЗМЕНЕНИЯ: Логика изменения размеров ---
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
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


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
        # Загружаем настройки улучшения
        settings = self.RECOMMENDED_DEFAULTS
        if IS_MANAGED_RUN and pysm_context:
            settings = pysm_context.get("enhancer_settings", self.RECOMMENDED_DEFAULTS)
        
        for key, widget_dict in self._get_all_sliders().items():
            value = float(settings.get(key, self.RECOMMENDED_DEFAULTS.get(key, 1.0)))
            widget_dict["slider"].setValue(int(value * 100))

# --- НАЧАЛО ИЗМЕНЕНИЯ: Загрузка параметров экспорта ---
        # 1. Сбрасываем размеры на оригинал текущего фото (безопасно)
        self.width_spin.setValue(self.original_size[0])
        self.height_spin.setValue(self.original_size[1])
        
        # 2. Загружаем сохраненные параметры экспорта или ставим дефолт
        # DPI
        saved_dpi = settings.get("export_dpi", self.original_dpi)
        self.dpi_spin.setValue(int(saved_dpi))
        
        # Качество
        saved_quality = settings.get("export_quality", 95)
        self.quality_spin.setValue(int(saved_quality))
        
        # Чекбоксы
        saved_watermarks = settings.get("export_watermarks", True)
        self.watermark_check.setChecked(bool(saved_watermarks))
        
        saved_ratio = settings.get("export_keep_ratio", True)
        self.ratio_check.setChecked(bool(saved_ratio))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


    def accept(self):
        if IS_MANAGED_RUN and pysm_context:
# --- НАЧАЛО ИЗМЕНЕНИЯ: Сохранение всех параметров в контекст ---
            # Берем текущие коэффициенты слайдеров
            settings = self.enhancement_factors.copy()
            
            # Добавляем параметры экспорта
            settings.update({
                "export_dpi": self.dpi_spin.value(),
                "export_quality": self.quality_spin.value(),
                "export_watermarks": self.watermark_check.isChecked(),
                "export_keep_ratio": self.ratio_check.isChecked()
                # Размеры (width/height) не сохраняем, они зависят от фото
            })
            
            pysm_context.set("enhancer_settings", settings)
# --- КОНЕЦ ИЗМЕНЕНИЯ ---
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
        # Сброс слайдеров
        for key, widget_dict in self._get_all_sliders().items():
            default_value = self.RECOMMENDED_DEFAULTS.get(key, 1.0)
            widget_dict["slider"].setValue(int(default_value * 100))
        
# --- НАЧАЛО ИЗМЕНЕНИЯ: Сброс параметров экспорта ---
        # Сброс размеров и DPI на оригинальные значения файла
        self.width_spin.setValue(self.original_size[0])
        self.height_spin.setValue(self.original_size[1])
        self.dpi_spin.setValue(self.original_dpi)
        
        # Сброс остальных настроек на "заводские"
        self.quality_spin.setValue(95)
        self.watermark_check.setChecked(True)
        self.ratio_check.setChecked(True)
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

    def get_export_settings(self) -> Dict[str, Any]:
        """Возвращает словарь со всеми настройками (улучшение + размер + dpi + качество + водяные знаки)."""
# --- НАЧАЛО ИЗМЕНЕНИЯ: Возврат новых параметров ---
        return {
            "factors": self.enhancement_factors,
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "dpi": self.dpi_spin.value(),
            "quality": self.quality_spin.value(),
            "watermarks": self.watermark_check.isChecked()
        }
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

    #def get_enhancement_factors(self) -> Dict[str, float]:
    #    return self.enhancement_factors

    def _get_all_sliders(self) -> Dict[str, Dict]:
        return {"brightness": self.brightness_slider, "contrast": self.contrast_slider,
                "color": self.color_slider, "sharpness": self.sharpness_slider}
                
                

class RenameDialog(QDialog):
    """
    Кастомный диалог для переименования кластера с использованием
    редактируемого выпадающего списка.
    """
    def __init__(self, predefined_names: List[str], current_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Переименование кластера")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        info_label = QLabel("Выберите имя из списка или введите новое:")
        layout.addWidget(info_label)

        self.combo_box = QComboBox(self)
        self.combo_box.setEditable(True)
        if predefined_names:
            self.combo_box.addItems(predefined_names)
        self.combo_box.setCurrentText(current_name)
        self.combo_box.lineEdit().selectAll() # Выделяем текст для удобства

        layout.addWidget(self.combo_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)

    def get_selected_name(self) -> str:
        """Возвращает итоговый текст из QComboBox."""
        return self.combo_box.currentText().strip()
        
# --- НАЧАЛО НОВОГО КЛАССА ---
class FaceSelectorDialog(QDialog):
    """
    Диалог для выбора конкретного лица на фотографии, если их найдено несколько.
    Показывает миниатюры всех обнаруженных лиц.
    """
    def __init__(self, image_path: Path, faces: List, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.faces = faces
        self.selected_index = -1
        
        self.setWindowTitle("Выберите лицо для перемещения")
        self.setMinimumSize(600, 400)
        
        self.layout = QVBoxLayout(self)
        
        # Инструкция
        info_lbl = QLabel(f"На изображении <b>{image_path.name}</b> обнаружено несколько лиц.<br>"
                          "Выберите, какое из них соответствует целевому кластеру:")
        self.layout.addWidget(info_lbl)
        
        # Список лиц (в виде иконок)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.layout.addWidget(self.list_widget)
        
        # Кнопки
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        self.layout.addWidget(btn_box)
        
        self._load_faces()

    def _load_faces(self):
        if not IS_PILLOW_AVAILABLE or not Image:
            for i, face in enumerate(self.faces):
                item = QListWidgetItem(f"Лицо #{i+1}")
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.list_widget.addItem(item)
            return

        try:
            # 1. Загружаем изображение
            # ВАЖНО: Убрали ImageOps.exif_transpose. 
            # Если координаты в JSON от OpenCV/dlib, они обычно по сырому файлу.
            full_img = Image.open(str(self.image_path)).convert("RGBA")
            img_w, img_h = full_img.size

            for i, face in enumerate(self.faces):
                bbox = face.bbox
                if len(bbox) == 4:
                    # --- ИЗМЕНЕНИЕ: Меняем порядок распаковки ---
                    # Было: top, right, bottom, left (face_recognition style)
                    # Стало: left, top, right, bottom (Common/PIL style)
                    # Попробуем этот формат, так как он чаще вызывает такие "сдвиги" при ошибке.
                    # Если у вас точно face_recognition, вернем обратно, но уберем transpose.
                    
                    # План А: Пробуем стандартный порядок [left, top, right, bottom]
                    v1, v2, v3, v4 = map(int, bbox)
                    
                    # Эвристика для определения формата:
                    # Обычно right > left и bottom > top.
                    # Если bbox[1] > bbox[3] (как в top, right, bottom, left где right > left), 
                    # то значения могут быть перепутаны.
                    
                    # Давайте попробуем универсальный подход:
                    # Pillow требует (left, top, right, bottom)
                    
                    # Вариант 1: JSON хранит [left, top, right, bottom]
                    x1, y1, x2, y2 = v1, v2, v3, v4
                    
                    # Вариант 2: Если JSON хранит [top, right, bottom, left], раскомментируйте это:
                    # y1, x2, y2, x1 = v1, v2, v3, v4 

                    # Валидация и исправление координат
                    # Гарантируем x1 < x2 и y1 < y2
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1

                    # Отступ 20%
                    width = x2 - x1
                    height = y2 - y1
                    padding = int(max(width, height) * 0.2)
                    
                    # Clamping (обрезка по границам фото)
                    final_x1 = max(0, x1 - padding)
                    final_y1 = max(0, y1 - padding)
                    final_x2 = min(img_w, x2 + padding)
                    final_y2 = min(img_h, y2 + padding)
                    
                    # Проверка на вырожденность
                    if final_x2 <= final_x1 or final_y2 <= final_y1:
                        self.list_widget.addItem(f"Лицо #{i+1} (Ошибка координат)")
                        continue

                    face_img = full_img.crop((final_x1, final_y1, final_x2, final_y2))
                    
                    if ImageQt:
                        qim = ImageQt.ImageQt(face_img)
                        pixmap = QPixmap.fromImage(qim)
                        icon = QIcon(pixmap)
                        
                        item = QListWidgetItem(f"Лицо {i+1}")
                        item.setIcon(icon)
                        item.setData(Qt.ItemDataRole.UserRole, i)
                        self.list_widget.addItem(item)
                    else:
                        self.list_widget.addItem(f"Лицо {i+1}")

        except Exception as e:
            logger.error(f"Ошибка при нарезке лиц для диалога: {e}")
            self.list_widget.clear()
            for i in range(len(self.faces)):
                item = QListWidgetItem(f"Лицо #{i+1} (Ошибка)")
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.list_widget.addItem(item)

    def _on_item_double_clicked(self, item):
        self.selected_index = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_accept(self):
        if len(self.list_widget.selectedItems()) > 0:
            item = self.list_widget.selectedItems()[0]
            self.selected_index = item.data(Qt.ItemDataRole.UserRole)
            self.accept()
        else:
            # Если ничего не выбрано, но нажали ОК - берем первое (или можно запретить)
            if self.list_widget.count() > 0:
                self.selected_index = 0
                self.accept()
            else:
                self.reject()

    def get_selected_index(self) -> int:
        return self.selected_index
# --- КОНЕЦ НОВОГО КЛАССА ---        