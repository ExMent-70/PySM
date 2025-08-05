# analize/cluster_editor/_lib/editor_workers.py
"""
Модуль, содержащий классы-воркеры для выполнения длительных операций
(загрузка галереи, экспорт) в фоновых потоках.
"""
import logging
import os
import random
import re
import shutil  # <-- ИЗМЕНЕНИЕ: Добавлен импорт
import sys
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QPixmap

# --- ИЗМЕНЕНИЕ: Исправлен блок импорта Pillow ---
try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageEnhance = None # <-- Добавлена недостающая строка
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

# Импортируем стили только для тайп-хинтинга
from .editor_styles import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)


class MoveWorker(QObject):
    """
    Выполняет физическое перемещение файлов в фоновом потоке.
    """
    task_finished = Signal(int, int) 
    finished = Signal()

    def __init__(self, tasks: List[Dict], main_window_ref):
        super().__init__()
        self.tasks = tasks
        self.main_window = main_window_ref
        self.num_threads = os.cpu_count() or 4

    def _process_single_move(self, task: Dict) -> Tuple[int, int]:
        """Обрабатывает перемещение для одного `filename`."""
        filename = task["filename"]
        source_id = task["source_id"]
        target_id = task["target_id"]
        
        moved_count = 0
        error_count = 0

        # --- ИЗМЕНЕНИЕ: Используем правильный метод _find_file ---
        source_path = self.main_window._find_file_globally(filename)
        if not source_path:
            logger.warning(f"Не удалось найти исходный путь для {filename} при перемещении.")
            return 0, 1

        source_dir = source_path.parent
        target_dir = self.main_window._get_folder_for_cluster_id(target_id)

        if not target_dir:
            logger.warning(f"Не удалось определить целевую папку для ID {target_id}.")
            return 0, 1

        if source_dir.resolve() == target_dir.resolve():
            return 0, 0

        file_stem = Path(filename).stem
        related_source_files = list(source_dir.glob(f"{file_stem}.*"))

        for source_file in related_source_files:
            target_file_path = target_dir / source_file.name
            try:
                shutil.move(str(source_file), str(target_file_path))
                moved_count += 1
            except Exception as e:
                error_count += 1
                # --- ИЗМЕНЕНИЕ: Убран некорректный аргумент ---
                logger.error(f"Не удалось переместить {source_file.name}: {e}")

        return moved_count, error_count

    def run(self):
        """Запускает многопоточное перемещение."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._process_single_move, task): task for task in self.tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    moved, errors = future.result()
                    self.task_finished.emit(moved, errors)
                except Exception as e:
                    task = future_to_task[future]
                    # --- ИЗМЕНЕНИЕ: Убран некорректный аргумент ---
                    logger.error(f"Критическая ошибка в потоке перемещения для {task['filename']}: {e}")
                    self.task_finished.emit(0, 1)
        
        self.finished.emit()


class GalleryLoadWorker(QObject):
    """
    Загружает изображения и создает для них QPixmap в фоновых потоках.
    """
    widget_ready = Signal(str, str, Path, QPixmap)
    finished = Signal()

    def __init__(self, tasks: List[Dict]):
        super().__init__()
        self.tasks = tasks
        self.num_threads = os.cpu_count() or 4
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True
        logger.debug("Получен запрос на прерывание GalleryLoadWorker.")

    def _process_single_image(self, task: Dict) -> Optional[Tuple[str, str, Path, QPixmap]]:
        if self._is_interruption_requested:
            return None
            
        filename = task["filename"]
        cluster_id = task["cluster_id"]
        full_path = task["full_path"]

        if full_path.is_file():
            pixmap = QPixmap(str(full_path))
            if pixmap.isNull():
                raise IOError(f"Не удалось загрузить QPixmap для {full_path}")
            scaled_pixmap = pixmap.scaled(
                THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            return filename, cluster_id, full_path, scaled_pixmap
        return None

    def run(self):
        """Запускает многопоточную загрузку pixmap'ов."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._process_single_image, task): task for task in self.tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                if self._is_interruption_requested:
                    logger.debug("Загрузка галереи прервана в цикле.")
                    for f in future_to_task: f.cancel()
                    break
                try:
                    result = future.result()
                    if result:
                        self.widget_ready.emit(*result)
                except Exception as e:
                    task = future_to_task[future]
                    # --- ИЗМЕНЕНИЕ: Убран некорректный аргумент ---
                    logger.error(f"Ошибка загрузки pixmap для {task['filename']}: {e}")
        self.finished.emit()


class ExportWorker(QObject):
    """
    Обрабатывает изображения для экспорта (улучшает, накладывает сетку, текст)
    в фоновых потоках.
    """
    progress_updated = Signal(int)
    finished = Signal(str)

# --- ИЗМЕНЕННЫЙ МЕТОД: __init__ ---
    def __init__(self, tasks: List[Dict], num_threads: int, enhancement_factors: Dict[str, float]):
        super().__init__()
        if Image is None:
            raise ImportError("Для экспорта необходима библиотека Pillow. Установите ее: pip install Pillow")
        self.tasks = tasks
        self.num_threads = num_threads
        self.enhancement_factors = enhancement_factors  # Сохраняем коэффициенты
        self.watermark_cache = None
        self._lock = threading.Lock()  # Лок для потокобезопасного доступа к кэшу



    def _create_watermark(self, size: Tuple[int, int]) -> Image.Image:
        with self._lock:
            if self.watermark_cache and self.watermark_cache.size == size:
                return self.watermark_cache
            width, height = size
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            spacing = 100
            for i in range(-width, width + height, spacing):
                draw.line([(i, 0), (i - height, height)], fill=(255, 0, 0, 90), width=2)
                draw.line([(i, 0), (i + height, height)], fill=(255, 0, 0, 90), width=2)
            self.watermark_cache = image
            return image

# --- ИЗМЕНЕННЫЙ МЕТОД: _process_single_task ---
    def _process_single_task(self, task: Dict):
        """Обрабатывает одну задачу (одно изображение)."""
        if ImageEnhance is None:  # Проверка, что модуль загрузился
            raise ImportError("Для улучшения изображений требуется Pillow (ImageEnhance).")

        source_path = task["source_path"]
        output_path = task["output_path"]
        child_name = task["child_name"]

        file_number_match = re.search(r'(\d{4})$', Path(source_path).stem)
        file_number = file_number_match.group(1) if file_number_match else "----"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path).convert("RGBA") as base_image:

            # --- БЛОК 1: Автоматическое улучшение изображения ---
            enhanced_image = base_image
            factors = self.enhancement_factors

            # Применяем только те улучшения, которые отличаются от значения по умолчанию (1.0)
            if factors.get("brightness", 1.0) != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(factors["brightness"])
            if factors.get("contrast", 1.0) != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(factors["contrast"])
            if factors.get("color", 1.0) != 1.0:
                enhancer = ImageEnhance.Color(enhanced_image)
                enhanced_image = enhancer.enhance(factors["color"])
            if factors.get("sharpness", 1.0) != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(factors["sharpness"])

            # --- БЛОК 2: Накладываем красную сетку (на улучшенное изображение) ---
            watermark = self._create_watermark(enhanced_image.size)
            combined = Image.alpha_composite(enhanced_image, watermark)

            # --- БЛОК 3: Накладываем надпись "Выбор фото" ---
            try:
                font_watermark = ImageFont.truetype("calibri.ttf", int(base_image.height / 18))
            except IOError:
                font_watermark = ImageFont.load_default()
            
            watermark_text = "Выбор фото"
            wm_bbox = font_watermark.getbbox(watermark_text)
            wm_width, wm_height = wm_bbox[2] - wm_bbox[0], wm_bbox[3] - wm_bbox[1]
            max_len = int((wm_width**2 + wm_height**2)**0.5) + 2 
            
            for _ in range(12):
                txt_layer = Image.new("RGBA", (max_len, max_len), (255, 255, 255, 0))
                draw_txt = ImageDraw.Draw(txt_layer)
                draw_txt.text(
                    ((max_len - wm_width) / 2, (max_len - wm_height) / 2), 
                    watermark_text, font=font_watermark, fill=(255, 255, 255, 85)
                )
                angle = random.randint(-45, 45)
                rotated_txt_layer = txt_layer.rotate(angle, expand=False, resample=Image.BICUBIC)
                rand_x = random.randint(0, int(base_image.width * 0.8))
                rand_y = random.randint(0, int(base_image.height * 0.8))
                combined.paste(rotated_txt_layer, (rand_x, rand_y), rotated_txt_layer)

            # --- БЛОК 4: Накладываем основную информацию (имя и номер) ---
            draw_main = ImageDraw.Draw(combined)
            font_size = int(base_image.height / 15)
            
            while font_size > 10:
                try:
                    font_main = ImageFont.truetype("arialbd.ttf", font_size)
                except IOError:
                    font_main = ImageFont.load_default(size=font_size)
                
                name_bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
                name_width = name_bbox[2] - name_bbox[0]
                if name_width < base_image.width * 0.9:
                    break
                font_size -= 2
            
            name_bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
            name_width, name_height = name_bbox[2] - name_bbox[0], name_bbox[3] - name_bbox[1]
            name_pos = ((base_image.width - name_width) / 2, (base_image.height - name_height) / 2 - 20)
            draw_main.text(name_pos, child_name, font=font_main, fill="white", stroke_width=3, stroke_fill="black")

            num_bbox = draw_main.textbbox((0, 0), file_number, font=font_main)
            num_width = num_bbox[2] - num_bbox[0]
            num_pos = ((base_image.width - num_width) / 2, name_pos[1] + name_height + 10)
            draw_main.text(num_pos, file_number, font=font_main, fill="white", stroke_width=3, stroke_fill="black")

            combined.convert("RGB").save(output_path, "JPEG", quality=95)

    def run(self):
        """Запускает многопоточный экспорт."""
        total = len(self.tasks)
        processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._process_single_task, task): task for task in self.tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    future.result()
                except Exception as e:
                    task = future_to_task[future]
                    # --- ИЗМЕНЕНИЕ: Убран некорректный аргумент ---
                    logger.error(f"Ошибка экспорта файла {task['source_path'].name}: {e}")

                processed_count += 1
                self.progress_updated.emit(processed_count)

        self.finished.emit(f"Экспорт завершен. Обработано {total} файлов.")