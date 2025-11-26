# 1. БЛОК: editor_workers.py (ПОЛНЫЙ ИСПРАВЛЕННЫЙ КОД)
# ==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль, содержащий классы-воркеры для выполнения длительных операций
(загрузка галереи, экспорт) в фоновых потоках.
"""
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QPixmap

try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
except ImportError:
    Image, ImageDraw, ImageFont, ImageEnhance = None, None, None, None

IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Ошибка импорта: {e}", file=sys.stderr)

from .editor_styles import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)


class GalleryPrepareWorker(QObject):
    """
    В фоновом потоке подготавливает списки изображений для отображения в галерее.
    """
    prepared = Signal(list, list) # cached_items, uncached_tasks
    finished = Signal()

    def __init__(self, data_manager, mode_config: Dict, cluster_id: str, pixmap_cache: Dict):
        super().__init__()
        self.data_manager = data_manager
        self.mode_config = mode_config
        self.cluster_id = cluster_id
        self.pixmap_cache = pixmap_cache
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True

    def run(self):
        if self._is_interruption_requested:
            self.finished.emit()
            return
            
        files_to_show = self.data_manager.get_files_for_cluster(self.mode_config, self.cluster_id)
        
        cached_items = []
        uncached_tasks = []

        for filename in files_to_show:
            if self._is_interruption_requested:
                break
            
            if filename in self.pixmap_cache:
                cached_items.append({
                    "filename": filename,
                    "pixmap": self.pixmap_cache[filename]
                })
            else:
                uncached_tasks.append({
                    "filename": filename,
                    "cluster_id": self.cluster_id
                })
        
        if not self._is_interruption_requested:
            self.prepared.emit(cached_items, uncached_tasks)
            
        self.finished.emit()


class FileReaderWorker(QObject):
    """Читает файлы с диска последовательно в одном потоке, чтобы избежать I/O contention."""
    finished = Signal(list)

    def __init__(self, tasks: List[Dict]):
        super().__init__()
        self.tasks = tasks
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True

    def run(self):
        read_data_tasks = []
        for task in self.tasks:
            if self._is_interruption_requested:
                break
            
            full_path = task.get("full_path")
            if not full_path or not full_path.is_file():
                continue

            try:
                raw_data = full_path.read_bytes()
                new_task = {
                    "filename": task["filename"],
                    "cluster_id": task["cluster_id"],
                    "raw_data": raw_data
                }
                read_data_tasks.append(new_task)
            except IOError as e:
                logger.error(f"Ошибка чтения файла {full_path}: {e}")
        
        if not self._is_interruption_requested:
            self.finished.emit(read_data_tasks)


class GalleryLoadWorker(QObject):
    """
    Создает QPixmap из данных в памяти и кэширует их.
    Сообщает о прогрессе и о завершении со списком обработанных файлов.
    """
    progress_updated = Signal(int)
    finished = Signal(list) 

    def __init__(self, tasks: List[Dict], pixmap_cache: Dict):
        super().__init__()
        self.tasks = tasks
        self.pixmap_cache = pixmap_cache
        self.num_threads = os.cpu_count() or 4
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True
        logger.debug("Получен запрос на прерывание GalleryLoadWorker.")

    def _process_single_image(self, task: Dict) -> Optional[Dict]:
        if self._is_interruption_requested:
            return None
            
        filename = task["filename"]
        raw_data = task["raw_data"]

        pixmap = QPixmap()
        if not pixmap.loadFromData(raw_data):
            logger.error(f"Не удалось создать QPixmap из данных для {filename}")
            return None
            
        scaled_pixmap = pixmap.scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.pixmap_cache[filename] = scaled_pixmap
        
        return {"filename": filename}

    def run(self):
        processed_tasks = []
        processed_count = 0
        total_tasks = len(self.tasks)
        progress_step = max(1, total_tasks // 100) 

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._process_single_image, task): task for task in self.tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                if self._is_interruption_requested:
                    for f in future_to_task: f.cancel()
                    break
                try:
                    result = future.result()
                    if result:
                        processed_tasks.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Ошибка обработки pixmap для {task['filename']}: {e}")
                
                processed_count += 1
                
                if processed_count % progress_step == 0 or processed_count == total_tasks:
                    self.progress_updated.emit(processed_count)

        if not self._is_interruption_requested:
            if processed_count % progress_step != 0:
                 self.progress_updated.emit(processed_count)
            self.finished.emit(processed_tasks)
        else:
            self.finished.emit([])


class ExportWorker(QObject):
    """
    Обрабатывает изображения для экспорта (улучшает, накладывает сетку, текст)
    в фоновых потоках.
    """
    progress_updated = Signal(int)
    finished = Signal(str)

    def __init__(self, tasks: List[Dict], num_threads: int, enhancement_factors: Dict[str, float], apply_watermarks: bool):
        super().__init__()
        if Image is None:
            raise ImportError("Для экспорта необходима библиотека Pillow. Установите ее: pip install Pillow")
        self.tasks = tasks
        self.num_threads = num_threads
        self.enhancement_factors = enhancement_factors
        self.apply_watermarks = apply_watermarks
        self.watermark_cache = None
        self._lock = threading.Lock()

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


    def _process_single_task(self, task: Dict):
        """Обрабатывает одну задачу (одно изображение)."""
        if ImageEnhance is None:
            raise ImportError("Для улучшения изображений требуется Pillow (ImageEnhance).")

        source_path = task["source_path"]
        output_path = task["output_path"]
        child_name = task["child_name"]

        file_number_match = re.search(r'(\d{4})$', Path(source_path).stem)
        file_number = file_number_match.group(1) if file_number_match else "----"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path).convert("RGBA") as base_image:
            enhanced_image = base_image
            factors = self.enhancement_factors
            if factors.get("brightness", 1.0) != 1.0:
                enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(factors["brightness"])
            if factors.get("contrast", 1.0) != 1.0:
                enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(factors["contrast"])
            if factors.get("color", 1.0) != 1.0:
                enhanced_image = ImageEnhance.Color(enhanced_image).enhance(factors["color"])
            if factors.get("sharpness", 1.0) != 1.0:
                enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(factors["sharpness"])

            final_image = enhanced_image
            
            if self.apply_watermarks:
                watermark_grid = self._create_watermark(enhanced_image.size)
                final_image = Image.alpha_composite(enhanced_image, watermark_grid)

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
                    final_image.paste(rotated_txt_layer, (rand_x, rand_y), rotated_txt_layer)

                draw_main = ImageDraw.Draw(final_image)
                font_size = int(base_image.height / 15)
                
                while font_size > 10:
                    try:
                        font_main = ImageFont.truetype("arialbd.ttf", font_size)
                    except IOError:
                        font_main = ImageFont.load_default(size=font_size)
                    
                    name_bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
                    if (name_bbox[2] - name_bbox[0]) < base_image.width * 0.9: break
                    font_size -= 2
                
                name_bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
                name_width, name_height = name_bbox[2] - name_bbox[0], name_bbox[3] - name_bbox[1]
                name_pos = ((base_image.width - name_width) / 2, (base_image.height - name_height) / 2 + 340)
                #name_pos = ((base_image.width - name_width) / 2, (base_image.height - name_height) / 2 - 20)
                draw_main.text(name_pos, child_name, font=font_main, fill="white", stroke_width=3, stroke_fill="black")

                num_bbox = draw_main.textbbox((0, 0), file_number, font=font_main)
                num_width = num_bbox[2] - num_bbox[0]
                num_pos = ((base_image.width - num_width) / 2, name_pos[1] + name_height + 10)
                draw_main.text(num_pos, file_number, font=font_main, fill="white", stroke_width=3, stroke_fill="black")
            
            final_image.convert("RGB").save(output_path, "JPEG", quality=95)

    def run(self):
        total = len(self.tasks)
        processed_count = 0
        logger.info(f"\nЭкспорт портретных фотографии(й) с водяными знаками...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._process_single_task, task): task for task in self.tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    future.result()
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Ошибка экспорта файла {task['source_path'].name}: {e}")
                processed_count += 1
                self.progress_updated.emit(processed_count)
        logger.info(f"Экспорт завершен. Обработано {total} файла(ов).")
        
        if IS_MANAGED_RUN:
            export_status = 1
            pysm_context.set("var_jpg_move", export_status)
            logger.info(f"\nПеременная контекста <b>var_jpg_move</b> установлена в значение <b>{export_status}</b>.")
            
        self.finished.emit(f"Экспорт завершен. Обработано {total} файлов.")