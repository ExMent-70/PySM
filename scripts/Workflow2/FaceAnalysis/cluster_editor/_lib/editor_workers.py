# analize/cluster_editor/_lib/editor_workers.py

import logging
import os
import concurrent.futures
import re
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QImage

# Импорт общей логики
try:
    from PIL import Image, ImageDraw, ImageFont
    from .image_processing import apply_color_corrections, create_watermark_layer
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None
    apply_color_corrections, create_watermark_layer = None, None

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

from .editor_delegates import THUMBNAIL_SIZE


logger = logging.getLogger(__name__)

TEXT_VERTICAL_ANCHOR_PCT = 0.85 
TEXT_LINE_SPACING = 15 

def run_export_task(task_data: Dict[str, Any]) -> str:
    """Функция обработки (выполняется в отдельном процессе)."""
    if Image is None: return "Pillow not installed"

    source_path = task_data["source_path"]
    output_path = task_data["output_path"]
    child_name = task_data["child_name"]
    raw_faces_bboxes = task_data.get("faces_bboxes", [])
    
    factors = task_data.get("factors", {})
    target_size = task_data.get("target_size")
    target_dpi = task_data.get("target_dpi", (300, 300))
    quality = task_data.get("quality", 95)
    apply_watermarks = task_data.get("apply_watermarks", False)

    try:
        source_path_obj = Path(source_path)
        output_path_obj = Path(output_path)
        
        file_number_match = re.search(r'(\d{4})$', source_path_obj.stem)
        file_number = file_number_match.group(1) if file_number_match else "----"

        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as src_img:
            # Конвертируем в RGBA для безопасной обработки слоев
            base_image = src_img.convert("RGBA")
            original_w, original_h = base_image.size

            # 1. ЦВЕТОКОРРЕКЦИЯ (Общая функция)
            enhanced_image = apply_color_corrections(base_image, factors)

            # 2. РЕСАЙЗ
            if target_size and target_size != enhanced_image.size:
                enhanced_image = enhanced_image.resize(target_size, Image.Resampling.LANCZOS)

            final_image = enhanced_image
            target_w, target_h = final_image.size

            # 3. ВОДЯНЫЕ ЗНАКИ
            if apply_watermarks:
                # Масштабируем BBox'ы лиц под новый размер
                scaled_bboxes = []
                if target_size:
                    scale_x = target_w / original_w
                    scale_y = target_h / original_h
                    for bbox in raw_faces_bboxes:
                        if len(bbox) == 4:
                            sb = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
                            scaled_bboxes.append(sb)
                else:
                    scaled_bboxes = raw_faces_bboxes

                # Генерируем слой (Общая функция)
                watermark_layer = create_watermark_layer((target_w, target_h), scaled_bboxes, factors, child_name)
                
                if watermark_layer:
                    final_image = Image.alpha_composite(final_image, watermark_layer)

                # ТЕКСТ (Имя и номер) - это рисуется только при экспорте, не в превью
                draw_main = ImageDraw.Draw(final_image)
                font_size = int(target_h / 15)
                font_main = None
                try: font_main = ImageFont.truetype("arialbd.ttf", font_size)
                except IOError:
                    try: font_main = ImageFont.truetype("arial.ttf", font_size)
                    except IOError: font_main = ImageFont.load_default()

                # Подгон размера шрифта имени
                if hasattr(font_main, 'getbbox'): 
                    while font_size > 10:
                        bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
                        if (bbox[2] - bbox[0]) < target_w * 0.9: 
                            break
                        font_size -= 2
                        try: font_main = ImageFont.truetype("arialbd.ttf", font_size)
                        except: pass

                # Позиционирование
                try:
                    name_bbox = draw_main.textbbox((0, 0), child_name, font=font_main)
                    name_w = name_bbox[2] - name_bbox[0]
                    name_h = name_bbox[3] - name_bbox[1]
                    
                    num_bbox = draw_main.textbbox((0, 0), file_number, font=font_main)
                    num_w = num_bbox[2] - num_bbox[0]
                    
                    # textsize legacy check omitted for brevity as PIL 10+ uses textbbox
                except AttributeError:
                    name_w, name_h = draw_main.textsize(child_name, font=font_main)
                    num_w, num_h = draw_main.textsize(file_number, font=font_main)

                anchor_y = int(target_h * TEXT_VERTICAL_ANCHOR_PCT)
                text_top_y = anchor_y - ((name_h + TEXT_LINE_SPACING + name_h) // 2) # approx height
                
                name_pos = ((target_w - name_w) / 2, text_top_y)
                num_pos = ((target_w - num_w) / 2, text_top_y + name_h + TEXT_LINE_SPACING)

                draw_main.text(name_pos, child_name, font=font_main, fill="white", stroke_width=3, stroke_fill="black")
                draw_main.text(num_pos, file_number, font=font_main, fill="white", stroke_width=3, stroke_fill="black")
            
            # 4. СОХРАНЕНИЕ
            final_image.convert("RGB").save(output_path_obj, "JPEG", quality=quality, dpi=target_dpi)
            return "OK"

    except Exception as e:
        return f"Error processing {Path(source_path).name}: {traceback.format_exc()}"


class ChunkedImageLoader(QObject):
    # Оставляем без изменений (только импорты PIL обновились глобально)
    chunk_ready = Signal(list) 
    progress_updated = Signal(int)
    finished = Signal()

    def __init__(self, tasks: List[Dict], pixmap_cache: Dict, num_threads: int):
        super().__init__()
        self.tasks = tasks
        self.pixmap_cache = pixmap_cache
        self.num_threads = min(num_threads, (os.cpu_count() or 4) * 2)

        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True

    def _load_single_image(self, task: Dict) -> Optional[Tuple[str, QImage]]:
        if self._is_interruption_requested: return None
        full_path = task.get("full_path")
        cache_key = task.get("cache_key", task["filename"])
        bbox = task.get("bbox")
        draw_rect = task.get("draw_face_rect", False)
        if not full_path or not full_path.exists(): return None
        try:
            if bbox and Image:
                with Image.open(str(full_path)) as pil_img:
                    img_w, img_h = pil_img.size
                    x1, y1, x2, y2 = map(int, bbox)
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1
                    face_w = x2 - x1; face_h = y2 - y1
                    pad = int(max(face_w, face_h) * 0.5)
                    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                    cx2 = min(img_w, x2 + pad); cy2 = min(img_h, y2 + pad)
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pil_img.crop((cx1, cy1, cx2, cy2))
                        if draw_rect:
                            if crop.mode != "RGBA": crop = crop.convert("RGBA")
                            draw = ImageDraw.Draw(crop)
                            rx1 = x1 - cx1; ry1 = y1 - cy1
                            rx2 = rx1 + face_w; ry2 = ry1 + face_h
                            for w in range(3):
                                draw.rectangle([rx1-w, ry1-w, rx2+w, ry2+w], outline=(255, 165, 0, 255))
                        if crop.mode != "RGBA": crop = crop.convert("RGBA")
                        data = crop.tobytes("raw", "RGBA")
                        qim = QImage(data, crop.width, crop.height, QImage.Format.Format_RGBA8888).copy()
                        thumb = qim.scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        return (cache_key, thumb)
            image = QImage()
            if not image.load(str(full_path)): return None
            thumb = image.scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            return (cache_key, thumb)
        except Exception as e:
            logger.error(f"Error loading {full_path}: {e}")
            return None

    def run(self):
        total_tasks = len(self.tasks)
        processed_count = 0
        current_batch_size = 5 
        max_batch_size = 20
        batch_buffer = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._load_single_image, task): task for task in self.tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                if self._is_interruption_requested: break
                try:
                    result = future.result()
                    if result: batch_buffer.append(result)
                except Exception: pass
                processed_count += 1
                if len(batch_buffer) >= current_batch_size:
                    self.chunk_ready.emit(batch_buffer)
                    batch_buffer = []
                    self.progress_updated.emit(processed_count)
                    if current_batch_size < max_batch_size: current_batch_size += 5
        if batch_buffer and not self._is_interruption_requested:
            self.chunk_ready.emit(batch_buffer)
            self.progress_updated.emit(processed_count)
        self.finished.emit()

class ExportWorker(QObject):
    progress_updated = Signal(int)
    finished = Signal(str)
    def __init__(self, tasks: List[Dict], num_threads: int, 
                 enhancement_factors: Dict, target_size: Tuple[int, int], 
                 target_dpi: Tuple[int, int], quality: int, apply_watermarks: bool):
        super().__init__()
        self.tasks = tasks
        self.num_processes = min(num_threads, os.cpu_count() or 4)
        self.common_settings = {
            "factors": enhancement_factors,
            "target_size": target_size,
            "target_dpi": target_dpi,
            "quality": quality,
            "apply_watermarks": apply_watermarks
        }
    def run(self):
        total = len(self.tasks)
        processed_count = 0
        watermarks_text = ""
        if self.common_settings["apply_watermarks"]:
            watermarks_text = " с водяными знаками"
        logger.info(f"\n{icon_info} Экспорт <b>{total}</b> файла(ов){watermarks_text}...")
        prepared_tasks = []
        for task in self.tasks:
            full_task = task.copy()
            full_task.update(self.common_settings)
            full_task["source_path"] = str(full_task["source_path"])
            full_task["output_path"] = str(full_task["output_path"])
            prepared_tasks.append(full_task)
        errors = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(run_export_task, task_data) for task_data in prepared_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res != "OK": errors.append(res)
                except Exception as e: errors.append(str(e))
                processed_count += 1
                self.progress_updated.emit(processed_count)
        if errors:
            logger.error(f"Errors during export ({len(errors)}):")
            for e in errors[:5]: logger.error(e)
        self.finished.emit(f"Экспорт завершен. Обработано {total} файлов. Ошибок: {len(errors)}.")