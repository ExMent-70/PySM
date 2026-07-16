
import logging
import os
import concurrent.futures
import re
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any

from PySide6.QtCore import QObject, Signal

# Импорт общей логики
try:
    from PIL import Image, ImageDraw, ImageFont
    from .image_processing import apply_color_corrections, create_watermark_layer
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None
    apply_color_corrections, create_watermark_layer = None, None

from _common import (
    icon_info,
)

logger = logging.getLogger(__name__)

TEXT_VERTICAL_ANCHOR_PCT = 0.85 
TEXT_LINE_SPACING = 15 


class DataLoadWorker(QObject):
    """Load and parse one data manager outside the GUI thread."""

    finished = Signal(bool, str)

    def __init__(self, data_manager) -> None:
        super().__init__()
        self.data_manager = data_manager

    def run(self) -> None:
        try:
            success, message = self.data_manager.load_data()
        except Exception as exc:
            success, message = False, str(exc)
        self.finished.emit(success, message)

def run_export_task(task_data: Dict[str, Any]) -> str:
    """Функция обработки (выполняется в отдельном процессе)."""
    if Image is None: return "Pillow not installed"

    source_path = task_data["source_path"]
    output_path = task_data["output_path"]
    student_name = task_data["student_name"]
    raw_faces_bboxes = task_data.get("faces_bboxes", list())
    
    factors = task_data.get("factors", dict())
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
                scaled_bboxes = list()
                if target_size:
                    scale_x = target_w / original_w
                    scale_y = target_h / original_h
                    for bbox in raw_faces_bboxes:
                        if len(bbox) == 4:
                            sb =[bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
                            scaled_bboxes.append(sb)
                else:
                    scaled_bboxes = raw_faces_bboxes

                # Генерируем слой (Общая функция)
                watermark_layer = create_watermark_layer(
                    (target_w, target_h), scaled_bboxes, factors, student_name
                )
                
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
                        bbox = draw_main.textbbox((0, 0), student_name, font=font_main)
                        if (bbox[2] - bbox[0]) < target_w * 0.9: 
                            break
                        font_size -= 2
                        try: font_main = ImageFont.truetype("arialbd.ttf", font_size)
                        except OSError:
                            pass

                # Позиционирование
                try:
                    name_bbox = draw_main.textbbox((0, 0), student_name, font=font_main)
                    name_w = name_bbox[2] - name_bbox[0]
                    name_h = name_bbox[3] - name_bbox[1]
                    
                    num_bbox = draw_main.textbbox((0, 0), file_number, font=font_main)
                    num_w = num_bbox[2] - num_bbox[0]
                except AttributeError:
                    name_w, name_h = draw_main.textsize(student_name, font=font_main)
                    num_w, num_h = draw_main.textsize(file_number, font=font_main)

                anchor_y = int(target_h * TEXT_VERTICAL_ANCHOR_PCT)
                text_top_y = anchor_y - ((name_h + TEXT_LINE_SPACING + name_h) // 2)
                
                name_pos = ((target_w - name_w) / 2, text_top_y)
                num_pos = ((target_w - num_w) / 2, text_top_y + name_h + TEXT_LINE_SPACING)

                draw_main.text(
                    name_pos, student_name, font=font_main, fill="white",
                    stroke_width=3, stroke_fill="black"
                )
                draw_main.text(num_pos, file_number, font=font_main, fill="white", stroke_width=3, stroke_fill="black")
            
            descriptor, temporary_name = tempfile.mkstemp(
                dir=output_path_obj.parent,
                prefix=f".{output_path_obj.name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            try:
                final_image.convert("RGB").save(
                    temporary_path,
                    "JPEG",
                    quality=quality,
                    dpi=target_dpi,
                )
                os.replace(temporary_path, output_path_obj)
            finally:
                temporary_path.unlink(missing_ok=True)
            return "OK"

    except Exception as e:
        return f"Error processing {Path(source_path).name}: {traceback.format_exc()}"


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
        
        self._is_interruption_requested = False
        self._executor = None
        self._futures = []

    def request_interruption(self) -> None:
        """Cancel queued jobs; running processes finish before ``finished``."""
        self._is_interruption_requested = True
        for future in list(self._futures):
            future.cancel()

    def run(self):
        total = len(self.tasks)
        processed_count = 0
        errors = []
        try:
            if self._is_interruption_requested:
                return

            watermarks_text = (
                " с водяными знаками"
                if self.common_settings["apply_watermarks"]
                else ""
            )
            logger.info(
                f"\n{icon_info} Экспорт <b>{total}</b> файла(ов)"
                f"{watermarks_text}..."
            )

            prepared_tasks = []
            for task in self.tasks:
                full_task = {**task, **self.common_settings}
                full_task["source_path"] = str(full_task["source_path"])
                full_task["output_path"] = str(full_task["output_path"])
                prepared_tasks.append(full_task)

            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_processes
            )
            self._futures = [
                self._executor.submit(run_export_task, task_data)
                for task_data in prepared_tasks
            ]

            for future in concurrent.futures.as_completed(self._futures):
                if self._is_interruption_requested:
                    break
                try:
                    res = future.result()
                    if res != "OK":
                        errors.append(res)
                except Exception as exc:
                    errors.append(str(exc))

                processed_count += 1
                self.progress_updated.emit(processed_count)

        except Exception as exc:
            errors.append(f"Ошибка инфраструктуры экспорта: {exc}")
            logger.exception("Экспорт аварийно остановлен")
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                errors.append(f"Ошибка завершения пула экспорта: {exc}")
                logger.exception("Не удалось корректно завершить пул экспорта")
            self._executor = None
            self._futures = []

            if errors:
                logger.error(f"Errors during export ({len(errors)}):")
                for error in errors[:5]:
                    logger.error(error)

            if self._is_interruption_requested:
                message = (
                    f"Экспорт прерван. Обработано {processed_count} "
                    f"из {total} файлов."
                )
            else:
                message = (
                    f"Экспорт завершен. Обработано {processed_count} "
                    f"из {total} файлов. Ошибок: {len(errors)}."
                )
            self.finished.emit(message)
