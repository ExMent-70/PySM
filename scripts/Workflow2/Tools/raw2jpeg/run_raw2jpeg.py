# --- Блок 1: Импорты стандартных и сторонних библиотек ---
# ==============================================================================
import argparse
import io
import logging
import os
import sys
# ИЗМЕНЕНИЕ: Возвращаем ThreadPoolExecutor вместо ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Попытка импорта необходимых библиотек для обработки изображений
try:
    import numpy as np
    import rawpy
    from PIL import Image, ImageOps
    from psd_tools import PSDImage
except ImportError as e:
    print(f"Критическая ошибка: Необходимая библиотека не найдена. {e}", file=sys.stderr)
    print("Пожалуйста, установите зависимости: pip install numpy rawpy Pillow psd-tools", file=sys.stderr)
    sys.exit(1)

# --- Блок 2: Настройка системного пути и импорт PySM ---
# ==============================================================================
try:
    current_script_path = Path(__file__).resolve()
    # Предполагаем, что структура папок .../analize/export_to_jpeg/
    project_root = current_script_path.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder

    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else None

# --- Блок 3: Константы и настройка логирования ---
# ==============================================================================
RAW_EXTENSIONS =[".arw", ".cr2", ".cr3", ".nef", ".dng"]
PSD_EXTENSIONS = [".psd", ".psb"]

# Убрали HTML из формата, чтобы логи в консоли читались чисто
logging.basicConfig(
    level=logging.INFO,
    #format="[%(levelname)s] %(message)s",
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Блок 4: Конфигурация ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Определяет аргументы командной строки и разрешает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(description="Конвертация RAW/PSD файлов в JPEG.")
    
    parser.add_argument(
        "--all_threads", 
        type=int, 
        default=os.cpu_count() or 4, 
        help="Количество процессов для обработки (по умолчанию: авто)."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Путь к исходной директории с файлами RAW/PSD."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Путь к директории для сохранения итоговых JPEG."
    )
    parser.add_argument(
        "--a_r2j_image_type", 
        type=str, 
        dest="a_r2j_image_type", 
        default="raw",
        choices=["raw", "psd", "all"], 
        help="Тип файлов для обработки."
    )
    parser.add_argument(
        "--a_r2j_preview_size", 
        type=int, 
        dest="a_r2j_preview_size", 
        default=4096,
        help="Максимальный размер длинной стороны для итогового JPEG."
    )
    parser.add_argument(
        "--a_r2j_jpeg_quality", 
        type=int, 
        dest="a_r2j_jpeg_quality", 
        default=95,
        help="Качество сохранения JPEG (0-100)."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


# --- Блок 5: Класс ImageExporter ---
# ==============================================================================
class ImageExporter:
    """
    Инкапсулирует логику конвертации и сохранения изображений в формат JPEG.
    Оптимизирован для работы с кириллическими путями и минимизации потребления памяти.
    """
    def __init__(self, jpeg_quality: int, preview_size: int):
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality должно быть в диапазоне от 0 до 100.")
        self.jpeg_quality = jpeg_quality
        self.preview_size = preview_size

    def export_file(self, source_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """Оркестратор: определяет тип файла, извлекает изображение и сохраняет его."""
        output_jpeg_path = output_dir / f"{source_path.stem}.jpg"
        suffix = source_path.suffix.lower()

        try:
            # 1. Извлечение изображения в объект PIL.Image
            if suffix in RAW_EXTENSIONS:
                pil_image = self._extract_raw_preview(source_path)
            elif suffix in PSD_EXTENSIONS:
                pil_image = self._extract_psd_preview(source_path)
            else:
                return False, f"Пропущен (неподдерживаемый формат): {source_path.name}"

            if not pil_image:
                return False, f"Не удалось извлечь изображение из {source_path.name}"

            # 2. Обработка (ресайз) и сохранение
            return self._resize_and_save(pil_image, source_path.name, output_jpeg_path)

        except Exception as e:
            return False, f"Ошибка при обработке {source_path.name}: {e}"

    def _extract_raw_preview(self, source_path: Path) -> Optional[Image.Image]:
            """Извлекает встроенное превью из RAW-файла с учетом EXIF-ориентации."""
            try:
                with source_path.open('rb') as f:
                    with rawpy.imread(f) as raw:
                        thumb = raw.extract_thumb()
                        
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                            # ДОБАВЛЕНО: Автоматический поворот на основе EXIF-тега Orientation
                            return ImageOps.exif_transpose(img)
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            # BITMAP обычно уже ориентирован корректно декодером LibRaw
                            return Image.fromarray(thumb.data)
                        else:
                            raise ValueError("Неизвестный формат превью.")
            except rawpy.LibRawNoThumbnailError:
                raise ValueError("В RAW-файле отсутствует превью.")
            except rawpy.LibRawUnsupportedThumbnailError:
                raise ValueError("Формат превью не поддерживается rawpy.")

    def _extract_psd_preview(self, source_path: Path) -> Optional[Image.Image]:
        """Извлекает изображение из PSD-файла."""
        with source_path.open('rb') as f:
            psd = PSDImage.open(f)
            pil_image = psd.topil()
            
            if pil_image and pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image

    def _resize_and_save(self, pil_image: Image.Image, source_name: str, output_path: Path) -> Tuple[bool, str]:
        """Выполняет масштабирование и сохранение изображения в JPEG."""
        try:
            width, height = pil_image.size
            long_side = max(width, height)

            if long_side > self.preview_size:
                scale = self.preview_size / long_side
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            icc_profile = pil_image.info.get("icc_profile")

            pil_image.save(
                output_path,
                format="JPEG",
                quality=self.jpeg_quality,
                dpi=(300, 300),
                icc_profile=icc_profile,
            )
            return True, f"{source_name} -> {output_path.name}"
        finally:
            if hasattr(pil_image, 'close'):
                pil_image.close()


# --- Блок 6: Выполнение скрипта (Оркестратор) ---
# ==============================================================================
def main():
    logger.info("<b>ЭКСПОРТ RAW/PSD В JPEG</b><br>")

    # 1. Получение конфигурации
    config = get_config()
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)

    # 2. Валидация путей
    if not input_dir.is_dir():
        logger.critical(f"Ошибка: Исходная папка не найдена или не является директорией: {input_dir}")
        sys.exit(1)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Ошибка: Не удалось создать выходную папку {output_dir}: {e}")
        sys.exit(1)

    logger.debug(f"Исходная папка: {input_dir.resolve()}")
    logger.debug(f"Выходная папка: {output_dir.resolve()}")

    # 3. Определение списка файлов для обработки
    allowed_extensions =[]
    if config.a_r2j_image_type in ["raw", "all"]:
        allowed_extensions.extend(RAW_EXTENSIONS)
    if config.a_r2j_image_type in ["psd", "all"]:
        allowed_extensions.extend(PSD_EXTENSIONS)
    
    files_to_process =[
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    if not files_to_process:
        logger.info("Не найдено файлов для обработки с указанными расширениями. Завершение работы.")
        return

    logger.info(f"Найдено файлов для конвертации: <b>{len(files_to_process)} ({config.a_r2j_image_type})</b>")
    logger.debug(f"Используется потоков (threads): {config.all_threads}")

    # 4. Запуск многопоточной обработки
    exporter = ImageExporter(
        jpeg_quality=config.a_r2j_jpeg_quality,
        preview_size=config.a_r2j_preview_size
    )

    success_count = 0
    error_count = 0
    is_interrupted = False

    # ИЗМЕНЕНИЕ: Используем ThreadPoolExecutor. Потоки мгновенно умирают при закрытии PySM
    executor = ThreadPoolExecutor(max_workers=config.all_threads)
    futures = {
        executor.submit(exporter.export_file, file_path, output_dir): file_path
        for file_path in files_to_process
    }

    try:
        with tqdm(total=len(files_to_process), desc="Экспорт в JPEG") as progress_bar:
            # Использование timeout=1.0 заставляет цикл регулярно "просыпаться"
            # и проверять системные сигналы (такие как нажатие кнопки Стоп)
            for future in as_completed(futures, timeout=None):
                path = futures[future]
                try:
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(message)
                except Exception as e:
                    error_count += 1
                    logger.error(f"Критическая ошибка при обработке файла {path.name}: {e}")
                finally:
                    progress_bar.update(1)
                    progress_bar.set_postfix(ok=success_count, failed=error_count)

    except KeyboardInterrupt:
        logger.warning("\n[ВНИМАНИЕ] Процесс был принудительно остановлен пользователем!")
        is_interrupted = True
        for future in futures.keys():
            future.cancel()

    finally:
        # Корректное гашение пула потоков
        executor.shutdown(wait=False, cancel_futures=True)

    if IS_MANAGED_RUN and pysm_context:
        tv_builder = StandardTreeBuilder(icon_size=28)
        root_node_input = ResourceNode("Исходная<br>папка", Path(input_dir), "folder", "Папка с исходными RAW/PSD файлами")
        root_node_output = ResourceNode("Целевая<br>папка", Path(output_dir), "folder", "Папка с конвертированными JPG-файлами ")
        tv_builder.add_section("<br>Рабочие папки и файлы", [root_node_input, root_node_output])
        # 4. Вывод
        pysm_context.log_html(tv_builder.get_html())
    else:
        logger.info("ЭКСПОРТ ЗАВЕРШЕН")
        logger.info(f"Успешно: {success_count}")
        logger.info(f"Ошибок: {error_count}")

    if error_count > 0 or is_interrupted:
        sys.exit(1)

# --- Блок 7: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    # В Windows для многопроцессорности обязательно использование freeze_support,
    # если скрипт компилируется в exe, но для обычного скрипта хватит if __name__ == '__main__':
    main()