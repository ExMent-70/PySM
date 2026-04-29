# --- Блок 1: Импорты стандартных и сторонних библиотек ---
# ==============================================================================
import argparse
import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Protocol

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
    project_root = current_script_path.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder

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
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".dng"}
PSD_EXTENSIONS = {".psd", ".psb"}

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Подавление спама ("Unknown image resource...") от библиотеки psd_tools
logging.getLogger("psd_tools").setLevel(logging.ERROR)


# --- Блок 4: Конфигурация ---
# ==============================================================================
def str2bool(value: str) -> bool:
    """Вспомогательная функция для безопасного парсинга булевых флагов из UI."""
    return str(value).lower() in ("yes", "true", "t", "1")


def get_config() -> argparse.Namespace:
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
    
    # Новые аргументы для управления рекурсией
    parser.add_argument(
        "--recursive_read", 
        type=str2bool, 
        nargs="?", 
        const=True, 
        default=False,
        help="Включить рекурсивный поиск файлов во вложенных папках."
    )
    parser.add_argument(
        "--keep_structure", 
        type=str2bool, 
        nargs="?", 
        const=True, 
        default=False,
        help="Сохранять структуру вложенных папок при сохранении."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


# --- Блок 5: Паттерн "Стратегия" для извлечения изображений ---
# ==============================================================================
class ImageExtractor(Protocol):
    """Интерфейс для классов извлечения изображений."""
    def extract(self, source_path: Path) -> Optional[Image.Image]:
        ...

class RawExtractor:
    """Извлекает превью из RAW файлов."""
    def extract(self, source_path: Path) -> Optional[Image.Image]:
        try:
            with source_path.open('rb') as f:
                with rawpy.imread(f) as raw:
                    thumb = raw.extract_thumb()
                    
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        img = Image.open(io.BytesIO(thumb.data))
                        return ImageOps.exif_transpose(img)
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        return Image.fromarray(thumb.data)
                    else:
                        raise ValueError("Неизвестный формат превью.")
        except rawpy.LibRawNoThumbnailError:
            raise ValueError("В RAW-файле отсутствует превью.")
        except rawpy.LibRawUnsupportedThumbnailError:
            raise ValueError("Формат превью не поддерживается rawpy.")

class PsdExtractor:
    """Извлекает изображения из PSD/PSB файлов."""
    def extract(self, source_path: Path) -> Optional[Image.Image]:
        with source_path.open('rb') as f:
            psd = PSDImage.open(f)
            pil_image = psd.topil()
            if pil_image and pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image


# --- Блок 6: Класс ImageExporter ---
# ==============================================================================
class ImageExporter:
    """
    Инкапсулирует логику масштабирования и сохранения изображений.
    Использует стратегии извлечения на основе расширения файла.
    """
    def __init__(self, jpeg_quality: int, preview_size: int):
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality должно быть в диапазоне от 0 до 100.")
        
        self.jpeg_quality = jpeg_quality
        self.preview_size = preview_size
        
        # Реестр стратегий (Open/Closed Principle)
        self.extractors = dict()
        for ext in RAW_EXTENSIONS:
            self.extractors[ext] = RawExtractor()
        for ext in PSD_EXTENSIONS:
            self.extractors[ext] = PsdExtractor()

    def export_file(
        self, 
        source_path: Path, 
        input_dir: Path, 
        output_dir: Path, 
        keep_structure: bool
    ) -> Tuple[bool, str]:
        """Оркестратор: извлекает, ресайзит и сохраняет файл."""
        try:
            output_jpeg_path = self._build_output_path(
                source_path, input_dir, output_dir, keep_structure
            )
        except Exception as e:
            return False, f"Ошибка при формировании пути для {source_path.name}: {e}"

        suffix = source_path.suffix.lower()
        extractor = self.extractors.get(suffix)

        if not extractor:
            return False, f"Пропущен (неподдерживаемый формат): {source_path.name}"

        try:
            # Получаем изображение через соответствующую стратегию
            pil_image = extractor.extract(source_path)
            
            if not pil_image:
                return False, f"Не удалось извлечь изображение из {source_path.name}"

            # Гарантированное закрытие ресурса (защита от утечек памяти)
            try:
                return self._resize_and_save(pil_image, source_path.name, output_jpeg_path)
            finally:
                if hasattr(pil_image, 'close'):
                    pil_image.close()

        except Exception as e:
            return False, f"Ошибка при обработке {source_path.name}: {e}"

    def _build_output_path(
        self, 
        source_path: Path, 
        input_dir: Path, 
        output_dir: Path, 
        keep_structure: bool
    ) -> Path:
        """Вычисляет целевой путь сохранения на основе флагов рекурсии."""
        if keep_structure:
            rel_path = source_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".jpg")
            # Безопасно создаем вложенные папки, если их нет
            out_path.parent.mkdir(parents=True, exist_ok=True)
            return out_path
        
        # Плоская структура
        return output_dir / f"{source_path.stem}.jpg"

    def _resize_and_save(self, pil_image: Image.Image, source_name: str, output_path: Path) -> Tuple[bool, str]:
        """Выполняет масштабирование и сохранение изображения в JPEG."""
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


# --- Блок 7: Вспомогательные функции (Бизнес-логика и UI) ---
# ==============================================================================
def get_files_to_process(input_dir: Path, allowed_exts: set, recursive: bool) -> list:
    """Собирает список файлов для обработки на основе заданных параметров."""
    iterator = input_dir.rglob('*') if recursive else input_dir.iterdir()
    return[f for f in iterator if f.is_file() and f.suffix.lower() in allowed_exts]

def build_ui_report(input_dir: Path, output_dir: Path) -> None:
    """Формирует и выводит HTML-отчет для графического интерфейса PySM."""
    if not (IS_MANAGED_RUN and pysm_context):
        return
    
    tv_builder = StandardTreeBuilder(icon_size=28)
    root_node_input = ResourceNode("Исходная<br>папка", input_dir, "folder", "Папка с исходными RAW/PSD файлами")
    root_node_output = ResourceNode("Целевая<br>папка", output_dir, "folder", "Папка с конвертированными JPG-файлами")
    tv_builder.add_section("<br>Рабочие папки и файлы",[root_node_input, root_node_output])
    pysm_context.log_html(tv_builder.get_html())


# --- Блок 8: Выполнение скрипта (Оркестратор) ---
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

    # 3. Определение списка файлов
    allowed_extensions = set()
    if config.a_r2j_image_type in ["raw", "all"]:
        allowed_extensions.update(RAW_EXTENSIONS)
    if config.a_r2j_image_type in ["psd", "all"]:
        allowed_extensions.update(PSD_EXTENSIONS)
    
    files_to_process = get_files_to_process(input_dir, allowed_extensions, config.recursive_read)

    if not files_to_process:
        logger.info("Не найдено файлов для обработки с указанными расширениями. Завершение работы.")
        return

    logger.info(
        f"Найдено файлов для конвертации: <b>{len(files_to_process)} "
        f"({config.a_r2j_image_type})</b>"
    )
    logger.debug(f"Используется потоков (threads): {config.all_threads}")
    logger.debug(f"Рекурсивное чтение: {config.recursive_read}")
    logger.debug(f"Рекурсивное сохранение (структура): {config.keep_structure}")

    # 4. Запуск многопоточной обработки
    exporter = ImageExporter(
        jpeg_quality=config.a_r2j_jpeg_quality,
        preview_size=config.a_r2j_preview_size
    )

    success_count = 0
    error_count = 0
    is_interrupted = False

    executor = ThreadPoolExecutor(max_workers=config.all_threads)
    futures = dict()
    
    for file_path in files_to_process:
        future = executor.submit(
            exporter.export_file, 
            source_path=file_path, 
            input_dir=input_dir, 
            output_dir=output_dir, 
            keep_structure=config.keep_structure
        )
        futures[future] = file_path

    try:
        with tqdm(total=len(files_to_process), desc="Экспорт в JPEG") as progress_bar:
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
        executor.shutdown(wait=False, cancel_futures=True)

    # 5. Формирование отчета
    build_ui_report(input_dir, output_dir)
    
    if not IS_MANAGED_RUN:
        logger.info("ЭКСПОРТ ЗАВЕРШЕН")
        logger.info(f"Успешно: {success_count}")
        logger.info(f"Ошибок: {error_count}")

    if error_count > 0 or is_interrupted:
        sys.exit(1)


# --- Блок 9: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()